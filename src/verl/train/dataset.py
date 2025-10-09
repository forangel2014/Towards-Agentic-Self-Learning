# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import copy
import os
import re
from io import BytesIO
from typing import Optional

import datasets
from loguru import logger
from omegaconf import DictConfig, ListConfig
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .data import BaseMMProcessor, MMProcessorContext, init_mm_processor


def extract_gsm8k_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


def process_image(image: dict | ImageObject, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math

    if isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))

    assert isinstance(image, ImageObject), f"image type: {type(image)}"

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def _filter_overlong_prompts(
    tokenizer,
    mm_tool: BaseMMProcessor,
    chat,
    images: list,
    max_prompt_length: int,
    max_images: int,
    image_resolution: Optional[int],
    fast: bool = True,
) -> bool:
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers.models.qwen2_vl import Qwen2VLProcessor

    from src.models import is_agivlm

    # drop if
    # 1. have image and exceed max_images
    # 2. vlm model but have no image ( avoid vlm zero3 hang )
    image_num = 0
    if images is None:
        images = []
    for img in images:
        # FIXME(wuhuan): 潜规则，直接吧tar里面的图片数写到文件名里，避免读取
        if isinstance(img, str):
            its = img.split("/")[-1].split(".")
            if img.endswith(".tar") and len(its) == 3 and its[-2].isdigit():
                image_num += int(its[-2])
                continue
        image_num += 1

    # 只有多模模型且指定 max_images 时才需要按图片数过滤
    drop_by_image = image_num > max_images if (max_images and mm_tool.is_multimodal()) else False
    if drop_by_image:
        logger.warning(f"Drop sample by image length: {image_num} > {max_images}")
        return False

    if (
        not (is_agivlm(mm_tool.processor) or isinstance(mm_tool.processor, Qwen2_5_VLProcessor | Qwen2VLProcessor))
        and image_num == 0
        and mm_tool.is_multimodal()
    ):
        # FIXME(wuhuan): 支持文本+图片混合训练，加 dummy image，会和 remove_padding 冲突因为没传 attn mask
        # qwen 和 agi 的实现在 verl/models/transformers/monkey_patch.py apply_monkey_patch_for_dummy_image 中
        logger.warning("Drop sample with no image which may cause zero3 hanging")
        return False

    prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    if mm_tool.is_multimodal():
        prompt_with_chat_template, placeholder_num = mm_tool.strip_image_placeholder(prompt_with_chat_template)
        if image_num != placeholder_num:
            logger.warning(f"Drop sample by image placeholder mismatch: {image_num=} != {placeholder_num=}")
            return False

    length = len(tokenizer(prompt_with_chat_template, add_special_tokens=False)["input_ids"])
    if image_num and mm_tool.is_multimodal():
        ctx = MMProcessorContext(image_paths=images)
        mm_tool.process(ctx, required_fields=["mm_placeholders"])
        assert ctx.mm_placeholders is not None, "mm_placeholders should not be None after MMProcessorContext.process"
        length += sum([len(tokenizer.encode(p)) for p in ctx.mm_placeholders])

    drop_by_prompt = length > max_prompt_length
    if drop_by_prompt:
        logger.warning(f"Drop sample by prompt length: {length} > {max_prompt_length}")
        return False

    return True


class RLHFDataset(Dataset):
    """We assume the dataset contains a column that contains prompts and other
    information."""

    def __init__(
        self,
        parquet_files: str | list[str],
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin] = None,
    ):
        from src.models import is_agivlm1_6

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser("~/.cache/verl/rlhf")
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = config.data.prompt_key
        self.image_key = config.data.image_key
        self.max_prompt_length = config.data.max_prompt_length
        self.return_raw_chat = config.data.return_raw_chat
        self.truncation = config.data.truncation
        self.image_resolution = config.data.image_resolution
        self.image_limit = config.actor_rollout_ref.rollout.limit_images
        self.image_placeholder = config.data.image_placeholder
        self.recompute_mm = config.actor_rollout_ref.model.get("recompute_mm", False)
        self.offset = getattr(config.data, "offset", None)
        self.default_system = getattr(config.data, "default_system", None)

        if not is_agivlm1_6(processor) and self.recompute_mm:
            raise ValueError(
                "recompute_mm is only supported for AgiVlm <= 1.6, please use it or set recompute_mm=False"
            )
        self.mm_tool = init_mm_processor(tokenizer, processor, self.image_placeholder)

        self.asycn_rollout = config.actor_rollout_ref.rollout.mode == "async"

        self.filter_overlong_prompts = config.data.filter_overlong_prompts
        self.filter_num_workers = config.data.filter_overlong_prompts_workers
        self.filter_cache = config.data.filter_overlong_prompts_cache

        assert "image" in config.data.image_placeholder
        mm_braces = tuple(config.data.image_placeholder.split("image"))
        assert len(mm_braces) == 2, f"braces should be a tuple of length 2, but got {mm_braces}"
        if processor is not None:
            processor.mm_braces = mm_braces

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _refine_row(self, row_dict: dict) -> dict:
        if self.image_key in row_dict:
            images = row_dict[self.image_key]
            if images is None:
                images = []
            elif isinstance(images, dict | str | ImageObject):
                images = [images]
            assert isinstance(images, list), f"images should be a list, but got {type(images)}"
            row_dict[self.image_key] = images

        chat = row_dict[self.prompt_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        assert isinstance(chat, list), f"chat should be a list, but got {type(chat)}"
        if chat[0]["role"] != "system" and self.default_system:
            chat.insert(0, {"role": "system", "content": self.default_system})
        row_dict[self.prompt_key] = chat
        return row_dict

    def _filter(self, row_dict) -> bool:
        row_dict = self._refine_row(row_dict)
        return _filter_overlong_prompts(
            self.tokenizer,
            self.mm_tool,
            row_dict[self.prompt_key],
            row_dict.get(self.image_key, []),
            self.max_prompt_length,
            self.image_limit or 0,
            image_resolution=self.image_resolution,
            fast=True,
        )

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)
        logger.info(f"dataset len: {len(self.dataframe)}")
        # filter out too long prompts
        logger.info(f"filter_overlong_prompts: {self.filter_overlong_prompts}")

        if self.offset:
            self.dataframe = self.dataframe.select(range(self.offset, len(self.dataframe)))
            logger.info(f"offset dataset len: {len(self.dataframe)}")

        if self.filter_overlong_prompts:
            self.dataframe = self.dataframe.filter(
                lambda doc: self._filter(doc),
                num_proc=self.filter_num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
                load_from_cache_file=self.filter_cache,
            )
            logger.info(f"filter dataset len: {len(self.dataframe)}")

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, "original_parquet_files") else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            logger.warning(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """Note that we also return the raw_input_ids so that it can be
        combined with other chat template."""
        row_dict: dict = self.dataframe[item]
        if self.image_key in row_dict and row_dict[self.image_key] is None:
            row_dict[self.image_key] = []
        row_dict = self._refine_row(row_dict)

        messages = row_dict.pop(self.prompt_key)

        if self.image_key in row_dict and row_dict[self.image_key] is None:
            row_dict[self.image_key] = []

        images = row_dict.pop(self.image_key, [])
        res = self.mm_tool.process_raw_messages(
            raw_messages=messages,
            images=images,
            max_prompt_length=self.max_prompt_length,
            truncation=self.truncation,
            enable_mm_inputs=not self.recompute_mm,
            enable_openai_messages=False,
            image_process_params=dict(image_resolution=self.image_resolution),
        )

        for k in [
            "input_ids",
            "attention_mask",
            "position_ids",
            "raw_prompt_ids",
            "multi_modal_data",
            "origin_multi_modal_data",
            "multi_modal_inputs",
            "raw_prompt",
        ]:
            if k in res:
                row_dict[k] = res[k]

        prompt_length = res["attention_mask"].sum()
        row_dict["input_id_list"] = res["input_ids"][-prompt_length:].tolist()

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state
        return self.__dict__.copy()
