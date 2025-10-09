import base64
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from io import BytesIO
from typing import (
    Any,
    Optional,
)

import torch
from PIL import Image, ImageFile
from PIL.Image import Image as ImageObject
from pillow_heif import register_heif_opener
from transformers import PreTrainedTokenizer, ProcessorMixin

register_heif_opener()

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_PLACEHOLDER = "<image>"


@dataclass
class MMProcessorContext:
    # input ###################################################################

    # 带 image placeholder(<image>) 的最原始的输入 message
    # [{"role": "user", "content": "<image>描述这张图片"},...]
    messages_raw: Optional[list[dict]] = None
    # 图片路径或 PIL object 或 url 列表
    image_paths: Optional[list[Any]] = None
    # 是否计算 multi_modal_inputs （比如 pixel_values 等），耗时且会占内存
    enable_mm_inputs: bool = True
    # 是否生成 openai 格式的 messages，此时图片会冗余一份 base64 插入到 messages 里，耗时且占内存
    enable_openai_messages: bool = False
    # 图片处理参数，目前支持 image_resolution, image_max_pixels+image_min_pixels,
    image_process_params: Optional[dict[str, Any]] = None

    # output ##################################################################

    # 不带 <image>, image token 不展开 <|vision_start|><|image_pad|><|vision_end|>，可以直接喂到 VLLM 中
    # DEPRECATED: 暂时不需要，只要 prompt 即可
    _messages: Optional[list[dict[str, Any]]] = None

    # 对 messages 做 apply_chat_template 后的 prompt
    prompt: Optional[str] = None

    # 原始 PIL 格式图片
    raw_images: Optional[list[ImageObject]] = None
    # clip/resize 处理后的 PIL 格式图片
    processed_images: Optional[list[ImageObject]] = None
    # mm_inputs: pixel_values, image_thw_grid
    mm_inputs: Optional[dict[str, Any]] = None

    # 不带 <image>, image token 展开 <|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>，用于训练
    _messages_expand_mm: Optional[list[dict[str, Any]]] = None
    # 对 messages_expand_mm 做 apply_chat_template + encode
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None

    # 不带 image token，openai 格式 message，可以作为请求调用 vllm
    # {"role": "user", "content": [{"type": "image_url", "image_url": ""},{text},...]
    messages_openai: Optional[Optional[list[dict[str, Any]]]] = None

    # 展开的 mm token placeholder，每个 <image> 对应 N 个 image token (含 start/end)
    # ["<|vision_start|><|image_pad|>...<|image_pad|><|vision_end|>"] x 图片数
    mm_placeholders: Optional[list[str]] = None


class BaseMMProcessor:
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        image_placeholder: str = IMAGE_PLACEHOLDER,
        **kwargs,
    ) -> None:
        from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
        from transformers.models.qwen2_vl import Qwen2VLProcessor

        from redaccel.models import is_agivlm1_6, is_agivlm1_7, is_agivlm1_8

        self.tokenizer = tokenizer
        self.processor = processor
        self.image_placeholder = image_placeholder
        self.mm_braces = tuple(image_placeholder.split("image"))
        assert len(self.mm_braces) == 2, f"braces should be a tuple of length 2, but got {self.mm_braces}"

        if "template" in kwargs and kwargs["template"]:
            template_name = kwargs["template"]
        elif processor is None:
            template_name = None
        elif is_agivlm1_6(processor):
            template_name = "AgiVlm"
        elif is_agivlm1_7(processor) or is_agivlm1_8(processor):
            template_name = "AgiVlm1_7"
        elif isinstance(processor, Qwen2VLProcessor | Qwen2_5_VLProcessor):
            template_name = "qwen2_vl"
        else:
            raise NotImplementedError(f"Processor {processor} is not supported for multimodal processing.")
        self.template_name = template_name

    def is_multimodal(self) -> bool:
        return self.processor is not None

    def process(self, ctx: MMProcessorContext, required_fields: Optional[list[str]] = None):
        raise NotImplementedError

    def strip_image_placeholder(self, prompt: str) -> tuple[str, int]:
        placeholder_num = len(re.findall(rf"{self.mm_braces[0]}image(?:_\d+)?{self.mm_braces[1]}", prompt))
        new_prompt = re.sub(rf"{self.mm_braces[0]}image(?:_\d+)?{self.mm_braces[1]}", "", prompt)
        return new_prompt, placeholder_num

    def process_raw_messages(
        self,
        raw_messages: list,
        images: list,
        max_prompt_length: int,
        truncation: str,
        enable_mm_inputs: bool = False,
        enable_openai_messages: bool = False,
        image_process_params: Optional[dict] = None,
    ) -> dict:
        import verl.utils.torch_functional as verl_F
        from verl.utils.dataset.rl_dataset import compute_position_id_with_mask

        res = {}
        mm_ctx = None
        if images and self.is_multimodal():
            mm_ctx = MMProcessorContext(
                image_paths=images,
                image_process_params=image_process_params,
                messages_raw=raw_messages,
                enable_mm_inputs=enable_mm_inputs,
                enable_openai_messages=enable_openai_messages,
            )

            self.process(mm_ctx)
            assert isinstance(mm_ctx.processed_images, list)
            res["multi_modal_data"] = {"image": mm_ctx.processed_images}
            res["origin_multi_modal_data"] = {"image": mm_ctx.raw_images}
            res["multi_modal_inputs"] = {} if mm_ctx.mm_inputs is None else dict(mm_ctx.mm_inputs)

            assert mm_ctx.prompt, "mm_ctx.prompt should not be None, please check the image_process_params."
            raw_prompt = mm_ctx.prompt
            input_ids = mm_ctx.input_ids
            attention_mask = mm_ctx.attention_mask
        else:
            if self.is_multimodal():
                res["multi_modal_data"] = {"image": []}
                res["origin_multi_modal_data"] = {"image": []}
                res["multi_modal_inputs"] = {}

            raw_prompt = self.tokenizer.apply_chat_template(raw_messages, add_generation_prompt=True, tokenize=False)
            assert isinstance(raw_prompt, str), f"expected str but got {type(raw_prompt)}"
            input_data = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = input_data.pop("input_ids")
            attention_mask = input_data.pop("attention_mask")

        assert isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor), (
            f"input_ids type: {type(input_ids)}, attention_mask type: {type(attention_mask)}"
        )

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        if (
            "multi_modal_inputs" in res
            and "image_grid_thw" in res["multi_modal_inputs"]
            and self.template_name in ("qwen2_vl",)
        ):
            # NOTE(wuhuan): dpskvlm 也是 Qwen2VLProcessor 但不是 mrope，所以不需要处理
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=res["multi_modal_inputs"]["image_grid_thw"],
                second_per_grid_ts=res["multi_modal_inputs"].get("second_per_grid_ts"),
                attention_mask=attention_mask[0],
            )  # (3, seq_len)

            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)
            if self.template_name in ("qwen2_vl",):
                position_ids = position_ids.view(1, 1, -1).expand(1, 3, -1)

        res["input_ids"] = input_ids[0]
        res["attention_mask"] = attention_mask[0]
        res["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        res["raw_prompt_ids"] = raw_prompt_ids
        res["raw_prompt"] = raw_messages
        res["chat_messages"] = raw_messages if mm_ctx is None else mm_ctx.messages_openai

        return res


class RedAccelMMProcessor(BaseMMProcessor):
    SUPPORTED_MM_TEMPLATES = ("AgiVlm", "AgiVlm1_7", "qwen2_vl")

    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        image_placeholder: str = IMAGE_PLACEHOLDER,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            image_placeholder=image_placeholder,
            **kwargs,
        )
        from ...tuner.data.template import TEMPLATES

        self.template = TEMPLATES.get(self.template_name or "empty")
        assert self.template is not None, f"Template {self.template_name} not found."
        self.mm_plugin = self.template.mm_plugin
        # NOTE(wuhuan): 图片先手动调用 _preprocess_image 处理好，无需后续重复处理
        self.mm_plugin.do_preprocess_image = False

    def process(self, ctx: MMProcessorContext, required_fields: Optional[list[str]] = None):
        def _check_required_fields() -> bool:
            if required_fields is None:
                return False

            for f in required_fields:
                if getattr(ctx, f, None) is None:
                    return False
            return True

        if ctx.image_paths:
            ctx.raw_images = self._load_images(ctx.image_paths)
        if _check_required_fields():
            return

        if ctx.image_process_params and ctx.raw_images:
            processed_images = [
                self.mm_plugin._preprocess_image(
                    img,
                    **ctx.image_process_params,
                )
                for img in ctx.raw_images
            ]
            ctx.processed_images = processed_images
        else:
            ctx.processed_images = ctx.raw_images

        if _check_required_fields():
            return

        if self.processor is not None and ctx.processed_images:
            if self.template_name in self.SUPPORTED_MM_TEMPLATES:
                ctx.mm_placeholders = self.mm_plugin._get_mm_placeholders(ctx.processed_images, self.processor)
            else:
                raise NotImplementedError(f"MM plugin {self.mm_plugin} is not supported.")
        if _check_required_fields():
            return

        if self.processor is not None and ctx.enable_mm_inputs and ctx.processed_images:
            ctx.mm_inputs = dict(self.mm_plugin.get_mm_inputs(ctx.processed_images, [], [], [], [], self.processor))
        else:
            ctx.mm_inputs = {}
        if _check_required_fields():
            return

        if self.processor is not None and ctx.messages_raw is not None:
            ctx._messages = []
            ctx.messages_openai = []
            image_idxs = list(range(len(ctx.processed_images or [])))
            for message_ in ctx.messages_raw:
                message = deepcopy(message_)

                if self.template_name in self.SUPPORTED_MM_TEMPLATES:
                    mm_repl = self.mm_plugin._get_mm_placeholders([None], self.processor, expand_mm_tokens=False)[0]
                else:
                    raise NotImplementedError(f"MM plugin {self.mm_plugin} is not supported.")
                message["content"] = message["content"].replace(self.image_placeholder, mm_repl)
                ctx._messages.append(message)

                if not ctx.enable_openai_messages:
                    continue

                # openai
                openai_message = deepcopy(message_)
                image_list = []
                new_content = ""
                for segment in re.split(f"({self.image_placeholder})", openai_message["content"]):
                    if segment == self.image_placeholder:
                        # NOTE(wuhuan): 填充 image base64
                        image_idx = image_idxs.pop(0)
                        buffered = BytesIO()
                        assert ctx.processed_images, f"processed_images is empty, but image_idx={image_idx}"
                        image = ctx.processed_images[image_idx]
                        image.save(buffered, format="PNG")
                        buffered.seek(0)
                        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        image_list.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        )
                        new_content += mm_repl
                    else:
                        # NOTE(wuhuan): add string 而不是 append dict，因为 vllm 里面会用 "\n" 拼 dict，导致不一致，见
                        # https://github.com/vllm-project/vllm/blob/aaa4ac1c95aaf70afab51582c56d80554a21bbd0/vllm/entrypoints/chat_utils.py#L1032
                        new_content += segment

                if image_list:
                    openai_message["content"] = [{"type": "text", "text": new_content}] + image_list
                else:
                    openai_message["content"] = new_content

                ctx.messages_openai.append(openai_message)

        if _check_required_fields():
            return

        if ctx._messages is not None:
            prompt = self.tokenizer.apply_chat_template(ctx._messages, add_generation_prompt=True, tokenize=False)
            assert isinstance(prompt, str), f"{prompt}"
            ctx.prompt = prompt
        if _check_required_fields():
            return

        if ctx.messages_raw is not None:
            ctx._messages_expand_mm = self.mm_plugin.process_messages(
                ctx.messages_raw, ctx.processed_images or [], [], self.processor
            )
        if _check_required_fields():
            return

        if ctx._messages_expand_mm is not None:
            prompt_expand_mm = self.tokenizer.apply_chat_template(
                ctx._messages_expand_mm, add_generation_prompt=True, tokenize=False
            )
            assert isinstance(prompt_expand_mm, str), f"expected str but got {type(prompt_expand_mm)}"
            input_data = self.tokenizer(prompt_expand_mm, return_tensors="pt", add_special_tokens=False)
            ctx.input_ids = input_data["input_ids"]
            ctx.attention_mask = input_data["attention_mask"]

    def _load_images(self, images: list[Any]) -> list[ImageObject]:
        from ...io.image import load_pillow_images

        res = []
        for image in images:
            for img in load_pillow_images(image):
                res.append(img)
        return res


class NativeMMProcessor(BaseMMProcessor):
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        processor: Optional["ProcessorMixin"] = None,
        image_placeholder: str = IMAGE_PLACEHOLDER,
        **kwargs,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            processor=processor,
            image_placeholder=image_placeholder,
            **kwargs,
        )

    def _load_images(self, images: list[Any]) -> list[ImageObject]:
        from verl.utils.dataset.vision_utils import process_image, process_image_for_qwen_processor

        if self.template_name and "qwen" in self.template_name and self.is_multimodal():
            return [process_image_for_qwen_processor(i) for i in images]

        return [process_image(i) for i in images]

    def _preprocess_image(self, image: ImageObject, **kwargs):
        image_resolution = kwargs.get("image_resolution")
        image_max_pixels = kwargs.get("image_max_pixels")
        image_min_pixels = kwargs.get("image_min_pixels")

        if image_max_pixels or image_min_pixels:
            if image_max_pixels and image.width * image.height > image_max_pixels:
                resize_factor = math.sqrt(image_max_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
            if image_min_pixels and image.width * image.height < image_min_pixels:
                resize_factor = math.sqrt(image_min_pixels / (image.width * image.height))
                width, height = int(image.width * resize_factor), int(image.height * resize_factor)
                image = image.resize((width, height))
        elif image_resolution and max(image.width, image.height) > image_resolution:
            resize_factor = image_resolution / max(image.width, image.height)
            width, height = int(image.width * resize_factor), int(image.height * resize_factor)
            image = image.resize((width, height), resample=Image.Resampling.NEAREST)

        if image.mode != "RGB":
            image = image.convert("RGB")

        return image

    def _build_messages(self, messages: list[dict], images: Optional[list] = None, videos: Optional[list] = None):
        if images or videos:
            messages = deepcopy(messages)
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split(f"({self.image_placeholder}|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == self.image_placeholder:
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def process(self, ctx: MMProcessorContext, required_fields: Optional[list[str]] = None):
        def _check_required_fields() -> bool:
            if required_fields is None:
                return False

            for f in required_fields:
                if getattr(ctx, f, None) is None:
                    return False
            return True

        if ctx.image_paths:
            ctx.raw_images = self._load_images(ctx.image_paths)
        if _check_required_fields():
            return

        if ctx.image_process_params and ctx.raw_images:
            processed_images = [
                self._preprocess_image(
                    img,
                    **ctx.image_process_params,
                )
                for img in ctx.raw_images
            ]
            ctx.processed_images = processed_images
        else:
            ctx.processed_images = ctx.raw_images
        if _check_required_fields():
            return

        if self.processor is not None and ctx.processed_images:
            if self.template_name == "AgiVlm":
                from redaccel.models import is_agivlm1_6

                assert is_agivlm1_6(self.processor), f"expected AgiVLProcessorV11 but got {type(self.processor)}"
                ctx.mm_placeholders = self.processor.get_mm_placeholders(ctx.processed_images)
            else:
                ctx.mm_placeholders = [
                    "<|vision_start|>"
                    + self.tokenizer.decode(
                        self.processor(
                            text=self.processor.image_token,
                            images=ctx.processed_images,
                        )["input_ids"][0]
                    )
                    + "<|vision_end|>"
                ]
        if _check_required_fields():
            return

        assert ctx.messages_raw, "messages_raw is None"
        ctx.messages_openai = self._build_messages(ctx.messages_raw, ctx.processed_images)

        if self.processor is not None and ctx.processed_images:
            if self.template_name == "AgiVlm":
                from redaccel.models import is_agivlm1_6

                prompt = self.tokenizer.apply_chat_template(
                    ctx.messages_raw, add_generation_prompt=True, tokenize=False
                )
                assert isinstance(prompt, str), f"expected str but got {type(prompt)}"
                assert is_agivlm1_6(self.processor), f"expected AgiVLProcessorV11 but got {type(self.processor)}"
                out = self.processor(prompt=prompt, images=ctx.processed_images)
                model_inputs = {
                    "input_ids": out.input_ids.unsqueeze(0),
                    "attention_mask": out.attention_mask.unsqueeze(0),
                    "pixel_values": out.pixel_values,
                }
                image_placeholder_reg = rf"{self.mm_braces[0]}image(?:_\d+)?{self.mm_braces[1]}"
                while re.findall(image_placeholder_reg, prompt):
                    prompt = re.sub(image_placeholder_reg, "<|imgpad|>", prompt, count=1)
                ctx.prompt = prompt
            else:
                ctx.prompt = self.processor.apply_chat_template(
                    ctx.messages_openai, add_generation_prompt=True, tokenize=False
                )
                model_inputs = self.processor(text=ctx.prompt, images=ctx.processed_images, return_tensors="pt")
            ctx.input_ids = model_inputs.pop("input_ids")
            ctx.attention_mask = model_inputs.pop("attention_mask")
            model_inputs.pop("second_per_grid_ts", None)
            if ctx.enable_mm_inputs:
                ctx.mm_inputs = model_inputs
        else:
            ctx.prompt = self.tokenizer.apply_chat_template(
                ctx.messages_openai, add_generation_prompt=True, tokenize=False
            )
            model_inputs = self.tokenizer(ctx.prompt, return_tensors="pt", add_special_tokens=False)
            ctx.input_ids = model_inputs.pop("input_ids")
            ctx.attention_mask = model_inputs.pop("attention_mask")
            ctx.mm_inputs = {}

        if _check_required_fields():
            return


def init_mm_processor(*args, **kwargs):
    from ...utils.configs import Configs

    has_redaccel = True
    try:
        import redaccel.tuner.data.mm_plugin as mm_plugin  # noqa
    except BaseException:
        has_redaccel = False

    if has_redaccel or Configs.RL_USE_NATIVE_PROCESSOR:
        return RedAccelMMProcessor(*args, **kwargs)
    return NativeMMProcessor(*args, **kwargs)
