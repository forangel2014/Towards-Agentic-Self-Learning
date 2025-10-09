# Copyright (c) 2024 RedNote Authors. All Rights Reserved.

import copy
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from ...utils.configs import Configs
from ...utils.log import logger


class RedData(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"no attribute {key}")

    def __setattr__(self, key, value):
        self[key] = value

    def keys(self):
        return list(super().keys())


class ImgType(Enum):
    PTL_IMG = 1
    BYTE = 2


class Pipeline:
    def __init__(self):
        self.functions = []

    def add_function(self, fn):
        if callable(fn):
            self.functions.append(fn)
        else:
            raise ValueError(f"The provided object {fn} is not callable.")

    def run(self, input_data):
        output = input_data
        for fn in self.functions:
            output = fn(output)
        return output


class _RedBaseDataset(Dataset):
    """RedBaseDataset is an iterable dataset designed for seamless integration
    with PyTorch data pipelines. It allows easy manipulation and retrieval of
    data from specified RedJson files.

    Attributes:
        dataset_files: List: A list of file paths to the datasets. Each element in the list is expected to be a string representing a file path.
        parse_data_fn: Callable[[Union[Dict, RedData]], Any] = None: An optional callable function that takes either a dictionary or an instance of RedData as input and returns any type of data. If not provided, it defaults to None.
        is_prefetch_dataset (bool): Indicator for prefetching capability.
        shuffle: bool = True: A boolean flag to determine if the dataset should be shuffled. Defaults to True.
        select_imgs_fn: Callable[[Dict[str, List[str]]], Dict[str, List[str]]] = None: An optional callable function that takes a dictionary with string keys and lists of strings as values, and returns a dictionary of the same structure.
        select_sample_fn: Callable[[Dict], List[Dict]] = None: An optional callable function that takes a dictionary as input and returns a list of dictionaries.
        repetitions: int = sys.maxsize: An integer specifying the maximum number of times the dataset should be iterated over. Defaults to sys.maxsize, which means the dataset will repeat indefinitely.
        use_as_dict: bool = False: A boolean flag indicating whether the dataset should be used as a dictionary.The default value is False, indicating the use of the RedData type.
        img_type: int = ImgType.PTL_IMG.value: An integer representing the type of image data. The default value is of the type PIL.Image.Image.
    """

    def __init__(
        self,
        dataset_files: List = None,
        parse_data_fn: Callable[[Union[Dict, RedData]], Any] = None,
        is_prefetch_dataset: bool = False,
        shuffle: bool = True,
        select_imgs_fn: Callable[[Dict[str, List[str]]], Dict[str, List[str]]] = None,
        select_sample_fn: Callable[Dict, List[Dict]] = None,
        repetitions: int = sys.maxsize,
        use_as_dict: bool = False,
        img_type: int = ImgType.PTL_IMG.value,
        dataset_format: str = "redjson",
        max_samples: int = sys.maxsize,
        delete_cache_dir_enable: bool = True,
        batch_size: int = 1,
    ):
        super().__init__()
        if isinstance(dataset_files, list):
            self.dataset_files = dataset_files
        else:
            self.dataset_files = [dataset_files]
        # Directory for caching data, configurable via environment variable.
        self.cache_dir = Configs.IO_CACHE_DIR
        # Function to parse data, if provided.
        self.parse_data_fn = Pipeline()
        self.parse_data_fn.add_function(parse_data_fn)
        self.select_imgs_fn = select_imgs_fn
        self.select_sample_fn = select_sample_fn
        # Boolean flag to determine if dataset supports prefetching.
        self.is_prefetch_dataset = is_prefetch_dataset
        # Set repetitions to sys.maxsize to allow indefinite iteration.
        self.repetitions = repetitions
        self._instance_id = None
        self.shuffle = shuffle
        self.use_as_dict = use_as_dict
        self._length_dataset = None
        self._epoch = 0
        self.rank = None
        self.world_size = None
        self.global_worker_idx = None
        self.set_img_type(img_type)
        self.dataset_format = dataset_format
        self.max_samples = max_samples if max_samples else sys.maxsize
        self.delete_cache_dir_enable = delete_cache_dir_enable
        self.batch_size = batch_size
        assert (
            isinstance(self.max_samples, int) and self.max_samples > 0
        ), f"max_sample={max_samples},type(max_sample)={type(max_samples)}"

    def set_img_type(self, value):
        values = [member.value for member in ImgType]
        assert (
            value in values
        ), f"Invalid img_type value: {value}. Please use one of the following enum values: {', '.join([f'{member.value} ({member.name})' for member in ImgType])}"
        self.img_type = value

    def set_dataset_id(self, worker_info=None, rank=None, rediter_count=None):
        logger.debug(f"{worker_info=},{rank=},{self.is_prefetch_dataset=},{self.cache_dir=},{Configs.IO_CACHE_DIR}")
        if rank is None:
            self.pytorch_worker_info(worker_info)
        self.cache_dir = Configs.IO_CACHE_DIR + f"epoch_{rediter_count}_dataset_{self._instance_id}_rank_{rank}_/"

    def set_epoch(self, epoch):
        self._epoch = epoch

    def set_rank_info(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def clean_disk_cache(self, rediter_count=None, rank=None):
        if not self.delete_cache_dir_enable:
            # note(jiajia): for "类目回刷"的个性化写法
            return
        target_dir = Configs.IO_CACHE_DIR + f"epoch_{rediter_count}_dataset_{self._instance_id}_rank_{rank}_/"

        if os.path.exists(target_dir) and os.path.isdir(target_dir):
            try:
                shutil.rmtree(target_dir)
                logger.debug(f"Disk cache has been deleted. dir={target_dir} {self.is_prefetch_dataset=}")
            except OSError as e:
                logger.debug(f"Error: {e.strerror}. File: {e.filename}")
            except Exception as e:
                logger.debug(f"An unexpected error occurred: {e}")
        else:
            logger.error(f"The disk cache directory does not exist. dir={target_dir}")

    def set_worker_info(self):
        self.rank, self.world_size, self.worker_id, self.num_workers = self.pytorch_worker_info()
        self.global_worker_idx = self.rank * self.num_workers + self.worker_id
        self.global_worker_num = self.world_size * self.num_workers
        self.worker_seed = self.rank * 1000 + self.worker_id + self._epoch * 10
        self.rng = random.Random(self.worker_seed)
        logger.debug(
            f"{self.world_size=} * {self.num_workers=} > {len(self.dataset_files)=}; {self.global_worker_idx=},{self.global_worker_num=}"
        )

    def read_image_from_alluxio_s3(self, noteid):
        from .file import QuickSilverFile

        QuickSilverFile(self.cache_dir).download_to_disk_cache(noteid)
        logger.debug(f"Note {noteid} download to disk cache from alluxio s3 {self.cache_dir+noteid}")

    def imageid_to_disk_path(self, imageid):
        image_disk_path = self.cache_dir + imageid
        if not os.path.exists(image_disk_path):
            logger.debug(f"Not Hit the disk. {image_disk_path}")
            try:
                self.read_image_from_alluxio_s3(imageid)
            except Exception as e:
                logging.error(f"Failed to read note from Alluxio S3: {e}")
                try:
                    pass
                    # TODO(jiajia)：self.read_note_from_oss_s3(imageid)
                except Exception as e:
                    logging.error(f"Failed to read image from OSS S3: {e}")
                    raise ValueError(f"Image {imageid} not found in any storage") from e
        else:
            logger.debug(f"Hit the disk. {image_disk_path}")
        if not os.path.exists(image_disk_path):
            logger.debug(f"image_disk_path={image_disk_path} not exist in disk")
            return None
        return image_disk_path

    def pytorch_worker_info(self, worker_info=None, group=None):
        """Return node and worker info for PyTorch and some distributed
        environments."""

        if self.rank is not None and self.world_size is not None:
            return self.rank, self.world_size, 0, 1

        rank = 0
        world_size = 1
        worker = 0
        num_workers = 1
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            try:
                import torch.distributed

                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    group = group or torch.distributed.group.WORLD
                    rank = torch.distributed.get_rank(group=group)
                    world_size = torch.distributed.get_world_size(group=group)
            except ModuleNotFoundError:
                logger.error("Failed to import torch.distributed. Using default values.")

        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info() if worker_info is None else worker_info
        if self.is_prefetch_dataset:
            from .worker import get_worker_info

            worker_info = get_worker_info()
        if worker_info is not None:
            worker = worker_info.id
            num_workers = worker_info.num_workers
        else:
            logger.error(
                f"Failed to get worker info. Using default values. {self.is_prefetch_dataset=} ,{worker_info=}"
            )
        self.rank = rank
        return rank, world_size, worker, num_workers

    def pick(self, buf):
        k = self.rng.randint(0, len(buf) - 1)
        sample = buf[k]
        buf[k] = buf[-1]
        buf.pop()
        return sample

    def iter_data(self, bufsize=1000, initial=100, rng=None, handler=None):
        """Shuffle the data in the stream.

        This uses a buffer of size `bufsize`. Shuffling at
        startup is less random; this is traded off against
        yielding samples quickly.

        data: iterator
        bufsize: buffer size for shuffling
        returns: iterator
        rng: either random module or random.Random instance
        """
        if not self.shuffle:
            while True:
                try:
                    yield next(self.dataset_index)
                except StopIteration:
                    break
        else:
            initial = min(initial, bufsize)
            buf = []
            while True:
                try:
                    data = next(self.dataset_index)
                except StopIteration:
                    break
                buf.append(data)
                if len(buf) < bufsize:
                    try:
                        buf.append(next(self.dataset_index))  # skipcq: PYL-R1708
                    except StopIteration:
                        pass
                if len(buf) >= initial:
                    yield self.pick(buf)
            while len(buf) > 0:
                yield self.pick(buf)

    def get_image(self, image_id):
        flag = "IMG_NOT_FOUND"
        if image_id is None or image_id.strip() == "":
            return flag + ": " + image_id
        image_disk_path = self.imageid_to_disk_path(image_id)
        if image_disk_path is None:
            logger.debug(f"IMG_NOT_FOUND,{image_id=},{image_disk_path=}")
            return flag + ": " + image_id
        try:
            if self.img_type == ImgType.PTL_IMG.value:
                image = Image.open(image_disk_path)
            elif self.img_type == ImgType.BYTE.value:
                with open(image_disk_path, "rb") as image_file:
                    image = image_file.read()
            else:
                raise ValueError("Unsupported image type. Only ImgType.PTL_IMG and ImgType.BYTE are allowed.")
        except Image.UnidentifiedImageError:
            return flag + ": " + image_id + " (Reason: The image is corrupted.)"
        if Configs.RED_ENABLE_IMAGE_CLEAR:
            if os.path.exists(image_disk_path):
                os.remove(image_disk_path)
        return image

    def get_image_path(self, image_id):
        flag = "IMG_NOT_FOUND"
        # NOTE(jiajia): for different image use different resloution
        if isinstance(image_id, dict):
            image_id, save_for_return = image_id.get("path"), image_id
        else:
            save_for_return = None

        if not image_id or image_id.strip() == "":
            return f"{flag}: {image_id}"

        image_disk_path = self.imageid_to_disk_path(image_id)
        if image_disk_path is None:
            logger.debug(f"IMG_NOT_FOUND, image_id={image_id}, image_disk_path=None")
            return f"{flag}: {image_id}"

        if save_for_return:
            save_for_return["path"] = image_disk_path
            return save_for_return
        return image_disk_path

    def get_src_imgs(self, data):
        if isinstance(data, dict):
            for key, value_list in data.items():
                data[key] = [self.get_image(v) for v in value_list]
            return data
        elif isinstance(data, list):
            return [self.get_image(v) for v in data]
        else:
            raise ValueError("Unsupported data type. Expected a dictionary or a list.")

    def get_src_imgs_path(self, data):
        if isinstance(data, dict):
            for key, value_list in data.items():
                data[key] = [self.get_image_path(v) for v in value_list]
            return data
        elif isinstance(data, list):
            return [self.get_image_path(v) for v in data]
        else:
            raise ValueError("Unsupported data type. Expected a dictionary or a list.")

    def get_full_sample(self, sample_empty_image, image_ids):
        sample_empty_image["src_imgids"] = copy.deepcopy(image_ids)
        src_imgs = self.get_src_imgs(image_ids)
        sample_empty_image["src_imgs"] = src_imgs
        if self.use_as_dict:
            return self.parse_data_fn.run(sample_empty_image)
        else:
            return self.parse_data_fn.run(RedData(sample_empty_image))

    def add_preprocess_fn(self, fn):
        self.parse_data_fn.add_function(fn)

    def get_full_sample_sharegpt(self, sample_empty_image, image_ids):
        # Note(jiajia): fix memory leaky issue
        src_imgs = self.get_src_imgs_path(image_ids)
        sample_empty_image["images"] = src_imgs
        if self.use_as_dict:
            return self.parse_data_fn.run(sample_empty_image)
        else:
            return self.parse_data_fn.run(RedData(sample_empty_image))

    def parse_redjson(self):
        if len(self.dataset_files) == 0:
            logger.error(
                "Message: No more dataset files to iterate over. Please note that NUM_WORKER * GPUS_PER_NODE * NNODES should be greater than or equal to the number of RedJson files."
            )
            raise StopIteration

        if self.shuffle and len(self.dataset_files) > 1:
            self.rng.shuffle(self.dataset_files)
        self.count_redjosn_line = -1
        for json_file in self.dataset_files:
            with open(json_file, "r") as file:
                if self.count_redjosn_line > self.max_samples:
                    break
                for line in file:
                    self.count_redjosn_line += 1
                    if self.count_redjosn_line > self.max_samples:
                        break
                    if (
                        self.global_worker_idx is not None
                        and (self.count_redjosn_line % self.global_worker_num) != self.global_worker_idx
                    ):
                        continue
                    try:
                        data = json.loads(line)
                        if not callable(self.select_sample_fn):
                            yield data
                        else:
                            sample_list = self.select_sample_fn(data)
                            for sample in sample_list:
                                yield sample
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON: {e}")

    def parse_srcjson(self, total_samples):
        assert (
            len(self.dataset_files) > 0
        ), "Message: No more dataset files to iterate over. Please note that NUM_WORKER * GPUS_PER_NODE * NNODES should be greater than or equal to the number of RedJson files."

        # Step 1: 计算每个 rank 应处理的batch数（忽略余数）
        samples_per_rank = total_samples // self.batch_size // self.world_size * self.batch_size

        # Step 2: 确定每个 rank 的数据范围
        start_idx_rank = self.rank * samples_per_rank
        end_idx_rank = start_idx_rank + samples_per_rank

        # Step 3: 计算每个 worker 在当前 rank 中应处理的数据量（考虑余数）
        if samples_per_rank // self.batch_size < self.num_workers:
            samples_per_worker = self.batch_size
        else:
            samples_per_worker = (
                (end_idx_rank - start_idx_rank) // self.batch_size // self.num_workers * self.batch_size
            )

        # Step 4: 确定每个 worker 的数据范围

        start_idx_worker = start_idx_rank + self.worker_id * samples_per_worker
        end_idx_worker = start_idx_worker + samples_per_worker

        # Step 5: 遍历文件并筛选数据
        self.count_redjson_line = 0
        for file_path in self.dataset_files:
            if file_path.endswith(".json"):
                with open(file_path, "r", encoding="utf-8") as file:
                    data = file.read()
                    lines = json.loads(data)
            elif file_path.endswith(".jsonl"):
                with open(file_path, "r", encoding="utf-8") as file:
                    lines = [json.loads(line) for line in file]
            else:
                raise ValueError(f"不支持的文件格式: {file_path}。请提供 .json 或 .jsonl 文件。")

            for data in lines:
                if self.count_redjson_line >= end_idx_rank:
                    break
                if start_idx_rank <= self.count_redjson_line < end_idx_rank:
                    if start_idx_worker <= self.count_redjson_line < end_idx_worker:
                        if not callable(self.select_sample_fn):
                            yield data
                        else:
                            sample_list = self.select_sample_fn(data)
                            for sample in sample_list:
                                yield sample
                self.count_redjson_line += 1

    def write_prefetch_order(self, note_list):  # for debuging prefetch order
        rank, world_size, worker, num_workers = self.pytorch_worker_info()
        dir_path = Configs.IO_CACHE_DIR + "/debug"
        os.makedirs(dir_path, exist_ok=True)
        file_path = f"{dir_path}/is_prefetch_{self.is_prefetch_dataset}_{rank}_{world_size}_{worker}_{num_workers}.log"

        with open(file_path, "a") as file:
            file.write(f"{str(note_list)}")
            file.write("\n")

    def get_sample_imageid(self, sample):
        if self.dataset_format in ["sharegpt", "alpaca"]:
            # NOTE(jiajia): for different image use different resloution
            if isinstance(sample["images"][0], dict):
                return [s["path"] for s in sample["images"]]
            elif isinstance(sample["images"][0], str):
                return sample["images"]
            else:
                raise ValueError("Unsupported data type. Expected a dictionary or a list about 'images'.")

        filtered_dict = {
            k.replace(Configs.IMG_DL_PREFIX, "", 1): v
            for k, v in sample.items()
            if k.startswith(Configs.IMG_DL_PREFIX)
        }
        if filtered_dict:
            return filtered_dict
        else:
            logging.error(
                f"No key-value pairs were found starting with the string '{Configs.IMG_DL_PREFIX}' in {filtered_dict=},{sample=}."
            )

    def get_imageid_list(self, data):
        if isinstance(data, dict):
            return sum(data.values(), [])
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unsupported data type. Expected a dictionary or a list.")

    def is_empty(self, var):
        if var is None:
            return True
        if isinstance(var, torch.Tensor):
            return var.numel() == 0 or torch.all(var == 0).item() or torch.all(torch.isnan(var)).item()
        if isinstance(var, np.ndarray):
            return var.size == 0 or np.all(var == 0) or np.all(np.isnan(var))
        if isinstance(var, (int, float)):
            return var == 0
        if isinstance(var, dict):
            return len(var) == 0
        if isinstance(var, list):
            return len(var) == 0
        return False


class _RedDataset(IterableDataset, _RedBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __len__(self):
        if self._length_dataset is not None:
            return self._length_dataset
        self._length_dataset = 0

        for json_file in self.dataset_files:
            if self.dataset_format in ["sharegpt", "alpaca"]:
                if json_file.endswith(".json"):
                    with open(json_file, "r", encoding="utf-8") as file:
                        self._length_dataset += len(json.load(file))
                elif json_file.endswith(".jsonl"):
                    with open(json_file, "r", encoding="utf-8") as file:
                        self._length_dataset += sum(1 for _ in file)
                else:
                    raise ValueError(f"不支持的文件格式: {json_file}。请提供 .json 或 .jsonl 文件。")
            else:
                result = subprocess.run(["wc", "-l", json_file], stdout=subprocess.PIPE, text=True)
                self._length_dataset += int(result.stdout.split()[0])
        self._length_dataset = min(self._length_dataset, self.max_samples)
        return self._length_dataset

    def __iter__(self):
        self.set_worker_info()
        for _ in range(self.repetitions):
            try:
                if self.dataset_format in ["sharegpt", "alpaca"]:
                    self.dataset_index = self.parse_srcjson(len(self))
                else:
                    self.dataset_index = self.parse_redjson()
            except StopIteration:
                break
            for sample_empty_image in self.iter_data():
                try:
                    if callable(self.select_imgs_fn):
                        sample_empty_image = self.select_imgs_fn(sample_empty_image)
                    image_ids = self.get_sample_imageid(sample_empty_image)
                    if Configs.LOG_LEVEL == "DEBUG":
                        self.write_prefetch_order(image_ids)
                    if image_ids is None:
                        continue
                    if self.is_prefetch_dataset:
                        imageid_list = self.get_imageid_list(image_ids)
                        if len(imageid_list) == 0:
                            continue
                        yield imageid_list
                    else:
                        if self.dataset_format in ["sharegpt", "alpaca"]:
                            full_sample = self.get_full_sample_sharegpt(sample_empty_image, image_ids)
                        else:
                            full_sample = self.get_full_sample(sample_empty_image, image_ids)
                        if self.is_empty(full_sample):
                            logger.debug(f"full_sample is empty,{full_sample=}")
                            continue
                        yield full_sample

                except StopIteration:
                    break
            self.sample_dataset = None


class RedDataset:
    instance_count = -1

    def __new__(cls, user_dataset_class=None, *args, **kwargs):
        RedDataset.instance_count += 1

        if user_dataset_class is None:
            kwargs["is_prefetch_dataset"] = False
            dataset = _RedDataset(*args, **kwargs)
            kwargs["is_prefetch_dataset"] = True
            prefetch_dataset = _RedDataset(*args, **kwargs)
        elif issubclass(user_dataset_class, _RedDataset):
            kwargs["is_prefetch_dataset"] = False
            dataset = user_dataset_class(*args, **kwargs)
            kwargs["is_prefetch_dataset"] = True
            prefetch_dataset = user_dataset_class(*args, **kwargs)
        else:
            raise ValueError("user_dataset_class must be a subclass of _RedDataset or None")
        dataset._instance_id = RedDataset.instance_count
        prefetch_dataset._instance_id = RedDataset.instance_count
        return (dataset, prefetch_dataset)
