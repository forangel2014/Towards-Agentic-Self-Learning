# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import os
import random

import torch
from loguru import logger


def set_basic_config(level):
    """
    This function sets the global logging format and level. It will be called when import verl
    """
    logging.basicConfig(format="%(levelname)s:%(asctime)s:%(message)s", level=level)


def log_to_file(string):
    print(string)
    if os.path.isdir("logs"):
        with open(f"logs/log_{torch.distributed.get_rank()}", "a+") as f:
            f.write(string + "\n")


@functools.lru_cache(None)
def info_once(*args, **kwargs):
    logger.info(*args, **kwargs)


def info_random(prob: float = 0.01, *args, **kwargs):
    if random.random() < prob:
        logger.info(*args, **kwargs)


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    logger.warning(*args, **kwargs)


@functools.lru_cache(None)
def error_once(*args, **kwargs):
    logger.error(*args, **kwargs)
