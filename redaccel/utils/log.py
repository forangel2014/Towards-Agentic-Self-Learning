# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.

import functools
import inspect
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Union

import loguru
from loguru import logger as _logger

from .configs import Configs


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# 注入所有默认 logger,会导致日志太多，先不开
# logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)


def patch_transformers_logger(logger):
    import transformers.utils.logging

    transformers.utils.logging._reset_library_root_logger()
    transformers.utils.logging._default_handler = InterceptHandler()


def patch_deepspeed_logger(logger: "loguru.Logger"):
    from deepspeed.utils.logging import logger as ds_logger

    ds_logger.handlers.clear()
    ds_logger.addHandler(InterceptHandler())


def add_distributed_logger_handler(logger: "loguru.Logger", sink: Any, rank: str, local_rank: str):
    """add handler for distributed logger.

    Parameters
    ----------
    logger : loguru.Logger
        logger of loguru
    sink : Any
        `sys.stderr` or filepath string
    rank : str
        rank or current training process, get from `os.environ["LOCAL_RANk"]`
    """

    if sink == sys.stderr:
        sink_type = "std"
    else:
        sink_type = "file"

    def dist_filter(record):
        if not rank:
            return True

        hit = True
        if "rank" in record["extra"]:
            hit &= str(record["extra"]["rank"]) == rank
        if "local_rank" in record["extra"]:
            hit &= str(record["extra"]["local_rank"]) == local_rank
        return hit

    logger.add(
        sink,
        level=Configs.LOG_LEVEL,
        format=Configs.LOG_FORMAT,
        filter=lambda r: dist_filter(r) and not r["extra"].get(f"disable_{sink_type}", False),
        diagnose=False,
        colorize=Configs.LOG_COLOR if sink_type == "std" else False,
    )
    return logger


def init_distributed_logger():
    import torch.distributed as dist

    if dist.is_initialized():
        rank = str(dist.get_rank())
    else:
        rank: str = os.environ.get("RANK", "")
    local_rank: str = os.environ.get("LOCAL_RANK", "")
    world_size: int = int(os.environ.get("WORLD_SIZE", "1"))

    if not rank and not local_rank:
        # no distributed training
        desc = ""
    elif world_size <= 1:
        desc = f"(rank={rank})" if rank else ""
    else:
        desc = f"(rank={local_rank}-{rank})" if rank and local_rank else ""

    _logger.configure(extra={"rank_desc": desc})
    dist_logger = _logger.bind()

    dist_logger.remove()
    add_distributed_logger_handler(dist_logger, sys.stderr, rank, local_rank)
    add_distributed_logger_handler(
        dist_logger,
        Path(Configs.LOG_FILE).with_suffix("").with_suffix(f".rank_{rank}.log") if rank else Configs.LOG_FILE,
        rank,
        local_rank,
    )

    patch_transformers_logger(dist_logger)
    patch_deepspeed_logger(dist_logger)
    return dist_logger


logger = init_distributed_logger()
logger.bind(rank=0).debug(f"configs: {Configs}")
rank0_logger = logger.bind(rank=0)


def add_log_dir(output_dir: str):
    rank: str = os.environ.get("RANK", "")
    local_rank: str = os.environ.get("LOCAL_RANK", "")
    add_distributed_logger_handler(
        logger,
        os.path.join(output_dir, f"redaccel.rank_{rank}.log" if rank else "redaccel.log"),
        rank,
        local_rank,
    )


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        t = time.time() - start
        rank0_logger.debug(f"execute {func.__name__} in {t:.4f}s.")
        return result

    return wrapper


def truncate_middle(s: Union[str, dict], n: int = 512, ellipsis: str = "...<TRUNCATED>..."):
    if isinstance(s, (dict, list)):
        s = json.dumps(s, ensure_ascii=False)

    if len(s) <= n:
        return s

    if n <= len(ellipsis) + 2:
        return ellipsis

    part_length = (n - len(ellipsis)) // 2
    front_len = part_length + (n - len(ellipsis)) % 2
    truncated = s[:front_len] + ellipsis + s[-part_length:]
    return truncated


def experimental(msg: str):
    def inner(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            rank0_logger.opt(colors=True).warning(f"<red>[EXPERIMENTAL]</red> {msg}")
            return func(*args, **kwargs)

        return wrapper

    return inner


def patch_print_to_log():
    # avoid circular import
    from .patch import PatchManager

    class StdOutLogger:
        def __init__(self, logger):
            self.logger = logger
            self.linebuf = ""

        def write(self, buf):
            self.logger.info(buf.rstrip())

        def flush(self):
            pass

    patch_mgr = PatchManager()
    patch_mgr.register_patch("sys.stdout", StdOutLogger(logger))
    return patch_mgr


@functools.lru_cache(None)
def warning_once(*args, **kwargs):
    logger.warning(*args, **kwargs)


@functools.lru_cache(None)
def info_once(*args, **kwargs):
    logger.info(*args, **kwargs)
