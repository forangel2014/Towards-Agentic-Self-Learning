# Copyright (c) 2024 RedNote Authors. All Rights Reserved.

import os
from distutils.dist import strtobool
from typing import Callable, Generic, Optional, TypeVar

from transformers.utils import is_torch_cuda_available


ENV_PREFIX = "RED_"

T = TypeVar("T")


def enabled_multi_cloud_ai_dataset():
    return os.getenv("MULTI_CLOUD_UNIFIED_AI_DATASET", "false") != "false"


def close_multi_cloud_ai_dataset():
    os.environ["MULTI_CLOUD_UNIFIED_AI_DATASET"] = "false"


class EnvProperty(Generic[T]):
    def __init__(
        self, env: str, dtype, default_value: Optional[T], default_factory: Optional[Callable[..., T]] = None
    ):
        if not env.startswith(ENV_PREFIX):
            env_with_prefix = ENV_PREFIX + env
            if env_with_prefix in os.environ:
                env = env_with_prefix

        self._type = dtype

        if default_value is None:
            if default_factory is None:
                raise ValueError("default value or default factory is required")
            default_value = default_factory()

        if dtype == bool:
            self._value: T = bool(strtobool(os.environ.get(env, str(default_value))))
        else:
            self._value: T = dtype(os.environ.get(env, default_value))

    def __set__(self, instance, value):
        if not isinstance(value, self._type):
            raise TypeError("excepting type: {}, got {}".format(self._type, type(value)))
        self.__dict__["_value"] = value

    def __get__(self, instance, clz) -> T:
        return self._value


class _Configs:
    # log
    LOG_LEVEL = EnvProperty("LOG_LEVEL", str, None, lambda: "DEBUG" if "re" in os.environ else "INFO")
    LOG_FILE = EnvProperty("LOG_FILE", str, "./log/python/src.log")
    LOG_COLOR = EnvProperty("LOG_COLOR", bool, True)
    LOG_FORMAT = EnvProperty(
        "LOG_FORMAT",
        str,
        "<blue>RedNote{extra[rank_desc]}</blue>|"
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green>|"
        "<level>{level: <7}</level>|"
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> <level>{message}</level>",
    )

    # io
    IO_DISABLE_GC = EnvProperty("IO_DISABLE_GC", bool, False)
    IO_CACHE_DIR = EnvProperty("IO_CACHE_DIR", str, "/data/temp/cache/quicksilver/")
    IO_USE_DISK_CACHE = EnvProperty("IO_USE_DISK_CACHE", bool, True)
    IO_USE_JUICEFS_SDK = EnvProperty("IO_USE_JUICEFS_SDK", bool, False)
    DEFAULT_DATA_NUM_WORKERS = EnvProperty("DEFAULT_DATA_NUM_WORKERS", int, 8)
    RED_ENABLE_IMAGE_CLEAR = EnvProperty("RED_ENABLE_IMAGE_CLEAR", bool, True)

    IMG_DL_PREFIX = "src_imgs__"

    IO_SAVER_HANDLER = EnvProperty(
        "IO_SAVER_HANDLER", str, None, lambda: "pin" if is_torch_cuda_available() else "paged"
    )
    IO_SAVER_TIMEOUT = EnvProperty("IO_SAVER_TIMEOUT", int, 600)  # in second

    EDS_HOST_ADDR = EnvProperty("EDS_HOST_ADDR", str, "http://10.11.177.52:8085")  # 服务发现地址
    EVENT_SERVER_ADDR = EnvProperty("EVENT_SERVER_ADDR", str, "")  # 埋点后端地址，如果为空，从服务发现里面取
    EVENT_TRACKER_ENABLE = EnvProperty("EVENT_TRACKER_ENABLE", bool, True)
    EVENT_UPDATE_INTERVAL = EnvProperty("EVENT_UPDATE_INTERVAL", int, 600)  # 上报间隔(单位s)

    REQUESTS_SESSION_POOL_SIZE = EnvProperty("HTTP_REQUESTS_POOL_SIZE", int, 16)  # request session pool size

    PLUGINS_DIR = EnvProperty("PLUGINS_DIR", str, "./plugins")  # plugins dir
    SINK_USER_EMAIL = EnvProperty("SINK_USER_EMAIL", str, "")  # user email for event sink, 逗号分隔
    SINK_WEBHOOK = EnvProperty("SINK_WEBHOOK", str, "")  # redcity webhook for event sink, 逗号分割

    # RAY #####################################################################
    RAY_DEBUG = EnvProperty("RAY_DEBUG", str, "0")  # or legacy
    RAY_RUNTIME_REQUIREMENTS = EnvProperty("RAY_RUNTIME_REQUIREMENTS", str, "")
    RAY_DEBUG_POST_MORTEM = EnvProperty("RAY_DEBUG_POST_MORTEM", str, "0")

    RL_USE_NATIVE_PROCESSOR = EnvProperty("RL_USE_NATIVE_PROCESSOR", bool, False)

    VLLM_USE_V1 = EnvProperty("VLLM_USE_V1", str, "1")  # or 0
    VLLM_LOGGING_LEVEL = EnvProperty("VLLM_LOGGING_LEVEL", str, "WARN")  # INFO / DEBUG / ERROR

    def __repr__(self) -> str:
        head = "=" * 16 + " RED CONFIGS " + "=" * 16
        body = "\n".join([f"{k} = {getattr(self, k)}" for k in dir(self) if not k.startswith("__")])
        tail = "=" * len(head)
        return "\n" + "\n".join([head, body, tail])


Configs = _Configs()
