# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.

import platform

import accelerate
import datasets
import peft
import torch
import transformers
from transformers.utils import is_torch_cuda_available, is_torch_npu_available

from .. import _version


VERSION = _version.get_versions()["version"]


def print_env() -> None:
    info = {
        "`redaccel` version": VERSION,
        "Platform": platform.platform(),
        "Python version": platform.python_version(),
        "PyTorch version": torch.__version__,
        "Transformers version": transformers.__version__,
        "Datasets version": datasets.__version__,
        "Accelerate version": accelerate.__version__,
        "PEFT version": peft.__version__,
    }

    if is_torch_cuda_available():
        info["PyTorch version"] += " (GPU)"
        info["GPU type"] = torch.cuda.get_device_name()

    if is_torch_npu_available():
        info["PyTorch version"] += " (NPU)"
        info["NPU type"] = torch.npu.get_device_name()
        info["CANN version"] = torch.version.cann

    try:
        import deepspeed  # type: ignore

        info["DeepSpeed version"] = deepspeed.__version__
    except Exception:
        pass

    try:
        import bitsandbytes

        info["Bitsandbytes version"] = bitsandbytes.__version__
    except Exception:
        pass

    print("\n" + "\n".join(["- {}: {}".format(key, value) for key, value in info.items()]) + "\n")


def str_to_bool(s):
    if isinstance(s, bool):  # 如果已经是布尔类型，直接返回
        return s
    elif isinstance(s, str):  # 如果是字符串类型，进行转换
        s = s.lower()
        true_values = ["true", "1", "t", "y", "yes"]
        false_values = ["false", "0", "f", "n", "no"]

        if s in true_values:
            return True
        elif s in false_values:
            return False
        else:
            raise ValueError(f"Invalid value for boolean conversion: '{s}'")
    else:
        raise TypeError(f"Expected a string or boolean value, got {type(s)}")
