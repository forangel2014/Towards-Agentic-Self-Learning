# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from hydra import compose, initialize
from loguru import logger
from rich import print

from ..hparams.parser import get_ray_worker_num
from .workflow import run_ppo


def get_output_dir(config) -> Path:
    if config.trainer.default_local_dir:
        out_dir = Path(config.trainer.default_local_dir).absolute()
        logger.info(f"output_dir is set to {out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    tmp_dir = (
        Path(config.trainer.working_dir or os.getcwd())
        / "saves"
        / f"{config.trainer.project_name}"
        / f"{config.trainer.experiment_name}"
    )
    logger.warning(f"output_dir is not set, using default dir: {tmp_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir


@contextmanager
def get_config(hydra_overrides):
    inst = initialize(version_base=None, config_path="../verl/trainer/config")
    config_name = os.getenv("RL_CONFIG_NAME", "ppo_trainer")
    config_name = f"_generated_{config_name}"

    config = compose(config_name=config_name, overrides=hydra_overrides)
    if config.trainer.working_dir is None:
        config.trainer.working_dir = os.getcwd()
    config.trainer.default_local_dir = str(get_output_dir(config))
    try:
        yield config
    finally:
        inst.__exit__(None, None, None)


def run_exp(args: Optional[dict[str, Any]] = None) -> None:
    """支持：

    1. train style 启动： redaccel-cli rl config.yaml
    2. verl style 启动：redaccel-cli rl a=1 b=2
    3. 混合启动：redaccel-cli rl config.yaml a=1 b=2
    """
    extra_args = sys.argv[1:]
    overrides = {}

    for arg in extra_args:
        eles = arg.split("=", 1)
        if len(eles) != 2:
            logger.warning(f"Invalid argument: {arg}")
            continue

        if eles[0] in overrides:
            logger.info(f"Overriding {eles[0]} from {overrides[eles[0]]} to {eles[1]}")
        overrides[eles[0]] = eles[1]

    ray_worker_num = get_ray_worker_num() or 1
    gpu_num = int(os.getenv("GPU_NUM", 8))
    overrides["trainer.nnodes"] = ray_worker_num
    overrides["trainer.n_gpus_per_node"] = gpu_num

    hydra_overrides = [f"{k}={v}" for k, v in overrides.items()]

    with get_config(hydra_overrides) as config:
        try:
            return run_ppo(config)
        except:
            print(overrides)
            raise
