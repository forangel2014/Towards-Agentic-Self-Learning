# Copyright (c) 2025 RedNote Authors. All Rights Reserved.


def main():
    from .train.tuner import run_exp

    run_exp()


def tool():
    from .tools import ray_entrypoint

    ray_entrypoint()
