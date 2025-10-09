# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import sys
from contextlib import contextmanager

from ...utils.log import logger


def get_ray_worker_num() -> int:
    try:
        import ray

        # init and connect to existing ray cluster
        if not ray.is_initialized():
            ray.init()

        nodes = ray.nodes()
        cnt = 0
        for node in nodes:
            if "GPU" in node["Resources"]:
                cnt += 1
        return cnt
    except BaseException as e:
        logger.error(f"Failed to get ray worker num: {e}")
        return 1


@contextmanager
def _set_argv(*argv):
    ori = sys.argv
    try:
        sys.argv = list(argv)
        yield
    finally:
        sys.argv = ori
