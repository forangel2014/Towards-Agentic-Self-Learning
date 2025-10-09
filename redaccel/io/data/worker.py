# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.
import os
import queue
import random
from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch._utils import ExceptionWrapper
from torch.utils.data._utils import HAS_NUMPY, MP_STATUS_CHECK_INTERVAL, signal_handling
from torch.utils.data._utils.worker import WorkerInfo

from redaccel.utils.log import logger

from .fetcher import RedIterableDatasetFetcher
from .file import QuickSilverFile
from .warmup import WarmUP


class ManagerWatchdog:  # type: ignore[no-redef]
    def __init__(self):
        self.manager_pid = os.getppid()
        self.manager_dead = False

    def is_alive(self):
        if not self.manager_dead:
            self.manager_dead = os.getppid() != self.manager_pid
        return not self.manager_dead


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    worker_id: int


r"""Dummy class used to resume the fetching when worker reuse is enabled"""


@dataclass(frozen=True)
class _ResumeIteration:
    seed: Optional[int] = None


def _generate_state(base_seed, worker_id):
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = 4 * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


def _prefetch_thread_loop(path_queue):
    try:
        signal_handling._set_worker_signal_handlers()

        while True:
            try:
                r = path_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            if r is None:
                break
            QuickSilverFile().stream_read(r, write_cache=True)
            del r  # save memory
    except KeyboardInterrupt:
        pass


def get_worker_info() -> Optional[WorkerInfo]:
    r"""Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code.
    """
    return _worker_info


def _prefetch_worker_loop(
    dataset_kind,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
    rediter_count,
    rank,
):
    # rank = os.environ.get("RANK", '-')
    logger.debug(f"Prefetch worker[ID:{worker_id}] loop start...")
    warmup = WarmUP()
    warmup.set_process_name(f"prefetch_{worker_id}")
    try:
        signal_handling._set_worker_signal_handlers()

        # torch.set_num_threads(1)
        seed = base_seed + worker_id
        random.seed(seed)
        torch.manual_seed(seed)
        path_queue = queue.Queue()
        thread_num = 64
        if os.environ.get("REDACCEL_PREFETCH_THREAD_NUM", None):
            thread_num = int(os.environ["REDACCEL_PREFETCH_THREAD_NUM"])

        warmup.set_thread_num(thread_num)
        warmup.run_consumer()
        if worker_id == 0:
            logger.info("Prefetch io thread num: {}".format(thread_num))

        if HAS_NUMPY:
            np_seed = _generate_state(base_seed, worker_id)
            import numpy as np

            np.random.seed(np_seed)

        from torch.utils.data import IterDataPipe
        from torch.utils.data.graph_settings import apply_random_seed

        shared_rng = torch.Generator()
        if isinstance(dataset, IterDataPipe):
            assert shared_seed is not None
            shared_rng.manual_seed(shared_seed)
            dataset = apply_random_seed(dataset, shared_rng)

        global _worker_info
        _worker_info = WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)
        warmup.set_cache_dir(dataset.cache_dir)

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)
            fetcher = RedIterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        except Exception:
            init_exception = ExceptionWrapper(where=f"in DataLoader prefetch worker process {worker_id}")

        # When using Iterable mode, some worker can exit earlier than others due
        # to the IterableDataset behaving differently for different workers.
        # When such things happen, an `_IterableDatasetStopIteration` object is
        # sent over to the main process with the ID of this worker, so that the
        # main process won't send more tasks to this worker, and will send
        # `None` to this worker to properly exit it.
        #
        # Note that we cannot set `done_event` from a worker as it is shared
        # among all processes. Instead, we set the `iteration_end` flag to
        # signify that the iterator is exhausted. When either `done_event` or
        # `iteration_end` is set, we skip all processing step and just wait for
        # `None`.
        iteration_end = False

        watchdog = ManagerWatchdog()
        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)

            except queue.Empty:
                continue
            if isinstance(r, _ResumeIteration):
                iteration_end = False

                if isinstance(dataset, IterDataPipe):
                    assert r.seed is not None
                    shared_rng.manual_seed(r.seed)
                    dataset = apply_random_seed(dataset, shared_rng)

                # Recreate the fetcher for worker-reuse policy

                fetcher = RedIterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
                logger.debug("init prefetch process.")
                continue
            elif r is None:
                logger.debug("r is None")

                # Received the final signal
                assert done_event.is_set() or iteration_end
                break
            elif done_event.is_set() or iteration_end:
                logger.debug(f"done_event.is_set()={done_event.is_set()} or iteration_end={iteration_end}:")

                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue

            idx, index = r
            data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
            if init_exception is not None:
                logger.debug(f"init_exception not none: {init_exception}")
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
                    if data is None:
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        item, path = None, None
                        for item in data:
                            if not isinstance(item, str):
                                for path in item:
                                    assert isinstance(path, str), f"path must be str, get f{type(path)}"
                                    if len(path) == 0:
                                        continue

                                    if isinstance(path, list):
                                        for i in path:
                                            warmup.add_task(i)
                                    else:
                                        warmup.add_task(path)
                            else:
                                path = item
                                if len(path) == 0:
                                    continue

                                if isinstance(path, list):
                                    for i in path:
                                        warmup.add_task(i)
                                else:
                                    warmup.add_task(path)
                        del item, path
                except Exception as e:
                    import traceback

                    logger.error("Error: {}".format(e))
                    logger.error(f"{traceback.format_exc()=}")
                    if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        logger.warning("Prefetch Warning: {}".format(e))
                        # It is important that we don't store exc_info in a variable.
                        # `ExceptionWrapper` does the correct thing.
                        # See NOTE [ Python Traceback Reference Cycle Problem ]
                        data = ExceptionWrapper(where=f"in DataLoader prefetch worker process {worker_id}")
            del data, idx, index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        dataset.clean_disk_cache(rediter_count, rank)
        pass
    for i in range(thread_num):
        path_queue.put(None)
    if done_event.is_set() and data_queue:
        data_queue.cancel_join_thread()
        data_queue.close()

    logger.debug(f"Prefetch process[pid:{os.getpid()}] exit...")
