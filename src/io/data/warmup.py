# Copyright (c) 2024 RedNote Authors. All Rights Reserved.
import threading
import time

from src.utils import logger, singleton


@singleton
class WarmUP:
    def __init__(self):
        from src import _src

        self._handler = _src.WarmUP()
        logger.debug(f"Init warmup. Handler {self._handler}")
        self.is_running = False
        self.lock = threading.Lock()

    def __del__(self):
        logger.debug(f"Exit warmup. Handler {self._handler}")
        self.quit()

    def set_pid(self, pid):
        self._handler.set_process_name(pid)

    def set_process_name(self, pname):
        self._handler.set_process_name(pname)

    def sef_max_pending_tasks(self, num):
        self._handler.set_max_pending_tasks(num)

    def set_cache_dir(self, path):
        self._handler.set_cache_dir(path)

    def add_task(self, task):
        self._handler.add_task(task)

    def run_consumer(self):
        with self.lock:
            if self.is_running:
                logger.warn(f"WarmUP is running, cannot run consumer. handler {self._handler}")
                return None
            self.is_running = True
            self._handler.run_consumer()

    def set_thread_num(self, num):
        if self.is_running:
            logger.warn(
                f"WarmUP is running, cannot set thread num. Please set thread num before run. handler {self._handler}"
            )
            return None
        self._handler.set_thread_num(num)

    def quit(self):
        with self.lock:
            if not self.is_running:
                logger.warn(f"WarmUP is not running, cannot quit. handler {self._handler}")
                return None
            logger.debug("Call warmup quit...")
            self._handler.quit()
            self.is_running = False
            logger.debug("Fin warmup quit...")

    # def run(self):
    # self._handler.run()

    def pending_task_num(self):
        if not self.is_running:
            logger.warn(f"WarmUP is not running, cannot get pending task num. handler {self._handler}")
            return 0
        return self._handler.pending_task_num()

    def wait(self):
        while self._handler.pending_task_num():
            logger.debug("WarmUP: Wait for pending tasks...")
            time.sleep(1)
        self._handler.quit()
