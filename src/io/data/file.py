# Copyright (c) 2024 RedNote Authors. All Rights Reserved.
import io
import os

from src.utils import logger, singleton
from src.utils.configs import Configs

from .cache import QuickSilverCache


@singleton
class QuickSilverFile:
    def __init__(self, cache_dir=None, pid=os.getpid()):
        from .handler import DefaultHandler

        self.enable_disk_cache = Configs.IO_USE_DISK_CACHE
        self._cache = QuickSilverCache(cache_dir)
        use_juicefs_sdk = Configs.IO_USE_JUICEFS_SDK
        self._handler = DefaultHandler(use_juicefs_sdk=use_juicefs_sdk)
        self._handler.set_process_name(str(pid))

    def set_pid(self, pid):
        self._handler.set_process_name(pid)

    def set_process_name(self, pname):
        self._handler.set_process_name(pname)

    def get_handler(self):
        return self._handler

    def _download_to_buffer(self, path):
        assert path.startswith("/mnt/"), f"only support quicksilver fs now. {path}"
        data = self._handler.read(path)
        return data, len(data)

    def _download(self, src, dst):
        # assert src.startswith("/mnt/"), "only support quicksilver fs now."
        size = self._handler.read_to_file(src, dst)
        return size

    def cache(self, path):
        if not self.enable_disk_cache:
            return None, 0
        dst = self._cache.get_disk_cache_path(path)
        if not os.path.isdir(os.path.dirname(dst)):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
        size = self._download(path, dst)
        return dst, size

    def get_cache_path(self, path):
        return self._cache.get_cache_path(path)

    def stream_read(self, path, base_dir=os.path.dirname(__file__), write_cache=False):
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        ret = self._cache.stream_read(path)
        if ret:
            return ret
        if write_cache:
            cachepath, size = self.cache(path)
            ret = self._cache.stream_read(path)
            return ret
        else:
            data, size = self._download_to_buffer(path)
            return io.BytesIO(data)

    def download_to_disk_cache(self, path):
        cachepath, size = self.cache(path)

    def read(self, path, base_dir=os.path.dirname(__file__), write_cache=False):
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        if self.enable_disk_cache:
            data = self._cache.read(path)
        if data:
            logger.debug("Read data from cache. ")  ##
            return data

        data, size = self._download_to_buffer(path)
        if write_cache:
            self._cache.write(path, data)
        return data

    def list(self, path, max_num=-1, recursive=False):
        path = os.path.abspath(path)
        prefix = "/".join(path.split("/")[:3])
        cursor = ""
        cnt = 0
        while True:
            (files, cursor) = self._handler.list(path, 65535, cursor, recursive)
            for i in files:
                abspath = i
                if os.path.join(prefix, i).startswith(path.rstrip("/")):
                    # alluxio
                    abspath = os.path.join(prefix, i)
                if not abspath.startswith(path):
                    abspath = os.path.join(path, i)
                if abspath and abspath.rstrip("/") != path.rstrip("/"):
                    yield abspath
                cnt += 1
                if max_num > 0 and cnt >= max_num:
                    return
            if len(cursor) == 0:
                break
