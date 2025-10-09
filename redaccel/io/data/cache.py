# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.
import os
import threading

from redaccel.utils import logger, singleton
from redaccel.utils.configs import Configs


@singleton
class QuickSilverCache:
    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir
        self.enable_disk_cache = Configs.IO_USE_DISK_CACHE
        self.disk_cache_dir = self._get_cache_directory()
        self.cache_usage_bytes = 0

    def _get_cache_directory(self):
        if self.cache_dir is None:
            self.cache_dir = Configs.IO_CACHE_DIR
        self._create_directory(self.cache_dir)
        return self.cache_dir

    def _get_disk_cache_path(self, path):
        return self.disk_cache_dir + path

    def get_disk_cache_path(self, path):
        return self._get_disk_cache_path(path)

    def _create_directory(self, directory):
        os.makedirs(directory, exist_ok=True)

    def get_cache_path(self, path):
        return self._get_disk_cache_path(path)

    def stream_read(self, path):
        cache_path = self._get_disk_cache_path(path)
        if os.path.exists(cache_path):
            try:
                fp = open(cache_path, "rb")
            except OSError:
                return None
            return fp
        return None

    def read(self, path):
        path = self._get_disk_cache_path(path)
        if os.path.exists(path):
            try:
                with open(path, "rb") as file:
                    return file.read()
            except OSError:
                return None
        return None

    def write(self, path, data, sync=True):
        path = self._get_disk_cache_path(path)
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            self._create_directory(dirname)
        try:
            tmp_file = path + f".tmp.{os.getpid()}.{threading.current_thread().ident}"
            with open(tmp_file, "wb") as file:
                ret = file.write(data)
            os.rename(tmp_file, path)
            return ret
        except Exception as err:
            logger.warn(f"Error writing to disk cache: {err}")
        return None

    def write_stream(self, path, stream):
        path = self._get_disk_cache_path(path)
        dirname = os.path.dirname(path)
        if not os.path.isdir(dirname):
            self._create_directory(dirname)
        try:
            tmp_file = path + f".tmp.{os.getpid()}.{threading.current_thread().ident}"
            with open(tmp_file, "wb") as file:
                for chunk in stream:
                    file.write(chunk)
            os.rename(tmp_file, path)
            return True
        except Exception as err:
            logger.warning(f"Error writing to disk cache: {err}")
        return False
