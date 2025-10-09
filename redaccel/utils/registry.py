# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import importlib.util
import os
import sys
from pathlib import Path
from typing import Generic, List, Optional, Type, TypeVar

from loguru import logger


T = TypeVar("T")


class ClassRegistry(Generic[T]):
    def __init__(self):
        self._registry = {}

    def register(self, alias: Optional[List[str]] = None, ignore_if_exist: bool = False):
        def inner(cls: Type[T]) -> Type[T]:
            nonlocal alias

            if alias is None:
                alias = []
            cls_names = alias + [cls.__name__]
            for cls_name in cls_names:
                if cls_name in self._registry:
                    if ignore_if_exist:
                        continue
                    raise ValueError(f"Class '{cls_name}' already registered.")
                self._registry[cls_name] = cls
            return cls

        return inner

    def have(self, cls_name) -> bool:
        return cls_name in self._registry

    def get(self, cls_name) -> Type[T]:
        if cls_name not in self._registry:
            raise KeyError(f"Class '{cls_name}' not found in registry.")
        return self._registry[cls_name]

    def create(self, cls_name, *args, **kwargs) -> T:
        cls = self.get(cls_name)
        return cls(*args, **kwargs)

    def list_registered_classes(self) -> List[str]:
        return list(self._registry.keys())


def load_plugins(plugin_dir_s: Optional[str]):
    if not plugin_dir_s:
        return

    plugin_dir = Path(plugin_dir_s)
    if not plugin_dir.exists():
        logger.warning(f"plugin dir {plugin_dir} does not exist, skip loading plugins")
        return

    if (plugin_dir / "__init__.py").exists():
        return load_plugin_module(plugin_dir)

    return load_plugin_files(plugin_dir)


def load_plugin_module(plugin_dir: Path):
    logger.info(f"loading plugin module from {plugin_dir}")

    init_file = plugin_dir / "__init__.py"
    if init_file.exists():
        module_path = init_file
    else:
        raise ImportError(f"Directory {plugin_dir} is not a Python package")

    module_name = os.path.splitext(os.path.basename(module_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None:
        raise ImportError(f"Could not load spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    dir_path = os.path.dirname(module_path)
    if os.path.basename(dir_path):
        module.__package__ = os.path.basename(dir_path)
    else:
        module.__package__ = module_name

    module.__name__ = module_name
    module.__file__ = str(module_path)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)


def load_plugin_files(plugin_dir: Path):
    logger.info(f"loading plugin files from {plugin_dir}")

    for file_path in plugin_dir.glob("**/*.py"):
        try:
            module_name = file_path.stem  # 获取文件名（不带后缀）
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            logger.info(f"loaded plugin {module_name} from {file_path}")
        except BaseException as e:
            logger.error(f"failed to load plugin {file_path}, error: {e}")
