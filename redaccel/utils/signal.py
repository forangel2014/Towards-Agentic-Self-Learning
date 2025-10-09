# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import os
import signal
import threading
from types import FrameType
from typing import Any, Callable, List, Union


_SIGNUM = Union[int, signal.Signals]
_HANDLER = Union[Callable[[_SIGNUM, FrameType], Any], int, signal.Handlers, None]


class _HandlersCompose:
    def __init__(self, signal_handlers: Union[list[_HANDLER], _HANDLER]) -> None:
        if not isinstance(signal_handlers, list):
            signal_handlers = [signal_handlers]
        self.signal_handlers = signal_handlers

    def __call__(self, signum: _SIGNUM, frame: FrameType) -> None:
        pid = os.getpid()
        for signal_handler in self.signal_handlers:
            if isinstance(signal_handler, int):
                signal_handler = signal.getsignal(signal_handler)
            if callable(signal_handler):
                signal_handler(signum, frame)
        raise SystemError(f"[{pid}] receive signal {signum}, exit")


def _has_handler(signum: _SIGNUM) -> bool:
    return signal.getsignal(signum) not in (None, signal.SIG_DFL)


def _register_signal(signum: _SIGNUM, handlers: _HANDLER) -> None:
    if threading.current_thread() is threading.main_thread():
        signal.signal(signum, handlers)


def register_handler(sig: _SIGNUM, handlers: List[_HANDLER]):
    """注册signal handler
    NOTE(wuhuan): 不要在自己的 handler 里 raise ，否则导致后续 handler 没法触发
    """
    if _has_handler(sig):
        handlers.append(signal.getsignal(sig))
    _register_signal(sig, _HandlersCompose(handlers))
