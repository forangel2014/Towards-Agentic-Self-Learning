# Copyright (c) 2024 RedAccel Authors. All Rights Reserved.
import functools


def singleton(class_):
    instances = {}

    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance


def once(func):
    """保证函数只执行一次."""
    result = None
    executed = False

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal result, executed
        if not executed:
            result = func(*args, **kwargs)
            executed = True
        return result

    return wrapper
