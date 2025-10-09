# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.


from ....utils.registry import ClassRegistry


class SessionRewards:
    def __init__(self, name: str = "", weight: float = 1.0, **kwargs) -> None:
        self._name = name
        self._weight = weight

    @property
    def weight(self) -> float:
        return getattr(self, "_weight", 1.0)

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    def __call__(self, trajectory_group: list[dict], **kwargs) -> dict:
        raise NotImplementedError


session_rewards_registry = ClassRegistry[SessionRewards]()
