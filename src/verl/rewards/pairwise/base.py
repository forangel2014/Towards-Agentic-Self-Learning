# Copyright (c) 2025 RedNote Authors. All Rights Reserved.


from ....utils.registry import ClassRegistry


class PairwiseRewards:
    def __init__(self, name: str = "", weight: float = 1.0, **kwargs) -> None:
        self._name = name
        self._weight = weight

    @property
    def weight(self) -> float:
        return getattr(self, "_weight", 1.0)

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    def __call__(
        self,
        prompt_1: str,
        prompt_2: str,
        completion_1: str,
        completion_2: str,
        solution: str,
        **kwargs,
    ) -> tuple[float, float, dict]:
        raise NotImplementedError


pairwise_rewards_registry = ClassRegistry[PairwiseRewards]()
