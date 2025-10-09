# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.


from ....utils.registry import ClassRegistry


class GRPORewards:
    def __init__(self, name: str = "", weight: float = 1.0) -> None:
        self._name = name
        self._weight = weight

    @property
    def weight(self) -> float:
        return getattr(self, "_weight", 1.0)

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    def __call__(
        self, prompts: list[str], completions: list[str], solutions: list[str], **kwargs
    ) -> list[float | tuple[float, dict]]:
        """reward class invoke.

        Parameters
        ----------
        prompts : List[str]
            user input prompts, len = batch size
            用户输入的 prompt
        completions : List[str]
            grpo completions
            policy 模型 rollout 输出的答案，长度和 prompts 一致
        solutions : List[str]
            ground truth answers
            用户输入的数据集中的标准答案，长度和 prompts 一致
        **kwargs :
            - metadata: List[dict] 如果用户输入数据集中带了 metadata，可以直接取用
            - ...

        Returns
        -------
        float reward score list, length must be equal to batch size
        """

        raise NotImplementedError


rewards_registry = ClassRegistry[GRPORewards]()
