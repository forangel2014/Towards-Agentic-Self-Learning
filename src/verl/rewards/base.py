from pathlib import Path
from typing import Callable

from loguru import logger
from omegaconf import DictConfig

from verl import DataProto


class BaseComposeReward:
    def __init__(
        self,
        rewards: list,
        output_dir: Path,
        config: DictConfig,
        is_val: bool = False,
        **kwargs,
    ) -> None:
        self.rewards = rewards
        self.config = config

        self.step = 1
        self.is_val = is_val
        self.stage = "validate" if is_val else "train"

        self.mock_reward = config.reward_model.mock
        self.num_threads = config.reward_model.num_threads
        self.num_examine = config.reward_model.num_examine
        self.skip_special_tokens = config.reward_model.skip_special_tokens
        self.reward_fn_key = config.data.reward_fn_key
        # DAPO
        self.max_resp_len = config.data.max_response_length
        self.overlong_buffer_cfg = config.reward_model.get("overlong_buffer", None)

        logger.info(f"{self.mock_reward=} {self.num_threads=} {self.num_examine=} {self.skip_special_tokens=}")

        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {self.overlong_buffer_cfg=}, but got None"
            )

    def compute(self, data: DataProto, tokenizer) -> Callable[..., dict]:
        raise NotImplementedError

    def set_step(self, step: int):
        self.step = step
