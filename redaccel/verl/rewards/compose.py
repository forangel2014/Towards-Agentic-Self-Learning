# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from omegaconf import DictConfig
from pydantic import BaseModel

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager

from ...utils.log import logger
from ...utils.registry import ClassRegistry, load_plugins
from .base import BaseComposeReward
from .pairwise import MultiThreadPairwiseComposeReward, pairwise_rewards_registry
from .session import MultiThreadSessionComposeReward, session_rewards_registry
from .std import MultiThreadComposeReward, WorkerGroupComposeReward, rewards_registry
from .utils import WithWorkerGroupMixin


class ConstructConfig(BaseModel):
    clz: str
    name: str = ""
    params: dict = {}


def load_compose_reward_manager(tokenizer, config: DictConfig, is_val: bool = True):
    if "reward_funcs" in config.reward_model.reward_kwargs:
        # reward funcs 优先级最高，直接用于构造
        # example: [{"clz": $NAME, "params": {...}}, {...}]
        reward_funcs = config.reward_model.reward_kwargs.pop("reward_funcs")
    else:
        # 如果没给 reward_funcs，则通过 reward_name 构造
        if config.reward_model.reward_name is None:
            # 兼容旧版本的 data_source
            reward_name = config.data.data_source
        else:
            reward_name = config.reward_model.reward_name

        reward_funcs = [{"clz": reward_name}]
        if config.reward_model.reward_params is not None:
            reward_funcs[0]["params"] = config.reward_model.reward_params

    load_plugins(config.trainer.plugin_dir)
    return ComposeRewardManager(
        tokenizer=tokenizer,
        config=config,
        reward_funcs=reward_funcs,
        output_dir=config.trainer.default_local_dir,
        is_val=is_val,
    )


T = TypeVar("T")


def _create_rewards(
    registry: ClassRegistry[T],
    reward_cfgs: list[ConstructConfig],
) -> list[T]:
    reward_funcs: list = []

    for cfg in reward_cfgs:
        if registry.have(cfg.clz):
            reward_funcs.append(registry.create(cfg.clz, name=cfg.name, **cfg.params))

    return reward_funcs


class ComposeRewardManager(AbstractRewardManager, WithWorkerGroupMixin):
    def __init__(
        self,
        tokenizer,
        config: DictConfig,
        reward_funcs: Optional[list],
        output_dir: str,
        is_val: bool = False,
    ) -> None:
        # init reward functions
        assert reward_funcs, "No reward function is provided."
        reward_cfgs = [ConstructConfig.model_validate(i) for i in reward_funcs]
        logger.info(f"init reward funcs from config: {reward_cfgs}")

        reward_funcs = []

        output_dir_ = Path(output_dir)
        if is_val:
            output_dir_ = output_dir_ / "val"

        self.rewards: dict[str, BaseComposeReward] = {}
        if reward_funcs := _create_rewards(rewards_registry, reward_cfgs):
            if isinstance(reward_funcs[0], WithWorkerGroupMixin):
                self.rewards["std"] = WorkerGroupComposeReward(reward_funcs, output_dir_, config, is_val)
                logger.info(f"init WorkerGroupComposeReward, funcs: {reward_funcs}")
            else:
                self.rewards["std"] = MultiThreadComposeReward(reward_funcs, output_dir_, config, is_val)
                logger.info(f"init MultiThreadComposeReward, funcs: {reward_funcs}")

        if reward_funcs := _create_rewards(pairwise_rewards_registry, reward_cfgs):
            self.rewards["pairwise"] = MultiThreadPairwiseComposeReward(
                _create_rewards(pairwise_rewards_registry, reward_cfgs), output_dir_, config, is_val
            )
            logger.info(f"init MultiThreadPairwiseComposeReward, funcs: {reward_funcs}")

        if reward_funcs := _create_rewards(session_rewards_registry, reward_cfgs):
            self.rewards["session"] = MultiThreadSessionComposeReward(
                _create_rewards(session_rewards_registry, reward_cfgs), output_dir_, config, is_val
            )
            logger.info(f"init MultiThreadSessionComposeReward, funcs: {reward_funcs}")

        self.tokenizer = tokenizer

        if not self.rewards:
            raise ValueError(
                "no valid rewards found in configs for data_source, please check you reward plugins and registry"
            )

    def update_step(self, step):
        for reward in self.rewards.values():
            reward.set_step(step)

    def set_worker_group(self, worker_group: dict[str, Any]):
        if isinstance(self.rewards["std"], WithWorkerGroupMixin):
            self.rewards["std"].set_worker_group(worker_group)

    def async_call(self, data: DataProto) -> Callable[..., dict]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch:
            # TODO(wuhuan): dump results
            return lambda: dict(reward_tensor=data.batch["rm_scores"])

        for k in self.rewards:
            # NOTE(wuhuan): 暂时只取第一个
            return self.rewards[k].compute(data, self.tokenizer)
        raise ValueError("rewards is empty")

    def __call__(self, data: DataProto, return_dict: bool = True) -> dict:
        res = self.async_call(data)()
        if return_dict:
            return res
        return res["reward_tensor"]
