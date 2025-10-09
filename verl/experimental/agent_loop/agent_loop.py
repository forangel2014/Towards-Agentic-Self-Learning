# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import heapq
import random
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, Optional

import hydra
import numpy as np
import ray
import torch
from cachetools import LRUCache
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tensordict import TensorDict
from transformers import AutoTokenizer

from verl.protocol import DataProto
from verl.single_controller.ray.base import RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils.profiler.performance import simple_timer
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr, rollout_trace_op
from verl.utils.tokenizer import hf_processor
from verl.utils.torch_functional import pad_sequence_to_length
from verl.workers.rollout.async_server import async_server_class


@ray.remote(concurrency_groups={"acquire": 1, "release": 10, "reset": 1})
class GlobalLoadBalancer:
    def __init__(self, config: DictConfig, num_servers: int, max_cache_size: int = 10000):
        from loguru import logger

        self.config = config
        self.num_servers = num_servers

        # TODO: add a config for this
        self.max_loads_per_server = 256
        self.total_capacity = self.max_loads_per_server * num_servers
        self._semaphore = threading.Semaphore(self.total_capacity)
        self._current_loads = defaultdict(lambda: [0] * num_servers)  # Track current load of each server
        self._lock = threading.Lock()  # Protect concurrent access to _current_loads

        logger.info(f"[GlobalLoadBalancer] max_loads_per_server: {self.max_loads_per_server}")
        logger.info(f"[GlobalLoadBalancer] total_capacity: {self.total_capacity}")

        # Least requests load balancing
        self.weighted_serveres = [[0, server_index] for server_index in range(num_servers)]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
        self.requests_on_the_fly = {}

        self.report_daemon = threading.Thread(target=self.report_workerloads, daemon=True)
        self.report_daemon.start()

    def report_workerloads(self):
        from loguru import logger

        if not self.config.actor_rollout_ref.rollout.get("load_balance", False):
            return

        while 1:
            try:
                request_ids = list(self.requests_on_the_fly.keys())
                request_num = len(request_ids)
                request_ids = list(set([i.split("-")[0] for i in request_ids]))  # 去重
                unique_request_num = len(request_ids)
                if request_num > 0:
                    logger.info(f"current workloads: {dict(self._current_loads)} {request_num=} {unique_request_num=}")
            except Exception:
                logger.exception("Error in report_workerloads")
            time.sleep(10)

    def get_workloads(self) -> dict:
        if not self.config.actor_rollout_ref.rollout.get("load_balance", False):
            return {}

        return {
            "workloads": deepcopy(dict(self._current_loads)),
            "request_ids": deepcopy(dict(self.requests_on_the_fly)),
        }

    @ray.method(concurrency_group="acquire")
    def get_server_index(self, request_id: str, lb_key: str = "default") -> int:
        """Get the server index that should be used"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # Acquire semaphore permission
            self._semaphore.acquire()

            # Select server with minimum load
            with self._lock:
                min_load_idx = min(range(self.num_servers), key=lambda i: self._current_loads[lb_key][i])
                self._current_loads[lb_key][min_load_idx] += 1
                server_index = min_load_idx

                self.requests_on_the_fly[request_id] = server_index
            return server_index
        else:
            return self._choose_server_index(request_id)

    @ray.method(concurrency_group="release")
    def release_server_index(self, request_id: str, server_index: int, lb_key: str = "default"):
        """Release server index"""
        if self.config.actor_rollout_ref.rollout.get("load_balance", False):
            # Decrease server load count
            with self._lock:
                if self._current_loads[lb_key][server_index] > 0:
                    self._current_loads[lb_key][server_index] -= 1

                self.requests_on_the_fly.pop(request_id, None)
            # Release semaphore permission
            self._semaphore.release()

    def _choose_server_index(self, request_id: str) -> int:
        # adapt oversampled requests
        request_id_prefix = request_id.split("-")[0]
        if request_id_prefix in self.request_id_to_server:
            return self.request_id_to_server[request_id_prefix]

        server_index = self.weighted_serveres[0][1]
        self.weighted_serveres[0][0] += 1
        heapq.heapreplace(self.weighted_serveres, self.weighted_serveres[0])
        self.request_id_to_server[request_id_prefix] = server_index
        return server_index

    @ray.method(concurrency_group="reset")
    def reset(self):
        """Reset load balancer state, including semaphore and load counts"""
        with self._lock:
            # Recreate semaphore
            self._semaphore = threading.Semaphore(self.total_capacity)
            # Reset load counts for all servers
            self._current_loads = defaultdict(lambda: [0] * self.num_servers)


class AsyncLLMServerManager:
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        global_load_balancer: ray.actor.ActorHandle,
        max_cache_size: int = 10000,
    ):
        """Initialize the AsyncLLMServerManager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
            max_cache_size (int, optional): max cache size for request_id to server mapping. Defaults to 10000.
        """
        self.config = config
        self.server_handles = server_handles
        random.shuffle(self.server_handles)

        # Least requests load balancing
        self.weighted_serveres = [[0, (hash(server), server)] for server in server_handles]
        heapq.heapify(self.weighted_serveres)

        # LRU cache to map request_id to server
        self.request_id_to_server = LRUCache(maxsize=max_cache_size)
        self.global_load_balancer = global_load_balancer

    async def monitor_longtail_workloads(self, long_tail_req_threshold: int = 5, long_tail_max_occur: int = 2):
        """
        - long_tail_req_threshold = 5  # 其他worker的请求数小于等于该值，则认为当前请求是长尾
        - long_tail_max_occur = 2  # 被判定为长尾次数大于等于该值，则直接杀掉
        """

        long_tail_count = 0
        while 1:
            try:
                await asyncio.sleep(30)

                # NOTE(wuhuan): 不 check 了，让服务自己抛异常
                # # check health
                # for server in self.server_handles:
                #     await server.check_health.remote()

                workloads_info = await self.global_load_balancer.get_workloads.remote()
                if "workloads" not in workloads_info:
                    continue
                global_workloads = workloads_info["workloads"]
                sum_workloads = [0] * len(self.server_handles)
                for k in global_workloads:
                    for i, v in enumerate(global_workloads[k]):
                        sum_workloads[i] += v

                request_ids = list(workloads_info["request_ids"].keys())
                request_ids = list(set([i.split("-")[0] for i in request_ids]))  # 去重

                if len(request_ids) <= long_tail_req_threshold:
                    long_tail_count += 1
                    logger.warning(
                        f"[AsyncLLMServerManager] Long tail detected {len(request_ids)=} {sum(sum_workloads)=} "
                        f"{long_tail_count=}/{long_tail_max_occur}"
                    )

                if long_tail_count >= long_tail_max_occur:
                    # 这里退出后，外面会自动 cancel generate 函数
                    logger.warning(
                        f"[AsyncLLMServerManager] Long tail detected {long_tail_count=} times, killing the server manager"
                    )
                    break
            except asyncio.CancelledError:
                break

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        load_balance_kwargs: Optional[dict] = None,
    ) -> dict:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.

        Returns:
            List[int]: List of generated token ids.
        """
        lb_key = (load_balance_kwargs or {}).get("lb_key", "default")
        server_index = await self.global_load_balancer.get_server_index.remote(request_id, lb_key)
        server = self.server_handles[server_index]
        self.request_id_to_server[request_id] = server_index
        output = {}
        try:
            output = await server.generate_with_cancel.remote(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
            )
        except asyncio.CancelledError:
            output = await server.cancel.remote(request_id)
            output_token_ids = output.get("token_ids", [])
            logger.warning(
                f"[AsyncLLMServerManager] Task cancelled: {request_id}, request length: {len(prompt_ids)} "
                f"resp length: {len(output_token_ids)}"
            )
        finally:
            await self.global_load_balancer.release_server_index.remote(request_id, server_index, lb_key)

        return output


class AgentLoopMetrics(BaseModel):
    """Agent loop performance metrics."""

    generate_sequences: float = 0.0
    tool_calls: float = 0.0
    compute_score: float = 0.0


class AgentLoopOutput(BaseModel):
    """Agent loop output."""

    prompt_ids: list[int]
    """Prompt token ids."""
    response_ids: list[int]
    """Response token ids including LLM generated token, tool response token."""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    num_turns: int = 0
    """Number of chat turns, including user, assistant, tool."""
    metrics: AgentLoopMetrics
    """Auxiliary performance metrics"""

    extra: dict


class _InternalAgentLoopOutput(AgentLoopOutput):
    """Internal agent loop output with padded sequences."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt_ids: torch.Tensor
    """Padded prompt token ids."""
    response_ids: torch.Tensor
    """Padded response token ids."""
    response_mask: torch.Tensor
    """Padded response mask."""
    attention_mask: torch.Tensor
    """Padded attention mask."""
    position_ids: torch.Tensor


# make hydra.utils.instantiate happy
class _DummyConfig:
    def __init__(self, config: DictConfig) -> None:
        self.config = config


@ray.remote(num_cpus=1)
class RewardManagerWorker:
    """Reward manager worker to compute reward score asynchronously to overlap with agent loop."""

    def __init__(self, config: DictConfig, local_path: str) -> None:
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.enable = True
        if config.reward_model.reward_manager == "compose":
            from redaccel.verl.rewards.compose import load_compose_reward_manager

            config.reward_model.agent_loop = True
            self.reward_manager = load_compose_reward_manager(tokenizer, config, is_val=False)

            # 只支持标准 reward, pairwise 等 reward 不支持
            for k in self.reward_manager.rewards:
                if k != "std":
                    self.enable = False
        else:
            raise NotImplementedError(f"Unsupported reward manager {config.reward_model.reward_manager}")

        self.loop = asyncio.get_event_loop()
        self.events = {}

    async def cancel_all(self):
        from loguru import logger

        logger.info("Cancelling all pending reward computations...")
        for v in self.events.values():
            v.set()
        self.evets = {}
        logger.info("All pending reward computations cancelled.")

    async def compute_score(self, output: AgentLoopOutput, kwargs: dict) -> dict:
        """Compute reward score for agent loop output.
        NOTE: Since `reward_manager.__call__` is blocking function, we run it in thread pool to
        compute multiple samples in parallel.
        Args:
            output (AgentLoopOutput): Agent loop output.
            kwargs (dict): Dataset fields from `verl.utils.dataset.RLHFDataset`.
        Returns:
            float: Reward score.
        """
        if not self.enable:
            return {}

        prompts = torch.tensor(output.prompt_ids, dtype=torch.long).unsqueeze(0)
        responses = torch.tensor(output.response_ids, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones((1, prompts.shape[1] + responses.shape[1]), dtype=torch.long)
        batch = TensorDict(
            {
                "prompts": prompts,  # [1, prompt_length]
                "responses": responses,  # [1, response_length]
                "attention_mask": attention_mask,  # [1, prompt_length + response_length]
            },
            batch_size=1,
        )
        non_tensor_batch = {
            **{k: np.array([v]) for k, v in output.extra.items()},
            **{k: np.array([v]) for k, v in kwargs.items()},
            "__num_turns__": np.array([output.num_turns]),
        }
        data = DataProto(
            batch=batch,
            non_tensor_batch=non_tensor_batch,
            meta_info={
                "global_steps": kwargs.get("global_steps", 1),
            },
        )
        active_event = asyncio.Event()
        cancel_handle = asyncio.create_task(active_event.wait())
        self.events[cancel_handle] = active_event
        reward_handle = self.loop.run_in_executor(
            None,
            self.reward_manager,
            data,
        )

        done, pending = await asyncio.wait([reward_handle, cancel_handle], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()

        if reward_handle in done:
            active_event.set()
            self.events.pop(cancel_handle)
            return reward_handle.result()

        # wait until all task are canceled
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass
        return {}


class AgentLoopBase(ABC):
    """An agent loop takes a input message, chat with OpenAI compatible LLM server and interact with various
    environments."""

    _class_initialized = False

    def __init__(
        self, trainer_config: _DummyConfig, server_manager: AsyncLLMServerManager, tokenizer: AutoTokenizer, **kwargs
    ):
        """Initialize agent loop, each sample will have its own loop instance.

        Args:
            trainer_config (_DummyConfig): trainer config.
            server_manager (AsyncLLMServerManager): OpenAI compatible LLM server manager.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.init_class(trainer_config.config, tokenizer, **kwargs)
        self.config = trainer_config.config
        self.server_manager = server_manager
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer, **kwargs):
        """This is used to do heavy initialization work that should shared across all instances. It's only called once.

        Args:
            config (DictConfig): trainer config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
            **kwargs: extra kwargs from config file passed in by `hydra.utils.instantiate`.
        """
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run agent loop to interact with LLM server and environment.

        Args:
            sampling_params (Dict[str, Any]): LLM sampling params.
            **kwargs: dataset fields from `verl.utils.dataset.RLHFDataset`.

        Returns:
            AgentLoopOutput: Agent loop output.
        """
        raise NotImplementedError


"""Agent loop registry: key is agent_name, value is a dict of agent loop config
used by hydra.utils.instantiate to initialize agent loop instance.

https://hydra.cc/docs/advanced/instantiate_objects/overview/
"""
_agent_loop_registry: dict[str, dict] = {}


def register(agent_name: str):
    """Register agent loop class."""

    def decorator(subclass: type[AgentLoopBase]) -> type[AgentLoopBase]:
        fqdn = f"{subclass.__module__}.{subclass.__qualname__}"
        _agent_loop_registry[agent_name] = {"_target_": fqdn}
        return subclass

    return decorator


@ray.remote
class AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        global_load_balancer: ray.actor.ActorHandle,
        tool_invoker: ray.actor.ActorHandle,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): YAML config.
            server_handles (List[ray.actor.ActorHandle]): OpenAI compatible LLM server actor handles.
        """
        from redaccel.models import register_all_models
        from redaccel.utils.registry import load_plugins

        register_all_models()
        load_plugins(config.actor_rollout_ref.rollout.plugin_dir)

        self.config = config
        self.server_manager = AsyncLLMServerManager(config, server_handles, global_load_balancer)
        self.tool_invoker = tool_invoker

        model_path = config.actor_rollout_ref.model.path
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        agent_loop_config_path = config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                _agent_loop_registry[agent_loop_config.name] = agent_loop_config

        self.reward_manager_worker = RewardManagerWorker.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=ray.get_runtime_context().get_node_id(),
                soft=False,
            ),
        ).remote(self.config, local_path)

        trace_config = self.config.actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
        )

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        """Generate sequences from agent loop.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            repetition_penalty=1.0,
        )

        # override sampling params for validation
        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["temperature"] = config.val_kwargs.temperature

        # by default, we assume it's a single turn agent
        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array(["redaccel_agent"] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index = batch.non_tensor_batch["index"]
        else:
            index = np.arange(len(batch))

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        from ray.experimental.tqdm_ray import tqdm

        tasks = []
        gen_events = {}
        pbar = tqdm(total=len(batch), desc="Async Rollout...")
        for i in range(len(batch)):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            kwargs["global_steps"] = batch.meta_info.get("global_steps")
            kwargs["validate"] = batch.meta_info.get("validate", False)
            gen_events[i] = asyncio.Event()
            tasks.append(
                asyncio.create_task(
                    self._run_agent_loop(pbar, gen_events[i], sampling_params, trajectory_info[i], **kwargs)
                )
            )
        tasks.append(asyncio.create_task(self._on_all_gen_done(gen_events)))
        outputs = await asyncio.gather(*tasks)
        pbar.close()

        output = self._postprocess(outputs[: len(batch)])
        return output

    async def _on_all_gen_done(self, events: dict[int, asyncio.Event]):
        for e in events.values():
            await e.wait()

        # NOTE(wuhuan): 全程传输 Future，无需 cancel all
        # logger.info(f"All {len(events)} generation done, cancelling pending reward computations...")
        # await self.reward_manager_worker.cancel_all.remote()

    async def _run_agent_loop(
        self,
        pbar,
        event: asyncio.Event,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=_DummyConfig(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                tool_invoker=self.tool_invoker,
            )
            output = await agent_loop.run(sampling_params, **kwargs)
            event.set()
            # compute reward
            score_metrics = {}
            with simple_timer("compute_score", score_metrics):
                if not kwargs.get("validate", False):
                    # reward_results = await self.reward_manager_worker.compute_score.remote(output, kwargs)
                    reward_results = self.reward_manager_worker.compute_score.remote(output, kwargs)
                    output.extra["reward_results"] = reward_results
            output.metrics.compute_score = score_metrics["compute_score"]

            # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
            # prompt_ids: left padded with zeros (e.g., [0,0,0,0,1,2,3,4])
            # response_ids: right padded with zeros (e.g., [5,6,7,8,0,0,0,0])
            # input_ids: concatenation of prompt + response
            # Mask:
            # For example, if the prompt is [1,2,3,4] and the response is [5,6,7,(tool start)8,9(tool end),10,11,12]
            # - prompt_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [0,0,0,0,1,1,1,1]
            # - response_attention_mask: 0s for padding, 1s for tokens
            #   e.g., [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]
            # attention_mask: concatenation of prompt_attention_mask and response_attention_mask
            #   e.g., [0,0,0,0,1,1,1,1(prompt),1,1,1,1,1,1,1,1,1,1,1,0,0,0,0(response)]
            # - response_mask: 1s for LLM generated tokens, 0 for tool response/padding tokens
            #   e.g., [1,1,1,1,1,1,1,(tool start),0,0(tool end),1,1,0,0,0,0]
            # - position_ids: sequential positions for tokens, starting at 0
            #   e.g., [0,0,0,0,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,0,0,0,0]

            self.tokenizer.padding_side = "left"
            prompt_output = self.tokenizer.pad(
                {"input_ids": output.prompt_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.prompt_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if prompt_output["input_ids"].dim() == 1:
                prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
                prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

            self.tokenizer.padding_side = "right"
            response_output = self.tokenizer.pad(
                {"input_ids": output.response_ids},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=True,
            )
            if response_output["input_ids"].dim() == 1:
                response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
                response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

            response_mask_output = self.tokenizer.pad(
                {"input_ids": output.response_mask},
                padding="max_length",
                max_length=self.config.actor_rollout_ref.rollout.response_length,
                return_tensors="pt",
                return_attention_mask=False,
            )
            if response_mask_output["input_ids"].dim() == 1:
                response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

            response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
            attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
            input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

            from redaccel.models import is_agivlm

            if (
                self.processor is not None
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
                and not is_agivlm(self.processor)
            ):
                from verl.models.transformers.qwen2_vl import get_rope_index

                multi_modal_inputs = output.extra.get("multi_modal_inputs", {})
                image_grid_thw = multi_modal_inputs.get("image_grid_thw")
                video_grid_thw = multi_modal_inputs.get("video_grid_thw")
                second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts")

                vision_position_ids = get_rope_index(
                    self.processor,
                    input_ids=input_ids.squeeze(0),
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask.squeeze(0),
                ).unsqueeze(0)  # (1, 3, seq_len)

                valid_mask = attention_mask[0].bool()
                text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
                text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
                text_position_ids = text_position_ids.unsqueeze(0)
                position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
            else:
                position_ids = compute_position_id_with_mask(attention_mask)  # (1, seq_len)

            pbar.update(1)
            return _InternalAgentLoopOutput(
                prompt_ids=prompt_output["input_ids"],
                response_ids=response_output["input_ids"],
                response_mask=response_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                num_turns=output.num_turns,
                metrics=output.metrics,
                extra=output.extra,
            )

    def _postprocess(self, inputs: list[_InternalAgentLoopOutput]) -> DataProto:
        """Process the padded outputs from _run_agent_loop and combine them into a batch."""
        # Convert lists back to tensors and stack them to create a batch.
        prompt_ids = torch.cat([input.prompt_ids for input in inputs], dim=0)
        response_ids = torch.cat([input.response_ids for input in inputs], dim=0)
        response_mask = torch.cat([input.response_mask for input in inputs], dim=0)
        attention_mask = torch.cat([input.attention_mask for input in inputs], dim=0)
        position_ids = torch.cat([input.position_ids for input in inputs], dim=0)

        input_ids = torch.cat([prompt_ids, response_ids], dim=1)

        batch = TensorDict(
            {
                "prompts": prompt_ids,  # [bsz, prompt_length]
                "responses": response_ids,  # [bsz, response_length]
                "response_mask": response_mask,  # [bsz, response_length]
                "input_ids": input_ids,  # [bsz, prompt_length + response_length]
                "attention_mask": attention_mask,  # [bsz, prompt_length + response_length]
                "position_ids": position_ids,  # [bsz, prompt_length + response_length]
            },
            batch_size=len(inputs),
        )

        num_turns = np.array([input.num_turns for input in inputs], dtype=np.int32)
        metrics = [input.metrics.model_dump() for input in inputs]

        # TODO(wuhuan): process reward tensor

        # # request level rewards
        # scores = [input.reward_ for input in inputs]
        # if all(score is not None for score in scores):
        #     prompt_length = prompt_ids.size(1)
        #     response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1
        #     rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        #     rm_scores[torch.arange(response_mask.size(0)), response_length] = torch.tensor(scores, dtype=torch.float32)
        #     batch["rm_scores"] = rm_scores

        non_tensor_batch = {
            "__num_turns__": num_turns,
            "request_id": np.array([input.extra.get("request_id", "") for input in inputs], dtype=object),
            "agent_history": np.array([input.extra.get("agent_history", []) for input in inputs], dtype=object),
            "reward_results": np.array([input.extra.get("reward_results", {}) for input in inputs], dtype=object),
        }

        if "multi_modal_data" in inputs[0].extra:
            non_tensor_batch["multi_modal_data"] = np.array(
                [input.extra.get("multi_modal_data", {"image": []}) for input in inputs], dtype=object
            )
        if "multi_modal_inputs" in inputs[0].extra:
            non_tensor_batch["multi_modal_inputs"] = np.array(
                [input.extra.get("multi_modal_inputs", {}) for input in inputs], dtype=object
            )

        for k in ["rollout_log_probs", "env_reward"]:
            if k in inputs[0].extra:
                batch[k] = torch.cat(
                    [
                        pad_sequence_to_length(
                            input.extra[k],
                            self.config.actor_rollout_ref.rollout.response_length,
                            0,
                        ).view(1, -1)
                        for input in inputs
                    ],
                    dim=0,
                )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={"metrics": metrics})


async def get_trajectory_info(step, index, validate):
    """Get trajectory info.

    Args:
        step (int): global steps in the trainer.
        index (list): form datastore extra_info.index column.
        validate (bool): whether is a validate step.

    Returns:
        list: trajectory.
    """
    trajectory_info = []
    rollout_n = 0
    for i in range(len(index)):
        if i > 0 and index[i - 1] == index[i]:
            rollout_n += 1
        else:
            rollout_n = 0
        trajectory_info.append({"step": step, "sample_index": index[i], "rollout_n": rollout_n, "validate": validate})
    return trajectory_info


class ToolExecutionWorker:
    def __init__(self, config) -> None:
        # NOTE(wuhuan): 为了控制 tool 全局并发，但是不该放这里，反向引用了，后面要把 envs 抽出来
        from redaccel.models import register_all_models
        from redaccel.verl.agent.parallel_env import execute_tool_call

        register_all_models()

        local_path = config.actor_rollout_ref.model.path
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)
        self.processor = hf_processor(local_path, trust_remote_code=True)

        self.execute_tool_call = execute_tool_call

    def ping(self):
        return True

    def execute(self, action, tool, agent_config):
        sample = {"action": action, "tool": tool}
        obs, reward, done, _ = self.execute_tool_call(sample, agent_config, self.tokenizer, self.processor)
        return obs, reward, done


class AgentLoopManager:
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(self, config: DictConfig, worker_group: RayWorkerGroup):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group.
        """
        self.config = config
        self.worker_group = worker_group

        self._initialize_llm_servers()
        self._init_global_server_manager()
        self._init_agent_loop_workers()

        # Initially we're in sleep mode.
        self.sleep()

    def _initialize_llm_servers(self):
        self.rollout_tp_size = self.config.actor_rollout_ref.rollout.tensor_model_parallel_size
        self.rollout_pp_size = self.config.actor_rollout_ref.rollout.pipeline_model_parallel_size
        self.rollout_mp_size = self.rollout_tp_size * self.rollout_pp_size
        self.rollout_dp_size = self.worker_group.world_size // self.rollout_mp_size

        register_center = ray.get_actor(f"{self.worker_group.name_prefix}_register_center")
        workers_info = ray.get(register_center.get_worker_info.remote())
        assert len(workers_info) == self.worker_group.world_size

        self.async_llm_servers = [None] * self.rollout_dp_size
        self.server_addresses = [None] * self.rollout_dp_size

        if self.config.actor_rollout_ref.rollout.agent.custom_async_server:
            server_class = async_server_class(
                rollout_backend=self.config.actor_rollout_ref.rollout.name,
                rollout_backend_module=self.config.actor_rollout_ref.rollout.agent.custom_async_server.path,
                rollout_backend_class=self.config.actor_rollout_ref.rollout.agent.custom_async_server.name,
            )
        else:
            server_class = async_server_class(rollout_backend=self.config.actor_rollout_ref.rollout.name)

        # Start all server instances, restart if address already in use.
        unready_dp_ranks = set(range(self.rollout_dp_size))
        while len(unready_dp_ranks) > 0:
            servers = {
                rollout_dp_rank: server_class.options(
                    # make sure AsyncvLLMServer colocates with its corresponding workers
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=workers_info[rollout_dp_rank * self.rollout_mp_size],
                        soft=False,
                    ),
                    name=f"async_llm_server_{rollout_dp_rank}",
                ).remote(self.config, self.rollout_dp_size, rollout_dp_rank, self.worker_group.name_prefix)
                for rollout_dp_rank in unready_dp_ranks
            }

            for rollout_dp_rank, server in servers.items():
                try:
                    address = ray.get(server.get_server_address.remote())
                    self.server_addresses[rollout_dp_rank] = address
                    self.async_llm_servers[rollout_dp_rank] = server
                    unready_dp_ranks.remove(rollout_dp_rank)
                except Exception:
                    ray.kill(server)
                    logger.exception(
                        f"rollout server {rollout_dp_rank} failed, maybe address already in use, restarting..."
                    )

        # All server instances are ready, init AsyncLLM engine.
        ray.get([server.init_engine.remote() for server in self.async_llm_servers])

    def _init_global_server_manager(self):
        """Create global AsyncLLMServerManager as Ray Actor"""
        self.global_load_balancer = GlobalLoadBalancer.options(
            name="global_async_llm_load_balancer",
        ).remote(self.config, self.rollout_dp_size)
        logger.info("[AgentLoopManager] Created global load balancer")

    def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.config.actor_rollout_ref.rollout.agent.num_workers
        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"]]
        for i in range(num_workers):
            # Round-robin scheduling over the all nodes
            node_id = node_ids[i % len(node_ids)]
            tool_invoker = (
                ray.remote(ToolExecutionWorker)
                .options(
                    name=f"tool_invoker_{i}",
                    max_concurrency=self.config.actor_rollout_ref.rollout.agent.concurrent_workers,
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                )
                .remote(self.config)
            )
            self.agent_loop_workers.append(
                AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(self.config, self.async_llm_servers, self.global_load_balancer, tool_invoker)
            )
            logger.info(f"[AgentLoopManager] Created agent loop worker {i} on node {node_id}")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Split input batch and dispatch to agent loop workers.

        Args:
            prompts (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
        """
        # Reset global load balancer at the beginning of each generate call
        ray.get(self.global_load_balancer.reset.remote())
        logger.info("[AgentLoopManager] Reset global load balancer")

        timing = {}
        if self.config.actor_rollout_ref.rollout.free_cache_engine:
            with marked_timer("reshard", timing):
                self.wake_up()

        with marked_timer("generate_sequences", timing):
            chunkes = prompts.chunk(len(self.agent_loop_workers))
            outputs = ray.get(
                [
                    worker.generate_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
                ]
            )
            output = DataProto.concat(outputs)

            if self.config.actor_rollout_ref.rollout.free_cache_engine:
                self.sleep()

            # calculate performance metrics
            metrics = [output.meta_info["metrics"] for output in outputs]  # List[List[Dict[str, str]]]
            timing = self._performance_metrics(timing, metrics, output)

        output.meta_info = {"timing": timing}
        return output

    def _performance_metrics(self, timing, metrics: list[list[dict[str, str]]], output: DataProto) -> dict[str, float]:
        t_generate_sequences = np.array([metric["generate_sequences"] for chunk in metrics for metric in chunk])
        t_tool_calls = np.array([metric["tool_calls"] for chunk in metrics for metric in chunk])
        t_compute_score = np.array([metric["compute_score"] for chunk in metrics for metric in chunk])

        timing["agent_loop/generate_sequences/min"] = t_generate_sequences.min()
        timing["agent_loop/generate_sequences/max"] = t_generate_sequences.max()
        timing["agent_loop/generate_sequences/mean"] = t_generate_sequences.mean()

        timing["agent_loop/tool_calls/min"] = t_tool_calls.min()
        timing["agent_loop/tool_calls/max"] = t_tool_calls.max()
        timing["agent_loop/tool_calls/mean"] = t_tool_calls.mean()

        timing["agent_loop/compute_score/min"] = t_compute_score.min()
        timing["agent_loop/compute_score/max"] = t_compute_score.max()
        timing["agent_loop/compute_score/mean"] = t_compute_score.mean()

        # batch sequence generation is bounded by the slowest sample
        slowest = np.argmax(t_generate_sequences + t_tool_calls)
        attention_mask = output.batch["attention_mask"][slowest]
        prompt_length = output.batch["prompts"].shape[1]
        timing["agent_loop/slowest/generate_sequences"] = t_generate_sequences[slowest]
        timing["agent_loop/slowest/tool_calls"] = t_tool_calls[slowest]
        timing["agent_loop/slowest/prompt_length"] = attention_mask[:prompt_length].sum().item()
        timing["agent_loop/slowest/response_length"] = attention_mask[prompt_length:].sum().item()

        return timing

    def wake_up(self):
        """Wake up all rollout server instances."""
        ray.get([server.wake_up.remote() for server in self.async_llm_servers])

    def sleep(self):
        """Sleep all rollout server instances."""
        ray.get([server.sleep.remote() for server in self.async_llm_servers])
