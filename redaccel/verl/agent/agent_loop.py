import asyncio
import pickle
import random
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Iterable, Optional
from uuid import uuid4

import numpy as np
import ray
import shortuuid
import torch
from loguru import logger
from pydantic import BaseModel

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.logging_utils import info_random
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op

from .parallel_env import (
    ToolBase,
    _get_mm_template,
    _merge_multi_modal_inputs,
)
from .utils import REDACCEL_AGENT_NAME, check_trailing_repetition


def init_tool(
    tool_name: str,
    config,
    processor,
    raw_prompt: list[dict],
    multi_modal_data: dict,
    origin_multi_modal_data: dict,
):
    if not tool_name:
        return

    def hack(cfg):
        if tool_name == "diandian":
            cfg["img_counter"] = 1
            cfg["note_id"] = 1
            cfg["img_id"] = 1
        return cfg

    tool_kwargs = {
        "parameters": config.params,
        "config": config,
        "mm_template": _get_mm_template(processor),
    }
    tool_kwargs = hack(tool_kwargs)

    tool = ToolBase.create(tool_name, **tool_kwargs)

    reset_kwargs = {
        "raw_prompt": raw_prompt,
        "multi_modal_data": deepcopy(multi_modal_data),
        "origin_multi_modal_data": deepcopy(origin_multi_modal_data),
        "mm_template": _get_mm_template(processor),
    }
    reset_kwargs = hack(reset_kwargs)
    tool.reset(**reset_kwargs)
    return tool


class ChunkRolloutScheduler:
    class Entity(BaseModel):
        idx: int
        max_tokens: int
        over_sample_rate: float = 0.0
        over_sample_num: int = 1

    def __init__(self, max_tokens: int, sample_rate_base: float) -> None:
        self.max_tokens = max_tokens
        self.sample_rate_base = sample_rate_base

    def chunk(self, part: int) -> Iterable["ChunkRolloutScheduler.Entity"]:
        """
        max_tokens=1024 part=4 => 512 170 170 172
        """
        if part <= 1:
            return [ChunkRolloutScheduler.Entity(idx=0, max_tokens=self.max_tokens, over_sample_rate=0)]

        length = self.max_tokens // 2
        splits = [length]
        if part > 2:
            splits += [length // (part - 1) for _ in range(part - 2)]

        splits += [self.max_tokens - sum(splits)]

        # TODO(wuhuan): as config
        res = []
        for i, s in enumerate(splits):
            res.append(
                ChunkRolloutScheduler.Entity(
                    idx=i,
                    max_tokens=s,
                    over_sample_rate=self.sample_rate_base if i > 0 else 0.0,
                    over_sample_num=i,
                )
            )
        return res


class ResponseLengthPredictor:
    def __init__(self, meta: dict, max_total_length: int) -> None:
        self.meta = meta
        self.max_total_length = max_total_length

    def predict(self, prompt_ids: list) -> str:
        if len(prompt_ids) >= (self.max_total_length * 0.7):
            return "short"

        # TODO(wuhuan): 暴露为配置
        if self.meta.get("data_source") in ("revisual-r1", "skywork-math"):
            return "long"
        return "short"


@register(REDACCEL_AGENT_NAME)
class RedAccelAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, tool_invoker, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        logger.info("Performing class-level RedAccelAgentLoop initialization")

        cls.recompute_mm = config.actor_rollout_ref.model.get("recompute_mm", False)
        cls.response_length = config.data.max_response_length
        cls.max_generated_tokens = cls.response_length
        cls.prompt_length = config.data.max_prompt_length
        cls.max_total_length = config.data.max_response_length + config.data.max_prompt_length
        cls.rollout_config = config.actor_rollout_ref.rollout
        cls.agent_config = config.actor_rollout_ref.rollout.agent
        cls.max_turns = config.actor_rollout_ref.rollout.agent.max_turns
        cls.image_placeholder = config.actor_rollout_ref.rollout.agent.image_placeholder

        # NOTE(wuhuan): hack for agent
        sampling_params = {}
        if cls.agent_config.activate_agent:
            sampling_params["skip_special_tokens"] = False
            sampling_params["spaces_between_special_tokens"] = False
            sampling_params["include_stop_str_in_output"] = True
            cls.max_generated_tokens = min(
                cls.agent_config.single_response_max_tokens,
                config.data.max_response_length,
            )
            if list(cls.agent_config.custom_stop):
                sampling_params["stop"] = list(cls.agent_config.custom_stop)
            if list(cls.agent_config.bad_words):
                sampling_params["bad_words"] = list(cls.agent_config.bad_words)

        sampling_params["max_tokens"] = cls.max_generated_tokens
        cls.sampling_params = sampling_params

        cls.tokenizer = tokenizer
        cls.processor = processor

        cls.execution_pool = tool_invoker

    def _remove_bad_words_from_response(self, responses: dict) -> dict:
        # https://github.com/vllm-project/vllm/issues/15764#issuecomment-2874920383
        text = self.tokenizer.decode(responses["token_ids"])
        for k in list(self.agent_config.bad_words):
            if k in text:
                logger.warning(f"Removing bad word '{k}' from response: {text[:100]}...")
                text = text.replace(k, "")
        responses["text"] = text
        responses["token_ids"] = self.tokenizer.encode(text)
        return responses

    def _check_consistency_of_prompt_ids(self, input_ids: list[int], prompt_ids: list[int]):
        """
        ```
        <|vision_start|><|image_pad|><|image_pad|>...<|image_pad|><|vision_end|>
        =>
        <|vision_start|><|image_pad|><|vision_end|>
        ```
        """
        from redaccel.models import is_agivlm1_7, is_agivlm1_8

        if is_agivlm1_7(self.processor) or is_agivlm1_8(self.processor):
            input_id_arr = np.array(input_ids)

            # Create a mask where True indicates elements to keep
            mask = np.ones(len(input_id_arr), dtype=bool)

            # Find where the array equals the value
            is_value = input_id_arr == self.processor.image_token_id

            # Find consecutive duplicates by checking if previous element is also the value
            mask[1:] &= ~(is_value[1:] & is_value[:-1])

            assert np.array_equal(input_id_arr[mask], np.array(prompt_ids))

    async def generate(
        self,
        request_id,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        resp_len_predictor: ResponseLengthPredictor,
    ):
        max_tokens = sampling_params["max_tokens"]
        prompt_ids = deepcopy(prompt_ids)
        sampling_params = deepcopy(sampling_params)

        # NOTE(wuhuan): 一开始算即可，不需要每个 chunk 算，保证同一个 request 的类型一致，方便后面做调度
        response_length_type = resp_len_predictor.predict(prompt_ids)

        if (
            self.rollout_config.chunk_rollout_threshold <= 0
            or max_tokens <= self.rollout_config.chunk_rollout_threshold
        ):
            res = await self._generate(request_id, prompt_ids, sampling_params, response_length_type)
            if res is None:
                # 此时是极小概率情况，即还在 prefill 时候被 cancel 了，没有任何输出
                res = {"token_ids": []}
            res["text"] = self.tokenizer.decode(res["token_ids"])
            return res, {
                "text": res["text"],
                "response_length_type": response_length_type,
                "finish_reason": res["finish_reason"],
            }

        generated_prompt_ids = []
        generated_log_probs = []
        history = []
        res = {}
        for chunk in ChunkRolloutScheduler(max_tokens, self.rollout_config.over_sample_rate).chunk(
            self.rollout_config.chunk_rollout_num_chunks
        ):
            info_random(chunk.idx, f"chunk rollout {len(prompt_ids)=} {max_tokens=}, {chunk=}")
            sampling_params["max_tokens"] = chunk.max_tokens
            start_ts = time()
            res_ = await self._generate(
                request_id, prompt_ids, sampling_params, response_length_type, chunk.over_sample_rate
            )
            if res_ is None:
                history.append(
                    {
                        "prompt_length": len(prompt_ids),
                        "elapsed": time() - start_ts,
                        "chunk": chunk.model_dump(),
                        "finish_reason": "abort",
                    }
                )
                break
            res = res_
            history.append(
                {
                    "prompt_length": len(prompt_ids),
                    "response_length": len(res["token_ids"]),
                    "elapsed": time() - start_ts,
                    "text": self.tokenizer.decode(res["token_ids"]),
                    "finish_reason": res["finish_reason"],
                    "chunk": chunk.model_dump(),
                }
            )

            generated_prompt_ids += res["token_ids"]
            generated_log_probs += res["log_probs"]
            prompt_ids += res["token_ids"]
            if res["finish_reason"] in ("stop", "abort"):
                break

            # detect repetition
            is_repetition = await ray.remote(check_trailing_repetition).remote("".join([i["text"] for i in history]))
            if is_repetition:
                history[-1]["trailing_repetition"] = True
                break

            # NOTE(wuhuan): 这里 sleep 给 vllm 一点时间，否则会挂，暂时不知道咋解
            await asyncio.sleep(1)

        res["token_ids"] = generated_prompt_ids
        res["log_probs"] = generated_log_probs
        res["text"] = self.tokenizer.decode(res["token_ids"])
        return res, {"gen_history": history, "response_length_type": response_length_type}

    async def _generate(
        self,
        request_id,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        response_length_type: str = "default",
        over_sample_rate: float = 0.0,
        over_sample_num: int = 1,
    ) -> Optional[dict]:
        def get_child_request_id(suffix: str = ""):
            short_id = shortuuid.ShortUUID().random(length=8)
            return f"{request_id}-{short_id}{suffix}"

        monitor = asyncio.create_task(
            self.server_manager.monitor_longtail_workloads(
                long_tail_req_threshold=self.rollout_config.long_tail_req_threshold
            )
        )
        tasks = [
            monitor,
            asyncio.create_task(
                self.server_manager.generate(
                    request_id=get_child_request_id(),
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                    load_balance_kwargs={"lb_key": response_length_type},
                )
            ),
        ]
        for i in range(over_sample_num):
            if random.random() < over_sample_rate:
                tasks.append(
                    asyncio.create_task(
                        self.server_manager.generate(
                            request_id=get_child_request_id(f"-{i + 1}"),
                            prompt_ids=prompt_ids,
                            sampling_params=sampling_params,
                            load_balance_kwargs={"lb_key": response_length_type},
                        )
                    ),
                )

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

        # wait until all task are canceled
        for t in pending:
            try:
                await t
            except asyncio.CancelledError:
                pass

        for task in done:
            outputs = await task
            if outputs:
                return outputs

        if monitor in done:
            for task in pending:
                outputs = await task
                if outputs:
                    return outputs
            return None
        raise ValueError("No valid response from the model.")

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        from redaccel.models import is_agivlm1_7, is_agivlm1_8

        step = kwargs.get("global_steps", 0)
        validate = kwargs.get("validate", False)
        messages = list(deepcopy(kwargs["raw_prompt"]))
        prompt_ids = list(deepcopy(kwargs["raw_prompt_ids"]))  # for vllm generate
        input_ids = list(deepcopy(kwargs["input_id_list"]))  # for model training
        multi_modal_data = deepcopy(kwargs.get("multi_modal_data", {"image": []}))
        origin_multi_modal_data = deepcopy(kwargs.get("origin_multi_modal_data", None))
        multi_modal_inputs = deepcopy(kwargs.get("multi_modal_inputs", {}))
        tool_name = kwargs.get(self.agent_config.tool_name_key, "")
        sampling_params = dict(sampling_params, **self.sampling_params)

        predictor = ResponseLengthPredictor(kwargs, self.max_total_length)

        tool = init_tool(
            tool_name,
            self.agent_config,
            self.processor,
            messages,
            multi_modal_data,
            origin_multi_modal_data,
        )

        request_id = uuid4().hex
        metrics = {}
        response_mask = []
        rewards = []
        tracer = AgentTrajectory(self.config, request_id, step, validate)
        log_probs = []

        while True:
            with simple_timer("generate_sequences", metrics), tracer.guard("generate") as event:
                if multi_modal_data.get("image", []):
                    sampling_params["multi_modal_data"] = multi_modal_data

                # refine max_tokens
                sampling_params["max_tokens"] = min(
                    sampling_params["max_tokens"], self.max_total_length - len(input_ids)
                )
                event.request = {
                    "prompts": self.tokenizer.decode(prompt_ids),
                    "sampling_params": deepcopy(sampling_params),
                }
                self._check_consistency_of_prompt_ids(input_ids, prompt_ids)
                for _ in range(3):
                    responses, history = await self.generate(request_id, prompt_ids, sampling_params, predictor)
                    # FIXME(wuhuan): 高并发下仅返回 eos，先重试解决，issue：https://github.com/vllm-project/vllm/issues/19437
                    if len(responses["token_ids"]) > 1:
                        break
                    logger.warning(f"[AgentLoop] Got too short response, retrying... {responses=}")
                    await asyncio.sleep(1)
                # 为了 token in token out，先关掉这个功能
                # responses = self._remove_bad_words_from_response(responses)
                response_ids = responses["token_ids"]
                finish_reason = responses["finish_reason"]
                action = responses["text"]
                event.response = history
                if not response_ids:
                    logger.error("Response IDs should not be empty. Check the model output or sampling parameters.")

            input_ids += response_ids
            prompt_ids += response_ids
            rewards += [0] * len(response_ids)
            response_mask += [1] * len(response_ids)
            log_probs += responses["log_probs"]

            if not tool_name:
                break

            if not response_ids:
                tracer.stop("Empty action, skipping tool execution.")
                break

            if len(response_mask) >= self.max_total_length or finish_reason == "length":
                tracer.stop(f"{finish_reason=} is length or {len(response_mask)=} >= {self.max_total_length}")
                break

            if tracer.turns >= self.max_turns:
                tracer.stop(f"Reached max turns {self.max_turns}")
                break

            try:
                with simple_timer("tool_calls", metrics), tracer.guard("tool_calls") as event:
                    obs, reward, done = await self.execution_pool.execute.remote(action, tool, self.agent_config)
                    rewards[-1] += reward
                    event.request = {"action": action}
                    obs_dump = deepcopy(obs)
                    if "prompt_token_ids_vllm" in obs_dump:
                        obs_dump["prompt_vllm"] = self.tokenizer.decode(obs_dump.pop("prompt_token_ids_vllm"))
                    if "prompt_token_ids_model" in obs_dump:
                        obs_dump["prompt_model"] = self.tokenizer.decode(obs_dump.pop("prompt_token_ids_model"))
                    if "multi_modal_inputs" in obs_dump and isinstance(obs_dump["multi_modal_inputs"], dict):
                        pixel_values = obs_dump["multi_modal_inputs"].get("pixel_values")
                        obs_dump["multi_modal_inputs"]["pixel_values"] = (
                            f"{pixel_values.shape=}" if isinstance(pixel_values, torch.Tensor) else None
                        )

                    event.response = {"observation": obs_dump, "reward": reward, "done": done}

            except Exception as e:
                logger.exception(f"[Tool] Execution failed: {e}")
                break
            finally:
                tracer.step()

            if done:
                tracer.stop("Tool execution done")
                break

            if "prompt_token_ids_vllm" not in obs.keys() or "prompt_token_ids_model" not in obs.keys():
                tracer.stop("Missing prompt token ids in observation")
                break

            obs_token_ids_vllm = obs["prompt_token_ids_vllm"].tolist()
            obs_token_ids_model = obs["prompt_token_ids_model"].tolist()

            if len(response_mask) + len(obs_token_ids_vllm) > self.response_length:
                tracer.stop(f"{len(response_mask)=} + {len(obs_token_ids_vllm)=} > {self.response_length=}")
                break

            if len(response_mask) + len(obs_token_ids_model) > self.response_length:
                tracer.stop(f"{len(response_mask)=} + {len(obs_token_ids_model)=} > {self.response_length=}")
                break

            prompt_ids += obs_token_ids_vllm
            input_ids += obs_token_ids_model
            response_mask += [0] * len(obs_token_ids_model)
            log_probs += [0] * len(obs_token_ids_model)
            rewards += [0] * len(obs_token_ids_model)

            mm_data = obs.get("multi_modal_data", {"image": []})
            if "image" in mm_data.keys() and len(mm_data["image"]):
                multi_modal_data["image"] += mm_data["image"]

            mm_input = obs.get("multi_modal_inputs", {})
            if mm_input and not self.recompute_mm:
                multi_modal_inputs = _merge_multi_modal_inputs(multi_modal_inputs, mm_input)

        prompt_ids = input_ids[: -len(response_mask)]
        response_ids = input_ids[-len(response_mask) :][: self.response_length]
        response_mask = response_mask[: self.response_length]

        if is_agivlm1_7(self.processor) or is_agivlm1_8(self.processor):
            img_token_sum = prompt_ids.count(self.processor.image_token_id) + response_ids.count(
                self.processor.image_token_id
            )
            grid_thw = multi_modal_inputs.get("image_grid_thw")
            mm_embed_size = 0 if grid_thw is None else (grid_thw.prod(-1).sum().item() // 4)
            assert img_token_sum == mm_embed_size, f"{img_token_sum=} does not match {mm_embed_size=}"

        extra = {
            "request_id": request_id,
            "agent_history": _to_jsonable(tracer.to_dict()),
            "env_reward": torch.tensor(rewards[: self.response_length], dtype=torch.float32),
        }
        if self.rollout_config.calculate_log_probs:
            extra["rollout_log_probs"] = torch.tensor(log_probs[: self.response_length], dtype=torch.float32)
        if self.recompute_mm:
            extra["multi_modal_data"] = multi_modal_data
        else:
            extra["multi_modal_inputs"] = multi_modal_inputs
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            num_turns=tracer.turns,
            metrics=metrics,
            extra=extra,
        )
        return output


class AgentTrajectory:
    class Event(BaseModel):
        event_type: str

        turn: int = 0
        start_ts: float
        end_ts: float
        elapsed: float = 0

        request: Optional[dict] = {}
        response: Optional[dict] = {}

    def __init__(self, config, request_id: str, global_steps: int, validate: bool = False) -> None:
        self.events: list[AgentTrajectory.Event] = []
        self.turns = 0
        self.request_id = request_id
        self.global_steps = global_steps
        self.validate = validate
        self.config = config

    def step(self):
        self.turns += 1

    def stop(self, msg: str):
        logger.info(f"[{self.request_id} turns={self.turns}] {msg}")
        self.events.append(
            AgentTrajectory.Event(
                event_type="stop",
                turn=self.turns,
                response={"text": msg},
                start_ts=time(),
                end_ts=time(),
            )
        )

    @contextmanager
    def guard(self, t: str):
        start_ts = time()
        event = AgentTrajectory.Event(
            event_type=t,
            turn=self.turns,
            start_ts=start_ts,
            end_ts=start_ts,
        )
        try:
            yield event
        finally:
            event.end_ts = time()
            event.elapsed = event.end_ts - event.start_ts
            self.events.append(event)

    def dump(self):
        self.dump_to_leveldb()

    def to_dict(self):
        return {"events": [i.model_dump() for i in self.events]}

    def dump_to_leveldb(self):
        try:
            import plyvel
        except ImportError:
            return

        output_dir = Path(self.config.trainer.default_local_dir) / "agent_trace"
        if self.validate:
            output_dir = output_dir / "val"
        output_dir.mkdir(parents=True, exist_ok=True)

        with plyvel.DB(str(output_dir), create_if_missing=True) as db:
            db.put(self.request_id.encode(), pickle.dumps(self.to_dict()))

    def dump_to_sqlite(self):
        try:
            from sqlitedict import SqliteDict
        except ImportError:
            return

        output_dir = Path(self.config.trainer.default_local_dir) / "agent_trace"
        if self.validate:
            output_dir = output_dir / "val"
        output_dir.mkdir(parents=True, exist_ok=True)

        db = SqliteDict(str(output_dir / f"{self.global_steps}.sqlite"))
        try:
            db[self.request_id] = _to_jsonable(self.to_dict())
        finally:
            db.commit()
            db.close()


def _to_jsonable(data):
    if isinstance(data, dict):
        return {key: _to_jsonable(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, list | tuple):
        return [_to_jsonable(item) for item in data]
    elif isinstance(data, float | str | bool | int):
        return data
    return str(data)
