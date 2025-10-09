import atexit
import gc
import json
import os
from collections import defaultdict
from concurrent.futures import Future, as_completed
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Callable, Iterable

import ray
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from verl import DataProto

from ..base import BaseComposeReward
from ..utils import AgentTraceDumper, WithWorkerGroupMixin, to_jsonable
from .base import GRPORewards


class MultiThreadComposeReward(BaseComposeReward):
    def __init__(
        self,
        rewards: list[GRPORewards],
        output_dir: Path,
        config: DictConfig,
        is_val: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(rewards, output_dir, config, is_val, **kwargs)
        if self.num_threads > 1:
            self.pool = ThreadPoolExecutor(max_workers=self.num_threads)
            atexit.register(self.pool.shutdown)
        else:
            self.pool = None
            logger.info("calculate rewards in main thread")

        self.is_request_level = config.reward_model.agent_loop
        self.dumper = AgentTraceDumper(self.output_dir / "agent_trace", enable=not self.is_request_level)
        self.dump_pool = ThreadPoolExecutor(max_workers=1)

    def compute(self, data: DataProto, tokenizer) -> Callable[..., dict]:
        if self.is_request_level:
            assert len(data) == 1, f"request level reward only support batch size 1, but got {len(data)}"

        results: list[Future | dict] = []
        for done, res in self._pre_process(data, tokenizer):
            if done:
                results.append(res)
            else:
                future = self._submit(**res)
                results.append(future)

        if self.is_request_level:

            def _request_level_wait() -> dict:
                for i, result in self._iter_results(results, tqdm_disable=True):
                    return result
                raise ValueError("No result returned")

            return _request_level_wait

        # dump 后不再需要这个字段
        data.non_tensor_batch.pop("agent_history", None)
        data.non_tensor_batch.pop("reward_results", None)  # agentloop 里面计算的
        logger.info(f"[{self.stage}] all reward task submitted, num={len(data)}")

        return partial(self._post_process, data, results)

    def _pre_process(self, data, tokenizer) -> Iterable[tuple[bool, dict]]:
        # yield is_done, result_dict
        already_print_data_sources = defaultdict(list)

        for i in tqdm(range(len(data)), desc="submit standard rewards", disable=self.is_request_level):
            data_item = data[i]  # DataProtoItem

            # 如果在 agentloop 里面已经算好了，这里就不用重新算了
            if data_item.non_tensor_batch.get("reward_results"):
                yield True, data_item.non_tensor_batch.get("reward_results")
                continue

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = tokenizer.decode(
                valid_prompt_ids,
                skip_special_tokens=self.skip_special_tokens,
            )
            completion_str = tokenizer.decode(
                valid_response_ids,
                skip_special_tokens=self.skip_special_tokens,
            )

            if "reward_model" in data_item.non_tensor_batch:
                ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            else:
                ground_truth = data_item.non_tensor_batch["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, None)
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            kwargs = {"solutions": "", "extra_info": extra_info, "global_steps": self.step}
            kwargs["images"] = data_item.non_tensor_batch.get("multi_modal_data", {"image": []})
            kwargs["request_id"] = data_item.non_tensor_batch.get("request_id", None)
            kwargs["agent_history"] = data_item.non_tensor_batch.get("agent_history", None)
            kwargs["valid_response_length"] = valid_response_length
            if isinstance(ground_truth, str):
                kwargs["solution"] = ground_truth
            elif isinstance(ground_truth, dict):
                kwargs.update(ground_truth)

            if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = int(valid_response_length) - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                kwargs["overlong_reward"] = overlong_reward
                kwargs["overlong"] = bool(overlong_reward < 0)

            yield (
                False,
                dict(
                    i=i,
                    prompts=prompt_str,
                    completions=completion_str,
                    data_source=data_source,
                    metadata=data_item.non_tensor_batch.get("metadata"),
                    **kwargs,
                ),
            )

            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = tokenizer.decode(sequences.tolist())
            if len(already_print_data_sources[data_source]) < self.num_examine and not self.is_request_level:
                already_print_data_sources[data_source].append(i)
                logger.info(f"[{self.stage}][{data_source=} {i=}] sequences_str={sequences_str}")
                logger.info(f"[{self.stage}][{data_source=} {i=}] ground_truth={ground_truth}")

    def _post_process(self, data: DataProto, results) -> dict:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        if "env_reward" in data.batch:
            reward_tensor += data.batch["env_reward"]
            logger.info(
                f"[DEBUG reward] mean={reward_tensor.mean().item()}, min={reward_tensor.min().item()}, max={reward_tensor.max().item()}"
            )

        result_dict = defaultdict(lambda: [None for _ in range(len(data))])
        reward_extra_info = {}
        for i, result in self._iter_results(results, tqdm_disable=self.is_request_level):
            score = result["score"]
            valid_len = result["valid_response_length"]
            reward_tensor[i, valid_len - 1] += score
            for key in result:
                result_dict[key][i] = result[key]

            # DAPO metric
            if self.config.algorithm.filter_groups.enable:
                metric_name = self.config.algorithm.filter_groups.metric
                if metric_name not in reward_extra_info.keys():
                    reward_extra_info[metric_name] = [None for _ in range(len(data))]
                reward_extra_info[metric_name][i] = result[metric_name]

        self._on_step_end(results)
        return dict(reward_tensor=reward_tensor, reward_extra_info=reward_extra_info, **result_dict)

    def _submit(self, i: int, prompts: str, completions: str, solutions: str, **kwargs) -> Future | dict:
        def calc(idx: int, **reward_kwargs):
            res = 0.0
            for reward in self.rewards:
                extra = {}
                if self.mock_reward:
                    score = 0
                else:
                    reward_res = reward(
                        prompts=[reward_kwargs["prompts"]],
                        completions=[reward_kwargs["completions"]],
                        solutions=[reward_kwargs["solutions"]],
                        **kwargs,
                    )[0]
                    if isinstance(reward_res, float | int):
                        score = reward_res
                    elif isinstance(reward_res, list | tuple):
                        score, extra = reward_res
                    elif isinstance(reward_res, dict):
                        score = reward_res.pop("score")
                        extra = reward_res
                    else:
                        raise ValueError(f"Invalid reward result type: {type(reward_res)}")
                res += score * reward.weight
            overlong_reward = kwargs.get("overlong_reward", 0.0)
            reward_kwargs.pop("images", None)
            return dict(idx=idx, score=res + overlong_reward, **reward_kwargs, **extra)

        if self.pool is not None:
            return self.pool.submit(
                calc, idx=i, prompts=prompts, completions=completions, solutions=solutions, **kwargs
            )

        return calc(idx=i, prompts=prompts, completions=completions, solutions=solutions, **kwargs)

    def _on_step_end(self, results):
        self.dump_pool.submit(self._dump, results=results)
        del results

    def _iter_results(self, results: list[Future | dict], tqdm_disable: bool = False) -> Iterable[tuple[int, dict]]:
        pbar = tqdm(total=len(results), desc="calculating standard rewards", disable=tqdm_disable)
        futures = {}
        for i, result in enumerate(results):
            if isinstance(result, Future):
                futures[result] = i
            elif isinstance(result, ray.ObjectRef):
                yield i, ray.get(result)
            else:
                yield i, result
                pbar.update(1)

        for f in as_completed(futures):
            res = f.result()
            idx = res.get("idx", futures[f])
            yield idx, res
            pbar.update(1)
        pbar.close()

    def _dump(self, results: list[Future | dict]):
        if self.is_request_level:
            return

        if not results:
            return

        dict_results = []
        for i, result in self._iter_results(results, tqdm_disable=True):
            assert isinstance(result, dict), f"result must be dict, but got {type(result)}"
            self.dumper.add(result["request_id"], result.pop("agent_history", None))
            dict_results.append(result)
        self.dumper.flush(self.step)

        data = {
            "step": self.step,
            "results": to_jsonable(dict_results),
        }

        reward_dir = os.path.join(self.output_dir, "reward_data")
        os.makedirs(reward_dir, exist_ok=True)
        logger.info(f"[val={self.is_val}] dumping rewards@step={self.step} to {reward_dir}")
        with open(os.path.join(reward_dir, f"{self.step}.jsonl"), "w", encoding="utf-8") as f:
            for res in data["results"]:
                f.write(json.dumps({"step": self.step, **res}, ensure_ascii=False) + "\n")
        del results, dict_results
        gc.collect()

    def _clear(self):
        gc.collect()


class WorkerGroupComposeReward(MultiThreadComposeReward, WithWorkerGroupMixin):
    def set_worker_group(self, worker_group):
        for reward in self.rewards:
            assert isinstance(reward, WithWorkerGroupMixin)
            reward.set_worker_group(worker_group)

    def compute(self, data: DataProto, tokenizer) -> Callable[..., dict]:
        if self.is_request_level:
            raise NotImplementedError("request level reward not supported in WorkerGroupComposeReward")

        assert len(self.rewards) == 1, f"Only one reward function is supported, but got {len(self.rewards)}"

        tasks = []
        for done, res in self._pre_process(data, tokenizer):
            assert not done, "done should be False in WorkerGroupComposeReward"
            res.update({"tokenizer": tokenizer, "data": data[res["i"]]})
            tasks.append(res)

        # batch calculate
        zipped_task = defaultdict(list)
        for sample in tasks:
            for k in sample:
                zipped_task[k].append(sample[k])

        results = [{"score": 0.0} for _ in range(len(tasks))]
        reward_output = self.rewards[0](**zipped_task)
        is_last_step = False
        for i, reward_res in enumerate(reward_output):
            if i == len(tasks):
                # TODO(wuhuan): 这个是干嘛用的?
                is_last_step = reward_res
            else:
                results[i].update(tasks[i])
                if isinstance(reward_res, float | int):
                    results[i]["score"] = float(reward_res)
                elif isinstance(reward_res, list | tuple):
                    results[i]["score"] = reward_res[0]
                    results[i].update(reward_res[1])
                elif isinstance(reward_res, dict):
                    results[i].update(reward_res)
                else:
                    raise ValueError(f"Invalid reward result type: {type(reward_res)}")
                results[i].pop("data")
                results[i].pop("image", None)

        for d in results:
            d.update({"is_last_step": is_last_step})

        # dump 后不再需要这个字段
        data.non_tensor_batch.pop("agent_history", None)
        data.non_tensor_batch.pop("reward_results", None)  # agentloop 里面计算的
        logger.info(f"[{self.stage}] all reward task submitted, num={len(data)}")

        return partial(self._post_process, data, results)
