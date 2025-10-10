import atexit
import gc
import json
import os
from collections import defaultdict
from concurrent.futures import Future, as_completed, wait
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from rich import print
from tqdm import tqdm
from loguru import logger

from ..base import BaseComposeReward
from .base import PairwiseRewards
from verl import DataProto
from ..utils import to_jsonable


class MultiThreadPairwiseComposeReward(BaseComposeReward):
    def __init__(
        self,
        rewards: list[PairwiseRewards],
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

        self._tasks: list[Future] = []
        self._results: list[dict] = []

    def compute(self, data: DataProto, tokenizer) -> Callable[..., dict]:
        assert len(data) % 2 == 0, f"Data length should be even, but got {len(data)}"

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        valid_response_len = []
        pair_num = len(data) // 2

        already_print_data_sources = defaultdict(int)
        for i in tqdm(range(pair_num), desc="submit pairwise reward..."):
            data_1_item = data[i]
            data_2_item = data[i + pair_num]
            prompt_1_ids, prompt_2_ids = data_1_item.batch["prompts"], data_2_item.batch["prompts"]
            prompt_1_length, prompt_2_length = prompt_1_ids.shape[-1], prompt_2_ids.shape[-1]

            valid_prompt_1_length = data_1_item.batch["attention_mask"][:prompt_1_length].sum()
            valid_prompt_2_length = data_2_item.batch["attention_mask"][:prompt_2_length].sum()
            valid_response_length_1 = data_1_item.batch["attention_mask"][prompt_1_length:].sum()
            valid_response_length_2 = data_2_item.batch["attention_mask"][prompt_2_length:].sum()

            valid_prompt_1_ids = prompt_1_ids[-valid_prompt_1_length:]
            valid_prompt_2_ids = prompt_2_ids[-valid_prompt_2_length:]

            valid_resp_1_ids = data_1_item.batch["responses"][:valid_response_length_1]
            valid_resp_2_ids = data_2_item.batch["responses"][:valid_response_length_2]

            prompt_str_1 = tokenizer.decode(valid_prompt_1_ids, skip_special_tokens=False)
            completion_str_1 = tokenizer.decode(valid_resp_1_ids, skip_special_tokens=False)
            prompt_str_2 = tokenizer.decode(valid_prompt_2_ids, skip_special_tokens=False)
            completion_str_2 = tokenizer.decode(valid_resp_2_ids, skip_special_tokens=False)

            ground_truth = data_1_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_1_item.non_tensor_batch[self.reward_fn_key]
            images1 = data_1_item.non_tensor_batch.get(
                "multi_modal_data", {"image": []}
            )  # [{"image": [pil0, pil1, ...]}, ]
            images2 = data_2_item.non_tensor_batch.get(
                "multi_modal_data", {"image": []}
            )  # [{"image": [pil0, pil1, ...]}, ]
            agent_history1 = data_1_item.non_tensor_batch.get("agent_history", None)
            agent_history2 = data_2_item.non_tensor_batch.get("agent_history", None)

            kwargs = {"solution": "", "global_steps": self.step}
            kwargs["images1"] = images1
            kwargs["images2"] = images2
            kwargs["agent_history1"] = agent_history1
            kwargs["agent_history2"] = agent_history2
            try:
                ground_truth = json.loads(ground_truth)
            except BaseException:
                pass

            if isinstance(ground_truth, str):
                kwargs["solution"] = ground_truth
            elif isinstance(ground_truth, dict):
                kwargs.update(ground_truth)

            # 为了计算 think length he answer length
            kwargs["completion_ids_1"] = valid_resp_1_ids.tolist()
            kwargs["completion_ids_2"] = valid_resp_2_ids.tolist()

            self._submit(
                i=i,
                prompt_1=prompt_str_1,
                prompt_2=prompt_str_2,
                completion_1=completion_str_1,
                completion_2=completion_str_2,
                data_source=data_source,
                **kwargs,
            )
            valid_response_len.append((valid_response_length_1, valid_response_length_2))

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"{prompt_str_1=} {completion_str_1=}")

        logger.info(f"[{self.stage}] all reward task submitted, num={len(data)}")

        def _wait():
            result_dict = defaultdict(lambda: [None for _ in range(len(data))])
            for i, result in self._iter_results():
                score_1, score_2 = result["score_1"], result["score_2"]
                valid_response_len_1, valid_response_len_2 = valid_response_len[i]
                reward_tensor[i, valid_response_len_1 - 1] = score_1
                reward_tensor[i + pair_num, valid_response_len_2 - 1] = score_2

                for key in result:
                    result_dict[key][i] = result[key]
                    result_dict[key][i + pair_num] = result[key]

            self._on_step_end()
            return dict(reward_tensor=reward_tensor, **result_dict)

        return _wait

    def _submit(
        self,
        i: int,
        prompt_1: str,
        prompt_2: str,
        completion_1: str,
        completion_2: str,
        solution: str,
        **kwargs,
    ):
        def calc(idx: int, **reward_kwargs):
            score1, score2 = 0.0, 0.0
            extra = {}
            for reward in self.rewards:
                if self.mock_reward:
                    scores = (0, 0, {})
                else:
                    scores = reward(**reward_kwargs)

                assert isinstance(scores[0], float | int), f"reward {reward} should return a float or int score, but got {scores[0]}"
                assert isinstance(scores[1], float | int), f"reward {reward} should return a float or int score, but got {scores[1]}"
                assert isinstance(scores[2], dict), f"reward {reward} should return a dict as extra info, but got {scores[2]}"

                score1 += scores[0] * reward.weight
                score2 += scores[1] * reward.weight
                extra.update(scores[2])

            out = dict(reward_kwargs, **extra)
            out.pop("score_1", None)
            out.pop("score_2", None)
            # 省内存
            out.pop("images1", None)
            out.pop("images2", None)
            return dict(idx=idx, score_1=score1, score_2=score2, **out)

        submit_kwargs = dict(
            prompt_1=prompt_1,
            prompt_2=prompt_2,
            completion_1=completion_1,
            completion_2=completion_2,
            solution=solution,
            **kwargs,
        )
        if self.pool is not None:
            future = self.pool.submit(calc, idx=i, **submit_kwargs)
            self._tasks.append(future)
        else:
            self._results.append(calc(idx=i, **submit_kwargs))

    def _on_step_end(self):
        if self.pool is not None:
            wait(self._tasks)

        self._dump()
        self._clear()

    def _get_results(self):
        if self.pool is not None:
            return [f.result() for f in self._tasks]
        return self._results

    def _iter_results(self):
        if self.pool is not None:
            pbar = tqdm(total=len(self._tasks), desc="calculating pairwise rewards")
            for f in as_completed(self._tasks):
                res = f.result()
                idx = res.pop("idx")
                yield idx, res
                pbar.update(1)
            pbar.close()
        else:
            yield from enumerate(self._results)

    def _dump(self):
        results = self._get_results()
        if not results:
            return
        data = {
            "step": self.step,
            "results": to_jsonable(results),
        }

        # reward_file = os.path.join(self.output_dir, "pairwise_reward_data.jsonl")
        # logger.info(f"[val={self._is_val}] dumping rewards@step={self._step} to {reward_file}")
        # with open(reward_file, "w" if self._step == 1 else "a", encoding="utf-8") as f:
        #     f.write(json.dumps(data, ensure_ascii=False) + "\n")

        reward_dir = os.path.join(self.output_dir, "reward_data")
        os.makedirs(reward_dir, exist_ok=True)
        logger.info(f"[val={self.is_val}] dumping rewards@step={self.step} to {reward_dir}")
        with open(os.path.join(reward_dir, f"{self.step}.jsonl"), "w", encoding="utf-8") as f:
            for res in data["results"]:
                f.write(json.dumps({"step": self.step, **res}, ensure_ascii=False) + "\n")

    def _clear(self):
        self._tasks.clear()
        self._results.clear()
        gc.collect()
