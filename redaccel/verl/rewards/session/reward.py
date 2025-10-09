import atexit
import json
import os
from collections import defaultdict
from concurrent.futures import Future, as_completed, wait
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from verl import DataProto

from ..base import BaseComposeReward
from ..utils import to_jsonable
from .base import SessionRewards


class MultiThreadSessionComposeReward(BaseComposeReward):
    def __init__(
        self,
        rewards: list[SessionRewards],
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
        assert len(self.rewards) == 1, (
            f"Only one reward function is supported for pairwise rewards, but got {len(self.rewards)}"
        )

    def compute(self, data: DataProto, tokenizer) -> Callable[..., dict]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)

        trajectory_groups = defaultdict(list)
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            uid = data_item.non_tensor_batch["uid"]
            raw_prompt = data_item.non_tensor_batch["raw_prompt"]
            session_turn = raw_prompt[-1].get("session_turn", 0)

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

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            images = data_item.non_tensor_batch.get("multi_modal_data", {"image": []})
            kwargs = {"solution": "", "global_steps": self.step}
            kwargs["images"] = images
            try:
                ground_truth = json.loads(ground_truth)
            except BaseException:
                pass

            if isinstance(ground_truth, str):
                kwargs["solution"] = ground_truth
            elif isinstance(ground_truth, dict):
                kwargs.update(ground_truth)

            kwargs["uid"] = uid
            kwargs["idx"] = i
            kwargs["prompt"] = prompt_str
            kwargs["completion"] = completion_str
            kwargs["data_source"] = data_source
            kwargs["session_turn"] = session_turn
            kwargs["valid_response_length"] = valid_response_length

            trajectory_groups[uid].append(kwargs)

        self._submit(trajectory_groups)
        logger.info(f"[{self.stage}] all reward task submitted, num={len(data)}")

        def _wait():
            for group in self._iter_results():
                for idx, traj in group.items():
                    score = traj["score"]
                    valid_response_length = traj["valid_response_length"]
                    reward_tensor[idx, valid_response_length - 1] = score
                    if "pairwise_reward" in traj:
                        pr = traj["pairwise_reward"]
                        reward_tensor[idx, pr["valid_response_length_1"] - 1] = pr["score_1"]
                        reward_tensor[idx, pr["valid_response_length_2"] - 1] = pr["score_2"]
            self._on_step_end()
            return dict(reward_tensor=reward_tensor)

        return _wait

    def _submit(self, trajectory_groups: dict[str, list[dict]], **kwargs):
        reward: SessionRewards = self.rewards[0]
        for uid in trajectory_groups:
            if self.pool is not None:
                future = self.pool.submit(reward, trajectory_group=trajectory_groups[uid], **kwargs)
                self._tasks.append(future)
            else:
                self._results.append(reward(trajectory_group=trajectory_groups[uid], **kwargs))

    def _on_step_end(self):
        if self.pool is not None:
            wait(self._tasks)

        self._dump()
        self._clear()

    def _get_results(self):
        if self.pool is not None:
            return [f.result() for f in self._tasks]
        return self._results

    def _iter_results(self) -> Iterable[dict]:
        if self.pool is not None:
            pbar = tqdm(total=len(self._tasks), desc="calculating pairwise rewards")
            for f in as_completed(self._tasks):
                res = f.result()
                yield res
                pbar.update(1)
            pbar.close()
        else:
            yield from enumerate(self._results)

    def _dump(self):
        groups = self._get_results()
        if not groups:
            return

        results = []
        for group in groups:
            for result in group.values():
                results.append(result)

        data = {
            "step": self.step,
            "results": to_jsonable(results),
        }

        reward_dir = os.path.join(self.output_dir, "reward_data")
        os.makedirs(reward_dir, exist_ok=True)
        logger.info(f"[val={self.is_val}] dumping rewards@step={self.step} to {reward_dir}")
        with open(os.path.join(reward_dir, f"{self.step}.jsonl"), "w", encoding="utf-8") as f:
            for res in data["results"]:
                f.write(json.dumps({"step": self.step, **res}, ensure_ascii=False) + "\n")

    def _clear(self):
        self._tasks.clear()
        self._results.clear()
