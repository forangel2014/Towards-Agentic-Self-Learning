import json
import os
import time
from copy import deepcopy

from src.utils.event_tracker import EventTracker
from verl import DataProto

_start_timestamp = time.time()


class TrainerEventTracker:
    @classmethod
    def wrap(cls, tracking, trainer):
        o = cls(trainer)
        tracking.logger["_event"] = o
        return o

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self._track_on_train_begin()
        self.log_history = []

    def log(self, data, step, batch=None, *args, **kwargs):
        self._track_on_step_end(data)

    def __del__(self):
        self._track_on_train_end()

    def _track_on_train_begin(self):
        trainer = self.trainer
        tracker = EventTracker.get_instance()
        config = trainer.config
        n_gpus = trainer.resource_pool_manager.get_n_gpus()

        tracker.update("qs_gpu_num_per_node", trainer.ctx.n_gpus_per_node, emit=False)
        tracker.update("qs_node_num", trainer.ctx.nnodes, emit=False)

        if trainer.train_dataset is not None:
            tracker.update_rt("train_data_size", len(trainer.train_dataset), emit=False)

        if getattr(trainer, "val_dataset", None) is not None:
            tracker.update_rt("val_data_size", len(trainer.val_dataset), emit=False)

        tracker.updates_rt(
            {
                "qs_gpu_num_per_node": trainer.ctx.n_gpus_per_node,
                "qs_node_num": trainer.ctx.nnodes,
                "batch_size": config.data.train_batch_size // n_gpus,
                "global_batch_size": config.data.train_batch_size,
                "total_steps": trainer.total_training_steps,
                "pre_elapsed": time.time() - _start_timestamp,
                "train_start_timestamp": time.time(),
            }
        )

    def _track_on_train_end(self):
        tracker = EventTracker.get_instance()
        tracker.updates_rt(
            {
                "main_elapsed": time.time() - _start_timestamp,
                "train_stop_time": time.time(),
            },
        )

    def _track_on_step_end(self, metrics: dict):
        key = "timing_s/step"
        if key not in metrics or metrics[key] == 0:
            return

        # dump data for CE
        data = deepcopy(metrics)
        data["step"] = self.trainer.global_steps
        data["loss"] = metrics["actor/entropy_loss"]
        if "actor/lr" in metrics:
            data["learning_rate"] = metrics["actor/lr"]
        if "actor/grad_norm" in metrics:
            data["grad_norm"] = metrics["actor/grad_norm"]
        self.log_history.append(data)

        with open(self.trainer.ctx.output_dir / "trainer_state.json", "w") as f:
            json.dump({"log_history": self.log_history}, f, indent=4)

        trainer = self.trainer
        tracker = EventTracker.get_instance()
        tracker.updates_rt(
            {
                "latency_per_step": metrics[key],
                "throughput": trainer.config.data.train_batch_size / metrics[key],
                "main_elapsed": time.time() - _start_timestamp,
            },
        )


class RLLoggingBoardLogger:
    def __init__(self, root_log_dir: str, project_name: str, experiment_name: str, tokenizer):
        self.save_path = os.path.join(root_log_dir, project_name, experiment_name)
        self.tokenizer = tokenizer
        try:
            os.makedirs(self.save_path, exist_ok=True)
        except BaseException:
            pass

    def log(self, data: dict, step: int, batch: DataProto, *args, **kwargs):
        if batch is None:
            return

        tokenizer = self.tokenizer

        rm_response_list = kwargs["rm_response_list"] if "rm_response_list" in kwargs else None
        with open(os.path.join(self.save_path, "rollout_data_rank0.jsonl"), "a") as f:
            for i in range(len(batch)):
                data_item = batch[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_length = prompt_ids.shape[-1]
                rm_response = rm_response_list[i] if rm_response_list else None

                valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
                valid_prompt_ids = prompt_ids[-valid_prompt_length:]

                response_ids = data_item.batch["responses"]
                valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
                valid_response_ids = response_ids[:valid_response_length]

                prompt_str = tokenizer.decode(valid_prompt_ids)
                response_str = tokenizer.decode(valid_response_ids)
                response_tokens = [tokenizer.decode([token]) for token in valid_response_ids]
                cur_sample = {
                    "step": step,
                    "prompt": prompt_str,
                    "response": response_str,
                    "response_tokens": response_tokens,
                    "logprobs": data_item.batch["old_log_probs"][:valid_response_length].cpu().tolist(),
                    # "values": data_item.batch['values'][:valid_response_length].cpu().tolist(),
                    "token_rewards": data_item.batch["token_level_rewards"][:valid_response_length]
                    .cpu()
                    .tolist(),  # with KL penalty
                    "reward": data_item.batch["token_level_scores"][:valid_response_length]
                    .cpu()
                    .sum()
                    .item(),  # without KL penalty"
                }
                if "ref_log_prob" in data_item.batch:
                    cur_sample["ref_logprobs"] = data_item.batch["ref_log_prob"][:valid_response_length].cpu().tolist()

                if "ground_truth" in data_item.non_tensor_batch["reward_model"]:
                    cur_sample["ground_truth"] = data_item.non_tensor_batch["reward_model"]["ground_truth"]

                if "values" in data_item.batch:
                    cur_sample["values"] = data_item.batch["values"][:valid_response_length].cpu().tolist()

                if rm_response is not None:
                    cur_sample["rm_response"] = rm_response

                for key in cur_sample:
                    try:
                        json.dumps(cur_sample[key])
                    except BaseException:
                        # print(f"[WARNING] key {key} of type {type(cur_sample[key])} is not JSON serializable")
                        cur_sample[key] = ""

                f.write(f"{json.dumps(cur_sample, ensure_ascii=False)}\n")
