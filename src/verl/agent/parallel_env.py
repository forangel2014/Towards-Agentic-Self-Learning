# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import asyncio
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from verl import DataProto
from verl.models.transformers.qwen2_vl import get_rope_index
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.vision_utils import process_image, process_image_for_qwen_processor
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import pad_2d_list_to_length

from .tool_envs import ToolBase


def _strip_system_block(text: str) -> str:
    """删除 text 中第一个 <|im_start|>system ...

    <|endofassistant|> 区块（含标签）， 并返回删除后的字符串。 如果找不到匹配的开始或结束标签，则返回原文。
    """
    # 非贪婪匹配，匹配跨行，支持 qwen 和 小地瓜(agivlm)
    pattern = r"<\|im_start\|>system.*?<\|im_end\|>|<\|system\|>.*?<\|endofsystem\|>|<\|systemprompt\|>.*?<\|endofsystemprompt\|>"
    # 替换为空
    result = re.sub(pattern, "", text, flags=re.S)
    return result.lstrip("\n")


def _concat_vllm_input(prompt_token_ids, response_token_ids, tokenizer=None, invalid_token_ids=None):
    # NOTE: temporarily fix qwen-base oov issue
    if tokenizer is not None:
        max_token_id = max(tokenizer.get_vocab().values())
        tokenizer_size = len(tokenizer)
        max_token_id = max(max_token_id, tokenizer_size)
        valid_token_mask = torch.le(response_token_ids, max_token_id)
        if invalid_token_ids:
            for bad_token in invalid_token_ids:
                valid_token_mask *= torch.ne(response_token_ids, bad_token)
        response_token_ids = torch.masked_select(response_token_ids, valid_token_mask)

    if isinstance(prompt_token_ids, torch.Tensor):
        output_tensor = torch.cat(
            [
                prompt_token_ids,
                response_token_ids.to(prompt_token_ids.device),
            ],
            dim=-1,
        )
        return output_tensor.cpu().numpy().flatten().tolist()
    else:
        output_array = np.concatenate(
            [
                prompt_token_ids,
                response_token_ids.cpu().numpy(),
            ],
            axis=-1,
        )
        return output_array.flatten().tolist()


def _merge_multi_modal_inputs(mm_input, other):
    if not mm_input and not other:
        return {}
    elif len(mm_input) == 0 and len(other) > 0:
        return other
    elif len(mm_input) > 0 and len(other) == 0:
        return mm_input

    output_dict = {}
    for key in mm_input.keys():
        if key not in other.keys():
            output_dict[key] = mm_input[key]
            continue

        mm_value = mm_input[key]
        other_value = other.pop(key)
        if isinstance(mm_value, np.ndarray) and isinstance(other_value, np.ndarray):
            merged_value = np.concatenate([mm_value, other_value], axis=0)
        elif isinstance(mm_value, torch.Tensor) and isinstance(other_value, torch.Tensor):
            merged_value = torch.cat([mm_value, other_value], dim=0)
        elif mm_value is None or other_value is None:
            continue
        else:
            raise ValueError(f"Invalid {type(mm_value)=}, {type(other_value)=}")

        output_dict[key] = merged_value
    return dict(**output_dict, **other)


def _get_mm_template(processor) -> str:
    if processor is None:
        return ""

    from src.models import is_agivlm

    if is_agivlm(processor):
        return "agivlm"
    if "Qwen2VLImageProcessor" in processor.image_processor.__class__.__name__:
        return "qwen2vl"
    raise ValueError(f"Unsupported processor type: {processor}")


def _preprocess_multi_modal_inputs(prompt_str, processor, image_placeholder: str = "<image>", **kwargs):
    from src.models import is_agivlm1_6, is_agivlm1_7, is_agivlm1_8

    if processor is None or "multi_modal_data" not in kwargs:
        return prompt_str, prompt_str, {}

    if is_agivlm1_6(processor):
        input_mm_data = kwargs.get("multi_modal_data", {"image": []})
        input_mm_data["image"] = [process_image(image) for image in input_mm_data["image"]]
        processor.image_tag = image_placeholder
        model_inputs = processor(
            prompt=prompt_str, images=input_mm_data["image"], return_tensors="pt", add_special_tokens=False
        )
        return (
            prompt_str.replace(image_placeholder, "<|imgpad|>"),
            model_inputs["input_ids"],
            {"pixel_values": model_inputs["pixel_values"]} if len(input_mm_data["image"]) else {},
        )

    elif is_agivlm1_7(processor) or is_agivlm1_8(processor):
        vllm_input_prompt = prompt_str.replace(image_placeholder, "<|img|><|imgpad|><|endofimg|>")
        input_mm_data = kwargs.get("multi_modal_data", {"image": []})
        input_mm_data["image"] = [process_image_for_qwen_processor(image) for image in input_mm_data["image"]]
        model_inputs = processor(
            text=[vllm_input_prompt],
            images=input_mm_data["image"] or None,
            return_tensors="pt",
            add_special_tokens=False,
        )

        input_ids = model_inputs["input_ids"][0]
        if input_ids[0] == 0:
            # strip bos, transformers llamatokenizer bug, 设置 add_bos_token=False 没用，离谱
            input_ids = input_ids[1:]
        mm_output = {}
        if len(input_mm_data["image"]):
            mm_output = {"pixel_values": model_inputs["pixel_values"], "image_grid_thw": model_inputs["image_grid_thw"]}
        return (
            vllm_input_prompt,
            input_ids,
            mm_output,
        )

    # qwen2 vl image processor
    vllm_input_prompt = prompt_str.replace(image_placeholder, "<|vision_start|><|image_pad|><|vision_end|>")
    input_mm_data = kwargs.get("multi_modal_data", {"image": []})
    input_mm_data["image"] = [process_image_for_qwen_processor(image) for image in input_mm_data["image"]]
    model_inputs = processor(
        text=[vllm_input_prompt], images=input_mm_data["image"], return_tensors="pt", add_special_tokens=False
    )
    input_ids = model_inputs.pop("input_ids")[0]
    _ = model_inputs.pop("attention_mask")[0]

    mm_inputs = dict(model_inputs)
    return vllm_input_prompt, input_ids, mm_inputs


def _strip_image_token(s: str, rep: str = ".") -> str:
    # TODO: use processor image token attribute
    return s.replace("<|imgpad|>", rep).replace("<|image_pad|>", rep)


@dataclass
class VLLMLikeOutput:
    token_ids: list[int]
    text: str
    finish_reason: str
    log_probs: Optional[list[int]] = None


@dataclass
class VLLMLikeOutputs:
    outputs: list[VLLMLikeOutput]


def _is_vllm_engine(engine) -> bool:
    # vllm may not be installed
    is_vllm = "vllm" in engine.__class__.__module__.lower()
    if is_vllm:
        from vllm import LLM

        assert isinstance(engine, LLM), f"Engine must be an instance of vllm.LLM, got {type(engine)}"
    return is_vllm


def generate(engine, config, tokenizer, sampling_params, vllm_like_inputs: list) -> list[VLLMLikeOutputs]:
    max_generated_tokens = min(config.agent.single_response_max_tokens, config.response_length)
    custom_stop = list(config.agent.custom_stop)
    if _is_vllm_engine(engine):
        from vllm import LLM, SamplingParams

        assert isinstance(engine, LLM), f"Engine must be an instance of vllm.LLM, got {type(engine)}"
        assert isinstance(sampling_params, SamplingParams), (
            f"SamplingParams must be an instance of SamplingParams, got {type(sampling_params)}"
        )

        agent_sampling_params = sampling_params.clone()
        agent_sampling_params.detokenize = True
        agent_sampling_params.skip_special_tokens = False
        agent_sampling_params.spaces_between_special_tokens = False
        agent_sampling_params.n = 1
        agent_sampling_params.include_stop_str_in_output = True
        agent_sampling_params.max_tokens = max_generated_tokens

        # support custom stop specified in dataset, like </search>, ```, etc.
        if custom_stop:
            prev_stop = (
                sampling_params.stop
                if isinstance(sampling_params.stop, list)
                else [sampling_params.stop]
                if isinstance(sampling_params.stop, str)
                else []
            )
            agent_sampling_params.stop = prev_stop + custom_stop

        if len(config.agent.bad_words):
            agent_sampling_params.bad_words = list(config.agent.bad_words)
        return engine.generate(prompts=vllm_like_inputs, sampling_params=agent_sampling_params, use_tqdm=False)

    assert isinstance(sampling_params, dict), f"SamplingParams must be a dict, got {type(sampling_params)}"
    sampling_params["max_new_tokens"] = max_generated_tokens
    sampling_params["n"] = 1
    sampling_params["skip_special_tokens"] = False
    sampling_params["spaces_between_special_tokens"] = False
    sampling_params["no_stop_trim"] = True

    if custom_stop:
        sampling_params["stop"] = custom_stop
    # if config.agent.bad_words:
    #     raise NotImplementedError(f"Bad words filtering is not supported in vllm-like engine, please use vllm engine with SamplingParams: {config.agent.bad_words=}")

    token_list = [i["prompt_token_ids"] for i in vllm_like_inputs]
    image_list = [i.get("multi_modal_data", {}).get("image", []) for i in vllm_like_inputs]

    # NOTE(wuhuan): will be fixed by https://github.com/sgl-project/sglang/pull/7887
    image_list = [i if i else None for i in image_list]
    loop = asyncio.get_event_loop()
    output = loop.run_until_complete(
        engine.async_generate(
            prompt=None,  # because we have already convert it to prompt token id
            sampling_params=sampling_params,
            return_logprob=True,
            input_ids=token_list,
            image_data=image_list,
        )
    )

    res = []
    for o in output:
        output_token_logprobs = o["meta_info"]["output_token_logprobs"]
        log_probs, token_ids = zip(
            *[(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs], strict=False
        )
        res.append(
            VLLMLikeOutputs(
                outputs=[
                    VLLMLikeOutput(
                        token_ids=token_ids,
                        log_probs=log_probs,
                        text=tokenizer.decode(token_ids),
                        finish_reason=o["meta_info"]["finish_reason"]["type"],
                    )
                ]
            )
        )
    return res


class NoOPTPGroup:
    @property
    def is_first_rank(self) -> bool:
        return True

    def broadcast_object(self, obj):
        return obj


def agent_rollout_loop(config, engine, vllm_inputs, prompts, multi_modal_inputs, sampling_params, tp_group):
    if tp_group is None:
        tp_group = NoOPTPGroup()
    tokenizer = hf_tokenizer(config.agent.vl_model_path)
    processor = hf_processor(config.agent.vl_model_path)

    if multi_modal_inputs is not None:
        multi_modal_inputs = multi_modal_inputs.tolist()
    else:
        multi_modal_inputs = [{} for _ in range(len(vllm_inputs))]

    batch_size = len(vllm_inputs)
    vllm_input_list = []
    running_states = []
    running_action_masks = []
    running_attn_masks = []
    reward_tensor_list = []
    active_mask = []
    mm_input_list = []
    mm_data_list = []  # for recompute_mm
    tool_call_cnt_list = []
    agent_history = []
    img_counter_list = []  # Track IMG counter for each rollout instance (backward compatibility)
    # Shared kwargs for each rollout instance, used to keep note_id and img_id counters in sync.
    kwargs_list = []
    timing = {"rollout": 0.0, "agent": 0.0, "post_process": 0.0}

    sample_n = sampling_params["n"] if isinstance(sampling_params, dict) else sampling_params.n
    env = ParallelEnv(config.agent, tokenizer, processor)
    env.reset(prompts, vllm_inputs, n=sample_n, parameters=config.agent.params)

    # interleaving inputs if sample_n > 1
    for i in range(batch_size):
        for _ in range(sample_n):
            vllm_input_list.append(deepcopy(vllm_inputs[i]))
            prompt_ids = prompts.batch["input_ids"][i, :].clone()
            running_states.append(prompt_ids)
            prompt_mask = prompts.batch["attention_mask"][i, :].clone()
            running_action_masks.append(prompt_mask)
            running_attn_masks.append(prompt_mask)
            reward_tensor = torch.zeros_like(prompt_ids, dtype=torch.float)
            reward_tensor_list.append(reward_tensor)
            active_mask.append(True)
            mm_input_list.append(deepcopy(multi_modal_inputs[i]))
            mm_data_list.append(deepcopy(vllm_inputs[i].get("multi_modal_data", {"image": []})))
            tool_call_cnt_list.append(0)
            agent_history.append({"action": [], "obs": [], "stop_reason": []})
            img_counter_list.append(0)  # Initialize IMG counter for each rollout
            kwargs_list.append(SharedKwargs(note_id=1, img_id=1))

    max_total_length = config.prompt_length + config.response_length
    bad_word_tokens = []
    for bw in list(config.agent.bad_words):
        bad_word_tokens += tokenizer.encode(bw)

    for step in range(config.agent.max_turns):
        # print(f"[DEBUG 000] {step=}, total={batch_size}, n={sampling_params.n}, num_active={sum(active_mask)}")
        if sum(active_mask) == 0:
            break

        active_indices = [idx for idx, is_active in enumerate(active_mask) if is_active]
        active_vllm_inputs = []
        for vinput, is_active in zip(vllm_input_list, active_mask, strict=False):
            if not is_active:
                continue

            max_token_id = max(vinput["prompt_token_ids"])
            if max_token_id >= len(tokenizer):
                logger.error(
                    f"[{step=}][{batch_size=}] Invalid token id in vllm_input: {max_token_id=}, {len(tokenizer)=}"
                )
                vinput["prompt_token_ids"] = [i for i in vinput["prompt_token_ids"] if i < len(tokenizer)]

            # 如果没有图片，剔除 multi_modal_data 字段(vllm V1 的要求)
            if not vinput.get("multi_modal_data", {}).get("image"):
                vinput.pop("multi_modal_data", None)
            active_vllm_inputs.append(vinput)

        # NOTE(wuhuan): DUMP vllm inputs for debug if needed
        if tp_group.is_first_rank and (dump_dir := os.getenv("ROLLOUT_PROMPTS_DUMP_DIR")):
            uid = uuid.uuid4()
            import pickle

            with open(f"{dump_dir}/step{step}_{uid}.pkl", "wb") as f:
                pickle.dump(dict(prompts=active_vllm_inputs, sampling_params=sampling_params), f)

        gen_start_ts = time.time()
        actions = generate(engine, config, tokenizer, sampling_params, active_vllm_inputs)
        agent_start_ts = time.time()
        if tp_group.is_first_rank:
            obs_results = env.step(active_indices, actions, kwargs_list)
        else:
            obs_results = None

        obs_results = tp_group.broadcast_object(obs_results)
        observations, rewards, dones, info = obs_results
        agent_end_ts = time.time()

        timing["rollout"] += agent_start_ts - gen_start_ts
        timing["agent"] += agent_end_ts - agent_start_ts

        if tp_group.is_first_rank:
            logger.info(
                f"{step=} gen time: {agent_start_ts - gen_start_ts:.3f}, agent time: {agent_end_ts - agent_start_ts:.3f}s"
            )

        for idx, obs, act, rew, done in zip(active_indices, observations, actions, rewards, dones, strict=False):
            # process response token ids
            response_token_ids = torch.tensor(
                act.outputs[0].token_ids, dtype=torch.int64, device=running_states[idx].device
            )
            running_states[idx] = torch.cat([running_states[idx], response_token_ids])
            vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
                vllm_input_list[idx]["prompt_token_ids"],
                response_token_ids,
                tokenizer=tokenizer,
                invalid_token_ids=bad_word_tokens,
            )
            agent_history[idx]["action"].append(tokenizer.decode(act.outputs[0].token_ids))

            action_reward = torch.zeros_like(
                response_token_ids, dtype=torch.float, device=reward_tensor_list[idx].device
            )
            reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], action_reward])
            reward_tensor_list[idx][-1] += rew

            action_mask = torch.ones_like(
                response_token_ids, dtype=torch.int64, device=running_action_masks[idx].device
            )
            running_action_masks[idx] = torch.cat([running_action_masks[idx], action_mask])
            running_attn_masks[idx] = torch.cat([running_attn_masks[idx], action_mask])

            # Ensure the last token is not obs
            if (
                running_states[idx].shape[-1] >= max_total_length
                or len(vllm_input_list[idx]["prompt_token_ids"]) >= max_total_length
            ):
                active_mask[idx] = False
                agent_history[idx]["stop_reason"].append(
                    f"[DEBUG] {idx=} {step=} max length reached {max_total_length=}, {len(vllm_input_list[idx]['prompt_token_ids'])=}, {running_states[idx].shape[-1]=}"
                )
                print(agent_history[idx]["stop_reason"][-1])
                continue

            if done or step == config.agent.max_turns - 1:
                active_mask[idx] = False
                agent_history[idx]["stop_reason"].append(
                    f"[DEBUG] {idx=} {step=} >= {config.agent.max_turns - 1=} or done={done}"
                )
                print(agent_history[idx]["stop_reason"][-1])
                continue
            tool_call_cnt_list[idx] += 1

            # Update counters if returned from diandian tool
            if "img_counter" in obs:
                img_counter_list[idx] = obs["img_counter"]
                kwargs_list[idx]["img_id"] = obs["img_counter"]

            if "note_id" in obs:
                kwargs_list[idx]["note_id"] = obs["note_id"]

            # process obs tokens and images
            if "prompt_token_ids_vllm" in obs.keys() and "prompt_token_ids_model" in obs.keys():
                obs_token_ids_vllm = obs["prompt_token_ids_vllm"]
                obs_token_ids_model = obs["prompt_token_ids_model"].to(running_states[idx].device)

                # strip image token to avoid compose reward save file too large
                agent_history[idx]["obs"].append(_strip_image_token(tokenizer.decode(obs_token_ids_model)))

                if len(vllm_input_list[idx]["prompt_token_ids"]) + len(obs_token_ids_vllm) >= max_total_length:
                    active_mask[idx] = False
                    agent_history[idx]["stop_reason"].append(
                        f"[DEBUG] {idx=} {step=} vllm input length exceeded {max_total_length=}, {len(vllm_input_list[idx]['prompt_token_ids'])=}, {len(obs_token_ids_vllm)=}"
                    )
                    print(agent_history[idx]["stop_reason"][-1])
                    continue
                if running_states[idx].shape[-1] + len(obs_token_ids_model) >= max_total_length:
                    active_mask[idx] = False
                    agent_history[idx]["stop_reason"].append(
                        f"[DEBUG] {idx=} {step=} model input length exceeded {max_total_length=}, {running_states[idx].shape[-1]=}, {len(obs_token_ids_model)=}"
                    )
                    print(agent_history[idx]["stop_reason"][-1])
                    continue

                vllm_input_list[idx]["prompt_token_ids"] = _concat_vllm_input(
                    vllm_input_list[idx]["prompt_token_ids"],
                    obs_token_ids_vllm,
                    tokenizer=tokenizer,
                )

                running_states[idx] = torch.cat([running_states[idx], obs_token_ids_model])
                obs_reward = torch.zeros(
                    len(obs_token_ids_model), dtype=torch.float, device=reward_tensor_list[idx].device
                )
                reward_tensor_list[idx] = torch.cat([reward_tensor_list[idx], obs_reward], dim=-1)

                obs_mask = torch.zeros(
                    len(obs_token_ids_model), dtype=torch.int64, device=running_action_masks[idx].device
                )
                running_action_masks[idx] = torch.cat([running_action_masks[idx], obs_mask])
                attn_mask = torch.ones(
                    len(obs_token_ids_model), dtype=torch.int64, device=running_attn_masks[idx].device
                )
                running_attn_masks[idx] = torch.cat([running_attn_masks[idx], attn_mask])

                mm_data = obs.get("multi_modal_data", {})
                if "image" in mm_data.keys():
                    if "multi_modal_data" not in vllm_input_list[idx].keys():
                        vllm_input_list[idx]["multi_modal_data"] = {"image": []}
                    vllm_input_list[idx]["multi_modal_data"]["image"] += mm_data["image"]

                    # for recompute mm
                    mm_data_list[idx]["image"] += mm_data["image"]
                    # print(f' [DEBUG img] {idx=} after update {len(vllm_input_list[idx]["multi_modal_data"]["image"])=}')

                mm_input = obs.get("multi_modal_inputs", {})
                if mm_input:
                    mm_input_list[idx] = _merge_multi_modal_inputs(mm_input_list[idx], mm_input)

            if (
                running_states[idx].shape[-1] >= max_total_length
                or len(vllm_input_list[idx]["prompt_token_ids"]) >= max_total_length
            ):
                agent_history[idx]["stop_reason"].append(
                    f"[DEBUG] {idx=} {step=} max length reached {max_total_length=}, {len(vllm_input_list[idx]['prompt_token_ids'])=}, {running_states[idx].shape[-1]=}"
                )
                active_mask[idx] = False

        timing["post_process"] += time.time() - agent_end_ts

    env.close()
    target_device = prompts.batch["input_ids"].device
    running_states = [state[:max_total_length] for state in running_states]
    state_tensor = pad_2d_list_to_length(running_states, tokenizer.pad_token_id, max_total_length).to(target_device)

    running_action_masks = [mask[:max_total_length] for mask in running_action_masks]
    action_mask_tensor = pad_2d_list_to_length(running_action_masks, 0, max_total_length).to(target_device)

    running_attn_masks = [mask[:max_total_length] for mask in running_attn_masks]
    attn_mask_tensor = pad_2d_list_to_length(running_attn_masks, 0, max_total_length).to(target_device)

    if processor is not None and _get_mm_template(processor) == "qwen2vl":
        # For Qwen-VL: (n*bs, 3, seq_len)
        position_ids_list = [
            get_rope_index(
                processor,
                input_ids=state_tensor[i, :],
                image_grid_thw=mm_input_list[i].get("image_grid_thw", None),
                video_grid_thw=mm_input_list[i].get("video_grid_thw", None),
                second_per_grid_ts=mm_input_list[i].get("second_per_grid_ts", None),
                attention_mask=attn_mask_tensor[i, :],
            )
            for i in range(batch_size * sample_n)
        ]
        position_ids_tensor = torch.stack(position_ids_list, dim=0)
    else:
        # For LM: (n*bs, seq_len)
        position_ids_tensor = compute_position_id_with_mask(attn_mask_tensor)

    reward_tensor_list = [reward[:max_total_length] for reward in reward_tensor_list]
    reward_tensor = pad_2d_list_to_length(reward_tensor_list, 0.0, max_total_length).to(target_device)

    tool_call_tensor = torch.tensor(tool_call_cnt_list, dtype=torch.float32).to(target_device).unsqueeze(1)

    non_tensors = {"agent_history": agent_history}
    if processor is not None:
        non_tensors.update({"multi_modal_inputs": mm_input_list, "multi_modal_data": mm_data_list})

    return DataProto.from_dict(
        tensors={
            "response": state_tensor[:, -config.response_length :],
            "action_mask": action_mask_tensor,
            "attention_mask": attn_mask_tensor,
            "position_ids": position_ids_tensor,
            "env_reward": reward_tensor[:, -config.response_length :],
            "tool_cnt": tool_call_tensor,
        },
        non_tensors=non_tensors,
        meta_info={"timing": timing},
    )


def execute_tool_call(sample, config: DictConfig, tokenizer=None, processor=None, pbar=None):
    from src.models import is_agivlm

    action_string = sample.get("action", "")
    tool = sample.get("tool", None)
    shared_kwargs = sample.get("kwargs", None)
    is_agi = is_agivlm(processor)

    # non-agent data
    if action_string == "" or tool is None:
        return {}, 0.0, True, {}

    if config.mock:
        tool_result, reward, done, info = "", 0.0, False, {}
    else:
        # Attach shared kwargs state to tool before execution so that counters are
        # preserved across steps.
        if shared_kwargs is not None:
            tool.kwargs = shared_kwargs
            # Keep img_counter in sync for backward-compat
            if hasattr(shared_kwargs, "img_id"):
                tool.img_counter = shared_kwargs["img_id"] if isinstance(shared_kwargs, dict) else shared_kwargs.img_id
        tool_result, reward, done, info = tool.execute(action_string)

    if tokenizer.eos_token_id not in tokenizer.encode(action_string) and "<tool_call>" in action_string:
        done = False

    # post-process
    if not tool_result:
        tool_result_info = {}

    elif isinstance(tool_result, str):
        # Format 1: text output
        obs_token_ids = tokenizer.encode(tool_result, add_special_tokens=False)
        tool_result_info = {
            "prompt_token_ids_vllm": torch.tensor(obs_token_ids),
            "prompt_token_ids_model": torch.tensor(obs_token_ids),
            "chat_messages": [{"role": "user", "content": tool_result}],
        }

    elif isinstance(tool_result, list) and isinstance(tool_result[0], dict) and is_agi:
        # Format 2: [{"role": "...", "content": "..."}, ...]
        obs_str = tokenizer.apply_chat_template(tool_result, add_generation_prompt=True, tokenize=False)
        obs_str = _strip_system_block(obs_str)
        obs_token_ids = tokenizer.encode(obs_str, add_special_tokens=False, return_tensors="pt")[0]
        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids,
            "prompt_token_ids_model": obs_token_ids,
            "chat_messages": tool_result,
        }

    elif isinstance(tool_result, list) and isinstance(tool_result[0], dict):
        # Format 2: [{"role": "...", "content": "..."}, ...]
        obs_token_ids = tokenizer.apply_chat_template(tool_result, add_generation_prompt=True, return_tensors="pt")[0]

        # NOTE: skip the sp (and the \n token that comes after it) added by Qwen tokenizer
        eos_start_idx = torch.nonzero(obs_token_ids == tokenizer.eos_token_id)
        if eos_start_idx.shape[0] > 0:
            eos_start_idx = eos_start_idx[0].item()
            obs_token_ids = obs_token_ids[eos_start_idx + 1 :]
        else:
            raise ValueError(
                f"tool [{tool.name}] returned type List[str] output must be in openai/qwen format : {tool_result}"
            )

        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids,
            "prompt_token_ids_model": obs_token_ids,
            "chat_messages": tool_result,
        }

    elif isinstance(tool_result, dict):
        # Format 3: {"prompt": "...", "chat": [{"role": "...", "content": "..."}, ...], "multi_modal_data": ...}
        assert "prompt" in tool_result or "chat" in tool_result, f"Neither `prompt` nor `chat` in {tool_result=}"
        prompt_str = tool_result.pop("prompt", "")
        chat_list = tool_result.pop("chat", [])

        if len(prompt_str) == 0 and len(chat_list) == 0:
            raise ValueError("Both prompt_str and chat_list are invalid")
        elif len(prompt_str) == 0 and len(chat_list) > 0:
            prompt_str = tokenizer.apply_chat_template(chat_list, add_generation_prompt=True, tokenize=False)
            prompt_str = _strip_system_block(prompt_str)

        prompt_str_vllm, obs_token_ids_model, mm_inputs = _preprocess_multi_modal_inputs(
            prompt_str, processor, config.image_placeholder, **tool_result
        )
        obs_token_ids_vllm = tokenizer.encode(prompt_str_vllm, add_special_tokens=False, return_tensors="pt")[0]
        tool_result_info = {
            "prompt_token_ids_vllm": obs_token_ids_vllm,
            "prompt_token_ids_model": obs_token_ids_model,
            "chat_messages": chat_list or [{"role": "user", "content": prompt_str}],
            **tool_result,  # multi_modal_data
        }
        if mm_inputs:
            tool_result_info["multi_modal_inputs"] = mm_inputs

    else:
        raise ValueError(f"Invalid tool_result type: {type(tool_result)=} -- {tool_result}")

    # Add updated counters for diandian tool so that caller can sync.
    if hasattr(tool, "name") and tool.name == "diandian":
        if hasattr(tool, "img_counter"):
            tool_result_info["img_counter"] = tool.img_counter
            # Update shared kwargs reference as well
            if shared_kwargs is not None:
                shared_kwargs["img_id"] = tool.img_counter
        if hasattr(tool, "kwargs") and isinstance(tool.kwargs, dict):
            note_id_val = tool.kwargs.get("note_id", None)
            if note_id_val is not None:
                tool_result_info["note_id"] = note_id_val
                if shared_kwargs is not None:
                    shared_kwargs["note_id"] = note_id_val

    if pbar is not None:
        pbar.update(1)
    return tool_result_info, reward, done, info


class SharedKwargs(dict):
    """Dictionary subclass that supports both key and attribute style access.
    Instances of this class are mutable and therefore can be shared across
    different steps in the agent rollout loop to keep states such as
    `note_id` and `img_id` in sync.
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        if item in self:
            del self[item]
        else:
            raise AttributeError(item)


class ParallelEnv:
    """
    The interface intentionally designed to be the similar to : https://github.com/openai/gym
    Hope this could be easier to use for RLers.
    """

    def __init__(self, env_config, tokenizer, processor, **kwargs):
        self.config = env_config
        self.tokenizer = tokenizer
        self.processor = processor

        # type: List[ Dict[ Str, ToolBase subclasses ] ]
        self.tools = []

    def step(self, active_indices, actions: list[VLLMLikeOutputs], kwargs_list):
        """
        Input:
        - actions: vllm.RequestOutput
        - kwargs_list: List[SharedKwargs], shared kwargs (note_id, img_id) for each rollout instance

        Output:
        - observations: List[Dict], content like {"prompt_token_ids": ..., "multi_modal_data": ...},
                multi_modal_data only appears when there are images/videos in obs
        - rewards: List[ float ].
                each time after an action being executed, procedure rewards can be assigned to
                the last valid token of model outputs. This might be useful for ...,
                e.g., invalid action, code execution error, format error,
                or video game envs where immediate feedback is available.
        - dones: List[ Boolean ]
        - infos: Dict, for debugging only
        """
        obs_list = [{}] * len(actions)
        reward_list = [0.0] * len(actions)
        done_list = []
        valid_indices = []
        real_indices = []
        valid_actions = []

        # 1. filtering valid actions
        for i, (idx, act) in enumerate(zip(active_indices, actions, strict=False)):
            # if act.outputs[0].finish_reason == "length":
            #    done_list.append(True)
            #    continue

            if len(act.outputs[0].token_ids) == 0:
                done_list.append(True)
                continue

            done_list.append(False)
            real_indices.append(i)
            valid_indices.append(idx)
            valid_actions.append(act.outputs[0].text)

        agent_inputs = []
        for i, idx, action in zip(real_indices, valid_indices, valid_actions, strict=False):
            agent_inputs.append(
                dict(
                    idx=i,
                    valid_idx=idx,
                    action=action,
                    tool=self.tools[idx],
                    kwargs=kwargs_list[idx],
                )
            )

        # 2. executing actions (sync or async)
        num_workers = min(self.config.concurrent_workers, len(valid_actions))
        pbar = (
            tqdm(total=len(valid_actions), desc=f"Tool calling on {num_workers} workers")
            if self.config.show_tqdm
            else None
        )
        if num_workers <= 1:
            for agi in agent_inputs:
                subidx = agi["idx"]
                obs, reward, done, info = execute_tool_call(agi, self.config, self.tokenizer, self.processor, pbar=pbar)
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done
        else:
            partial_tool_func = partial(
                execute_tool_call, config=self.config, tokenizer=self.tokenizer, processor=self.processor, pbar=pbar
            )
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                raw_outputs = list(executor.map(partial_tool_func, agent_inputs))
            for agi, raw in zip(agent_inputs, raw_outputs, strict=False):
                obs, reward, done = raw[0], raw[1], raw[2]
                subidx = agi["idx"]
                obs_list[subidx] = obs
                reward_list[subidx] = reward
                done_list[subidx] |= done

        if pbar is not None:
            pbar.close()
        return obs_list, reward_list, done_list, {}

    def reset(self, prompts, vllm_inputs, n=1, **kwargs):
        self.tools = []
        reset_output_list = []
        assert len(prompts) == len(vllm_inputs), f"{len(prompts)=}, {len(vllm_inputs)=}"

        num_agent, num_non_agent = 0, 0
        for i in range(len(prompts)):
            data_item = prompts[i]  # DataProtoItem
            tool_name = data_item.non_tensor_batch.pop(self.config.tool_name_key, "")
            raw_prompt = data_item.non_tensor_batch.pop("raw_prompt", None)

            vllm_input_item = vllm_inputs[i]  # {"prompt_token_ids": ..., "multi_modal_data": ...}
            multi_modal_data = vllm_input_item.get("multi_modal_data", None)
            origin_multi_modal_data = data_item.non_tensor_batch.pop("origin_multi_modal_data", None)
            for _ in range(n):
                if tool_name:
                    # init tools from config field `tool_name_key`
                    # Pass img_counter=0 for diandian tools
                    tool_kwargs = dict(**kwargs)
                    tool_kwargs["config"] = self.config
                    tool_kwargs["mm_template"] = _get_mm_template(self.processor)
                    if tool_name == "diandian":
                        tool_kwargs["img_counter"] = 1
                        tool_kwargs["note_id"] = 1
                        tool_kwargs["img_id"] = 1

                    tool_fns = ToolBase.create(tool_name, **tool_kwargs)

                    # Pass img_counter to reset method only for diandian tools
                    reset_kwargs = {
                        "raw_prompt": raw_prompt,
                        "multi_modal_data": deepcopy(multi_modal_data),
                        "origin_multi_modal_data": deepcopy(origin_multi_modal_data),
                        "mm_template": _get_mm_template(self.processor),
                    }
                    reset_kwargs.update(kwargs)
                    if tool_name == "diandian":
                        reset_kwargs["img_counter"] = 1
                        reset_kwargs["note_id"] = 1
                        reset_kwargs["img_id"] = 1

                    reset_output = tool_fns.reset(**reset_kwargs)
                    self.tools.append(tool_fns)
                    reset_output_list.append(reset_output)
                    num_agent += 1
                else:
                    # non-agent data
                    self.tools.append(None)
                    reset_output_list.append(None)
                    num_non_agent += 1

        print(f" [DEBUG agent] {num_agent=}, {num_non_agent=}")
        return reset_output_list

    def close(self):
        self.tools = []
