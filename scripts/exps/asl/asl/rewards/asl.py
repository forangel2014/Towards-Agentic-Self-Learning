# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import json
import os
import random
import re
import string
from collections import Counter
from typing import Sequence

import datasets
import numpy as np
import torch.nn.functional as F

import verl.utils.torch_functional as verl_F
from src.verl.rewards.std.base import GRPORewards, rewards_registry
from src.verl.rewards.utils import WithWorkerGroupMixin
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask


RETRIEVAL_SYS = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
## Tools
You are provided with function signatures within <tools></tools> tags:
<tools>
Name:retrieve
Description: Retrieve relevant information from the locally deployed knowledge base based on the provided list of search terms.
Input:{'query': {'Optional/Required': 'required', 'Parameter Description': 'search terms', 'Parameter Type': 'str'}}
</tools>
For each function call, you should call and then include the json format inputs within <tool_call></tool_call> tags, for example:
<tool_call>{\n  \"name\": tool['name'],\n  \"arguments\": tool['arguments']\n}</tool_call>
For each function call, the result will be returned in the <tool_response></tool_response> tags.
## Formats
Your output should be a combination of the following formats:
1. <think>your reasoning thoughts</think>
2. <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": \"Beijing cuisine\"\n    }\n}\n</tool_call>
3. <answer>YOUR ANSWER</answer>
## Tasks
Answer the user's question.
You can use <think></think> and <tool_call></tool_call> as many times as you want, but you must use <answer></answer> once and only once at the end of your response.
Your answer should be concise and to the point, without detailed illustrations. For example, <answer>Beijing</answer>.
"""

PROMPT_GENERATOR_SYS = """
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.
## Tools
You are provided with function signatures within <tools></tools> tags:
<tools>
Name:retrieve
Description: Retrieve relevant information from the locally deployed knowledge base based on the provided list of search terms.
Input:{'query': {'Optional/Required': 'required', 'Parameter Description': 'search terms', 'Parameter Type': 'str'}}
</tools>
For each function call, you should call and then include the json format inputs within <tool_call></tool_call> tags, for example:
<tool_call>{\n  \"name\": tool['name'],\n  \"arguments\": tool['arguments']\n}</tool_call>
For each function call, the result will be returned in the <tool_response></tool_response> tags.
## Tasks
generate a question and answer that satisfying the user's request.
You can use <think></think> and <tool_call></tool_call> as many times as you want, but you must use <question></question> and <answer></answer> once and only once at the end of your response.
Your question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Your answer should be concise and to the point, without detailed illustrations.
For example:
<question>What is the capital of China?</question>
<answer>Beijing</answer>
## Formats
Your output should be a combination of the following formats:
1. <think>your reasoning thoughts</think>
2. <tool_call>\n{\n    \"name\": \"retrieve\",\n    \"arguments\": {\n        \"query\": \"Beijing cuisine\"\n    }\n}\n</tool_call>
3. <question>YOUR GENERATED QUESTION</question>
4. <answer>THE ANSWER TO THE GENERATED QUESTION</answer>
"""

THINKING_PROMPT_TEMPLATE = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me verify the correctness of my student's answer.
Use thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: correct/wrong
</answer>
"""

BINARY_PROMPT_TEMPLATE = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me verify the correctness of my student's answer.
Use the tool and thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: correct/wrong
</answer>
"""

BINARY_PROMPT_TEMPLATE_UNCERTAIN = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me verify the correctness of my student's answer.
If the student does not provide a specific entity as answer, you should draw your conclusion as "wrong".
If you are uncertain whether this question is solvable or you are able to judge the correctness of the answer, you should always draw your conclusion as "uncertain".
Use the tool and thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: correct/wrong/uncertain
</answer>
"""

BINARY_PROMPT_TEMPLATE_WITH_ANSWER = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

An reference answer is:
{solution}

I want you to help me verify the correctness of my student's answer.
Use the tool and thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: correct/wrong
</answer>
"""

RANKING_PROMPT_TEMPLATE = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

I want you to help me rank the answer with one of the following options (from best to worst):

A: The answer is perfectly correct.

B: The answer is correct but have missed some important information.

C: The answer is incorrect, but there are some correct points.

D: The answer is completely incorrect.

E: The question is not solvable.

Use the tool and thinking to analyze, then draw your conclusion in the following format:
<answer>
conclusion: <A/B/C/...>
</answer>
"""

online_prompt_generation_template = """
I asked a question to my student:
{question_prompt}

My student's answer is:
{completion}

This answer is {answer_type}, therefore, to better train my student, I want you to generate a new {question_type} question based on the current question.
The new question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Use the tool to help you generate the question, ensure that the question is solvable and the tool is necessary to answer the question.
Output with the following format:
<question>
...a new question...
</question>
<answer>
...the answer to the new question...
</answer>
"""

online_prompt_generation_template_multi_students = """
I asked a question to my students:
{question_prompt}

My students' answers are:
{completion}

Most answers are {answer_type}, therefore, to better train my students, I want you to generate a new {question_type} question based on the current question.
The new question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Use the tool to help you generate the question, ensure that the new question is solvable and the tool is necessary to answer the new question.
Generate the new question by {question_constraint} one-hop reasoning steps to the current question.
Do not generate the new question based on the answer to the current question.
Output with the following format:
<question>
...a new question...
</question>
<answer>
...the answer to the new question...
</answer>
"""

online_prompt_generation_template_depth = """
I asked a question to my students:
{question_prompt}

The true answer to this question is:
{answer_to_question}

Most student {answer_type}LY answer this question, therefore, to better train my students, I want you to generate a {question_type} question based on the current question.
You should generate the new question by {question_constraint} one-hop reasoning steps to the current question.
The new question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
Use the tool to help you generate the question, ensure that the question is solvable and the tool is necessary to answer the question.
e.g. {demo}
Output with the following format:
<question>
...a new question...
</question>
<answer>
...the answer to the new question...
</answer>
"""

# online_prompt_generation_template_width = """
# I asked a question to my students:
# {question_prompt}

# The true answer to this question is:
# {answer_to_question}

# Most student {answer_type}LY answer this question, therefore, to better train my students, I want you to generate a {question_type} question based on the current question.
# You should generate the new question that is similar to the current question but requires new knowledge and is {question_type} to answer.
# The new question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
# Use the tool to help you generate the question, ensure that the question is solvable and the tool is necessary to answer the question.
# e.g. {demo}
# Output with the following format:
# <question>
# ...a new question...
# </question>
# <answer>
# ...the answer to the new question...
# </answer>
# """

online_prompt_generation_template_width = """
I asked a question to my students:
{question_prompt}

The true answer to this question is:
{answer_to_question}

Most student {answer_type}LY answer this question, therefore, to better train my students, I want you to generate a {question_type} question based on the current question.
You should generate the new question that is similar to the current question but requires new knowledge and is {question_type} to answer.
1. The new question should be about factual knowledge and can be answered with ONLY ONE concreate entity.
2. Use the tool to help you generate the new question.
3. After generating the new question and answer, you MUST verify that the question is solvable and the tool is necessary to answer the question.
4. If you find that the question is not solvable with the current tool, you should generate a new question and answer again, until you find a solvable question and answer.
Your final output should be of the following formats:
<question>
...a new question...
</question>
<answer>
...the answer to the new question...
</answer>
"""


def extract_tool_response(completions):
    # completions = [completion.replace("<|im_end|>", "").split("<|im_start|>assistant")[-1] for completion in completions]
    tool_response_pattern = r".*<tool_response>(.*?)</tool_response>"
    extracted_completions = []
    for completion in completions:
        matches = re.search(tool_response_pattern, completion, re.DOTALL)
        if matches:
            match = matches.group(1).strip()
            if match:
                extracted_completions.append(match)
            else:
                extracted_completions.append("No tool response found.")
        else:
            extracted_completions.append("No tool response found.")
    return extracted_completions


def extract_question(completions):
    # completions = [completion.replace("<|im_end|>", "").split("<|im_start|>assistant")[-1] for completion in completions]
    question_pattern = r".*<question>(.*?)</question>"
    extracted_completions = []
    for completion in completions:
        matches = re.search(question_pattern, completion, re.DOTALL)
        if matches:
            match = matches.group(1).strip()
            if match:
                extracted_completions.append(match)
            else:
                extracted_completions.append("No question found.")
        else:
            extracted_completions.append("No question found.")
    return extracted_completions


def extract_answer(completions):
    # completions = [completion.replace("<|im_end|>", "").split("<|im_start|>assistant")[-1] for completion in completions]
    answer_pattern = r".*<answer>(.*?)</answer>"
    extracted_completions = []
    for completion in completions:
        matches = re.search(answer_pattern, completion, re.DOTALL)
        if matches:
            match = matches.group(1).strip()
            if match:
                extracted_completions.append(match)
            else:
                extracted_completions.append("No answer found.")
        else:
            extracted_completions.append("No answer found.")
    return extracted_completions


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)

    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0:
        return None

    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if em_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_subem(solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print("--------------------------------")
        print(f"Golden answers: {ground_truth}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")

    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth):
            return score
        else:
            return format_score


def compute_score_gt(completions, solutions):
    scores = [
        compute_score_subem(completion, solution) for completion, solution in zip(completions, solutions, strict=False)
    ]
    return scores


def compute_score_em_verify(completions, solutions):
    scores = [
        int(normalize_answer(completion) == normalize_answer(solution))
        for completion, solution in zip(completions, solutions, strict=False)
    ]
    return scores


def compute_score_subem_verify(completions, solutions):
    scores = [
        int(normalize_answer(solution) in normalize_answer(completion))
        for completion, solution in zip(completions, solutions, strict=False)
    ]
    return scores


def pad_sequence_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """pad a 2D tensors (e.g. responses, logprobs) in the last dim to
    max_seq_length. input shape: [bs, seq_length] output shape: [bs,
    max_seq_length]

    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (max_seq_len - tensors.shape[-1], 0) if left_pad else (0, max_seq_len - tensors.shape[-1])
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)


def tokenize_and_postprocess_data(
    prompt: str, tokenizer, max_length: int, pad_token_id: int, left_pad=True, truncation="error"
):
    """input_data is the output from tokenizer."""
    assert truncation in ["left", "right", "error"]

    input_data = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    input_ids = input_data["input_ids"]
    attention_mask = input_data["attention_mask"]

    assert input_ids.ndim == 2

    sequence_length = input_ids.shape[-1]
    if sequence_length < max_length:
        input_ids = pad_sequence_to_length(
            input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
        )
        attention_mask = pad_sequence_to_length(
            attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
        )
    elif sequence_length > max_length:
        if truncation == "left":
            # actually, left truncation may not be reasonable
            input_ids = input_ids[:, -max_length:]
            attention_mask = attention_mask[:, -max_length:]
        elif truncation == "right":
            input_ids = input_ids[:, :max_length]
            attention_mask = attention_mask[:, :max_length]
        elif truncation == "error":
            raise NotImplementedError(f"{sequence_length=} is larger than {max_length=}")
        else:
            raise NotImplementedError(f"Unknown truncation method {truncation}")

    return input_ids, attention_mask


def parse_prompts(prompts, is_agentic):
    sys_prompts, question_prompts = [], []
    for prompt in prompts:
        sys_prompt, question_prompt = prompt.split("<|im_end|>\n<|im_start|>user")
        sys_prompt = sys_prompt.split("<|im_start|>system\n")[1]
        if not is_agentic:
            sys_prompt = sys_prompt.split("\n## Tools")[0]
        question_prompt = question_prompt.split("<|im_end|>\n<|im_start|>assistant")[0]
        sys_prompts.append(sys_prompt)
        question_prompts.append(question_prompt)
    return sys_prompts, question_prompts


def build_gen_batch_with_prompts(config, prompts, completions, solutions, tokenizer, is_agentic, logger):
    """批量构建输入批次.

    :param prompts: 提示列表
    :param tokenizer: 分词器
    :return: 包含所有提示的批处理字典
    """
    all_input_ids = []
    all_attention_masks = []
    all_position_ids = []
    raw_prompt_ids = []
    sys_prompts, question_prompts = parse_prompts(prompts, is_agentic)
    chat_messages = []
    if config.reward_model.agentic:
        if config.reward_model.gen_rm_verification_method == "binary":
            verify_prompt_template = BINARY_PROMPT_TEMPLATE
        elif config.reward_model.gen_rm_verification_method == "binary_with_answer":
            verify_prompt_template = BINARY_PROMPT_TEMPLATE_WITH_ANSWER
        elif config.reward_model.gen_rm_verification_method == "binary_uncertain":
            verify_prompt_template = BINARY_PROMPT_TEMPLATE_UNCERTAIN
        else:
            verify_prompt_template = RANKING_PROMPT_TEMPLATE
    else:
        verify_prompt_template = THINKING_PROMPT_TEMPLATE
    verify_prompts = [
        verify_prompt_template.format(question_prompt=question_prompt, completion=completion, solution=solution)
        for question_prompt, completion, solution in zip(question_prompts, completions, solutions, strict=False)
    ]

    for i in range(len(sys_prompts)):
        chat = [{"content": sys_prompts[i], "role": "system"}, {"content": verify_prompts[i], "role": "user"}]
        chat_messages.append(chat)
        prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if i == 0:
            logger.write("******************reward model prompt******************\n")
            logger.write(prompt_with_chat_template)

        raw_prompt_id = tokenizer.encode(prompt_with_chat_template)
        raw_prompt_ids.append(raw_prompt_id)
        input_ids, attention_mask = tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=1024,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation="right",
        )
        position_ids = compute_position_id_with_mask(attention_mask)

        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_position_ids.append(position_ids)

    # 将列表转换为张量批次
    batch_input_ids = verl_F.torch.stack(all_input_ids).squeeze(1)
    batch_attention_mask = verl_F.torch.stack(all_attention_masks).squeeze(1)
    batch_position_ids = verl_F.torch.stack(all_position_ids).squeeze(1)

    batch_dict = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "position_ids": batch_position_ids,
    }

    batch_dict["raw_prompt"] = np.array(chat_messages)
    batch_dict["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)

    return batch_dict


def get_policy_rm_output_batch(
    config,
    prompts,
    example_batch,
    completions,
    solutions,
    tokenizer,
    policy_model,
    is_agentic,
    rollout_n,
    logger,
    new_rollout_n=1,
):
    """批量获取模型输出.

    :param prompts: 提示列表
    :param tokenizer: 分词器
    :param policy_model: 策略模型
    :return: 生成的文本列表
    """
    batch_dict = build_gen_batch_with_prompts(config, prompts, completions, solutions, tokenizer, is_agentic, logger)
    batch = complete_gen_batch(batch_dict, example_batch, rollout_n, is_agentic, new_rollout_n=new_rollout_n)

    batch_padded, pad_size = pad_dataproto_to_divisor(batch, policy_model.world_size)
    # open("batch_padded.json", "w").write(str(batch_padded))
    output_gen_batch_padded = policy_model.generate_sequences(batch_padded)
    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

    output_ids = output_gen_batch.batch["responses"]
    output_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
    ]  # .split("<|im_start|>assistant")[-1]
    return output_texts


def complete_gen_batch(batch_dict, example_batch, rollout_n, is_agentic, new_rollout_n=1):
    if is_agentic:
        for key in example_batch.keys():
            if key not in batch_dict.keys():
                # batch_dict[key] = np.tile(example_batch[key], rollout_n)[:len(batch_dict['raw_prompt'])]
                batch_dict[key] = np.concatenate([example_batch[key] for _ in range(rollout_n * new_rollout_n)])[
                    : len(batch_dict["raw_prompt"])
                ]

    try:
        batch = DataProto.from_single_dict(batch_dict)
    except Exception as e:
        print(f"Error: {e}")
        open("batch_dict.json", "w").write(str(batch_dict))
        raise e

    # pop those keys for generation
    batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
    non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
    if "multi_modal_inputs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.extend(["multi_modal_data", "multi_modal_inputs", "origin_multi_modal_data"])
    if "raw_prompt" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("raw_prompt")
    if "chat_messages" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("chat_messages")
    if "tools_kwargs" in batch.non_tensor_batch:
        non_tensor_batch_keys_to_pop.append("tools_kwargs")
    gen_batch = batch.pop(
        batch_keys=batch_keys_to_pop,
        non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
    )

    gen_batch.meta_info["n"] = 1

    if is_agentic:
        tool_name_key = "env_name"
        if tool_name_key and tool_name_key in batch.non_tensor_batch.keys():
            gen_batch.non_tensor_batch[tool_name_key] = batch.non_tensor_batch.pop(tool_name_key)

    return gen_batch


def compute_score(config, output_texts):
    def get_score_binary(eval_text):
        eval_text = eval_text.lower()
        if "conclusion" in eval_text:
            eval_text = eval_text.split("conclusion")[-1].strip()
            if "incorrect" in eval_text:
                return 0.0
            elif "correct" in eval_text:
                return 1.0
            elif "uncertain" in eval_text:
                return 0.0
            elif "wrong" in eval_text:
                return 0.0
            else:
                return 0.0
        else:
            return 0.0

    def get_score_ranking(eval_text):
        eval_text = eval_text  # .lower()
        if "conclusion" in eval_text:
            eval_text = eval_text.split("conclusion")[-1].strip()
            if "A" in eval_text:
                return 1.0
            elif "B" in eval_text:
                return 0.8
            elif "C" in eval_text:
                return 0.6
            elif "D" in eval_text:
                return 0.4
            elif "E" in eval_text:
                return 0.2
            else:
                return 0.0
        else:
            return 0.0

    scores = [
        get_score_binary(text)
        if "binary" in config.reward_model.gen_rm_verification_method
        else get_score_ranking(text)
        for text in output_texts
    ]
    return scores


def save_new_qa_data(config, questions, logger, style, answers=None):
    if os.path.exists(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt"):
        meta_iter = int(open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt").read())
    else:
        meta_iter = 1
    if answers is None:
        answers = ["unknown" for _ in questions]
    qa_data = [
        {
            "data_source": "asl",  # data_source,
            "prompt": [
                {
                    "role": "system",
                    "content": RETRIEVAL_SYS
                    if meta_iter % 2 == 1
                    else PROMPT_GENERATOR_SYS,  # RETRIEVAL_SYS if style == "gen_rm" else PROMPT_GENERATOR_SYS
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
            "ability": "fact-reasoning",
            "reward_model": {"style": style, "ground_truth": answer},
            "extra_info": {
                "split": "train",
                "index": 0,
            },
            "env_name": "Retrieval",
        }
        for question, answer in zip(questions, answers, strict=False)
    ]
    qa_data_path = f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}.jsonl"

    os.makedirs(os.path.dirname(qa_data_path), exist_ok=True)
    # 以增量形式写入JSONL文件
    with open(qa_data_path, "a", encoding="utf-8") as f:
        for qa_item in qa_data:
            f.write(json.dumps(qa_item, ensure_ascii=False) + "\n")

    return meta_iter


def load_qa_data(config, meta_iter):
    qa_data = []
    with open(f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line.strip()))
    return qa_data


def build_gen_batch_with_prompts_qa(questions, tokenizer, is_agentic, logger, new_rollout_n=1):
    """批量构建输入批次.

    :param prompts: 提示列表
    :param tokenizer: 分词器
    :return: 包含所有提示的批处理字典
    """
    all_input_ids = []
    all_attention_masks = []
    all_position_ids = []
    raw_prompt_ids = []
    chat_messages = []
    all_prompts = []
    sys_prompts = [RETRIEVAL_SYS for _ in range(len(questions))]
    logger.write("******************sys_prompts_number******************\n")
    logger.write(str(len(sys_prompts)) + "\n")
    for i in range(len(sys_prompts)):
        for j in range(new_rollout_n):
            chat = [{"content": sys_prompts[i], "role": "system"}, {"content": questions[i], "role": "user"}]
            chat_messages.append(chat)
            prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
            all_prompts.append(prompt_with_chat_template)
            if i == 0 and j == 0:
                logger.write("******************qa prompt******************\n")
                logger.write(prompt_with_chat_template)

            raw_prompt_id = tokenizer.encode(prompt_with_chat_template)
            raw_prompt_ids.append(raw_prompt_id)
            input_ids, attention_mask = tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=tokenizer,
                max_length=1024,
                pad_token_id=tokenizer.pad_token_id,
                left_pad=True,
                truncation="right",
            )
            position_ids = compute_position_id_with_mask(attention_mask)

            all_input_ids.append(input_ids)
            all_attention_masks.append(attention_mask)
            all_position_ids.append(position_ids)

    # 将列表转换为张量批次
    batch_input_ids = verl_F.torch.stack(all_input_ids).squeeze(1)
    batch_attention_mask = verl_F.torch.stack(all_attention_masks).squeeze(1)
    batch_position_ids = verl_F.torch.stack(all_position_ids).squeeze(1)

    batch_dict = {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "position_ids": batch_position_ids,
    }

    batch_dict["raw_prompt"] = np.array(chat_messages)
    batch_dict["raw_prompt_ids"] = np.array(raw_prompt_ids, dtype=object)

    return batch_dict, all_prompts


def get_policy_rm_output_batch_qa(example_batch, questions, tokenizer, policy_model, is_agentic, rollout_n, logger):
    """批量获取模型输出.

    :param prompts: 提示列表
    :param tokenizer: 分词器
    :param policy_model: 策略模型
    :return: 生成的文本列表
    """
    batch_dict, all_prompts = build_gen_batch_with_prompts_qa(
        questions, tokenizer, is_agentic, logger, new_rollout_n=10
    )
    batch = complete_gen_batch(batch_dict, example_batch, rollout_n, is_agentic, new_rollout_n=10)

    # logger.write("******************batch_dict_size******************\n")
    # logger.write(str(len(batch_dict["input_ids"])) + "\n")
    # #open("batch_dict_qa.json", "w").write(str(batch_dict))
    # logger.write("******************world_size******************\n")
    # logger.write(str(policy_model.world_size) + "\n")

    batch_padded, pad_size = pad_dataproto_to_divisor(batch, policy_model.world_size)
    # open("batch_padded.json", "w").write(str(batch_padded))
    output_gen_batch_padded = policy_model.generate_sequences(batch_padded)
    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)

    output_ids = output_gen_batch.batch["responses"]
    output_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids
    ]  # .split("<|im_start|>assistant")[-1]
    return output_texts, all_prompts


def get_question_quality(
    config,
    example_batch,
    proposed_questions,
    proposed_answers,
    tokenizer,
    rollout_model,
    is_agentic,
    rollout_n,
    logger,
):
    # meta_scores = [0 for _ in meta_completions]
    question_scores = [-1 for _ in proposed_questions]
    failed_questions_idx = [
        i
        for i, question in enumerate(proposed_questions)
        if (
            "No question found" in question
            or len(question) < 10
            or "?" not in question[-5:]
            or "No answer found" in proposed_answers[i]
        )
    ]
    success_questions_idx = [i for i, question in enumerate(proposed_questions) if i not in failed_questions_idx]
    success_questions = [question for i, question in enumerate(proposed_questions) if i in success_questions_idx]
    success_answers = []
    for i in success_questions_idx:
        success_answers.extend([proposed_answers[i]] * 10)

    # logger.write("******************success_questions_number******************\n")
    # logger.write(str(len(success_questions)) + "\n")
    if len(success_questions):
        question_output_texts, question_prompts = get_policy_rm_output_batch_qa(
            example_batch, success_questions, tokenizer, rollout_model, is_agentic, rollout_n, logger
        )
        question_answers = extract_answer(question_output_texts)

        if config.reward_model.pg_type == "gen_rm" and config.reward_model.pg_reward != "absolute_zero":
            logger.write("******************question_output_texts******************\n")
            logger.write(question_output_texts[0] + "\n")
            try_answering_texts = get_policy_rm_output_batch(
                config,
                question_prompts,
                example_batch,
                question_answers,
                ["unknown" for _ in question_output_texts],
                tokenizer,
                rollout_model,
                is_agentic,
                rollout_n,
                logger,
                new_rollout_n=10,
            )
            # logger.write("******************try_answering_texts_number******************\n")
            # logger.write(str(len(try_answering_texts)) + "\n")
            logger.write("******************try_answering_texts******************\n")
            logger.write(try_answering_texts[0] + "\n")
            try_answering_scores = compute_score(config, try_answering_texts)
        elif config.reward_model.pg_type == "rule" or config.reward_model.pg_reward == "absolute_zero":
            logger.write("******************question_output_texts******************\n")
            logger.write(question_output_texts[0] + "\n")
            logger.write("******************success_answers******************\n")
            logger.write(success_answers[0] + "\n")
            logger.write("******************question_output_texts******************\n")
            logger.write(question_output_texts[-1] + "\n")
            logger.write("******************success_answers******************\n")
            logger.write(success_answers[-1] + "\n")
            try_answering_completions = extract_answer(question_output_texts)
            try_answering_scores = compute_rule_scores(
                question_prompts, success_answers, try_answering_completions, config, logger
            )
            logger.write("******************try_answering_scores******************\n")
            logger.write(str(try_answering_scores[0]) + "\n")
            logger.write("******************try_answering_scores******************\n")
            logger.write(str(try_answering_scores[-1]) + "\n")
        else:
            raise ValueError(f"Invalid pg_type: {config.reward_model.pg_type}")

    success_question_scores = []
    for proposed_question in success_questions:
        # idx = [proposed_question in question_prompt for question_prompt in question_prompts].index(True)
        idx = [
            i
            for i, contains_question in enumerate(
                [proposed_question in question_prompt for question_prompt in question_prompts]
            )
            if contains_question
        ]

        scores_for_question = [try_answering_scores[i] for i in idx]
        # score_for_question = np.mean(scores_for_question)
        # question_quality = min([score_for_question, 1-score_for_question])
        score_counts = Counter(scores_for_question)
        total = len(scores_for_question)
        if "entropy" in config.reward_model.pg_reward:
            probabilities = [score_counts[score] / total for score in sorted(score_counts)]
            question_quality = np.sum([-np.log(p) * p for p in probabilities])
        elif config.reward_model.pg_reward == "absolute_zero":
            avg_score = np.mean(scores_for_question)
            question_quality = 1 - avg_score if (0 < avg_score and avg_score < 1) else 0
        elif config.reward_model.pg_reward == "r_zero":
            avg_score = np.mean(scores_for_question)
            question_quality = 1 - 2 * abs(avg_score - 0.5)
        else:
            raise ValueError(f"Unknown pg_reward: {config.reward_model.pg_reward}")

        success_question_scores.append(question_quality)

        if proposed_question == success_questions[0]:
            logger.write("******************proposed_question******************\n")
            logger.write(proposed_question + "\n")
            # logger.write("******************idx******************\n")
            # logger.write(str(idx) + "\n")
            logger.write("******************scores_for_question******************\n")
            logger.write(str(scores_for_question) + "\n")
            logger.write("******************question_quality******************\n")
            logger.write(str(question_quality) + "\n")

    for i in range(len(success_questions_idx)):
        question_scores[success_questions_idx[i]] = success_question_scores[i]

    if config.reward_model.pg_reward == "r_zero":
        # 应用重复性惩罚
        if hasattr(config.reward_model, "repetition_penalty") and config.reward_model.repetition_penalty:
            lambda_scale = getattr(config.reward_model, "repetition_penalty_lambda", 1.0)
            tau_bleu = getattr(config.reward_model, "repetition_penalty_tau_bleu", 0.3)

            # 只对成功的问题计算重复性惩罚
            if len(success_questions) > 1:
                repetition_penalties = compute_repetition_penalty(success_questions, lambda_scale, tau_bleu)
                logger.write("******************repetition_penalties******************\n")
                logger.write(str(repetition_penalties) + "\n")
                logger.write("******************average_repetition_penalty******************\n")
                logger.write(str(np.mean(repetition_penalties)) + "\n")

                # 将重复性惩罚应用到问题质量分数上
                for i in range(len(success_questions_idx)):
                    original_score = success_question_scores[i]
                    penalty = repetition_penalties[i]
                    # 从原始分数中减去重复性惩罚
                    success_question_scores[i] = max(0, original_score - penalty)

                    if i == 0:  # 只记录第一个问题的详细信息
                        logger.write("******************original_score******************\n")
                        logger.write(str(original_score) + "\n")
                        logger.write("******************penalty******************\n")
                        logger.write(str(penalty) + "\n")
                        logger.write("******************final_score******************\n")
                        logger.write(str(success_question_scores[i]) + "\n")

                # 更新question_scores
                for i in range(len(success_questions_idx)):
                    question_scores[success_questions_idx[i]] = success_question_scores[i]

    for i in failed_questions_idx:
        question_scores[i] = 0.0

    return question_scores


def filter_and_duplicate_removal(questions):
    valid_questions = []
    for question in questions:
        if len(question) > 20 and "?" == question[-1]:
            valid_questions.append(question)
    return valid_questions


def compute_bleu_score(text1, text2):
    import sacrebleu

    """
    计算两个文本之间的BLEU分数

    Args:
        text1: 第一个文本
        text2: 第二个文本

    Returns:
        BLEU分数 (0-100)
    """
    try:
        # 清理文本，移除特殊字符
        text1_clean = text1.strip()
        text2_clean = text2.strip()

        # 如果文本为空或相同，返回100
        if not text1_clean or not text2_clean:
            return 0.0
        if text1_clean == text2_clean:
            return 100.0

        # 使用sacrebleu计算BLEU分数
        score = sacrebleu.sentence_bleu(text1_clean, [text2_clean]).score
        # 确保分数在0-100范围内
        return max(0.0, min(100.0, score))
    except Exception as e:
        # 如果计算失败，返回0
        print(f"BLEU calculation error: {e}")
        return 0.0


def compute_pairwise_bleu_distances(questions):
    """计算问题之间的成对BLEU距离矩阵.

    Args:
        questions: 问题列表

    Returns:
        距离矩阵，dij = 1 - BLEU(xi, xj)
    """
    n = len(questions)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0.0
            else:
                bleu_score = compute_bleu_score(questions[i], questions[j])
                # 将BLEU分数从0-100转换为0-1，然后计算距离
                # 确保距离值非负
                normalized_bleu = min(bleu_score / 100.0, 1.0)
                distance_matrix[i][j] = max(0.0, 1.0 - normalized_bleu)

    return distance_matrix


def cluster_questions_by_similarity(questions, tau_bleu=0.3):
    from sklearn.cluster import DBSCAN

    """
    基于BLEU距离将相似问题聚类

    Args:
        questions: 问题列表
        tau_bleu: BLEU距离阈值，小于此值的问题会被聚类到一起

    Returns:
        聚类结果，每个问题属于的聚类ID列表
    """
    if len(questions) <= 1:
        return [0] * len(questions)

    # 计算距离矩阵
    distance_matrix = compute_pairwise_bleu_distances(questions)

    # 检查距离矩阵是否包含负值
    min_distance = np.min(distance_matrix)
    max_distance = np.max(distance_matrix)

    if min_distance < 0:
        print(f"Warning: Distance matrix contains negative values. Min: {min_distance}, Max: {max_distance}")
        # 将所有负值设为0
        distance_matrix = np.maximum(distance_matrix, 0.0)

    # 确保距离矩阵是对称的
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    try:
        # 使用DBSCAN进行聚类
        # eps参数对应tau_bleu阈值
        # min_samples=2表示至少需要2个样本才能形成一个聚类
        clustering = DBSCAN(eps=tau_bleu, min_samples=2, metric="precomputed")
        cluster_labels = clustering.fit_predict(distance_matrix)

        # 将噪声点（标签为-1）分配为单独的聚类
        max_label = max(cluster_labels) if len(set(cluster_labels)) > 1 else -1
        for i, label in enumerate(cluster_labels):
            if label == -1:
                max_label += 1
                cluster_labels[i] = max_label

        return cluster_labels.tolist()
    except Exception as e:
        print(f"Clustering error: {e}")
        # 如果聚类失败，返回每个问题单独的聚类
        return list(range(len(questions)))


def compute_repetition_penalty(questions, lambda_scale=1.0, tau_bleu=0.3):
    """计算重复性惩罚.

    Args:
        questions: 问题列表
        lambda_scale: 缩放因子λ，默认为1.0
        tau_bleu: BLEU距离阈值

    Returns:
        每个问题的重复性惩罚值列表
    """
    if len(questions) <= 1:
        return [0.0] * len(questions)

    # 对问题进行聚类
    cluster_labels = cluster_questions_by_similarity(questions, tau_bleu)

    # 计算每个聚类的大小
    cluster_sizes = Counter(cluster_labels)

    # 计算每个问题的重复性惩罚
    batch_size = len(questions)
    repetition_penalties = []

    for i, cluster_id in enumerate(cluster_labels):
        cluster_size = cluster_sizes[cluster_id]
        # rrep(xi) = λ * |Ck| / B
        penalty = lambda_scale * cluster_size / batch_size
        repetition_penalties.append(penalty)

    return repetition_penalties


def prepare_meta_prompt(prompts, solutions, scores, is_agentic, logger):
    # _, gen_rm_question_prompts = parse_prompts(gen_rm_prompts, is_agentic)
    # answer_types = ["CORRECT" if score == 1.0 else "WRONG" for score in gen_rm_scores]
    # question_types = ["HARDER" if score == 1.0 else "EASIER" for score in gen_rm_scores]
    # online_prompt_generation_new_prompts = [online_prompt_generation_template.format(question_prompt=gen_rm_question_prompts[i], completion=gen_rm_completions[i], answer_type=answer_types[i], question_type=question_types[i]) for i in range(len(gen_rm_prompts))]
    # meta_iter = save_new_qa_data(config, online_prompt_generation_new_prompts, logger, style="meta")

    _, gen_rm_question_prompts = parse_prompts(prompts, is_agentic)
    all_questions = list(set(gen_rm_question_prompts))
    # answers_for_questions = [[gen_rm_completions[i] for i in range(len(gen_rm_prompts)) if gen_rm_question_prompts[i] == question] for question in all_questions]
    accuracy_for_questions = [
        np.mean([scores[i] for i in range(len(prompts)) if gen_rm_question_prompts[i] == question])
        for question in all_questions
    ]
    gt_for_questions = [
        [solutions[i] for i in range(len(prompts)) if gen_rm_question_prompts[i] == question][0]
        for question in all_questions
    ]
    # logger.write("******************answers_for_questions******************\n")
    # logger.write(str(answers_for_questions[0]) + "\n")
    # answers_for_questions = [answers_for_questions[i][0] for i in range(len(answers_for_questions))]
    answer_types = ["CORRECT" if score >= 0.5 else "WRONG" for score in accuracy_for_questions]
    question_types = ["HARDER" if score >= 0.5 else "EASIER" for score in accuracy_for_questions]
    question_constraints = ["ADDING" if score >= 0.5 else "REDUCING" for score in accuracy_for_questions]
    demos_harder_depth = [
        '"what is the capital of China?" -> "what is the most famous university in the capital of China?"',
        '"what is the capital of China?" -> "which city is bigger? Beijing or Shanghai?"',
    ]
    demos_easier_depth = [
        '"what is the most famous university in the capital of China?" -> "what is the capital of China?"',
        '"which city is bigger? Beijing or Shanghai?" -> "what is the area of Beijing?"',
    ]
    demos_harder_width = ['"what is the capital of China?" -> "what is the capital of Zimbabwe?"']
    # "\"who is the most famous banker in US?\" -> \"who is the most famous science fiction writer in US?\""]
    demos_easier_width = ['"what is the capital of Zimbabwe?" -> "what is the capital of China?"']
    # "\"who is the most famous science fiction writer in US?\" -> \"who is the most famous banker in US?\""]

    # demos = [demo_harder if score >= 0.5 else demo_easier for score in accuracy_for_questions]
    logger.write("******************all_questions******************\n")
    logger.write(str(all_questions[0]) + "\n")
    # logger.write("******************answers_for_questions******************\n")
    # logger.write(str(answers_for_questions[0]) + "\n")
    logger.write("******************accuracy_for_questions******************\n")
    logger.write(str(accuracy_for_questions[0]) + "\n")

    online_prompt_generation_new_prompts = []
    answer_to_save = []
    for i in range(len(all_questions)):
        for pg_type in ["width"]:
            if pg_type == "depth":
                pg_template = online_prompt_generation_template_depth
                demo_harder = random.choice(demos_harder_depth)
                demo_easier = random.choice(demos_easier_depth)
            elif pg_type == "width":
                pg_template = online_prompt_generation_template_width
                demo_harder = random.choice(demos_harder_width)
                demo_easier = random.choice(demos_easier_width)
            demo = demo_harder if accuracy_for_questions[i] >= 0.5 else demo_easier

            prompt = pg_template.format(
                question_prompt=all_questions[i],
                # completion=answers_for_questions[i],
                answer_to_question=gt_for_questions[i],
                answer_type=answer_types[i],
                question_type=question_types[i],
                question_constraint=question_constraints[i],
                demo=demo,
            )
            online_prompt_generation_new_prompts.append(prompt)
            answer_to_save.append(gt_for_questions[i])

    return online_prompt_generation_new_prompts, answer_to_save


def compute_rule_scores(rule_prompts, rule_solutions, rule_completions, config, logger):
    if config.reward_model.rule_verification_method == "em":
        rule_scores = compute_score_em_verify(rule_completions, rule_solutions)
    elif config.reward_model.rule_verification_method == "subem":
        rule_scores = compute_score_subem_verify(rule_completions, rule_solutions)
    elif config.reward_model.rule_verification_method == "random":
        rule_scores = [random.choice([0, 1]) for _ in range(len(rule_completions))]
    elif config.reward_model.rule_verification_method == "browsecomp":
        from src.verl.rewards.std.browsecomp import evaluator

        rule_questions = [
            rule_prompts[i].split("<|im_start|>user")[1].split("<|im_end|>")[0].strip()
            for i in range(len(rule_completions))
        ]
        rule_scores = evaluator.scores(rule_questions, rule_solutions, rule_completions)
    else:
        raise ValueError(f"Unknown rule verification method: {config.reward_model.rule_verification_method}")
    return rule_scores


@rewards_registry.register(alias=["asl_reward"], ignore_if_exist=True)
class AgenticSelfLearningReward(WithWorkerGroupMixin, GRPORewards):
    def __init__(self, name: str = "", weight: float = 1) -> None:
        super().__init__(name, weight)
        self.worker_group = {}

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        solutions: list[str],
        **kwargs,
    ) -> Sequence[float | tuple[float, dict]]:
        config = self.worker_group["config"]
        rollout_model = self.worker_group["rollout_model"]
        example_batch = self.worker_group["example_batch"]

        if config.trainer.val_before_train:
            split = "test"
        else:
            split = "train"
        reward_history_file = f"./exp/{config.trainer.experiment_name}/rewards_history_{split}.jsonl"
        logger = open(f"./exp/{config.trainer.experiment_name}/reward_{split}.log", "a")
        is_agentic = config.reward_model.agentic
        rollout_n = config.actor_rollout_ref.rollout.n
        tokenizer = kwargs["tokenizer"][0]
        data = kwargs["data"]

        batchsize = len(prompts)
        # logger.write("******************batchsize******************\n")
        # logger.write(str(batchsize) + "\n")
        assert len(completions) == batchsize
        assert len(solutions) == batchsize

        # example_batchsize = len(example_batch["input_ids"])
        # logger.write("******************example_batchsize******************\n")
        # logger.write(str(example_batchsize) + "\n")

        logger.write("******************prompts******************\n")
        logger.write(prompts[0] + "\n")
        logger.write("******************raw_completions******************\n")
        logger.write(completions[0] + "\n")

        extracted_completions = extract_answer(completions)

        reward_styles = [data[i].non_tensor_batch["reward_model"]["style"] for i in range(len(data))]
        logger.write("******************reward_styles******************\n")
        logger.write(str(reward_styles) + "\n")

        rule_index = [i for i, style in enumerate(reward_styles) if style == "rule"]
        gen_rm_index = [i for i, style in enumerate(reward_styles) if style == "gen_rm"]
        meta_index = [i for i, style in enumerate(reward_styles) if style == "meta"]
        logger.write("******************rule_num******************\n")
        logger.write(str(len(rule_index)) + "\n")
        logger.write("******************gen_rm_num******************\n")
        logger.write(str(len(gen_rm_index)) + "\n")
        logger.write("******************meta_num******************\n")
        logger.write(str(len(meta_index)) + "\n")
        assert len(rule_index) + len(gen_rm_index) + len(meta_index) == len(reward_styles)

        rule_prompts = [prompts[i] for i in rule_index]
        gen_rm_prompts = [prompts[i] for i in gen_rm_index]
        meta_prompts = [prompts[i] for i in meta_index]

        # for prompt in rule_prompts:
        #     assert "Please carefully evaluate" in prompt, f"prompt: {prompt}"
        # for prompt in gen_rm_prompts:
        #     assert "Please carefully evaluate" not in prompt, f"prompt: {prompt}"

        rule_completions = [extracted_completions[i] for i in rule_index]
        rule_solutions = [solutions[i] for i in rule_index]
        gen_rm_completions = [extracted_completions[i] for i in gen_rm_index]
        gen_rm_solutions = [solutions[i] for i in gen_rm_index]
        # meta_completions = [extracted_completions[i] for i in meta_index]
        # meta_solutions = [solutions[i] for i in meta_index]
        meta_completions = [extract_question([completions[i]])[0] for i in meta_index]
        meta_answers = [extracted_completions[i] for i in meta_index]
        meta_solutions = [solutions[i] for i in meta_index]

        do_rule_verification = len(rule_completions) > 0
        do_gen_rm_verification = len(gen_rm_completions) > 0
        do_meta_verification = len(meta_completions) > 0

        rewards_history = []
        if os.path.exists(reward_history_file):
            with open(reward_history_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rewards_history.append(json.loads(line.strip()))

        if do_rule_verification:
            logger.write("******************rule_prompts******************\n")
            logger.write(rule_prompts[0] + "\n")
            logger.write("******************rule_completions******************\n")
            logger.write(rule_completions[0] + "\n")
            logger.write("******************rule_solutions******************\n")
            logger.write(rule_solutions[0] + "\n")
        if do_gen_rm_verification:
            logger.write("******************gen_rm_prompts******************\n")
            logger.write(gen_rm_prompts[0] + "\n")
            logger.write("******************gen_rm_completions******************\n")
            logger.write(gen_rm_completions[0] + "\n")
            logger.write("******************gen_rm_solutions******************\n")
            logger.write(gen_rm_solutions[0] + "\n")
        if do_meta_verification:
            logger.write("******************meta_prompts******************\n")
            logger.write(meta_prompts[0] + "\n")
            logger.write("******************meta_completions******************\n")
            logger.write(meta_completions[0] + "\n")
            logger.write("******************meta_answers******************\n")
            logger.write(meta_answers[0] + "\n")
            logger.write("******************meta_solutions******************\n")
            logger.write(meta_solutions[0] + "\n")

        if do_gen_rm_verification:
            gen_rm_output_texts = get_policy_rm_output_batch(
                config,
                gen_rm_prompts,
                example_batch,
                gen_rm_completions,
                gen_rm_solutions,
                tokenizer,
                rollout_model,
                is_agentic,
                rollout_n,
                logger,
            )
            logger.write("******************gen_rm_output_texts******************\n")
            logger.write(gen_rm_output_texts[0] + "\n")
            gen_rm_scores = compute_score(config, gen_rm_output_texts)

            for i in range(len(gen_rm_completions)):
                if "No answer found" in gen_rm_completions[i]:
                    gen_rm_scores[i] = 0

            if config.reward_model.online_prompt_generation:
                online_prompt_generation_new_prompts, answer_to_save = prepare_meta_prompt(
                    gen_rm_prompts, gen_rm_solutions, gen_rm_scores, is_agentic, logger
                )
                meta_iter = save_new_qa_data(
                    config, online_prompt_generation_new_prompts, logger, style="meta", answers=answer_to_save
                )

        if do_rule_verification:
            rule_scores = compute_rule_scores(rule_prompts, rule_solutions, rule_completions, config, logger)

            if config.reward_model.online_prompt_generation:
                online_prompt_generation_new_prompts, answer_to_save = prepare_meta_prompt(
                    rule_prompts, rule_solutions, rule_scores, is_agentic, logger
                )
                meta_iter = save_new_qa_data(
                    config, online_prompt_generation_new_prompts, logger, style="meta", answers=answer_to_save
                )

        if do_meta_verification:
            # verify the quality of generated prompts, select the high quality prompts to save as new training data

            if config.reward_model.online_prompt_generation:
                ## entropy based reward
                if config.reward_model.pg_reward == "entropy":
                    meta_scores = get_question_quality(
                        config,
                        example_batch,
                        meta_completions,
                        meta_answers,
                        tokenizer,
                        rollout_model,
                        is_agentic,
                        rollout_n,
                        logger,
                    )
                    logger.write("******************question_scores******************\n")
                    logger.write(str(meta_scores) + "\n")
                    filtered_meta_completions = [
                        meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0
                    ]
                    valid_questions = filter_and_duplicate_removal(filtered_meta_completions)

                if config.reward_model.pg_reward == "absolute_zero" or config.reward_model.pg_reward == "r_zero":
                    meta_scores = get_question_quality(
                        config,
                        example_batch,
                        meta_completions,
                        meta_answers,
                        tokenizer,
                        rollout_model,
                        is_agentic,
                        rollout_n,
                        logger,
                    )
                    logger.write("******************question_scores******************\n")
                    logger.write(str(meta_scores) + "\n")
                    filtered_meta_completions = [
                        meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0
                    ]
                    valid_questions = filter_and_duplicate_removal(filtered_meta_completions)

                if config.reward_model.pg_reward == "entropy+tool":
                    meta_scores = get_question_quality(
                        config,
                        example_batch,
                        meta_completions,
                        meta_answers,
                        tokenizer,
                        rollout_model,
                        is_agentic,
                        rollout_n,
                        logger,
                    )
                    for i in range(len(meta_index)):
                        if meta_answers[i] != "No answer found." and meta_completions[i] != "No question found.":
                            meta_scores[i] += int(
                                meta_answers[i] in extract_tool_response([completions[meta_index[i]]])[0]
                            ) * (1 - np.log(2))

                    logger.write("******************question_scores******************\n")
                    logger.write(str(meta_scores) + "\n")
                    filtered_meta_completions = [
                        meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0
                    ]
                    valid_questions = filter_and_duplicate_removal(filtered_meta_completions)

                ## naive tool using reward
                # meta_scores = [int("<tool_response>" in completions[meta_index[i]]) for i in range(len(meta_completions))]
                # filtered_meta_completions = [meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0]
                # valid_questions = filter_and_duplicate_removal(filtered_meta_completions)

                ## answer in tool response reward
                if config.reward_model.pg_reward == "tool_gt":
                    meta_scores = []
                    for i in range(len(meta_index)):
                        if meta_answers[i] != "No answer found." and meta_completions[i] != "No question found.":
                            meta_scores.append(
                                (
                                    int(meta_answers[i] in extract_tool_response([completions[meta_index[i]]])[0])
                                    + int(not (compute_score_subem_verify([meta_answers[i]], [meta_solutions[i]])[0]))
                                    + int(
                                        not (compute_score_subem_verify([meta_completions[i]], [meta_solutions[i]])[0])
                                    )
                                )
                                / 3
                            )
                        else:
                            meta_scores.append(0)

                filtered_meta_completions = [
                    meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0
                ]
                valid_questions = filter_and_duplicate_removal(filtered_meta_completions)
                valid_answers = [meta_answers[meta_completions.index(question)] for question in valid_questions]
                meta_iter = save_new_qa_data(
                    config, valid_questions, logger, style=config.reward_model.pg_type, answers=valid_answers
                )

            else:
                meta_scores = [
                    int("<tool_response>" in completions[meta_index[i]]) for i in range(len(meta_completions))
                ]
                filtered_meta_completions = [
                    meta_completions[i] for i in range(len(meta_completions)) if meta_scores[i] > 0
                ]
                valid_questions = filter_and_duplicate_removal(filtered_meta_completions)
                meta_iter = save_new_qa_data(config, valid_questions, logger, style="gen_rm")

        if do_gen_rm_verification:
            logger.write("******************gen_rm_scores******************\n")
            logger.write(str(gen_rm_scores[0]) + "\n")
        if do_rule_verification:
            logger.write("******************rule_scores******************\n")
            logger.write(str(rule_scores[0]) + "\n")
        if do_meta_verification:
            logger.write("******************meta_scores******************\n")
            logger.write(str(meta_scores[0]) + "\n")

        scores = [-1 for i in range(len(data))]
        if do_gen_rm_verification:
            for i in range(len(gen_rm_scores)):
                scores[gen_rm_index[i]] = gen_rm_scores[i]
        if do_rule_verification:
            for i in range(len(rule_scores)):
                scores[rule_index[i]] = rule_scores[i]
        if do_meta_verification:
            for i in range(len(meta_scores)):
                scores[meta_index[i]] = meta_scores[i]

        assert -1 not in scores

        if config.reward_model.format_reward:
            rollout_texts = [prompt + completion for prompt, completion in zip(prompts, completions, strict=False)]

            def check_think_format(text):
                """计算文本中<|im_start|>assistant\n后紧跟了一个<think>xxx</think>块的比例
                返回0到1之间的数值，表示符合格式的比例."""
                import re

                # 找到所有的<|im_start|>assistant\n位置
                assistant_pattern = r"<\|im_start\|>assistant\n"
                assistant_matches = list(re.finditer(assistant_pattern, text))

                if not assistant_matches:
                    return 0  # 没有找到assistant标记，返回0

                total_count = len(assistant_matches)
                valid_count = 0

                # 检查每个assistant标记后是否紧跟think块
                for match in assistant_matches:
                    start_pos = match.end()  # assistant标记结束的位置
                    remaining_text = text[start_pos:]

                    # 检查是否以<think>开头
                    if remaining_text.strip().startswith("<think>"):
                        # 找到对应的</think>结束标记
                        think_end_pattern = r"</think>"
                        think_end_match = re.search(think_end_pattern, remaining_text)

                        if think_end_match:
                            valid_count += 1

                return valid_count / total_count  # 返回比例

            format_scores = [
                (check_think_format(text) + int("<tool_response>" in completions[i])) / 4
                for i, text in enumerate(rollout_texts)
            ]

            # format_scores作为独立的评分系统，不替换原有的scores
            logger.write("******************format_scores******************\n")
            logger.write(str(format_scores[0]) + "\n")
            logger.write("*******************average format_scores*******************\n")
            logger.write(str(np.mean(format_scores)) + "\n")

            scores = [score + format_score for score, format_score in zip(scores, format_scores, strict=False)]

        if do_gen_rm_verification:
            logger.write("*******************average gen_rm scores*******************\n")
            logger.write(str(np.mean(gen_rm_scores)) + "\n")
        if do_rule_verification:
            logger.write("*******************average rule_scores*******************\n")
            logger.write(str(np.mean(rule_scores)) + "\n")
        if do_meta_verification:
            logger.write("*******************average meta_scores*******************\n")
            logger.write(str(np.mean(meta_scores)) + "\n")

        logger.write("*******************average score*******************\n")
        logger.write(str(np.mean(scores)) + "\n")

        # 获取当前step数，如果文件存在则读取最后一行获取step
        if os.path.exists(reward_history_file):
            step = 0
            try:
                # with open(reward_history_file, 'r', encoding='utf-8') as f:
                #     for line in f:
                #         if line.strip():
                #             last_record = json.loads(line.strip())
                #             step = last_record.get("step", 0)
                step = rewards_history[-1]["step"]
                step += 1
            except BaseException:
                step = 0
        else:
            step = 0

        # 创建新的记录
        new_record = {
            "gen_rm_scores": np.mean(gen_rm_scores) if do_gen_rm_verification else -1,
            "rule_scores": np.mean(rule_scores) if do_rule_verification else -1,
            "meta_scores": np.mean(meta_scores) if do_meta_verification else -1,
            "format_scores": np.mean(format_scores) if config.reward_model.format_reward else -1,
            "scores": np.mean(scores),
            "step": step,
        }
        rewards_history.append(new_record)

        # 以增量形式写入JSONL文件
        with open(reward_history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(new_record, ensure_ascii=False) + "\n")

        if config.trainer.val_before_train:
            scores.append(False)
            return scores

        # is_last_step
        if os.path.exists(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt"):
            meta_iter = int(open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt").read())
        else:
            meta_iter = 1

        if os.path.exists(f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}.jsonl"):
            meta_dataset = datasets.load_dataset("parquet", data_files=config.reward_model.meta_train_file)["train"]
            qa_data = load_qa_data(config, meta_iter)
            if meta_iter == 1:
                last_dataset_len = sum(1 for sample in meta_dataset if sample["reward_model"]["style"] == "meta")
            else:
                last_dataset = load_qa_data(config, meta_iter - 1)
                last_dataset_len = len(last_dataset)
                last_dataset = datasets.Dataset.from_list(last_dataset)
            logger.write("******************last_dataset_len******************\n")
            logger.write(str(last_dataset_len) + "\n")
            logger.write("******************len(qa_data)******************\n")
            logger.write(str(len(qa_data)) + "\n")

            if config.reward_model.online_prompt_generation:
                if meta_iter % 2 == 1:  # this condition should be carefully designed
                    # recent_meta_scores = []
                    # for i in range(1, len(rewards_history)):
                    #     if rewards_history[-i]["meta_scores"] >= 0:
                    #         recent_meta_scores.append(rewards_history[-i]["meta_scores"])
                    #     else:
                    #         break
                    # if len(recent_meta_scores) > 200:
                    #     if np.mean(recent_meta_scores[:100]) <= np.mean(recent_meta_scores[100:200]): #len(qa_data) > last_dataset_len * 0.5: #rollout_n:
                    #         new_dataset = datasets.Dataset.from_list(qa_data)
                    #         logger.write("******************len(new_dataset)******************\n")
                    #         logger.write(str(len(new_dataset)) + "\n")
                    #         mixed_dataset = new_dataset
                    #         mixed_dataset = mixed_dataset.shuffle(seed=37)
                    #         mixed_dataset.to_parquet(f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_train.parquet")
                    #         mixed_dataset.select(range(10)).to_parquet(f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_test.parquet")
                    #         open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt", "w").write(str(meta_iter + 1))
                    #         scores.append(True)
                    #     else:
                    #         scores.append(False)
                    # else:
                    #     scores.append(False)

                    if len(qa_data) > last_dataset_len * rollout_n / 2:
                        new_dataset = datasets.Dataset.from_list(qa_data)
                        logger.write("******************len(new_dataset)******************\n")
                        logger.write(str(len(new_dataset)) + "\n")
                        mixed_dataset = new_dataset
                        mixed_dataset = mixed_dataset.shuffle(seed=37)
                        mixed_dataset.to_parquet(
                            f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_train.parquet"
                        )
                        mixed_dataset.select(range(10)).to_parquet(
                            f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_test.parquet"
                        )
                        open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt", "w").write(
                            str(meta_iter + 1)
                        )
                        scores.append(True)
                    else:
                        scores.append(False)

                else:
                    recent_scores = []
                    for i in range(1, len(rewards_history)):
                        if rewards_history[-i][f"{config.reward_model.pg_type}_scores"] >= 0:
                            recent_scores.append(rewards_history[-i][f"{config.reward_model.pg_type}_scores"])
                        else:
                            break
                    if len(recent_scores) > 200:
                        ratio = (np.mean(recent_scores[:100]) - np.mean(recent_scores[100:200])) / np.mean(
                            recent_scores[100:200]
                        )
                        print(ratio)
                        logger.write("******************ratio******************\n")
                        logger.write(str(ratio) + "\n")
                        if ratio < 0.005:
                            new_dataset = datasets.Dataset.from_list(qa_data)
                            logger.write("******************len(new_dataset)******************\n")
                            logger.write(str(len(new_dataset)) + "\n")
                            mixed_dataset = meta_dataset  # new_dataset
                            mixed_dataset = mixed_dataset.shuffle(seed=37)
                            mixed_dataset.to_parquet(
                                f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_train.parquet"
                            )
                            mixed_dataset.select(range(10)).to_parquet(
                                f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_test.parquet"
                            )
                            open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt", "w").write(
                                str(meta_iter + 1)
                            )
                            scores.append(True)
                        else:
                            scores.append(False)
                    else:
                        scores.append(False)
            else:
                if (
                    len(qa_data) > config.reward_model.offline_data_size
                ):  # last_dataset_len * (rollout_n if meta_iter % 2 == 1 else 1): # this condition should be carefully designed
                    new_dataset = datasets.Dataset.from_list(qa_data)
                    if meta_iter == 1:
                        logger.write("******************len(new_dataset)******************\n")
                        logger.write(str(len(new_dataset)) + "\n")
                        mixed_dataset = new_dataset
                        mixed_dataset = mixed_dataset.shuffle(seed=37)
                        mixed_dataset.to_parquet(
                            f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_train.parquet"
                        )
                        mixed_dataset.select(range(10)).to_parquet(
                            f"./exp/{config.trainer.experiment_name}/meta_asl/qa_data_{meta_iter}_test.parquet"
                        )
                        open(f"./exp/{config.trainer.experiment_name}/meta_asl/meta_iter.txt", "w").write(
                            str(meta_iter + 1)
                        )
                        scores.append(True)
                    else:
                        scores.append(False)
                else:
                    scores.append(False)
        else:
            scores.append(False)

        return scores
