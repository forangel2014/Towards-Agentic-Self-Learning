import concurrent
import json
import random
import re

import requests

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask

IP_LIST = ["10.39.6.230", "10.39.27.184"]
URL_LIST = []
for ip in IP_LIST:
    URL_LIST.append(f"http://{ip}:80/v1/chat/completions")

DEFAULT_SP = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

REWARD_MODEL_BASE_PROMPT = """
[TASK]
Given:
1. A question about a specific task
2. One or more answers to that question
3. a reference answer wich is the correct answer to the question
Please carefully evaluate the correctness of the provided answer(s) to the quetion
"""

ILLUSTRATION_PROMPT = """
[INPUT]
<question>:
{}
</question>
<answer>:
{}
</answer>
<reference answer>:
{}
</reference answer>
"""

RANKING_PROMPT = """
Please rank the answer with one of the following options (from best to worst):

A: The answer is perfectly correct, even better than the reference answer.

B: The answer is correct as the reference answer.

C: The answer is correct, but it is not as good as the reference answer.

D: The answer is incorrect, but there are points that are valuable and worth further exploration.

E: The answer is incorrect, and there are a few correct points.

F: The answer is incorrect, but at least it is related to the question.

G: The answer is completely incorrect or not related to the question.

please first analyze the correctness of the provided answer(s) to the quetion, and then draw your conclusion in the following format:
<Ranking>: <A/B/C/...>
"""

# BINARY_PROMPT = """
# Please determine whether the answer is correct or wrong:

# # output format
# <think>
# ...(all your analysis and reasoning process)
# </think>
# <Conclusion>: correct/wrong
# """

BINARY_PROMPT = """
please first analyze the correctness of the provided answer(s) to the quetion, and then draw your conclusion in the following format:
<Conclusion>: correct/wrong
"""


def get_solution(solution_str):
    solution = solution_str.split("<|assistant|>")[-1].split("<|endofassistant|>")[0]
    boxed_content = re.findall(r"\\boxed{(.*?)}", solution)
    if len(boxed_content) > 0:
        solution = boxed_content[0]
    return solution


def get_format_score(res_text):
    """
    获取文本结果的格式得分
    :param res_text: 模型生成的文本结果
    :return: 格式得分（符合要求返回0.5，否则返回0.0）
    """
    return 0
    try:
        # 提取助手回复部分
        res_text = res_text.split("<|endofassistant|>\n<|assistant|>\n")[1]

        # 检查是否包含思考标签
        if "<think>" in res_text and "</think>" in res_text:
            # 获取思考部分的内容
            think_content = res_text.split("<think>")[1].split("</think>")[0].strip()

            # 检查思考内容是否至少包含3个词
            if len(think_content.split()) >= 3:
                # 获取</think>之后的内容
                after_think = res_text.split("</think>")[1].strip()

                # 检查</think>之后是否有内容（不直接是结束标记）
                if after_think and not after_think.startswith("<|endofassistant|>"):
                    return 0.5

        return 0.0
    except Exception as e:
        print(f"Error in get_format_score: {e}")
        return 0.0


def get_score_binary(eval_text):
    """
    Get the score from the text result
    :param res_text: The text result from the model
    :return: The score
    """
    if "Conclusion" in eval_text:
        eval_text = eval_text.split("Conclusion")[-1]
        if "correct" in eval_text:
            return 1.0
        elif "wrong" in eval_text:
            return -1.0
        else:
            return 0.0
    else:
        return 0.0


def get_score_ranking(eval_text):
    """
    Get the score from the text result
    :param res_text: The text result from the model
    :return: The score
    """
    if "Ranking" in eval_text:
        eval_text = eval_text.split("Ranking")[1].strip()
        score_mapping = {
            "A": 1.0,
            "B": 0.66,
            "C": 0.33,
            "D": 0.0,
            "E": -0.33,
            "F": -0.66,
            "G": -1.0,
        }
        # 遍历截断后的文本，寻找第一个有效评分字母
        for char in eval_text:
            if char in score_mapping:
                return score_mapping[char]
        return 0.0
    return 0.0


def build_gen_batch_with_prompt(prompt, tokenizer):
    """
    Build a batch of inputs for the model
    :param prompt: The prompt to be used for generation
    :param tokenizer: The tokenizer to be used
    :return: A dictionary containing the input IDs and attention mask
    """
    chat = [{"content": DEFAULT_SP, "role": "system"}, {"content": prompt, "role": "user"}]
    prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
    input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
        prompt=prompt_with_chat_template,
        tokenizer=tokenizer,
        max_length=4096,
        pad_token_id=tokenizer.pad_token_id,
        left_pad=True,
        truncation="right",
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": compute_position_id_with_mask(attention_mask),
    }


def build_gen_batch_with_prompts(prompts, tokenizer):
    """
    批量构建输入批次
    :param prompts: 提示列表
    :param tokenizer: 分词器
    :return: 包含所有提示的批处理字典
    """
    all_input_ids = []
    all_attention_masks = []
    all_position_ids = []

    for prompt in prompts:
        chat = [{"content": "You are a helpful AI assistant", "role": "system"}, {"content": prompt, "role": "user"}]
        prompt_with_chat_template = tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=tokenizer,
            max_length=4096,
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

    return {"input_ids": batch_input_ids, "attention_mask": batch_attention_mask, "position_ids": batch_position_ids}


def get_policy_rm_output_batch(prompts, tokenizer, policy_model):
    """
    批量获取模型输出
    :param prompts: 提示列表
    :param tokenizer: 分词器
    :param policy_model: 策略模型
    :return: 生成的文本列表
    """
    batch = build_gen_batch_with_prompts(prompts, tokenizer)
    batch = DataProto.from_dict(batch)
    batch.meta_info = {
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "recompute_log_prob": False,
        "do_sample": False,
        "validate": True,
    }
    # pad to be divisible by dp_size
    batch_padded, pad_size = pad_dataproto_to_divisor(batch, policy_model.world_size)
    # print("in anchor 7, batch_padded shape: ", batch_padded.batch['input_ids'].shape)
    output_gen_batch_padded = policy_model.generate_sequences(batch_padded)
    # unpad
    output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
    output_ids = output_gen_batch.batch["responses"]
    output_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return output_texts


def batch_compute_verify_score(
    solution_str_list, ground_truth_list, extra_info=None, policy_model=None, tokenizer=None, *args, **kwargs
):
    """The scoring function for verification.

    Args:
        solution_str_list: the solution text
        ground_truth_list: the ground truth
        policy_model: the policy model to use

    Returns:
        The score of the solution
    """
    batch_size = len(solution_str_list)
    questions = []
    prompts = []
    solutions = []

    # 解析每对解决方案
    for i in range(batch_size):
        # TODO支持多轮
        question = extra_info.get("question", "")
        questions.append(question)
        solution = get_solution(solution_str_list[i])
        solutions.append(solution)
        reference = ground_truth_list[i]
        prompt = REWARD_MODEL_BASE_PROMPT + ILLUSTRATION_PROMPT.format(question, solution, reference) + BINARY_PROMPT
        prompts.append(prompt)

    if policy_model:
        all_eval_texts = get_policy_rm_output_batch(prompts, tokenizer, policy_model)
    else:
        all_eval_texts = get_remote_rm_output_batch(prompts)

    eval_scores = [get_score_binary(text) for text in all_eval_texts]
    final_scores = eval_scores

    return final_scores, [
        {"prompt": prompt, "response": response, "score": score}
        for prompt, response, score in zip(prompts, all_eval_texts, final_scores, strict=False)
    ]


def call_remote_gen_rm(prompt, sp=DEFAULT_SP, max_len=20000):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(
        {
            "model": "qwen3-8b",
            "messages": [
                {"role": "system", "content": sp},
                {"role": "user", "content": prompt[:max_len]},
            ],
            "temperature": 0.7,
            "top_p": 0.7,
            "max_tokens": 12240,
        }
    )
    url = random.choice(URL_LIST)
    response = requests.post(url, headers=headers, data=data, timeout=360)
    resjson = response.json()
    text_result = resjson["choices"][0]["message"]["content"]
    return text_result


def get_remote_rm_output_batch(prompts):
    """
    并行批量获取远程模型输出，并保持与输入顺序一致
    :param prompts: 提示列表
    :return: 按原始顺序排列的生成文本列表
    """
    # 使用多线程或异步请求来并行处理多个提示
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 为每个prompt添加索引标记
        indexed_results = []
        futures = []

        for idx, prompt in enumerate(prompts):
            future = executor.submit(call_remote_gen_rm, prompt)
            futures.append((idx, future))

        # 收集结果
        for idx, future in futures:
            try:
                result = future.result()
                indexed_results.append((idx, result))
            except Exception as exc:
                print(f"生成异常: {exc}")
                # 发生异常时添加空结果，保持索引完整
                indexed_results.append((idx, ""))

    # 按原始索引排序
    indexed_results.sort(key=lambda x: x[0])

    # 只返回结果部分
    results = [item[1] for item in indexed_results]
    return results
