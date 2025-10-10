import random
from typing import Optional

import requests
from openai import OpenAI

from ..utils import with_reward_metrics

openai_api_key = "EMPTY"


def get_chat_template():
    chat_template = """
Below are two answers to a question. Question is [Question], [Standard Answer] is the standard answer to the question, and [Model_answer] is the answer extracted from a model's output to this question.  Determine whether these two answers are consistent.
Note that [Model Answer] is consistent with [Standard Answer] whenever they are essentially the same. If the meaning is expressed in the same way, it is considered consistent, for example, 'pink' and 'it is pink'.
If they are consistent, Judement is 1; if they are different, Judement is 0. Just output Judement and don't output anything else.\n\n
"""
    return chat_template


def get_gpt4_score_ICE():
    example_1 = """
[Question]: Is the countertop tan or blue?
[Standard Answer]: The countertop is tan.
[Model_answer] : tan
Judgement: 1
"""  # noqa

    example_2 = """
[Question]: On which side of the picture is the barrier?
[Standard Answer]: The barrier is on the left side of the picture.
[Model_answer] : left
Judgement: 1
"""  # noqa

    example_3 = """
[Question]: Is the kite brown and large?
[Standard Answer]: Yes, the kite is brown and large.
[Model_answer] : Yes
Judgement: 1
"""  # noqa

    example_4 = """
[Question]: Are the spots on a giraffe?
[Standard Answer]: No, the spots are on a banana.
[Model_answer] : no
Judgement: 1
"""  # noqa

    example_5 = """
[Question]: Who is wearing pants?
[Standard Answer]: The boy is wearing pants.
[Model_answer] : The person in the picture is wearing pants.
Judgement: 1
"""  # noqa

    example_6 = """
[Question]: Is the man phone both blue and closed?
[Standard Answer]: Yes, the man phone is both blue and closed.
[Model_answer] : No.
Judgement: 0
"""  # noqa

    example_7 = """
[Question]: What color is the towel in the center of the picture?
[Standard Answer]: The towel in the center of the picture is blue.
[Model_answer] : The towel in the center of the picture is pink.
Judgement: 0
"""  # noqa

    return [example_1, example_2, example_3, example_4, example_5, example_6, example_7]


def get_prompt(predict_str, ground_truth, question):
    examples = get_gpt4_score_ICE()
    chat_template = get_chat_template()
    demo_prompt = chat_template
    for example in examples:
        demo_prompt += example + "\n\n"
    test_prompt = f"""
[Question]: {question}
[Standard Answer]: {ground_truth}
[Model_answer] : {predict_str}
Judgement:"""
    full_prompt = f"{demo_prompt}{test_prompt}"

    return full_prompt


def _compute_score(
    predict_str: str,
    ground_truth: str,
    client_list: list[OpenAI],
    model_name_list: list[str],
    weights: list[float],
    extra_info: Optional[dict],
    **kwargs,
) -> dict:
    assert len(weights) == 4, f"{weights=}"

    is_format_error = False
    # count_think_1 = predict_str.count("<think>")
    # count_think_2 = predict_str.count("</think>")
    # if count_think_1 != count_think_2:
    #     is_format_error = True

    # NOTE(wuhuan): 适配 qwen 和 小地瓜
    count_vision_1 = predict_str.count("<|vision_start|><|image_pad|>") or predict_str.count("<|img|><|imgpad|>")
    count_vision_2 = predict_str.count("<|image_pad|><|vision_end|>") or predict_str.count("<|imgpad|><|endofimg|>")
    if count_vision_1 != count_vision_2:
        is_format_error = True

    predict_no_think = predict_str.split("</think>")[-1].strip()
    count_answer_1 = predict_no_think.count("<answer>")
    count_answer_2 = predict_no_think.count("</answer>")
    if count_answer_1 != count_answer_2:
        is_format_error = True

    answer_text = predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    question_text = extra_info.get("question", "") if extra_info else ""
    full_prompt = get_prompt(answer_text, ground_truth, question_text)

    if client_list:
        client_idx = random.randint(0, len(client_list) - 1)
        client = client_list[client_idx]
        model_name = model_name_list[client_idx]

        chat_response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt},
            ],
            seed=random.randint(0, 1000000),
            temperature=0.3,
        )
        response = chat_response.choices[0].message.content.strip()
        if "Judgement:" in response:
            response = response.split("Judgement:")[-1].strip()
            if "1" in response:
                acc_reward = 1.0
            elif "0" in response:
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
        else:
            if response == "1":
                acc_reward = 1.0
            elif response == "0":
                acc_reward = 0.0
            else:
                print(f" [WARNING] resp format error {response=}")
                acc_reward = 0.0
    else:
        acc_reward = 0

    # Penalize for model trying to predict longer answer to hack llm-as-judge
    if len(answer_text) >= 300:
        acc_reward = 0.0
        is_format_error = True

    tool_reward_base = 1.0 if count_vision_1 > 0 else 0.0
    tool_reward = 1.0 if count_vision_1 > 0 and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0

    final_score = weights[0] * acc_reward + weights[1] * format_reward + weights[2] * tool_reward + weights[3] * tool_reward_base

    metrics = {
        "rewards/vl_acc": [acc_reward],
    }
    return with_reward_metrics(
        {
            "score": final_score,
            "acc_reward": acc_reward,
            "format_reward": format_reward,
            "tool_reward": tool_reward,
            "tool_reward_base": tool_reward_base,
            "judge_response": chat_response.choices[0].message.content.strip() if client_list else "",
        },
        metrics=metrics,
        agg_labels={"data_source": kwargs["data_source"]},
    )


g_client_list = []
g_model_name_list = []


def compute_score(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """枫原原来的hardcode 逻辑，不要用 !!!"""
    global g_client_list, g_model_name_list

    if not g_client_list or not g_model_name_list:
        openai_api_base_list = [
            "http://10.39.11.28:10000/v1",
            "http://10.39.11.27:10000/v1",
        ]

        for api_base in openai_api_base_list:
            client = OpenAI(
                api_key=openai_api_key,
                base_url=api_base,
            )
            g_client_list.append(client)

            response = requests.get(f"{api_base}/models")
            models = response.json()
            g_model_name_list.append(models["data"][0]["id"])

    return _compute_score(
        predict_str=predict_str,
        ground_truth=ground_truth,
        client_list=g_client_list,
        model_name_list=g_model_name_list,
        weights=[1.0, 0.2, 1.0, 0.2],
        extra_info=extra_info,
    )["score"]
