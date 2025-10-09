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

import os
import random
import re

import requests
from openai import OpenAI

try:
    from math_verify import parse, verify
    from math_verify.errors import TimeoutException  # noqa
    from math_verify.metric import math_metric  # noqa
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig  # noqa
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

# OpenAI API setup for generative verification
openai_api_key = "EMPTY"
openai_api_base_list = [
    os.environ.get("LLM_AS_A_JUDGE_BASE", "http://10.39.6.230:80/v1"),
]

client_list = []
model_name_list = []
for api_base in openai_api_base_list:
    try:
        client = OpenAI(
            api_key=openai_api_key,
            base_url=api_base,
        )
        client_list.append(client)
        response = requests.get(f"{api_base}/models")
        models = response.json()
        model_name_list.append(models["data"][0]["id"])
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI client for {api_base}: {e}")

MATH_VERIFY_PROMPT = """# CONTEXT #
I am a teacher, and I have some high-level math problems. I am tasked with evaluating the correctness of a student's answer. 
Below, I am provided with a problem and a reference answer. Additionally, a student's answer is provided. My job is to assess whether the student's answer captures the same meaning as the reference answer, even when expressed with different wording or format.

# OBJECTIVE #
I need you to judge whether the student's answer is correct given the ground truth answer.

Your tasks include:
1. Identify Mathematical or Notational Equivalence: Pay special attention to any LaTeX expressions in both answers. Confirm that the mathematical relationships, variables, and operations conveyed are equivalent.

# TONE #
Professional, scientific.

# RESPONSE: MARKDOWN REPORT #
## Equivalence Judgement
[Whether the student's answer share the same meaning with the reference answer. (TRUE or FALSE)]

# ATTENTION #
 - The reference answer is ALWAYS correct. You should carefully judge whether the student gives the same answer as reference answer.
 - The Equivalence Judgement is only TRUE or FALSE. The answer is FALSE even if the student's final answer almost correct with a minor mistakes.
 - Don't give extra explanation.

**Reference Answer**
{gold_ans}

## Student Final Answer
{pred_ans}"""


def rule_math_verify(ground_truth, model_answer):
    """Verify math answers using symbolic parsing and verification."""
    try:
        gold = parse(ground_truth)
        answer = parse(model_answer)
        return verify(gold, answer)
    except Exception:
        return False


def generative_verify(ground_truth, model_answer):
    """Verify math answers using LLM-as-a-judge."""
    if not client_list:
        print("Warning: No OpenAI clients available for generative verification")
        return False

    client_idx = random.randint(0, len(client_list) - 1)
    client = client_list[client_idx]
    model_name = model_name_list[client_idx]

    full_prompt = MATH_VERIFY_PROMPT.format(
        gold_ans=ground_truth,
        pred_ans=model_answer,
    )

    response = ""
    for it in range(2):
        try:
            chat_response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": full_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.0,
            )
            response = chat_response.choices[0].message.content.strip()
            break
        except Exception as e:
            print(f" [ERROR math] generative_verify error: {e}")
            continue

    judgement = response.split("## Equivalence Judgement")[-1].lower()
    if "true" in judgement and "false" not in judgement:
        return True
    elif "false" in judgement and "true" not in judgement:
        return False
    else:
        print(f" [ERROR math] verify bug output: {response}")
        return False


def extract_boxed_content(text):
    """
    Extract content from \boxed{...} expressions using balanced brace counting.
    This correctly handles nested braces of arbitrary depth.
    """
    matches = []

    # Find all \boxed{ starting positions
    for match in re.finditer(r"\\boxed\{", text):
        start_brace_pos = match.end() - 1  # Position of opening {
        brace_count = 0
        pos = start_brace_pos

        # Count braces to find matching closing brace
        while pos < len(text):
            char = text[pos]
            if char == "\\" and pos + 1 < len(text):
                # Skip escaped characters (like \{ or \})
                pos += 2
                continue
            elif char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    # Found matching closing brace
                    content = text[start_brace_pos + 1 : pos]
                    matches.append(content)
                    break
            pos += 1

    return matches


def compute_score_math(predict_str: str, ground_truth: str, extra_info=None) -> float:
    """Enhanced compute_score function with format checking and dual verification."""
    is_format_error = False
    # Check for matching think tags
    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    model_answer = ""
    predict_no_think = predict_str.split("</think>")[-1].strip()

    # Use the robust extraction function instead of regex
    answer_list = extract_boxed_content(predict_no_think)

    if len(answer_list) == 0:
        acc_reward = 0.0
        is_format_error = True
    else:
        if len(answer_list) > 1:
            is_format_error = True

        model_answer = answer_list[-1]
        if rule_math_verify(ground_truth, model_answer):
            acc_reward = 1.0
        else:
            acc_reward = 1.0 if generative_verify(ground_truth, model_answer) else 0.0

    if len(model_answer) == 0:
        print(f" [MATH DEBUG] model answer is empty, {predict_str=}")

    format_reward = -1.0 if is_format_error else 0.0
    print(f" [MATHDEBUG] {ground_truth=} {model_answer=}, {acc_reward=}, {format_reward=}")
    return 1.2 * acc_reward + 0.4 * format_reward
