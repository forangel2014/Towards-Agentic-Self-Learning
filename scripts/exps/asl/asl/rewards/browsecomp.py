# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

"""
BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents
Authors: Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, Mia Glaese
https://openai.com/index/browsecomp/
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import openai
import requests


# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_model_predictions.py#L11
QUERY_TEMPLATE = """
{Question}

Your response should be in the following format:
Explanation: {{your explanation for your final answer}}
Exact Answer: {{your succinct, final answer}}
Confidence: {{your confidence score between 0% and 100% for your answer}}
""".strip()

# from: https://github.com/centerforaisafety/hle/blob/7b6be5aad6f9b43af3857de7867f3b52f6e4acb3/hle_eval/run_judge_results.py#L16-L33
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available.
""".strip()

CHOICE_STRINGS = ["yes", "no"]


Message = dict[str, Any]  # keys role, content
MessageList = list[Message]


class SamplerBase:
    """Base class for defining a sampling model, which can be evaluated, or
    used as part of the grading process."""

    def __call__(self, message_list: MessageList) -> str:
        raise NotImplementedError


def get_completion_gpt(messages, api_key, temperature=0.7, max_tokens=2048):
    api_version = "2023-03-15-preview"
    while True:
        try:
            completion = requests.post(
                f"https://runway.devops.xiaohongshu.com/openai/chat/completions?api-version={api_version}",
                json={"messages": messages, "temperature": temperature, "max_tokens": max_tokens},
                headers={
                    "Content-Type": "application/json",
                    "api-key": api_key,
                },
                allow_redirects=True,
            ).json()
            # print(completion)
            return completion["choices"][0]["message"]["content"]
        except Exception as e:
            print(e)
            try:
                if "Please try again with a different prompt" in completion["error"]["message"]:
                    print(messages)
                    return ""
                time.sleep(2)
            except BaseException:
                time.sleep(2)


client = openai.OpenAI(
    api_key="sk-Y8Eek9eur0AeYc2DuGwpHfiy1Kik52cdgxqCr8qdiWn8KBRv", base_url="https://maxflyhub.com/v1"
)


def get_completion_maxflyhub(messages, api_key, temperature=0.7, max_tokens=2048):
    answer = None
    while answer is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14", messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            answer = response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
    return answer


class GPTChatCompletionSampler(SamplerBase):
    """Sample from OpenAI's chat completion API."""

    def __init__(
        self,
        # model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        api_key=None,
    ):
        self.api_key = api_key
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list

        return get_completion_gpt(
            messages=message_list, api_key=self.api_key, temperature=self.temperature, max_tokens=self.max_tokens
        )
        # return get_completion_maxflyhub(messages = message_list,
        #         api_key = self.api_key,
        #         temperature = self.temperature, max_tokens = self.max_tokens)


class BrowseCompEval:
    def __init__(self, grader_model: SamplerBase):
        self.grader_model = grader_model

    def grade_sample(self, question: str, correct_answer: str, response: str) -> str:
        grader_prompt = GRADER_TEMPLATE.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )

        prompt_messages = [self.grader_model._pack_message(content=grader_prompt, role="user")]
        # print(prompt_messages)

        grading_response = self.grader_model(prompt_messages)

        # print(grading_response)

        match = re.search(r"correct: (yes|no)", grading_response)
        return match.group(0) if match else "no"  # Default to "no" if no match

    def score(self, problem, answer, response_text):
        grade_result = self.grade_sample(problem, answer, response_text)

        # Metrics based on grading response
        is_correct = grade_result == "yes"
        # is_incorrect = grade_result == "no"

        score = int(is_correct)

        return score

    def scores(self, questions: list[str], answers: list[str], responses: list[str]) -> list[int]:
        """使用多线程并行评测多个样本.

        Args:
            questions: 问题列表
            answers: 正确答案列表
            responses: 模型回答列表

        Returns:
            评分结果列表，每个元素为0或1
        """
        if len(questions) != len(answers) or len(answers) != len(responses):
            raise ValueError("questions, answers, and responses must have the same length")

        scores_list = [0] * len(questions)

        # 使用线程池进行并行评测
        with ThreadPoolExecutor(max_workers=min(32, len(questions))) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(self.score, questions[i], answers[i], responses[i]): i for i in range(len(questions))
            }

            # 收集结果
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    score_result = future.result()
                    scores_list[index] = score_result
                except Exception as exc:
                    print(f"Sample {index} generated an exception: {exc}")
                    scores_list[index] = 0  # 出错时默认为0分

        return scores_list


grader_model = GPTChatCompletionSampler(api_key="ec38315430e64b129fbf24d634045004")
evaluator = BrowseCompEval(grader_model=grader_model)
