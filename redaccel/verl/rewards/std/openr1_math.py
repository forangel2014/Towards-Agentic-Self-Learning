# Copyright (c) 2025 RedAccel Authors. All Rights Reserved.

import re
from typing import Sequence

from loguru import logger
from math_verify import parse, verify

from .base import GRPORewards, rewards_registry


def _calc_math_reward(completion: str, sol: str):
    reward = 0.0
    try:
        answer = parse(completion)
        if float(verify(answer, parse(sol))) > 0:
            reward = 1.0
    except Exception as e:
        logger.error(f"Error in string matching: {e}")
        pass  # Continue to next verification method if this fails

    if reward == 0.0:
        try:
            sol_match = re.search(r"<answer>(.*?)</answer>", sol)
            ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

            content_match = re.search(r"<answer>(.*?)</answer>", completion)
            student_answer = content_match.group(1).strip() if content_match else completion.strip()

            if student_answer == ground_truth:
                reward = 1.0
        except Exception as e:
            logger.error(f"Error in string matching: {e}")
            pass  # Keep reward as 0.0 if both methods fail
    return reward


def _ensure_think_prefix(s):
    s = s.strip()
    think = "<think>"
    if s[: len(think)] != think:
        return think + s
    return s


def _calc_format_reward(completion: str, sol: str):
    try:
        completion = _ensure_think_prefix(completion)
        # Check if the format is correct
        regex = r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\n<answer>([\s\S]*?)<\/answer>$"

        m = re.search(regex, completion, re.DOTALL)
        # if the format is not correct, reward is 0
        if m is None or len(m.groups()) != 2:
            return 0
        else:
            return 1
    except Exception as e:
        logger.error(f"Error in calc format reward: {e}")
        return 0


def remove_marked_text(text):
    # 使用正则表达式匹配 <|...|> 并替换为空字符串
    return re.sub(r"<\|.*?\|>", "", text)


@rewards_registry.register(alias=["openr1_math", "openr1_mm"])
class OpenR1MathReward(GRPORewards):
    def __init__(self, name: str = "", weight: float = 1) -> None:
        super().__init__(name, weight)

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        solutions: list[str],
        data_source: str,
        **kwargs,
    ) -> Sequence[float | tuple[float, dict]]:
        res = []
        for c, s in zip(completions, solutions, strict=False):
            c = remove_marked_text(c)
            math_reward = _calc_math_reward(c, s)
            format_reward = _calc_format_reward(c, s)
            score = 0.9 * math_reward + 0.1 * format_reward
            res.append(
                {
                    "score": score,
                    "math_reward": math_reward,
                    "format_reward": format_reward,
                }
            )

        return res
