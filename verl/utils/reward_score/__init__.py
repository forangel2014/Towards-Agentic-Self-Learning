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
# from . import gsm8k, math, prime_math, prime_code

from loguru import logger

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "ToRL",
        "math500",
        "aime24",
        "aime25",
        "open-r1/Big-Math-RL-Verified-Processed",
    ]:
        from . import math_verifier

        res = math_verifier.compute_score(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from . import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from . import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["INK-USC/riddle_sense", "splat"]:
        from . import general_verify

        res, res_detail = general_verify.batch_compute_verify_score([solution_str], [ground_truth])
        res = res[0]
    elif data_source in ["open-r1/codeforces"]:
        from src.verl.rewards.reward_score import code_verify

        res, pass_rate = code_verify.compute_score(solution_str, ground_truth)

        return {
            "score": res,
            "pass_wrate": pass_rate,
        }

    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        # from . import search_r1_like_qa_em
        pass

    elif data_source in ["rag_v2-train", "R1-Searcher"]:
        from src.verl.rewards.reward_score import agent

        res = agent.compute_score_qwen_tool(solution_str, ground_truth)

    elif data_source in ["rag_v2-test"]:
        from src.verl.rewards.reward_score import agent

        res = agent.compute_score_eval(solution_str, ground_truth)

    elif data_source in ["vl_agent"]:
        from src.verl.rewards.reward_score import vl_agent

        res = vl_agent.compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ["vstar", "chart", "arxivqa"]:
        from . import vl_agent_v2

        res = vl_agent_v2.compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ["thinklite_eureka", "xince", "thinklite"]:
        from . import vl_agent_v2

        res = vl_agent_v2.compute_score_math(solution_str, ground_truth, extra_info)

    elif data_source in ["frozenlake"]:
        res = 0.0

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, list | tuple):
        logger.warning(f"[DEBUG 000] bad res: {data_source=}, {solution_str=}, {ground_truth=}, {extra_info=}, {res=}")
        res = res[0]

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])

    return res


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
