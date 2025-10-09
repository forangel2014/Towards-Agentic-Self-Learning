from loguru import logger


def default_compute_score(data_source, prompt_str, solution_str, ground_truth, extra_info=None) -> float | dict:
    prompt_str = prompt_str.split("<|user|>")[-1].replace("<|endofuser|>", "")
    if data_source == "openai/gsm8k":
        from verl.utils.reward_score import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)

    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval"]:
        from verl.utils.reward_score import math

        res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    # elif data_source == "math_dapo" or data_source.startswith("aime"):
    #     from verl.utils.reward_score import math_dapo

    #     res = math_dapo.compute_score(solution_str, ground_truth)

    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from verl.utils.reward_score import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        from verl.utils.reward_score import prime_code

        res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    # elif data_source in ["INK-USC/riddle_sense", "splat"]:
    #    from verl.utils.reward_score import general_verify
    #
    #    res, res_detail = general_verify.batch_compute_verify_score([solution_str], [ground_truth], {"question": prompt_str})
    #    res = res[0]
    elif data_source in ["open-r1/codeforces", "BigCodeBench"] or "skywork-code" in data_source or "skywork_code" in data_source:
        from . import code_verify

        res = code_verify.compute_score(solution_str, ground_truth)

    elif data_source in ["hiyouga/geometry3k"]:
        from verl.utils.reward_score import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)

    elif data_source in ["rag_v2-train"]:
        from . import agent

        res = agent.compute_score_qwen_tool(solution_str, ground_truth)

    elif data_source in ["rag_v2-test"]:
        from . import agent

        res = agent.compute_score_eval(solution_str, ground_truth)

    elif data_source in ["vstar", "vl_agent", "chart", "browsecomp", "r1-searcher-v3", "seekworld"]:
        from . import vl_agent_v3

        res = vl_agent_v3.compute_score(solution_str, ground_truth, extra_info)

    elif data_source in ["skywork-math", "revisual-r1", "thinklite_eureka", "thinklite_eureka-no_tool", "xince"]:
        from . import vl_agent_v3

        res = vl_agent_v3.compute_score_math_with_boxed(solution_str, ground_truth, extra_info)

    # elif data_source in ["vstar", "chart", "arxivqa", "Chinese-SimpleQA", "browsecomp", "R1-Searcher", "INK-USC/riddle_sense", "splat", "IOR_Bench", "katielink/nejm-medqa-diagnostic-reasoning-dataset", "BiomixQA", "CFLUE", "FinEval", "LeCaRDv2", "huatuo", "ecthr_cases"]:
    #     from . import vl_agent_v2

    #     res = vl_agent_v2.compute_score(solution_str, ground_truth, {"question": prompt_str})

    # elif data_source in ["thinklite_eureka", "thinklite_eureka-no_tool", "xince"]:
    #     from . import vl_agent_v3

    #     res = vl_agent_v3.compute_score_math(solution_str, ground_truth, extra_info)

    # elif data_source in ["thinklite_eureka", "xince", "thinklite", "OlympicArena", "olympic_reason_no_tool", "olympic_reason", "phys_reason_no_tool", "SARA"]:
    #     from . import vl_agent_v2

    #     res = vl_agent_v2.compute_score(solution_str, ground_truth, {"question": prompt_str})

    elif data_source in ["seekworld-test", "vstar-test", "visulogic-test", "ocr_reasoning-test"]:
        from . import vl_agent_v3

        res = vl_agent_v3.compute_score_acc(solution_str, ground_truth, extra_info)

    elif data_source in ["aime24", "aime25"]:
        from . import vl_agent_v3

        res = vl_agent_v3.evaluate_aime(solution_str, ground_truth, extra_info)

    elif data_source in [
        "mmsearch-test",
        "simpleqa-openai-test",
        "chinese_simpleqa-test",
        "browsecomp-zh-test",
        "browsecomp-openai-test",
        "simple-vqa-test",
        "zero-bench-test",
        "zero-bench-no-tools",
    ]:
        from .benchmark import evaluate_browsecomp_zh, evaluate_chinese_simpleqa, evaluate_openai_brosecomp_hle, evaluate_openai_simpleqa

        if data_source in ["simpleqa-openai-test", "mmsearch-test", "zero-bench-test", "zero-bench-no-tools"]:
            res = evaluate_openai_simpleqa(solution_str, ground_truth, extra_info)
        elif data_source in ["chinese_simpleqa-test", "simple-vqa-test"]:
            res = evaluate_chinese_simpleqa(solution_str, ground_truth, extra_info)
        elif data_source == "browsecomp-openai-test":
            res = evaluate_openai_brosecomp_hle(solution_str, ground_truth, extra_info)
        elif data_source == "browsecomp-zh-test":
            res = evaluate_browsecomp_zh(solution_str, ground_truth, extra_info)
        else:
            raise ValueError(f" [ERROR] invalid {data_source=}")

    elif data_source in ["frozenlake"]:
        res = 0.0
    
    elif data_source in ["eng_wiki_data", "moegirlsdata", "simpleqa-hj", "chinese_simpleqa-hj"]:
        from .deepsearch import compute_score_simple, compute_score_eval_simple
        
        if data_source == "eng_wiki_data":
            res = compute_score_simple(solution_str, ground_truth, extra_info)
        elif data_source == "moegirlsdata":
            res = compute_score_simple(solution_str, ground_truth, extra_info)
        elif data_source == "simpleqa-hj":
            res = compute_score_eval_simple(solution_str, ground_truth, extra_info)
        elif data_source == "chinese_simpleqa-hj":
            res = compute_score_eval_simple(solution_str, ground_truth, extra_info)
        else:
            raise ValueError(f" [ERROR] invalid {data_source=}")

    # elif data_source in ["Chinese-SimpleQA", "browsecomp"]:
    #    from . import search
    #
    #    res = search.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, (list, tuple)):
        logger.warning(f"[DEBUG 000] bad res: {data_source=}, {solution_str=}, {ground_truth=}, {extra_info=}, {res=}")
        res = res[0]

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)

    return res
