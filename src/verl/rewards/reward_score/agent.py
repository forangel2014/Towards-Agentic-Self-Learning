import re
import string
from collections import Counter

import evaluate

exact_match = None


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))


def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_score(predict_str: str, ground_truth: str) -> float:
    is_format_error = False
    predict_str = "<think>" + predict_str
    count_1 = predict_str.count("<|begin_of_documents|>\n")
    count_2 = predict_str.count("<|end_of_documents|>\n")
    count_3 = predict_str.count("<|begin_of_query|>")
    count_4 = predict_str.count("<|end_of_query|>")
    count_5 = predict_str.count("<|begin_of_documents|>")
    count_6 = predict_str.count("<|end_of_documents|>")
    count_7 = predict_str.count("<|begin_of_documents|>\n(1)")
    if not (count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7):
        is_format_error = True

    count_assiatant_1 = predict_str.count("Assistant:")
    count_assiatant_2 = predict_str.count("assistant:")
    if count_assiatant_1 != 0 or count_assiatant_2 != 0:
        is_format_error = True

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2:
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != 1 or count_answer_2 != 1:
        is_format_error = True

    answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    if "begin_of_query" in answer_text or "begin_of_documents" in answer_text:
        is_format_error = True

    answer_len = len(answer_text.split())
    if answer_len > 10:
        is_format_error = True

    # if count_7 == 0:
    #     is_format_error = True

    retrieval_pattern = re.compile(r"<\|begin_of_query\|>(.*?)<\|end_of_query\|>", re.DOTALL)
    retrieval_match = re.search(retrieval_pattern, predict_str)
    doc_pattern = re.compile(r"<\|begin_of_documents\|>(.*?)<\|end_of_documents\|>", re.DOTALL)
    doc_match = re.search(doc_pattern, predict_str)

    retrieval_reward = 1.0 if count_7 >= 1 else -1.0
    # em_score = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
    acc_reward, _, _ = f1_score(answer_text, ground_truth)
    acc_reward = 2.0 * acc_reward

    format_reward = -1.0 if is_format_error else 0.0
    return format_reward + retrieval_reward + acc_reward


def compute_score_qwen_tool(predict_str: str, ground_truth: str) -> float:
    # 检查格式
    is_format_error = False

    # 检查<answer>和</answer>标签是否各出现一次且按顺序出现
    count_answer_start = predict_str.count("<answer>")
    count_answer_end = predict_str.count("</answer>")

    if count_answer_start != 1 or count_answer_end != 1:
        is_format_error = True

    # 检查<answer>是否在</answer>之前出现
    if "<answer>" in predict_str and "</answer>" in predict_str:
        if predict_str.find("<answer>") > predict_str.find("</answer>"):
            is_format_error = True

    # 提取answer部分
    try:
        answer_text = predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
    except BaseException:
        answer_text = ""
        is_format_error = True

    # 检查answer长度
    answer_len = len(answer_text.split())
    is_length_error = answer_len > 16

    # 1. 如果format直接错了，无论如何直接set到-2分
    if is_format_error:
        return -2.0

    # 2. 如果format没有错，answer_length > 16直接set到-1分
    if is_length_error:
        return -1.0

    # 3. 如果format没有错，给出f1 score基准，标准化到-1和1之间
    f1, _, _ = f1_score(answer_text, ground_truth)

    # 将f1 score标准化到-1到1之间（f1原始范围是0到1）
    normalized_score = 2.0 * f1 - 1.0

    return normalized_score


def compute_score_eval(predict_str: str, ground_truth: str) -> float:
    global exact_match

    predict_no_think = predict_str.split("</think>")[-1].strip()
    answer_text = predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
    if exact_match is None:
        exact_match = evaluate.load("exact_match")

    score_info = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
    acc_reward = float(score_info["exact_match"])
    return acc_reward
