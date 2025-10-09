import numpy as np

RED_AGENT_NAME = "src_agent"


def process_agent_gen_batch(config, batch, gen_batch):
    if "input_id_list" in batch.non_tensor_batch:
        gen_batch.non_tensor_batch["input_id_list"] = batch.non_tensor_batch.pop("input_id_list")

    if "data_source" in batch.non_tensor_batch:
        # 后面还要用，不要 pop
        gen_batch.non_tensor_batch["data_source"] = batch.non_tensor_batch.get("data_source")

    if hasattr(config.actor_rollout_ref.rollout, "agent") and config.actor_rollout_ref.rollout.agent.activate_agent:
        gen_batch.non_tensor_batch["agent_name"] = np.array([RED_AGENT_NAME] * len(gen_batch), dtype=object)
        tool_name_key = config.actor_rollout_ref.rollout.agent.tool_name_key
        if tool_name_key and tool_name_key in batch.non_tensor_batch.keys():
            gen_batch.non_tensor_batch[tool_name_key] = batch.non_tensor_batch.pop(tool_name_key)


def check_trailing_repetition(text: str, min_repetition_length: int = 10, max_repetition_times: int = 5):
    """判断是否有末尾重复字符

    Parameters
    ----------
    text : str
        文本
    min_repetition_length : int
        最短重复字符的长度（防止正常的输出空格等）
    max_repetition_times : int
        最大判定重复次数，当大于该次数时判定为重复

    """
    from loguru import logger

    if not text or max_repetition_times <= 1:
        return False

    n = len(text)

    for unit_length in range(min_repetition_length, n // max_repetition_times + 1):
        # 检查是否可能形成重复
        if n < unit_length * max_repetition_times:
            continue

        # 提取末尾的重复单元
        unit = text[-unit_length:]

        # 检查是否连续重复
        is_repeating = True
        for i in range(1, max_repetition_times):
            start_idx = -unit_length * (i + 1)
            end_idx = -unit_length * i
            if text[start_idx:end_idx] != unit:
                is_repeating = False
                break

        if is_repeating:
            logger.warning(
                f"Detected trailing repetition: {unit}, len={len(unit)} repeated at least {max_repetition_times} times."
            )
            return True

    return False
