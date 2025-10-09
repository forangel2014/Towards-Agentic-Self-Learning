import os
from pathlib import Path

from rich import filesize, print

from ...utils.configs import Configs
from ...utils.log import info_once
from ...utils.singleton import once


@once
def get_ray_runtime_env() -> dict:
    cur_dir = Path(__file__).resolve().parent
    repo_dir = cur_dir.parent.parent.parent
    verl_dir = cur_dir.parent
    reqs = [
        "evaluate",
        "qwen_vl_utils",
        # match verify with multi processing fixed @haojiao
        "https://image-url-2-feature-1251524319.cos.ap-shanghai.myqcloud.com/wuhuan2/lib/math-verify/fix-mp/math_verify-0.7.0.post1-py3-none-any.whl",
        "debugpy",
        "memray",
        "eas_prediction",
        "mathruler",
        "shortuuid",
    ]
    if Configs.RAY_RUNTIME_REQUIREMENTS.strip():
        reqs += Configs.RAY_RUNTIME_REQUIREMENTS.split(",")

    runtime_env = {
        "env_vars": {
            "REDACCEL_LOG_LEVEL": Configs.LOG_LEVEL,
            "TOKENIZERS_PARALLELISM": os.getenv("TOKENIZERS_PARALLELISM", "true"),
            "NCCL_DEBUG": "INFO",
            "NCCL_IB_DISABLE": "0",
            "NCCL_PXN_DISABLE": "1",
            "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
            "SINK_USER_EMAIL": Configs.SINK_USER_EMAIL,
            "SINK_WEBHOOK": Configs.SINK_WEBHOOK,
            "VLLM_USE_V1": Configs.VLLM_USE_V1,
            "VLLM_LOGGING_LEVEL": Configs.VLLM_LOGGING_LEVEL,
            "VERL_PPO_LOGGING_LEVEL": "INFO",
            "RAY_DEBUG_POST_MORTEM": Configs.RAY_DEBUG_POST_MORTEM,
            "RAY_DEBUG": Configs.RAY_DEBUG,
            "PYTHONPATH": ":".join([str(verl_dir), str(repo_dir), os.getenv("PYTHONPATH", "")]),
        },
        "pip": reqs,
    }

    # 支持透传所有 RAY_RUNTIME_ 前缀的 env
    for key in os.environ:
        if key.startswith("RAY_RUNTIME_"):
            new_key = key[len("RAY_RUNTIME_") :]
            if new_key not in runtime_env["env_vars"]:
                runtime_env["env_vars"][new_key] = os.environ[key]
                info_once(f"add RAY RUNTIME env var: {new_key}={os.environ[key]}")

    # 使 ray debug 起在 worker ip 上而不是 localhost
    if Configs.RAY_DEBUG == "legacy":
        runtime_env["env_vars"]["RAY_DEBUGGER_EXTERNAL"] = "1"

    info_once(f"ray runtime env: {runtime_env}")
    return runtime_env


@once
def get_worker_affinity() -> dict:
    import ray

    if not ray.is_initialized():
        ray.init(runtime_env=get_ray_runtime_env())

    nodes = ray.nodes()
    worker_node_info = []
    for node in nodes:
        if "GPU" not in node["Resources"]:
            continue

        mem = node["Resources"].get("memory", 0)
        worker_info = {
            "node_id": node["NodeID"],
            "node_ip": node["NodeManagerAddress"],
            "memory": mem,
            "memroy_human": filesize.decimal(mem),
        }
        worker_node_info.append(worker_info)

    print(f"worker num: {len(worker_node_info)}")

    if not worker_node_info:
        raise RuntimeError(f"no worker fuond, {nodes=}")

    try:
        cur_node_id = ray._private.worker.global_worker.node._node_id
        for info in worker_node_info:
            if info["node_id"] == cur_node_id:
                return info
    except BaseException:
        pass

    return max(worker_node_info, key=lambda x: x["memory"])
