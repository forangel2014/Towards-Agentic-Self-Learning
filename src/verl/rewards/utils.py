import functools
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import requests
import torch
from loguru import logger
from openai import OpenAI

REWARD_METRICS_KEY = "reward_metrics"
REWARD_AGG_LABELS_KEY = "reward_agg"


def with_reward_metrics(
    reward: float | dict,
    metrics: Mapping[str, float | list],
    agg_labels: Optional[Mapping[str, str]] = None,
) -> dict:
    if not isinstance(reward, dict):
        return {
            "score": reward,
            REWARD_METRICS_KEY: metrics,
            REWARD_AGG_LABELS_KEY: agg_labels or {},
        }
    reward[REWARD_METRICS_KEY] = metrics
    reward[REWARD_AGG_LABELS_KEY] = agg_labels or {}
    return reward


def to_jsonable(data):
    if isinstance(data, dict):
        return {key: to_jsonable(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, torch.Tensor):
        return data.tolist()
    elif isinstance(data, list | tuple):
        return [to_jsonable(item) for item in data]
    elif isinstance(data, float | str | bool | int):
        return data
    return str(data)


class AgentTraceDumper:
    def __init__(self, output_dir: Path, enable: bool = True) -> None:
        self.output_dir = output_dir
        self.enable = enable
        self.data = {}

    def add(self, request_id, agent_trace: Optional[dict]):
        if agent_trace and self.enable:
            self.data[request_id] = agent_trace

    def flush(self, step: int):
        if not self.enable:
            return

        if (self.output_dir / f"step_{step}.sqlite").exists():
            shutil.rmtree(self.output_dir / f"step_{step}.sqlite", ignore_errors=True)

        try:
            from sqlitedict import SqliteDict
        except ImportError:
            logger.error("Please install sqlitedict to use _AgentTraceDumper.")
            self.data.clear()
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        db = SqliteDict(str(self.output_dir / f"step_{step}.sqlite"))
        try:
            for request_id in self.data:
                agent_trace = self.data[request_id]
                db[request_id] = agent_trace
        finally:
            db.commit()
            db.close()
            self.data.clear()


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None,
):
    """
    通用retry装饰器，可以套用不同的class

    Args:
        max_attempts: 最大重试次数
        delay: 初始延迟时间（秒）
        backoff_factor: 延迟时间的增长因子，默认2.0（指数增长）
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:  # 不是最后一次尝试
                        # 指数增长：delay * (backoff_factor ^ attempt)
                        current_delay = delay * (backoff_factor**attempt)

                        if on_retry:
                            on_retry(attempt + 1, e, current_delay)

                        time.sleep(current_delay)
                    else:
                        # 最后一次尝试失败，返回None而不是抛出异常
                        print(f"所有{max_attempts}次尝试都失败了，返回None ,last_exception={last_exception}")
                        return None

            return None  # 这行代码实际上不会执行到

        return wrapper

    return decorator


def split_llm_result(result):
    """
    将llm的result切分为think和response
    """
    if "</think>" in result:
        think, response = result.rsplit("</think>", 1)
        return think, response
    else:
        return None, result


def parse_messages(sp, prompt):
    """
    将sp和prompt构造为messages
    """
    messages = [{"role": "system", "content": sp}, {"role": "user", "content": prompt}]
    return messages


class AllinModel:
    def __init__(
        self,
        model_name,
        api_key="QST2cc6de76054a22d0f6446772811b7050",
        base_url="",
    ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.max_tokens = 4096
        self.temperature = 0.9

    @retry(max_attempts=3, delay=1.0)
    def generate_with_messages(self, messages, stream=False):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout=600,
        )
        return json.loads(completion.model_dump_json())["choices"][0]["message"]["content"]

    def generate(self, system_prompt, user_prompt, stream=False):
        messages = parse_messages(system_prompt, user_prompt)
        return self.generate_with_messages(messages, stream)


class HTTPModel:
    def __init__(self, model_name: str, ip: str, port: int = 8000):
        """
        初始化VLLM模型客户端

        Args:
            model_name: 模型名称
            ip: 服务器IP地址
            port: 服务器端口，默认8000
        """
        self.model_name = model_name
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}/v1/chat/completions"

    @retry(max_attempts=3, delay=1.0)
    def generate(
        self, system_prompt: str, user_prompt: str, temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = 8196
    ) -> Optional[str]:
        """
        调用自己部署的VLLM模型生成回复
        """
        headers = {"Content-Type": "application/json"}
        messages = parse_messages(system_prompt, user_prompt)
        data = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        response = requests.post(self.base_url, headers=headers, data=json.dumps(data))
        response_data = response.json()
        return response_data["choices"][0]["message"]["content"]


class WithWorkerGroupMixin:
    worker_group: dict[str, Any]

    def set_worker_group(self, worker_group: dict[str, Any]):
        self.worker_group = worker_group


if __name__ == "__main__":
    # 测试AllinModel
    print("=== 测试AllinModel ===")
    model = AllinModel(model_name="qwen3-32b")
    result = model.generate("你是一个有用的AI助手。", "你好呀")
    print(result)
