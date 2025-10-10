# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import functools
import json
import os
import queue
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .._version import get_versions
from .configs import Configs
from .log import logger
from .signal import register_handler
from .singleton import once


EVENT_SERVER_NAME = "qs-src-event-default"


def get_service_list(eds_host_addr: str, service_name: str, retry_cnt=2) -> List[str]:
    url = f"{eds_host_addr}/endpoints?serviceName={service_name}"
    for i in range(0, retry_cnt):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            ret = resp.json()
            if ret["code"] != 0:
                logger.warning("[EventTracker] invalid code %s msg %s" % (ret["code"], ret["msg"]))
                continue
            return ["http://" + i["address"] for i in ret["data"]["endpoints"]]
        except Exception as e:
            logger.error("[EventTracker] requests.GET %s except %s" % (url, e))
        time.sleep(1)

    return []


class EventSink:
    def send(self, data: Dict):
        pass


class WebHookEventSink(EventSink):
    """ref to
    https://docs.xiaohongshu.com/doc/5f02dcdd594a81bffdf497e142e57ed7."""

    def __init__(self, email: str, webhooks: str = "") -> None:
        self.emails = email.split(",") if email else []
        self.webhooks = webhooks.split(",") if webhooks else []
        self.reported = False
        logger.info(
            f"[EventTracker] WebHookEventSink initialized with email: {self.emails} and webhooks: {self.webhooks}"
        )

    def send(self, data: dict):
        if not self.webhooks:
            return

        if data.get("src_rt_status") != RunStatus.FAIL or self.reported:
            return

        for webhook_url in self.webhooks:
            self._send(webhook_url, data)
        self.reported = True

    def _send(self, webhook_url, data: Dict):
        elapsed = data.get("src_rt_elapsed")
        error_message = data.get("src_rt_error_message", "")
        cur_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        qs_url = ""
        if (job_id := os.getenv("QS_JOB_ID")) and (trial_id := os.getenv("QS_TRIAL_ID")):
            qs_url = f"https://qs2.devops.xiaohongshu.com/trainning/detail/{job_id}/{trial_id}"
            qs_url = f"- QS 任务地址：[{qs_url}]({qs_url})"

        markdown = f"""## RedNote 任务失败

- 用户：{self.emails}
- 时间：{cur_date}
- 耗时：{elapsed:.2f} s
{qs_url}

错误详情：

```python
{error_message}
```
        """

        payload = {"msgtype": "markdown", "markdown": {"content": markdown, "mentioned_list": self.emails}}

        header = {"Content-Type": "application/json"}
        res = requests.post(webhook_url, headers=header, json=payload, timeout=5)
        try:
            res.raise_for_status()
            data = res.json()
            if data.get("errcode", -1) == 0:
                logger.info(f"[EventTracker] send webhook {webhook_url} success, resp: {data}")
            else:
                logger.warning(f"[EventTracker] send webhook {webhook_url} failed, resp: {data}")
        except Exception as e:
            logger.exception(f"[EventTracker] send webhook {webhook_url} failed: {e}")


class FileEventSink(EventSink):
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def send(self, data: Dict):
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")


class RemoteEventSink(EventSink):
    def __init__(self, services: List[str], dry: bool = False) -> None:
        self.services = [i.rstrip("/") for i in services]
        self.dry = dry
        self._check()

    def send(self, data: Dict):
        for k in data:
            if isinstance(data[k], int):
                data[k] = min(data[k], 2147483647)  # INT_MAX for int32

        body = {
            "biz_name": "src",
            "data": [{"key": k, "value": v} for k, v in data.items()],
            "dry": self.dry,
        }
        resp = requests.post(f"{self.services[0]}/v1/event", json=body, timeout=2)
        resp.raise_for_status()

    def _check(self):
        if not self.services:
            return False

        url = f"{self.services[0]}/v1/health"
        req = requests.get(url, timeout=2)
        req.raise_for_status()


class RunStatus(str, Enum):
    START = "START"
    SUCCESS = "SUCCESS"
    FAIL = "FAIL"
    INTERRUPT = "INTERRUPT"
    TERMINATE = "TERMINATE"


class EventConsumer(threading.Thread):
    def __init__(self, sinks: List[EventSink], q: queue.Queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sinks = sinks
        self.q = q
        # NOTE(wuhuan): 不需要设为守护线程，在 track main 处显式等待结束，否则会漏消息
        # self.daemon = True  # 设置为守护线程

        self.last_emit_time = time.time()
        self.emit_interval = Configs.EVENT_UPDATE_INTERVAL

    def _run(self, item: dict):
        for sink in self.sinks:
            try:
                sink.send(item)
            except BaseException as e:
                logger.error(f"[EventTracker] emit data to sink({sink}) failed, error: {e}")
        self.last_emit_time = item["_timestamp"]
        self.q.task_done()

    def run(self):
        logger.info("[EventTracker] EventConsumer start")
        while 1:
            try:
                item = self.q.get(timeout=1, block=True)
                if item is None:  # 如果队列中放入了 None，表示停止消费
                    break

                if item.get("_must_emit"):
                    self._run(item)
                elif item["_timestamp"] - self.last_emit_time > self.emit_interval:
                    self._run(item)

            except (queue.Empty, TimeoutError):
                continue
        logger.info("[EventTracker] EventConsumer stopped")


class EventTracker:
    _instance: Optional["EventTracker"] = None
    _RT_PREFIX = "src_rt_"
    _CFG_PREFIX = "src_cfg_"

    def __init__(self, sinks: List[EventSink], enable: bool = True, **init_kwargs) -> None:
        self.enable = enable and sinks
        self.sinks = sinks
        self.data = self._init(**init_kwargs)

        self.q = queue.Queue()
        if self.enable:
            EventTracker.register_signal_handler()
            self.__consumer = EventConsumer(sinks, self.q)
            self.__consumer.start()

    def _init(self, **init_kwargs):
        if not self.enable:
            return {}

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))

        data = {
            "qs_trial_id": os.getenv("QS_TRIAL_ID"),
            "qs_job_id": os.getenv("QS_JOB_ID"),
            "qs_biz_type": os.getenv("QS_BIZ_TYPE"),
            "qs_biz_id": os.getenv("QS_BIZ_ID"),
            "qs_biz_execution_id": os.getenv("QS_BIZ_EXECUTION_ID"),
            "qs_region": os.getenv("REGION"),
            "qs_gpu_type": os.getenv("GPU_TYPE"),
            "qs_gpu_num_per_node": local_world_size,
            "qs_node_num": world_size // max(local_world_size, 1),
            "qs_user": os.getenv("QS_USER"),
            "qs_docker": None,
            "src_version": get_versions()["version"],
            # 在 trial 内可唯一标识即可，防止 int32 溢出
            "src_rt_run_id": int(time.time()) % 114514666,
        }
        return dict(data, **init_kwargs)

    def update(self, key: str, value: Any, *, emit: bool = True):
        """更新单条埋点数据.

        Parameters
        ----------
        key : str
            埋点键
        value : str
            埋点值
        emit : bool
            是否立刻上报该埋点，如果为 False，埋点会累积到下一次满足一定时间间隔(emit_interval)上报
        """
        if not self.enable:
            return

        self.data[key] = value
        logger.debug(f"[EventTracker] update tracking event {key=}, {value=}")
        self.emit(must=emit)

    def update_rt(self, key: str, value: Any, *, emit: bool = True):
        key = self._with_rt_prefix(key)
        self.update(key, value, emit=emit)

    @once
    def update_rt_once(self, key: str, value: Any, *, emit: bool = True):
        self.update_rt(key, value, emit=emit)

    def updates(self, data: Dict[str, Any], *, emit: bool = True):
        """批量更新埋点数据.

        Parameters
        ----------
        data : Dict[str, Any]
            埋点数据
        emit : bool
            是否立刻上报该埋点，如果为 False，埋点会累积到下一次满足一定时间间隔(emit_interval)上报
        """
        logger.debug(f"[EventTracker] update tracking event {data=}")
        self.data.update(data)
        self.emit(must=emit)

    def updates_rt(self, data: Dict[str, Any], *, emit: bool = True):
        d = {self._with_rt_prefix(k): v for k, v in data.items()}
        self.updates(d, emit=emit)

    @once
    def updates_rt_once(self, data: Dict[str, Any], *, emit: bool = True):
        self.updates_rt(data, emit=emit)

    def track_args(self, args: List[Any]):
        if not self.enable:
            return {}

        data = {}
        for arg in args:
            arg_dict = asdict(arg)
            for k, v in arg_dict.items():
                if isinstance(v, (int, float, bool, str)):
                    data[self._with_cfg_prefix(k)] = v

        # ADHOC(wuhuan) #############################################################
        # for multi cloud unified ai dataset
        if data.get("src_cfg_use_multi_cloud_unified_ai_dataset") or data.get(
            "src_cfg_use_multi_cloud_unified_ai_dataset_v2"
        ):
            data["src_cfg_use_multi_cloud_ai_dataset"] = True

        # for ds zero
        for arg in args:
            if hasattr(arg, "deepspeed_plugin") and arg.deepspeed_plugin is not None:
                data["src_cfg_ds_zero"] = arg.deepspeed_plugin.zero_stage
                data["src_cfg_ds_offload"] = arg.deepspeed_plugin.offload_optimizer_device != "none"
        self.updates(data)

    @contextmanager
    def with_timer(self, key: str, accumulate: bool = False, *, emit: bool = True):
        start = time.time()
        try:
            yield
        finally:
            v = time.time() - start
            if accumulate:
                v += self.data.get(self._with_rt_prefix(key), 0)

            self.update_rt(key, v)

    @contextmanager
    def catch_main(self):
        start_time = time.time()
        self.updates_rt(
            {
                "start_timestamp": start_time,
                "status": RunStatus.START,
            },
            emit=False,
        )
        payload = {}
        try:
            yield
            payload["status"] = RunStatus.SUCCESS
        except KeyboardInterrupt:
            payload["status"] = RunStatus.INTERRUPT
            raise
        except BaseException as e:
            payload["status"] = RunStatus.FAIL
            payload["error_message"] = str(e)
            raise
        finally:
            stop_time = int(time.time())
            payload["stop_timestamp"] = stop_time
            payload["elapsed"] = stop_time - start_time
            if train_end_time := self.data.get("src_rt_train_stop_time"):
                payload["post_elapsed"] = stop_time - train_end_time
            self.updates_rt(payload)

            if self.enable:
                self.q.put(None)
                self.__consumer.join()

    def emit(self, must: bool = True):
        if not self.enable:
            return

        self.data["_timestamp"] = time.time()
        self.data["_must_emit"] = must
        self.data["src_rt_event_timestamp"] = self.data["_timestamp"]

        self.q.put(self.data)

    @classmethod
    def from_env(cls, command: Optional[str] = None):
        sinks = []

        command = command or os.environ.get("RED_COMMAND", "train")
        # TODO(wuhuan): 暂时只支持 train 和 rl
        if command not in ("train", "rl"):
            return cls(sinks, src_rt_command=command)

        # 只在 rank0 执行即可
        if os.environ.get("RANK", "0") != "0" or not Configs.EVENT_TRACKER_ENABLE:
            return cls(sinks, src_rt_command=command)

        try:
            if Configs.EVENT_SERVER_ADDR:
                sinks.append(RemoteEventSink([Configs.EVENT_SERVER_ADDR]))
            else:
                sinks.append(RemoteEventSink(get_service_list(Configs.EDS_HOST_ADDR, EVENT_SERVER_NAME)))
        except BaseException as e:
            logger.warning(f"remote event server is not accessible, error: {e}")

        local_sink = FileEventSink(Path(Configs.LOG_FILE).with_suffix("").with_suffix(".events.log"))
        sinks.append(local_sink)
        webhook_sink = WebHookEventSink(Configs.SINK_USER_EMAIL, Configs.SINK_WEBHOOK)
        sinks.append(webhook_sink)
        return cls(sinks, src_rt_command=command)

    @classmethod
    def clear(cls):
        if cls._instance is not None and cls._instance.enable:
            cls._instance.q.put(None)
        cls._instance = None

    @classmethod
    def get_instance(cls, **kwargs):
        if cls._instance is None:
            cls._instance = cls.from_env(**kwargs)
        return cls._instance

    def _with_prefix(self, prefix: str, key: str):
        if key.startswith(prefix):
            return key
        return prefix + key

    _signal_registered = False

    @classmethod
    def register_signal_handler(cls):
        if cls._signal_registered:
            return

        def _close_before_exiting(signum, _):
            pid = os.getpid()
            logger.warning(f"{pid=} receive signal SIGTERM/SIGINT, {signum=}")
            if cls._instance is not None:
                cls.get_instance().update_rt("status", RunStatus.INTERRUPT)

        # 这里只需要捕捉 SIGINT 就行， SIGTERM 会走上面 catch_main() 的逻辑
        register_handler(signal.SIGINT, [_close_before_exiting])
        cls._signal_registered = True

    _with_cfg_prefix = functools.partialmethod(_with_prefix, _CFG_PREFIX)
    _with_rt_prefix = functools.partialmethod(_with_prefix, _RT_PREFIX)


def track_main(command: Optional[str] = None):
    def wrap(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            with EventTracker.get_instance(command=command).catch_main():
                return func(*args, **kwargs)

        return inner

    return wrap
