import atexit
import os

import torch
import transformers

from verl.utils.logging_utils import info_random


def get_multi_modal_inputs(micro_batch):
    multi_modal_inputs = {}
    if "multi_modal_inputs" in micro_batch:
        all_keys = {}
        for elem in micro_batch["multi_modal_inputs"]:
            for key in elem.keys():
                if key not in all_keys:
                    all_keys[key] = elem[key]

        for key in all_keys:
            key_inputs = []
            for inputs in micro_batch["multi_modal_inputs"]:
                if key in inputs:
                    key_inputs.append(inputs[key])

            multi_modal_inputs[key] = torch.cat(key_inputs, dim=0)
            info_random(0.1, f"{key=} {multi_modal_inputs[key].type()} {multi_modal_inputs[key].shape}")

    return multi_modal_inputs


class TorchProfiler:
    def __init__(
        self,
        disable: bool = False,
        output_dir: str = "./profiles",
        skip_first: int = 1,
        profile_mem: bool = True,
    ):
        from loguru import logger

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if local_rank or disable:
            self.prof = None
            return

        logger.info(f"Profiling is enabled! Output directory: {os.path.abspath(output_dir)}")
        os.makedirs(output_dir, exist_ok=True)
        if transformers.is_torch_npu_available():
            import torch_npu

            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                l2_cache=False,
            )
            self.prof = torch_npu.profiler.profile(
                activities=[
                    torch_npu.profiler.ProfilerActivity.CPU,
                    torch_npu.profiler.ProfilerActivity.NPU,
                ],
                schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=1, repeat=1, skip_first=skip_first),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
                record_shapes=True,
                profile_memory=profile_mem,
                with_stack=True,
                with_flops=False,
                with_modules=False,
                experimental_config=experimental_config,
            )
        else:
            self.prof = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(skip_first=skip_first, wait=1, warmup=1, active=1, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir, use_gzip=True),
                profile_memory=profile_mem,
                with_stack=True,
                record_shapes=True,
            )
        self.prof.start()
        atexit.register(self.prof.stop)

    def step(self):
        if self.prof is None:
            return
        self.prof.step()

    def on_step_end(self, args, state, control, **kwargs):
        self.step()
