# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

import os
from pathlib import Path

import ray
from loguru import logger
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from ...utils.event_tracker import track_main
from ..hparams.runtime_env import get_ray_runtime_env, get_worker_affinity
from ..rewards.compose import load_compose_reward_manager

worker_affinity = get_worker_affinity()
worker_affinity_strategy = NodeAffinitySchedulingStrategy(node_id=worker_affinity["node_id"], soft=False)
logger.info(f">>>>>> {worker_affinity=}")


def run_ppo(config: DictConfig):
    if not ray.is_initialized():
        ray.init(runtime_env=get_ray_runtime_env())

    affinity_ip = worker_affinity["node_ip"]
    logger.info(
        f">>>>>> current node ip: {ray._private.worker.global_worker.node._node_ip_address}, affinity node ip: {affinity_ip}"
    )
    runner = Runner.options(resources={f"node:{affinity_ip}": 1}).remote()
    ray.get(runner.run.remote(config))


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1, runtime_env=get_ray_runtime_env(), scheduling_strategy=worker_affinity_strategy)
def ppo_task(config: DictConfig):
    _ppo_task(config)


@ray.remote(num_cpus=1, runtime_env=get_ray_runtime_env(), scheduling_strategy=worker_affinity_strategy)
class Runner:
    def run(self, config: DictConfig):
        _ppo_task(config)


def _update_config(config: DictConfig, working_dir: Path, output_dir: Path, is_multimodal: bool = False):
    if is_multimodal:
        # NOTE(wuhuan): vllm 0.7.3 多模开 chunked prefill 会导致 image token vs placeholder 数目对不上！
        # config.actor_rollout_ref.rollout.enable_chunked_prefill = False
        config.actor_rollout_ref.rollout.limit_images = max(config.actor_rollout_ref.rollout.limit_images, 1)
        config.actor_rollout_ref.rollout.image_resolution = config.data.image_resolution

    # 兼容老的配置
    if os.getenv("RED_MOCK_REWARD") == "1":
        config.reward_model.mock = True

    if config.actor_rollout_ref.model.path and config.actor_rollout_ref.model.path.endswith("/"):
        config.actor_rollout_ref.model.path = config.actor_rollout_ref.model.path.rstrip("/")

    if config.critic.model.path and config.critic.model.path.endswith("/"):
        config.critic.model.path = config.critic.model.path.rstrip("/")


@track_main(command="rl")
def _ppo_task(config: DictConfig):
    logger.info(
        f">>>>>> runtime node id: {ray._private.worker.global_worker.node._node_id} ip: {ray._private.worker.global_worker.node._node_ip_address}"
    )
    # print initial config
    from omegaconf import OmegaConf

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer
    from verl.trainer.ppo.reward import load_reward_manager
    from verl.utils.fs import copy_to_local

    from ...models import register_all_models
    from .dataset import RLHFDataset

    register_all_models()

    working_dir = Path(config.trainer.working_dir)
    output_dir = Path(config.trainer.default_local_dir)

    # NOTE(wuhuan): force change working dir for worker node
    os.chdir(working_dir)
    print(f"current working dir: {working_dir}")

    # Download the checkpoint from HDFS to the local machine.
    # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )

    # instantiate tokenizer
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(
        local_path, trust_remote_code=trust_remote_code, use_fast=True
    )  # used for multimodal LLM, could be none

    _update_config(config, working_dir, output_dir, is_multimodal=processor is not None)

    print(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    with open(output_dir / "runtime_config.yaml", "w") as f:
        OmegaConf.save(config=config, f=f)
    print(f"config file dump to: {output_dir / 'runtime_config.yaml'}")

    # define worker classes
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        ray_worker_group_cls = RayWorkerGroup
        actor_rollout_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker

        actor_rollout_cls = (
            AsyncActorRolloutRefWorker if config.actor_rollout_ref.rollout.mode == "async" else ActorRolloutRefWorker
        )
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    import ray.cloudpickle as pickle

    for k in role_worker_mapping:
        _ = pickle.dumps(role_worker_mapping[k])

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }

    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    # use reference model
    if (
        config.algorithm.use_kl_in_reward
        or config.actor_rollout_ref.actor.use_kl_loss
        or config.reward_model.as_reward_model == "ref"
    ):
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id

    if config.reward_model.reward_manager == "compose":
        reward_fn = load_compose_reward_manager(tokenizer, config, is_val=False)
        val_reward_fn = load_compose_reward_manager(tokenizer, config, is_val=True)
    else:
        reward_kwargs = config.reward_model.get("reward_kwargs", {})
        reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)
        val_reward_fn = load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs)

    # Note that we always use function-based RM for validation
    if config.trainer.test_freq <= 0 and not config.trainer.val_before_train and not config.data.val_files:
        val_reward_fn = None

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    train_dataset = RLHFDataset(
        parquet_files=config.data.train_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
    )

    if val_reward_fn is not None:
        val_dataset = RLHFDataset(
            parquet_files=config.data.val_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )
    else:
        val_dataset = train_dataset

    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.init_workers()
    trainer.fit()
    return trainer
