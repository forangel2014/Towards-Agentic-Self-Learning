#!/bin/bash

# Copyright (c) 2025 RedNote Authors. All Rights Reserved.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo DIR: ${DIR}

export PYTHONPATH=$DIR:${PYTHONPATH}

set -exo pipefail

EXP_NAME=${1:-"asl_grm"}
DATA_DIR=/diancpfs/user/sunwangtao/data/asl/meta
MODEL_PATH=/diancpfs/user/tiandao/models/Qwen2.5-7B-Instruct
AGENT_STOP='["</tool_call>","</answer>"]'
META_TRAIN_FILE=${DATA_DIR}/train.parquet
META_VAL_FILE=${DATA_DIR}/test.parquet
VERIFY_TRAIN_FILE=${DATA_DIR}/verify_train.parquet
VERIFY_VAL_FILE=${DATA_DIR}/verify_test.parquet
# Function to build cmd array
build_cmd() {
    local train_file=$1
    local val_file=$2
    local meta_iter=$3
    
    # 根据meta_iter模3的结果设置max_prompt_length
    local max_prompt_length=1024
    if [[ $((meta_iter % 3)) -eq 2 ]]; then
        max_prompt_length=2048
    fi
    
    cmd=(
        trainer.plugin_dir=${DIR}/asl
        data.train_files=$train_file
        data.val_files=$val_file
        data.template=qwen
        data.num_workers=4
        data.train_batch_size=32
        data.max_prompt_length=$max_prompt_length
        data.max_response_length=10240
        data.filter_overlong_prompts=True
        data.truncation=right
        data.return_raw_chat=True
        algorithm.adv_estimator=gae
        algorithm.kl_ctrl.kl_coef=0.0
        algorithm.lam=1.0
        reward_model.skip_special_tokens=False
        reward_model.num_examine=1
        reward_model.reward_name=asl_reward_3
        reward_model.as_reward_model=policy
        +reward_model.agentic=True
        +reward_model.format_reward=True
        +reward_model.pg_type=gen_rm
        +reward_model.pg_reward=entropy+verify

        +reward_model.rule_verification_method=subem
        +reward_model.meta_train_file=${META_TRAIN_FILE}
        +reward_model.meta_val_file=${META_VAL_FILE}
        +reward_model.online_prompt_generation=True
        +reward_model.gen_rm_verification_method=binary

        actor_rollout_ref.model.path=${MODEL_PATH}
        actor_rollout_ref.model.enable_gradient_checkpointing=True
        actor_rollout_ref.model.freeze_vision_tower=True
        actor_rollout_ref.actor.clip_ratio_c=3.0
        actor_rollout_ref.actor.optim.lr=1e-6
        actor_rollout_ref.actor.ppo_mini_batch_size=32
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
        actor_rollout_ref.actor.entropy_coeff=0.001

        actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
        actor_rollout_ref.model.use_remove_padding=True
        critic.model.use_remove_padding=True

        actor_rollout_ref.actor.fsdp_config.param_offload=True
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.n=4
        actor_rollout_ref.rollout.temperature=1
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8
        actor_rollout_ref.rollout.enable_chunked_prefill=True
        actor_rollout_ref.rollout.enforce_eager=True
        actor_rollout_ref.rollout.free_cache_engine=True
        actor_rollout_ref.rollout.max_num_batched_tokens=32768
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
        actor_rollout_ref.ref.fsdp_config.param_offload=True
        #actor_rollout_ref.actor.use_kl_loss=True
        actor_rollout_ref.rollout.agent.activate_agent=True
        actor_rollout_ref.rollout.agent.tool_name_key=env_name
        actor_rollout_ref.rollout.agent.custom_stop=${AGENT_STOP}
        actor_rollout_ref.rollout.agent.single_response_max_tokens=2048
        actor_rollout_ref.rollout.agent.max_turns=9
        actor_rollout_ref.rollout.agent.concurrent_workers=4
        actor_rollout_ref.rollout.agent.show_tqdm=False
        actor_rollout_ref.rollout.agent.mock=False
        #algorithm.use_kl_in_reward=True
        # actor_rollout_ref.rollout.agent.params=${AGENT_PARAMETERS}
        critic.optim.lr=1e-5
        critic.cliprange_value=10
        critic.model.path=${MODEL_PATH}
        critic.model.fsdp_config.param_offload=True
        critic.model.fsdp_config.optimizer_offload=True
        critic.ppo_micro_batch_size_per_gpu=4
        trainer.val_before_train=False
        trainer.test_freq=0
        trainer.save_freq=64
        trainer.rotate_ckpt=False
        trainer.total_epochs=128
        trainer.max_actor_ckpt_to_keep=10000
        trainer.max_critic_ckpt_to_keep=10000
        trainer.experiment_name=${EXP_NAME}
        trainer.default_local_dir=./exp/${EXP_NAME}/ckpt
        trainer.logger=[console,tensorboard]
        trainer.rl_logging_board_dir=./exp/${EXP_NAME}/rl_logging_board
    )
}

# Function to run asl
run_asl() {
    dt=`date '+%Y-%m-%d_%H-%M-%S'`
    log_dir=./exp/${EXP_NAME}/log/${dt}
    mkdir -p ${log_dir}
    python -m src.cli rl "${cmd[@]}" 2>&1 | tee ${log_dir}/main.log
}

# Main execution loop
while true; do
    # Check if meta_iter.txt exists and read the iteration number
    meta_iter_file="./exp/${EXP_NAME}/meta_asl/meta_iter.txt"
    if [[ -f "${meta_iter_file}" ]]; then
        meta_iter=$(cat "${meta_iter_file}")
        echo "Found meta iteration: ${meta_iter}"
        
        # Update data files based on meta iteration
        train_file="./exp/${EXP_NAME}/meta_asl/qa_data_$((meta_iter-1))_train.parquet"
        val_file="./exp/${EXP_NAME}/meta_asl/qa_data_$((meta_iter-1))_test.parquet"
        
        echo "Using updated data files:"
        echo "  Train file: ${train_file}"
        echo "  Val file: ${val_file}"
        
        # Build cmd with updated files
        # if [[ $((meta_iter % 3)) -eq 2 ]]; then
        #     build_cmd "${VERIFY_TRAIN_FILE}" "${VERIFY_VAL_FILE}" "${meta_iter}"
        # else
        #     build_cmd "${train_file}" "${val_file}" "${meta_iter}"
        # fi
        build_cmd "${train_file}" "${val_file}" "${meta_iter}"

    else
        echo "Meta iteration file not found, using default data files"
        meta_iter=1
        # Build cmd with default files
        build_cmd ${META_TRAIN_FILE} ${META_VAL_FILE} ${meta_iter}
    fi
    
    # Run asl
    echo "Starting asl training..."
    run_asl
    
    # Check if asl completed successfully
    if [[ $? -eq 0 ]]; then
        echo "asl training completed successfully"
        # Check if meta_iter.txt was updated
        if [[ -f "${meta_iter_file}" ]]; then
            new_meta_iter=$(cat "${meta_iter_file}")
            if [[ "${new_meta_iter}" != "${meta_iter}" ]]; then
                echo "Meta iteration updated from ${meta_iter} to ${new_meta_iter}, continuing..."
                last_ckpt=$(cat "./exp/${EXP_NAME}/ckpt/latest_checkpointed_iteration.txt")
                echo "删除旧数据文件: ./exp/${EXP_NAME}/ckpt/global_step_${last_ckpt}/data.pt"
                rm -f "./exp/${EXP_NAME}/ckpt/global_step_${last_ckpt}/data.pt"
                echo "删除意外产生的下一轮文件: ./exp/${EXP_NAME}/meta_asl/qa_data_${new_meta_iter}.json"
                rm -f "./exp/${EXP_NAME}/meta_asl/qa_data_${new_meta_iter}.json"

            else
                echo "Meta iteration unchanged, stopping..."
                break
            fi
        else
            echo "Meta iteration file not found after completion, stopping..."
            break
        fi
    else
        echo "asl training failed with exit code $?, stopping..."
        break
    fi
done

echo "Script execution completed"
