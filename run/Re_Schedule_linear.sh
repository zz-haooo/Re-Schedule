#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set PYTHONPATH to use local verl version
export PYTHONPATH="${PROJECT_ROOT}/verl:$PYTHONPATH"

# Verify which VERL version is being used
echo "Current VERL path:"
python3 -c "import verl; print(verl.__file__)"

# Set training parameters
export PYTHONHASHSEED=42
export PYTORCH_SEED=42
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

project_name='new_GRPO_DAPO'
train_path="${PROJECT_ROOT}/datasets/data_ranked.parquet"
test_path="${PROJECT_ROOT}/datasets/aime24.parquet"
model_path="${MODEL_PATH:-Qwen2.5-Math-7B}"
run_name=Qwen2.5-Math-7B_verl_data_linear_0.5_2_10

export WANDB_INIT_TIMEOUT=300
export WANDB_TIMEOUT=300
export WANDB_RETRY_DELAY=60
export WANDB_MAX_RETRIES=10

save_contents="['hf_model']"
current_datetime=$(date +"%Y%m%d_%H%M%S")
run_name="${run_name}_${current_datetime}"
train_files="['$train_path']"
test_files="['$test_path']"

# Run training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    data.truncation='left'  \
    +data.use_dynamic_weights=True \
    +data.metric_column=metric \
    +data.alpha_mode="linear" \
    +data.dynamic_weights_min=0.5 \
    +data.dynamic_weights_max=2 \
    +data.dynamic_weights_total_epochs=10 \
    +data.dynamic_weights_k=10 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.actor.checkpoint.save_contents=${save_contents} \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.rollout_data_dir="${PROJECT_ROOT}/rollout_data/$project_name/$run_name" \
    trainer.critic_warmup=0 \
    trainer.logger="['console','wandb']" \
    trainer.project_name=$project_name \
    trainer.experiment_name=$run_name \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=50 \
    +trainer.save_best_only=False \
    +trainer.delete_old_best_checkpoint=False \
    +trainer.save_after=60 \
    +trainer.best_metric_key=val-core/math_dapo/acc/mean@32 \
    trainer.resume_mode=disable