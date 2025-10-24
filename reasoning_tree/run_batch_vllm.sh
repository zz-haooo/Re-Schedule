TIMESTAMP=$(date +'%Y%m%d_%H%M%S')
RESULT_DIR="eval_logs/batch_vllm_run_${TIMESTAMP}"
mkdir -p "$RESULT_DIR"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Evaluate base model only
MODEL_PATHS=(
    "${MODEL_PATH:-Qwen2.5-Math-7B}"
)

DATASET="dapo_converted"

# GPU configuration
GPU_DEVICES="0,1,2,3"           # GPU devices
# GPU_DEVICES="4,5,6,7"           # GPU devices
IFS=',' read -ra GPU_ARRAY <<< "$GPU_DEVICES"
TENSOR_PARALLEL_SIZE=${#GPU_ARRAY[@]}

# Batch parallel parameters
BRANCH_COUNT=4 # Branching factor
NUM_QUESTIONS=17398  # For testing, can be set to larger value
BATCH_SIZE=100   # Batch size
MAX_BRANCH_DEPTH=4
ROLLOUT_TIMES=1
HEURISTIC_BRANCH_INTERVAL=100  # Branch every 100 tokens

# Sample range parameters (optional, defaults to all)
START_INDEX=0  # Start index, set to null or comment out to remove limit
END_INDEX=4000    # End index, set to null or comment out to remove limit

# 17398

TEMPERATURE=1.0
TOP_P=1.0
MAX_TOKENS=4096

echo "Starting batch parallel VLLM heuristic branching inference evaluation"
echo "================================================"
echo "   - Dataset: ${DATASET}"
echo "   - Batch size: ${BATCH_SIZE}"
echo "   - Heuristic interval: ${HEURISTIC_BRANCH_INTERVAL} tokens (branch every 100 tokens)"
echo "   - Using VLLM: true"
echo "   - Branch count: ${BRANCH_COUNT}"
echo "   - Max depth: ${MAX_BRANCH_DEPTH}"
echo "   - Rollout times: ${ROLLOUT_TIMES}"
echo "   - Question count: ${NUM_QUESTIONS}"
echo "   - GPU devices: ${GPU_DEVICES} (${TENSOR_PARALLEL_SIZE} GPUs)"
if [ ! -z "$START_INDEX" ] && [ ! -z "$END_INDEX" ]; then
    echo "   - Sample range: [${START_INDEX}, ${END_INDEX})"
else
    echo "   - Sample range: All samples"
fi
echo "================================================"

for model_path in "${MODEL_PATHS[@]}"; do
    model_name=$(basename "$model_path")
    echo "Starting evaluation for $model_name"
    
    # Call batch parallel evaluation script
    python "eval_batch_vllm.py" \
        --model_path "$model_path" \
        --log_dir "eval_logs" \
        --result_dir "$RESULT_DIR" \
        --dataset_name "$DATASET" \
        --dataset_split "test" \
        --branch_count "$BRANCH_COUNT" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --max_tokens "$MAX_TOKENS" \
        --timestamp "$TIMESTAMP" \
        --num_questions "$NUM_QUESTIONS" \
        --max_branch_depth "$MAX_BRANCH_DEPTH" \
        --rollout_times "$ROLLOUT_TIMES" \
        --heuristic_branch_interval "$HEURISTIC_BRANCH_INTERVAL" \
        --batch_size "$BATCH_SIZE" \
        --gpu_devices "$GPU_DEVICES" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        $([ ! -z "$START_INDEX" ] && echo "--start_index $START_INDEX" || echo "") \
        $([ ! -z "$END_INDEX" ] && echo "--end_index $END_INDEX" || echo "") \
        2>&1 | tee "${RESULT_DIR}/${model_name}_batch.log"
    
    if [ $? -eq 0 ]; then
        echo -e "\033[1;32m$model_name batch parallel evaluation completed\033[0m"
    else
        echo -e "\033[1;31m$model_name batch parallel evaluation failed\033[0m"
    fi
    echo "-----------------------------------------------"
done

echo "Batch parallel VLLM heuristic branching inference evaluation completed!"
echo "Results saved to: $RESULT_DIR" 