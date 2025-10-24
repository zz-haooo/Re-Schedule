
#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set parameters
DATA_DIR="${PROJECT_ROOT}/reasoning_tree/eval_logs/depth_4_branch_4"
DATASET_PATH="${PROJECT_ROOT}/datasets/DAPO-Math-17k.parquet"

# Metric type: pruning_improvement_topM_sum | parent_selection_improvement_topM_sum
METRIC_TYPE="parent_selection_improvement_topM_sum"

TOP_M=10

echo "Using metric: $METRIC_TYPE (Top-M: $TOP_M)"

# Set output path based on metric type
case $METRIC_TYPE in
  "pruning_improvement_topM_sum")
    OUTPUT_DATASET_PATH="../datasets/data_pruning.parquet"
    ;;
  "parent_selection_improvement_topM_sum")
    OUTPUT_DATASET_PATH="../datasets/data_fixing.parquet"
    ;;
  *)
    echo "Error: Unknown metric type: $METRIC_TYPE"
    exit 1
    ;;
esac

# Run analysis
python add_metrics_to_dataset.py \
    "$DATA_DIR" \
    --dataset-path "$DATASET_PATH" \
    --output-path "$OUTPUT_DATASET_PATH" \
    --metric "$METRIC_TYPE" \
    --topM "$TOP_M"

echo ""
echo "Completed! Dataset saved to: $OUTPUT_DATASET_PATH"
