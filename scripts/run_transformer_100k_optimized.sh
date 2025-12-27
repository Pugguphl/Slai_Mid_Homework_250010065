#!/usr/bin/env bash
set -euo pipefail

# Run Transformer Optimized configuration for 100k dataset.
# This configuration is designed to prevent overfitting on 100k samples.
# Usage:
#   bash scripts/run_transformer_100k_optimized.sh [GPUS]
#
# Arguments:
#   GPUS: Number of GPUs to use (default: 1)
#
# Example:
#   bash scripts/run_transformer_100k_optimized.sh 1
#   bash scripts/run_transformer_100k_optimized.sh 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

CONFIG="$PROJECT_ROOT/experiments/configs/transformer_100k_optimized.yaml"
GPUS=${1:-1}
PORT=${PORT:-29508}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config file: $CONFIG"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p experiments/results

echo "=========================================="
echo "Transformer Optimized Training (100k)"
echo "=========================================="
echo "Config: $CONFIG"
echo "GPUs: $GPUS"
echo "Model: d_model=256, layers=4+4, heads=4"
echo "Parameters: ~15M"
echo "Batch size: 256"
echo "Learning rate: 0.001"
echo "Epochs: 100"
echo "=========================================="

if [[ "$GPUS" -gt 1 ]];
then
    echo "Launching distributed training on $GPUS GPUs..."
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node="$GPUS" --master_port="$PORT" src/transformer/train.py --config "$CONFIG"
else
    echo "Launching single-GPU training..."
    python src/transformer/train.py --config "$CONFIG"
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
