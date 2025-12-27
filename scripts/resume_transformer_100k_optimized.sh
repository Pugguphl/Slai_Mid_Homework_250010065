#!/usr/bin/env bash
set -euo pipefail

# Resume training from checkpoint.
# Usage:
#   bash scripts/resume_transformer_100k_optimized.sh [GPUS]
#
# Arguments:
#   GPUS: Number of GPUs to use (default: 1)
#
# Example:
#   bash scripts/resume_transformer_100k_optimized.sh 1
#   bash scripts/resume_transformer_100k_optimized.sh 2

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

CONFIG="$PROJECT_ROOT/experiments/configs/transformer_100k_optimized.yaml"
CHECKPOINT="$PROJECT_ROOT/experiments/logs/transformer_100k_optimized_best.pt"
GPUS=${1:-1}
PORT=${PORT:-29509}

if [[ ! -f "$CONFIG" ]]; then
  echo "Missing config file: $CONFIG"
  exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing checkpoint file: $CHECKPOINT"
  exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p experiments/results

echo "=========================================="
echo "Resume Transformer Optimized Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPUS"
echo "=========================================="

if [[ "$GPUS" -gt 1 ]];
then
    echo "Launching distributed training on $GPUS GPUs..."
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node="$GPUS" --master_port="$PORT" src/transformer/train.py --config "$CONFIG" --resume "$CHECKPOINT"
else
    echo "Launching single-GPU training..."
    python src/transformer/train.py --config "$CONFIG" --resume "$CHECKPOINT"
fi

echo "=========================================="
echo "Training completed!"
echo "=========================================="
