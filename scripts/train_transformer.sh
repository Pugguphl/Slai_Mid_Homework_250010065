#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
CONFIG="${1:-$PROJECT_ROOT/experiments/configs/transformer_base.yaml}"
GPUS=${GPUS:-1}
PORT=${PORT:-29507}

cd "$PROJECT_ROOT"

if [[ "$GPUS" -gt 1 ]];
then
    echo "Launching distributed transformer training on $GPUS GPUs"
    torchrun --nproc-per-node="$GPUS" --master_port="$PORT" src/transformer/train.py --config "$CONFIG"
else
    echo "Launching single-GPU transformer training"
    python src/transformer/train.py --config "$CONFIG"
fi