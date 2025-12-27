#!/usr/bin/env bash
# Run RNN experiment matrix (3 attention types × 2 RNN types = 6 experiments)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

echo "=================================================="
echo "RNN Experiment Matrix"
echo "Matrix: 3 (attention) × 2 (RNN type) = 6 experiments"
echo "=================================================="
echo ""

# Configuration
GPUS=${GPUS:-1}  # Default to 1 GPU for distributed training
PORT_BASE=29400

# Array of configurations
CONFIGS=(
    "experiments/configs/rnn_multiplicative.yaml"
    "experiments/configs/rnn_additive.yaml"
    "experiments/configs/rnn_base.yaml"
    "experiments/configs/rnn_multiplicative_no_teacher.yaml"
    "experiments/configs/rnn_additive_no_teacher.yaml"
    "experiments/configs/rnn_no_teacher.yaml"
)

NAMES=(
    "GRU + Multiplicative Attention (with teacher forcing)"
    "GRU + Additive Attention (with teacher forcing)"
    "GRU + Dot Attention (with teacher forcing)"
    "GRU + Multiplicative Attention (no teacher forcing)"
    "GRU + Additive Attention (no teacher forcing)"
    "GRU + Dot Attention (no teacher forcing)"
)

# Run experiments sequentially to avoid resource contention
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    PORT=$((PORT_BASE + i))

    echo "=================================================="
    echo "Experiment $((i+1))/6: $NAME"
    echo "Config: $CONFIG"
    echo "=================================================="
    echo ""

    if [[ "$GPUS" -gt 1 ]]; then
        echo "Launching distributed training on $GPUS GPUs (port $PORT)"
        torchrun --nproc-per-node="$GPUS" --master_port="$PORT" \
            src/rnn/train.py --config "$CONFIG"
    else
        echo "Launching single-GPU training"
        python src/rnn/train.py --config "$CONFIG"
    fi

    echo ""
    echo "Experiment $((i+1))/6 completed: $NAME"
    echo ""
    sleep 5  # Brief pause between experiments
done

echo "=================================================="
echo "All 6 RNN experiments completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run analysis: python scripts/analyze_rnn_results.py"
echo "2. Check results in: experiments/results/"
echo ""
