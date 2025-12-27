#!/usr/bin/env bash
# Run Transformer ablation experiments (2x2 matrix: positional_encoding × norm_type)
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Transformer Ablation Experiments"
echo "Matrix: 2 (pos: abs/rel) × 2 (norm: LN/RMS) = 4 experiments"
echo "=================================================="
echo ""

# Configuration
GPUS=${GPUS:-2}  # Default to 2 GPUs for distributed training
PORT_BASE=29500

# Array of configurations
CONFIGS=(
    "experiments/configs/transformer_abs_ln.yaml"
    "experiments/configs/transformer_abs_rms.yaml"
    "experiments/configs/transformer_rel_ln.yaml"
    "experiments/configs/transformer_rel_rms.yaml"
)

NAMES=(
    "Absolute PE + LayerNorm (baseline)"
    "Absolute PE + RMSNorm"
    "Relative PE + LayerNorm"
    "Relative PE + RMSNorm"
)

# Run experiments sequentially to avoid resource contention
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    PORT=$((PORT_BASE + i))

    echo "=================================================="
    echo "Experiment $((i+1))/4: $NAME"
    echo "Config: $CONFIG"
    echo "=================================================="
    echo ""

    if [[ "$GPUS" -gt 1 ]]; then
        echo "Launching distributed training on $GPUS GPUs (port $PORT)"
        torchrun --nproc-per-node="$GPUS" --master_port="$PORT" \
            src/transformer/train.py --config "$CONFIG"
    else
        echo "Launching single-GPU training"
        python src/transformer/train.py --config "$CONFIG"
    fi

    echo ""
    echo "Experiment $((i+1))/4 completed: $NAME"
    echo ""
    sleep 5  # Brief pause between experiments
done

echo "=================================================="
echo "All 4 ablation experiments completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run analysis: python scripts/analyze_transformer_ablations.py"
echo "2. Check results in: experiments/results/"
echo ""
