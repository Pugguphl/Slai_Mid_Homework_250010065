#!/usr/bin/env bash
# Run hyperparameter sensitivity sweep experiments
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

cd "$PROJECT_ROOT"

echo "=================================================="
echo "Hyperparameter Sensitivity Sweep Experiments"
echo "3 sweep experiments + 1 baseline = 4 total runs"
echo "=================================================="
echo ""

# Configuration
GPUS=${GPUS:-1}  # Default to 1 GPU
PORT_BASE=29600

# Array of configurations
CONFIGS=(
    "experiments/configs/transformer_abs_ln.yaml"  # Baseline (batch=128, lr=5e-4, d=512)
    "experiments/configs/sweep_batch_256.yaml"      # Sweep 1: batch_size=256
    "experiments/configs/sweep_lr_1e3.yaml"         # Sweep 2: learning_rate=1e-3
    "experiments/configs/sweep_d768.yaml"           # Sweep 3: d_model=768
)

NAMES=(
    "Baseline (batch=128, lr=5e-4, d=512)"
    "Sweep: Batch Size = 256"
    "Sweep: Learning Rate = 1e-3"
    "Sweep: Model Dimension = 768"
)

# Run experiments sequentially
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    NAME="${NAMES[$i]}"
    PORT=$((PORT_BASE + i))

    echo "=================================================="
    echo "Experiment $((i+1))/4: $NAME"
    echo "Config: $CONFIG"
    echo "=================================================="
    echo ""

    # Skip baseline if already exists
    if [[ $i -eq 0 ]] && [[ -f "experiments/results/transformer_abs_ln_metrics.csv" ]]; then
        echo "Baseline results already exist, skipping..."
        echo ""
        continue
    fi

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
echo "All hyperparameter sweep experiments completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run analysis: python scripts/analyze_hyperparam_sweeps.py"
echo "2. Check results in: experiments/results/"
echo ""
