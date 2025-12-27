#!/usr/bin/env bash
# Train T5-small for Chinese-English translation
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HOME="/mnt/afs/share/phl"
export HF_HUB_CACHE="/mnt/afs/share/phl/hub"
export TRANSFORMERS_CACHE="/mnt/afs/share/phl/transformers"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "T5-small Fine-tuning for Chinese-English NMT"
echo "=================================================="
echo ""

CONFIG="experiments/configs/t5_small.yaml"

echo "Config: $CONFIG"
echo ""

# Run training
python src/t5/finetune.py --config "$CONFIG"

echo ""
echo "=================================================="
echo "T5 training completed!"
echo "=================================================="
echo ""
echo "Next steps:"
echo "1. Run inference: python src/t5/infer.py --model_path experiments/logs/t5_small_best \\"
echo "                     --input data/processed/test.jsonl \\"
echo "                     --output experiments/results/t5_predictions.jsonl \\"
echo "                     --beam_size 4"
echo ""
