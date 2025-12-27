#!/usr/bin/env bash
set -euo pipefail

# Run Transformer ablation experiments (sinusoidal PE vs learned PE) in parallel.
# Usage:
#   bash scripts/run_transformer_ablations.sh
#
# Notes:
# - Assumes your training launcher is scripts/train_transformer.sh
# - If you have fewer GPUs, edit CUDA_VISIBLE_DEVICES mapping or run serially.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CFG_BASE="experiments/configs/transformer_base.yaml"
CFG_LEARNED="experiments/configs/transformer_learned_pe.yaml"

if [[ ! -f "$CFG_BASE" ]]; then
  echo "Missing $CFG_BASE"
  exit 1
fi
if [[ ! -f "$CFG_LEARNED" ]]; then
  echo "Missing $CFG_LEARNED"
  exit 1
fi

mkdir -p experiments/results

echo "[run_transformer_ablations] Launching baseline (sinusoidal) + learned PE in parallel..."

CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_transformer.sh "$CFG_BASE" &
PID1=$!
CUDA_VISIBLE_DEVICES=2,3 bash scripts/train_transformer.sh "$CFG_LEARNED" &
PID2=$!

echo "[run_transformer_ablations] PIDs: baseline=$PID1 learned=$PID2"
wait $PID1 $PID2
echo "[run_transformer_ablations] Done."
