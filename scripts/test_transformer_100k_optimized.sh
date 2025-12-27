#!/usr/bin/env bash
set -euo pipefail

# Test Transformer Optimized model on test set.
# Evaluates BLEU scores with greedy and beam search decoding.
# Usage:
#   bash scripts/test_transformer_100k_optimized.sh [CHECKPOINT_PATH]
#
# Arguments:
#   CHECKPOINT_PATH: Path to model checkpoint (default: experiments/logs/transformer_100k_optimized_best.pt)
#
# Example:
#   bash scripts/test_transformer_100k_optimized.sh
#   bash scripts/test_transformer_100k_optimized.sh experiments/logs/transformer_100k_optimized_best.pt

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

CHECKPOINT="${1:-$PROJECT_ROOT/experiments/logs/transformer_100k_optimized_best.pt}"
TEST_FILE="$PROJECT_ROOT/data/processed/test.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/experiments/results"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Error: Checkpoint not found: $CHECKPOINT"
  echo "Please train the model first using:"
  echo "  bash scripts/run_transformer_100k_optimized.sh"
  exit 1
fi

if [[ ! -f "$TEST_FILE" ]]; then
  echo "Error: Test file not found: $TEST_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Testing Transformer Optimized Model"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "=========================================="

cd "$PROJECT_ROOT"

# Extract Chinese sentences from test file
echo "Extracting test sentences..."
python -c "
import json
import sys
with open('$TEST_FILE', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        print(data['zh'])
" > /tmp/test_src.txt

# Extract English references
echo "Extracting reference translations..."
python -c "
import json
import sys
with open('$TEST_FILE', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        print(data['en'])
" > /tmp/test_ref.txt

# Test with greedy decoding
echo ""
echo "Testing with greedy decoding..."
python src/transformer/infer.py \
  --checkpoint "$CHECKPOINT" \
  --input_file /tmp/test_src.txt \
  --output_file /tmp/test_greedy_pred.txt \
  --strategy greedy

# Calculate BLEU for greedy
echo "Calculating BLEU (greedy)..."
python -c "
import sys
from sacrebleu import corpus_bleu

with open('/tmp/test_greedy_pred.txt', 'r', encoding='utf-8') as f:
    preds = [line.strip() for line in f]
with open('/tmp/test_ref.txt', 'r', encoding='utf-8') as f:
    refs = [line.strip() for line in f]

bleu = corpus_bleu(preds, [refs])
print(f'Greedy BLEU: {bleu.score:.4f}')
print(f'Greedy BLEU (details): {bleu.format()}')
"

# Test with beam search (beam_size=4)
echo ""
echo "Testing with beam search (beam_size=4)..."
python src/transformer/infer.py \
  --checkpoint "$CHECKPOINT" \
  --input_file /tmp/test_src.txt \
  --output_file /tmp/test_beam4_pred.txt \
  --strategy beam \
  --beam_size 4

# Calculate BLEU for beam4
echo "Calculating BLEU (beam=4)..."
python -c "
import sys
from sacrebleu import corpus_bleu

with open('/tmp/test_beam4_pred.txt', 'r', encoding='utf-8') as f:
    preds = [line.strip() for line in f]
with open('/tmp/test_ref.txt', 'r', encoding='utf-8') as f:
    refs = [line.strip() for line in f]

bleu = corpus_bleu(preds, [refs])
print(f'Beam (4) BLEU: {bleu.score:.4f}')
print(f'Beam (4) BLEU (details): {bleu.format()}')
"

# Test with beam search (beam_size=8)
echo ""
echo "Testing with beam search (beam_size=8)..."
python src/transformer/infer.py \
  --checkpoint "$CHECKPOINT" \
  --input_file /tmp/test_src.txt \
  --output_file /tmp/test_beam8_pred.txt \
  --strategy beam \
  --beam_size 8

# Calculate BLEU for beam8
echo "Calculating BLEU (beam=8)..."
python -c "
import sys
from sacrebleu import corpus_bleu

with open('/tmp/test_beam8_pred.txt', 'r', encoding='utf-8') as f:
    preds = [line.strip() for line in f]
with open('/tmp/test_ref.txt', 'r', encoding='utf-8') as f:
    refs = [line.strip() for line in f]

bleu = corpus_bleu(preds, [refs])
print(f'Beam (8) BLEU: {bleu.score:.4f}')
print(f'Beam (8) BLEU (details): {bleu.format()}')
"

# Save results to JSON
echo ""
echo "Saving results..."
python -c "
import json
from sacrebleu import corpus_bleu

# Greedy
with open('/tmp/test_greedy_pred.txt', 'r', encoding='utf-8') as f:
    greedy_preds = [line.strip() for line in f]
with open('/tmp/test_ref.txt', 'r', encoding='utf-8') as f:
    refs = [line.strip() for line in f]
greedy_bleu = corpus_bleu(greedy_preds, [refs])

# Beam4
with open('/tmp/test_beam4_pred.txt', 'r', encoding='utf-8') as f:
    beam4_preds = [line.strip() for line in f]
beam4_bleu = corpus_bleu(beam4_preds, [refs])

# Beam8
with open('/tmp/test_beam8_pred.txt', 'r', encoding='utf-8') as f:
    beam8_preds = [line.strip() for line in f]
beam8_bleu = corpus_bleu(beam8_preds, [refs])

results = {
    'checkpoint': '$CHECKPOINT',
    'test_file': '$TEST_FILE',
    'num_examples': len(refs),
    'greedy_bleu': greedy_bleu.score,
    'beam4_bleu': beam4_bleu.score,
    'beam8_bleu': beam8_bleu.score,
    'best_strategy': 'beam4' if beam4_bleu.score >= greedy_bleu.score and beam4_bleu.score >= beam8_bleu.score else 'beam8' if beam8_bleu.score >= greedy_bleu.score else 'greedy',
    'best_bleu': max(greedy_bleu.score, beam4_bleu.score, beam8_bleu.score)
}

output_file = '$OUTPUT_DIR/transformer_100k_optimized_test_results.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to: {output_file}')
print(f'Best BLEU: {results[\"best_bleu\"]:.4f} ({results[\"best_strategy\"]})')
"

# Clean up
rm -f /tmp/test_src.txt /tmp/test_ref.txt /tmp/test_greedy_pred.txt /tmp/test_beam4_pred.txt /tmp/test_beam8_pred.txt

echo ""
echo "=========================================="
echo "Testing completed!"
echo "=========================================="
