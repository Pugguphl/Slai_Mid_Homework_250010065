# Step 4 – Decoding Strategy Comparison

## Experiment setup
- **Model:** `experiments/logs/best_rnn_model.pt` (teacher-forcing decay baseline).
- **Dataset:** `data/processed/valid.jsonl` (492 sentence pairs, cleaned & tokenized with shared SentencePiece vocab).
- **Script:** `scripts/compare_decoding.py` (see `--help` for options). Default greedy batch size is 32; beam search is evaluated sequentially because the current implementation only supports batch size 1.
- **Metrics:** SacreBLEU v2.4.3, latency measured as pure decoding time (CUDA synchronized) per sentence, throughput = sentences / decoding time.

Command that produced the final results:
```
python scripts/compare_decoding.py \
  --checkpoint experiments/logs/best_rnn_model.pt \
  --dataset data/processed/valid.jsonl \
  --beam_sizes 4 8 \
  --num_examples 10
```

## Aggregate metrics (validation set)
| Strategy | Beam | BLEU | Avg latency (s) | Throughput (sent/s) |
| --- | --- | --- | --- | --- |
| Greedy | – | **1.47** | **0.0013** | **743.7** |
| Beam-4 | 4 | 1.22 | 0.2560 | 3.91 |
| Beam-8 | 8 | 1.45 | 0.6278 | 1.59 |

Notes:
- Greedy decoding is both faster and slightly better in BLEU for this RNN baseline, hinting that the model’s probability landscape is too flat/noisy for wider beams.
- Beam-8 recovers some BLEU vs beam-4 but remains below greedy despite >400× more latency, so beam expansion is currently unjustified.

## Qualitative findings
- A sample of 10 sentences with side-by-side outputs lives in `experiments/results/decoding_examples.json`.
- Beams tend to repeat spurious phrases ("the united states" loops) because the decoder lacks strong coverage/penalty heuristics; greedy avoids some loops simply by terminating earlier.
- Errors are dominated by content hallucination rather than fluency alone, underscoring the need for stronger models (Step 5) and possibly length/coverage penalties if we revisit beam search later.

## Artifacts
- Metrics CSV: `experiments/results/decoding_metrics.csv`
- Raw translations: `experiments/results/decoding_outputs/*.txt`
- Case studies: `experiments/results/decoding_examples.json`

## Next steps
1. Move on to Step 5 by modernizing the Transformer implementation and rerunning the decoding comparison once a stronger model is available.
2. Explore lightweight heuristics (length norm, coverage penalty) only after a better model exists to avoid masking modeling deficiencies with decoding tricks.
