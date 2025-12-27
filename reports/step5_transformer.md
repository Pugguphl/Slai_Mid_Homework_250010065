# Step 5 – Transformer Baseline

## Implementation highlights
- **Model:** `src/transformer/model.py` implements a configurable encoder-decoder wrapper around PyTorch's Transformer stack with tied embeddings, sinusoidal or learned positional encodings, greedy + beam decoding, and exportable `TransformerConfig` for future ablations.
- **Training:** `src/transformer/train.py` mirrors the RNN pipeline—supports distributed execution, SentencePiece tokenization, Noam warmup scheduler, gradient accumulation, label smoothing, automatic checkpointing, and CSV logging.
- **Inference:** `src/transformer/infer.py` + `scripts/infer_transformer.py` provide batch translation with greedy/beam strategies compatible with saved checkpoints.

## Experiment setup
- **Config:** `experiments/configs/transformer_base.yaml`
  - d_model=512, nhead=8, ff=2048, 6×6 layers, dropout=0.1, max position 256, label smoothing 0.1.
  - Batch size 96, gradient accumulation 2 (effective 192), warmup 4k, AdamW β=(0.9,0.98), weight decay 0.01.
  - Data: `data/processed/{train,valid}.jsonl`, tokenizer `data/vocab/tokenizer_config.json`.
- **Command:**
  ```bash
  torchrun --nproc-per-node=2 --master_port=29505 \
    src/transformer/train.py --config experiments/configs/transformer_base.yaml
  ```
- **Artifacts:** best weights `experiments/logs/best_transformer_model.pt`, metrics `experiments/results/transformer_metrics.csv`.

## Results (validation set)
| Epoch | Train loss | Valid loss | BLEU |
| --- | --- | --- | --- |
| 1 | 18138.62 | 6331.27 | 0.015 |
| 7 | 854.62 | 104.37 | 0.026 |
| 14 | 23.09 | 8.55 | 0.027 |
| **20** | **15.08** | **7.24** | **0.119** |

- Loss plummets quickly thanks to the Noam schedule, but BLEU lags, indicating the model still produces low-quality sequences despite lower perplexity.
- Evaluation currently uses greedy decoding; future Step 4-style comparisons should be rerun once BLEU improves, as wider beams may help when probabilities are more calibrated.

## Observations & follow-ups
1. **Normalization / positional variants (Step 6):** the refactor already exposes `norm_first` and positional encoding hooks; next, add RMSNorm + relative/rotary encodings to test stability and BLEU impact.
2. **Regularization tweaks:** consider dropout scheduling, label smoothing sweeps, and minimum learning-rate clamps to avoid the late-epoch BLEU oscillations seen in epochs 14–19.
3. **Data scale:** the baseline only sees the cleaned 100k split; adding curriculum (train_10k warm start plus full train) or on-the-fly augmentation might unlock higher BLEU before moving on to hyper-parameter sweeps.
4. **Diagnostics:** capture qualitative translations and BLEU on the test set once decoding comparisons are updated, feeding Step 9's report tables.
