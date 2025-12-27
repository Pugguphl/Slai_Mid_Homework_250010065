# Execution Roadmap

This roadmap tracks the nine required steps for the zh→en NMT midterm project. Each step lists concrete tasks, expected artifacts, and dependency notes to keep the project reproducible and auditable.

## Step Overview

| Step | Focus | Key Work Items | Primary Outputs | Dependencies | Status |
| --- | --- | --- | --- | --- | --- |
| 0 | Data schema + stats | Parse JSONL schema, length distributions, charset coverage, duplicate detection, logging | `experiments/results/data_profile.json`, histogram figures, schema summary note | Raw dataset availability | ✅ complete |
| 1 | Baseline preprocessing + RNN pass | Cleaning rules, tokenization pipeline, vocab export, baseline GRU/LSTM training run | `scripts/preprocess_data.py` revamp, vocab artifacts, baseline BLEU log | Step 0 stats, tokenizer choice | ✅ complete |
| 2 | Attention variants | Implement dot/mul/add attention, config switches, comparative experiments | RNN attention modules, config files, metrics table | Step 1 model | ✅ complete |
| 3 | Training strategy comparison | Teacher forcing schedule vs free running, logging TF rate, experiments | Training scripts with TF rate flag, result curves | Step 2 model | ✅ complete |
| 4 | Decoding strategy comparison | Greedy + beam search (variable beam), latency measurements | `scripts/compare_decoding.py`, `experiments/results/decoding_metrics.csv`, case studies JSON | Step 2 model | ✅ complete |
| 5 | Transformer baseline | Scratch encoder-decoder, training pipeline, baseline BLEU | `src/transformer/*` implementation refresh, configs, metrics CSV | Step 1 preprocessing | ✅ complete |
| 6 | Transformer ablations | Abs vs rel positional encoding, LayerNorm vs RMSNorm experiments | Config variants, metrics, discussion notes | Step 5 baseline | pending |
| 7 | Hyperparameter sensitivity | Batch size, lr, model scale sweeps, trend plots | Sweep configs, consolidated plots, analysis bullets | Step 5 baseline tooling | pending |
| 8 | T5 fine-tuning | Data conversion, HuggingFace training, comparison vs scratch Transformer | `src/models/t5_finetune.py`, configs, metrics | Step 1 preprocessing | pending |
| 9 | Unified analysis & reporting | Inference CLI, master results table, qualitative examples, report draft | `inference.py`, report figures, markdown tables | Steps 1-8 | pending |

## Step 4 decoding comparison snapshot

- Script: `scripts/compare_decoding.py` evaluates greedy and configurable beam sizes, computes corpus BLEU, latency, and throughput, and saves raw translations under `experiments/results/decoding_outputs/`.
- Metrics: `experiments/results/decoding_metrics.csv`

| Strategy | Beam | BLEU | Avg latency (s) | Throughput (sent/s) |
| --- | --- | --- | --- | --- |
| Greedy | – | 1.47 | 0.0013 | 743.7 |
| Beam-4 | 4 | 1.22 | 0.2560 | 3.91 |
| Beam-8 | 8 | 1.45 | 0.6278 | 1.59 |

- Case studies (10 examples) live in `experiments/results/decoding_examples.json` showing qualitative differences per strategy.

Observations: greedy decoding remains best for latency and even BLEU on this baseline RNN, while wider beams collapse due to low-quality model probabilities; future transformer work should revisit this comparison once model quality improves.

## Step 5 transformer baseline snapshot

- Implementation: `src/transformer/model.py` (configurable encoder-decoder with greedy/beam decoding), `src/transformer/train.py` (distributed training, Noam scheduler), `scripts/train_transformer.sh`, `scripts/infer_transformer.py`.
- Config: `experiments/configs/transformer_base.yaml` (SentencePiece vocab, batch 96×2 GPUs, 20 epochs, warmup 4k, smoothing 0.1).
- Training command:
	```bash
	torchrun --nproc-per-node=2 --master_port=29505 \
		src/transformer/train.py --config experiments/configs/transformer_base.yaml
	```
- Metrics (`experiments/results/transformer_metrics.csv`): best checkpoint at epoch 20 with BLEU 0.119 / valid loss 7.24 (checkpoint saved to `experiments/logs/best_transformer_model.pt`).

| Epoch | Train loss | Valid loss | BLEU | Notes |
| --- | --- | --- | --- | --- |
| 7 | 854.62 | 104.37 | 0.026 | Loss collapses rapidly after warmup; still low BLEU |
| 14 | 23.09 | 8.55 | 0.027 | Model begins to stabilize but remains underfit |
| **20** | **15.08** | **7.24** | **0.119** | Current best baseline checkpoint |

Observations: the scratch Transformer converges much faster in loss than BLEU, indicating exposure bias + limited training data (100k) and the lack of advanced regularization; Step 6 ablations (positional encoding, normalization) plus scheduled evaluation on larger corpora should target BLEU improvements before moving to hyper-parameter sweeps.

## Immediate Next Actions

1. Design Step 6 experiments: relative vs absolute positional encodings and LayerNorm vs RMSNorm toggles using the new `TransformerConfig` hooks; document expected config changes.
2. Prepare automation for Step 7 sweeps (batch size, d_model, learning rate) leveraging the refactored training script + configs.
3. Capture qualitative outputs for the Transformer baseline (for Step 9) and schedule a rerun on the test split once decoding comparisons are updated.
