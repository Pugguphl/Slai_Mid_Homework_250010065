#!/usr/bin/env python3
"""Compare decoding strategies (greedy vs. beam) for the RNN translator."""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.dataset import TranslationDataset, collate_translation_batch  # noqa: E402
from src.rnn.infer import build_model_from_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare greedy vs beam decoding for RNN NMT")
    parser.add_argument("--checkpoint", default="experiments/logs/best_rnn_model.pt", help="Path to trained checkpoint")
    parser.add_argument("--dataset", help="Path to JSONL dataset (default: checkpoint valid file)")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of samples to evaluate")
    parser.add_argument("--greedy_batch_size", type=int, default=32, help="Batch size for greedy decoding")
    parser.add_argument("--beam_sizes", type=int, nargs="*", default=[4, 8], help="Beam sizes to evaluate")
    parser.add_argument("--output_metrics", default="experiments/results/decoding_metrics.csv", help="CSV file for aggregate metrics")
    parser.add_argument("--case_studies", default="experiments/results/decoding_examples.json", help="JSON file for qualitative examples")
    parser.add_argument("--pred_dir", default="experiments/results/decoding_outputs", help="Directory to store raw translations")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of qualitative examples to store")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Force device selection")
    parser.add_argument("--disable_greedy", action="store_true", help="Skip greedy decoding evaluation")
    return parser.parse_args()


def collate_with_text(batch, tokenizer):
    data = collate_translation_batch(batch, tokenizer)
    data["src_text"] = [item["src_text"] for item in batch]
    data["tgt_text"] = [item["tgt_text"] for item in batch]
    return data


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_greedy(model, tokenizer, dataset, batch_size: int, device: torch.device) -> Tuple[Dict, List[str], List[str]]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_with_text, tokenizer=tokenizer),
    )
    predictions: List[str] = []
    references: List[str] = []
    decode_time = 0.0

    model.eval()
    with torch.no_grad():
        for batch in loader:
            src = batch["src"].to(device)
            lengths = batch["src_lengths"].to(device)
            synchronize(device)
            start = time.time()
            preds = model.greedy_decode(src, lengths)
            synchronize(device)
            decode_time += time.time() - start
            texts = [tokenizer.decode_tgt(seq.tolist()) for seq in preds]
            predictions.extend(texts)
            references.extend(batch["tgt_text"])

    bleu = corpus_bleu(predictions, [references]).score if predictions else 0.0
    latency = decode_time / len(predictions) if predictions else 0.0
    throughput = len(predictions) / decode_time if decode_time > 0 else 0.0
    return {
        "strategy": "greedy",
        "beam_size": 0,
        "bleu": bleu,
        "latency_sec": latency,
        "throughput_sps": throughput,
        "num_sentences": len(predictions),
    }, predictions, references


def evaluate_beam(model, tokenizer, dataset, beam_size: int, device: torch.device) -> Tuple[Dict, List[str], List[str]]:
    predictions: List[str] = []
    references: List[str] = []
    decode_time = 0.0

    model.eval()
    with torch.no_grad():
        for sample in dataset:
            src_tensor = torch.tensor(sample["src_ids"], dtype=torch.long, device=device).unsqueeze(0)
            lengths = torch.tensor([len(sample["src_ids"])], dtype=torch.long, device=device)
            synchronize(device)
            start = time.time()
            pred = model.beam_search(src_tensor, lengths, beam_size=beam_size)
            synchronize(device)
            decode_time += time.time() - start
            predictions.append(tokenizer.decode_tgt(pred.squeeze(0).tolist()))
            references.append(sample["tgt_text"])

    bleu = corpus_bleu(predictions, [references]).score if predictions else 0.0
    latency = decode_time / len(predictions) if predictions else 0.0
    throughput = len(predictions) / decode_time if decode_time > 0 else 0.0
    return {
        "strategy": f"beam_{beam_size}",
        "beam_size": beam_size,
        "bleu": bleu,
        "latency_sec": latency,
        "throughput_sps": throughput,
        "num_sentences": len(predictions),
    }, predictions, references


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, ckpt, tokenizer = build_model_from_checkpoint(args.checkpoint, device)
    data_cfg = ckpt.get("config", {}).get("data", {})
    dataset_path = args.dataset or data_cfg.get("valid_file")
    if not dataset_path:
        raise ValueError("Dataset path must be provided via --dataset when checkpoint config lacks data.valid_file")

    dataset = TranslationDataset(
        file_path=dataset_path,
        tokenizer=tokenizer,
        src_key=data_cfg.get("src_key", "zh"),
        tgt_key=data_cfg.get("tgt_key", "en"),
        add_bos_eos=True,
        max_samples=args.max_samples,
    )

    results = []
    predictions_by_strategy: Dict[str, List[str]] = {}
    references_cache: List[str] | None = None

    if not args.disable_greedy:
        greedy_metrics, greedy_preds, greedy_refs = evaluate_greedy(model, tokenizer, dataset, args.greedy_batch_size, device)
        results.append(greedy_metrics)
        predictions_by_strategy[greedy_metrics["strategy"]] = greedy_preds
        references_cache = greedy_refs
        print(json.dumps(greedy_metrics, ensure_ascii=False))

    for beam_size in args.beam_sizes:
        metrics, preds, refs = evaluate_beam(model, tokenizer, dataset, beam_size, device)
        if references_cache is None:
            references_cache = refs
        results.append(metrics)
        predictions_by_strategy[metrics["strategy"]] = preds
        print(json.dumps(metrics, ensure_ascii=False))

    ensure_dir(Path(args.output_metrics).parent)
    pd.DataFrame(results).to_csv(args.output_metrics, index=False)

    ensure_dir(Path(args.pred_dir))
    for strategy, preds in predictions_by_strategy.items():
        with open(Path(args.pred_dir) / f"{strategy}.txt", "w", encoding="utf-8") as f:
            for line in preds:
                f.write(line + "\n")

    if references_cache is None:
        return

    case_examples = []
    total_examples = min(args.num_examples, len(references_cache))
    for idx in range(total_examples):
        sample = dataset.samples[idx]
        example = {
            "index": idx,
            "source": sample.get(data_cfg.get("src_key", "zh"), sample.get("zh")),
            "reference": references_cache[idx],
            "predictions": {
                strategy: predictions_by_strategy[strategy][idx]
                for strategy in predictions_by_strategy
            },
        }
        case_examples.append(example)

    ensure_dir(Path(args.case_studies).parent)
    with open(args.case_studies, "w", encoding="utf-8") as f:
        json.dump(case_examples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
