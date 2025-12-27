#!/usr/bin/env python
"""Dataset profiling utility for the zh→en NMT midterm project.

This script inspects every JSONL file in the specified directory, verifies the
schema, and computes descriptive statistics required for Step 0:
- length distributions (source characters + target tokens)
- character set coverage
- duplicate sentence rates
- schema validation and summary logging

Outputs are written to `experiments/results/data_profile.json` (machine
readable) plus optional CSV/plot artifacts so downstream steps can reference the
same profiling results.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless environments
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

DEFAULT_RAW_DIR = "data/raw/AP0004_Midterm&Final_translation_dataset_zh_en"
DEFAULT_RESULTS_DIR = "experiments/results"
DEFAULT_FIG_DIR = "reports/figures/data_profile"


def describe_lengths(lengths: List[int]) -> Dict[str, float]:
    if not lengths:
        return {key: 0.0 for key in ["min", "max", "mean", "median", "p90", "p95", "p99"]}
    arr = np.array(lengths)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def save_histogram(lengths: List[int], title: str, xlabel: str, output_path: Path, bins: int = 80) -> None:
    if not lengths:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(lengths, bins=min(bins, max(10, int(math.sqrt(len(lengths))))), color="#4C72B0", alpha=0.85)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def detect_schema(keys_counter: Counter) -> Dict[str, List[str]]:
    """Return sorted schema keys and their frequencies."""
    total = sum(keys_counter.values())
    distribution = {
        key: {
            "count": count,
            "ratio": round(count / total, 4) if total else 0.0,
        }
        for key, count in keys_counter.most_common()
    }
    return distribution


def analyze_file(path: Path, src_key: str, tgt_key: str) -> Tuple[Dict, List[int], List[int]]:
    schema_counter = Counter()
    src_char_counter = Counter()
    tgt_token_counter = Counter()
    zh_lengths = []
    en_token_lengths = []
    en_char_lengths = []
    pair_counter = Counter()
    src_counter = Counter()
    tgt_counter = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            schema_counter.update(record.keys())
            if src_key not in record or tgt_key not in record:
                raise KeyError(f"Missing `{src_key}` or `{tgt_key}` in file={path} line={line[:80]}")
            src_text = record[src_key]
            tgt_text = record[tgt_key]

            zh_lengths.append(len(src_text))
            en_token_lengths.append(len(tgt_text.split()))
            en_char_lengths.append(len(tgt_text))

            src_char_counter.update(src_text)
            tgt_token_counter.update(tgt_text.split())

            pair_counter[(src_text, tgt_text)] += 1
            src_counter[src_text] += 1
            tgt_counter[tgt_text] += 1

    total_samples = sum(pair_counter.values())
    duplicate_pairs = sum(count - 1 for count in pair_counter.values() if count > 1)
    duplicate_src = sum(count - 1 for count in src_counter.values() if count > 1)
    duplicate_tgt = sum(count - 1 for count in tgt_counter.values() if count > 1)

    stats = {
        "file": path.name,
        "num_samples": total_samples,
        "schema": detect_schema(schema_counter),
        "language_direction": f"{src_key}->{tgt_key}",
        "source_char_vocab_size": len(src_char_counter),
        "target_token_vocab_size": len(tgt_token_counter),
        "source_top_chars": src_char_counter.most_common(20),
        "target_top_tokens": tgt_token_counter.most_common(20),
        "source_char_lengths": describe_lengths(zh_lengths),
        "target_token_lengths": describe_lengths(en_token_lengths),
        "target_char_lengths": describe_lengths(en_char_lengths),
        "duplicate_pair_rate": round(duplicate_pairs / total_samples, 6) if total_samples else 0.0,
        "duplicate_source_rate": round(duplicate_src / total_samples, 6) if total_samples else 0.0,
        "duplicate_target_rate": round(duplicate_tgt / total_samples, 6) if total_samples else 0.0,
    }
    return stats, zh_lengths, en_token_lengths


def run_profile(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    output_dir = Path(args.output_dir)
    fig_dir = Path(args.figure_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_stats: List[Dict] = []
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No JSONL files matched glob `{args.pattern}` under {input_dir}")

    for path in tqdm(files, desc="Profiling JSONL files"):
        stats, zh_lengths, en_lengths = analyze_file(path, args.src_key, args.tgt_key)
        all_stats.append(stats)
        save_histogram(zh_lengths, f"{path.name}: Source length", "Number of characters", fig_dir / f"{path.stem}_zh_char_hist.png")
        save_histogram(en_lengths, f"{path.name}: Target length", "Number of tokens", fig_dir / f"{path.stem}_en_token_hist.png")

    json_output = output_dir / "data_profile.json"
    with json_output.open("w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    table_records = []
    for item in all_stats:
        row = {
            "file": item["file"],
            "samples": item["num_samples"],
            "src_chars_mean": round(item["source_char_lengths"]["mean"], 2),
            "src_chars_p95": round(item["source_char_lengths"]["p95"], 2),
            "tgt_tokens_mean": round(item["target_token_lengths"]["mean"], 2),
            "tgt_tokens_p95": round(item["target_token_lengths"]["p95"], 2),
            "dup_pairs%": round(item["duplicate_pair_rate"] * 100, 4),
        }
        table_records.append(row)
    pd.DataFrame(table_records).to_csv(output_dir / "data_profile_summary.csv", index=False)

    print(f"Saved detailed profile to {json_output}")
    print(f"Saved summary CSV to {output_dir / 'data_profile_summary.csv'}")
    print(f"Histograms available under {fig_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile zh→en JSONL datasets for the NMT project")
    parser.add_argument("--input_dir", default=DEFAULT_RAW_DIR, help="Directory containing JSONL files")
    parser.add_argument("--pattern", default="*.jsonl", help="Glob pattern to match JSONL files")
    parser.add_argument("--src_key", default="zh", help="Key name for source (Chinese) text")
    parser.add_argument("--tgt_key", default="en", help="Key name for target (English) text")
    parser.add_argument("--output_dir", default=DEFAULT_RESULTS_DIR, help="Directory to store JSON/CSV outputs")
    parser.add_argument("--figure_dir", default=DEFAULT_FIG_DIR, help="Directory to store histogram figures")
    return parser.parse_args()


if __name__ == "__main__":
    run_profile(parse_args())
