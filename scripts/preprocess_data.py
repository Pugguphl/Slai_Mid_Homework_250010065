#!/usr/bin/env python
"""Full preprocessing pipeline: cleaning, tokenization, reproducible splits."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.preprocess import CleaningConfig, clean_parallel_record
from src.common.tokenizer import (
    SentencePieceTokenizer,
    train_char_word_tokenizer,
    train_sentencepiece_tokenizer,
)
from src.common.utils import create_directory, set_seed


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def deduplicate(records: List[Dict], src_key: str, tgt_key: str) -> Tuple[List[Dict], int]:
    seen = set()
    unique_records = []
    removed = 0
    for record in records:
        key = (record[src_key], record[tgt_key])
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        unique_records.append(record)
    return unique_records, removed


def process_split(
    files: List[Path],
    src_key: str,
    tgt_key: str,
    cfg: CleaningConfig,
    dedup: bool,
) -> Tuple[List[Dict], Dict[str, int]]:
    stats = Counter()
    cleaned: List[Dict] = []
    for path in files:
        for record in read_jsonl(path):
            stats["raw"] += 1
            cleaned_record, reason = clean_parallel_record(record, src_key, tgt_key, cfg)
            if reason:
                stats[f"filtered_{reason}"] += 1
                continue
            cleaned.append({
                "index": record.get("index"),
                src_key: cleaned_record[src_key],
                tgt_key: cleaned_record[tgt_key],
            })
    if dedup:
        cleaned, removed = deduplicate(cleaned, src_key, tgt_key)
        stats["filtered_duplicates"] += removed
    stats["kept"] = len(cleaned)
    return cleaned, stats


def train_tokenizer(train_records: List[Dict], args) -> Dict:
    create_directory(args.tokenizer_dir)
    if args.tokenizer_type == "sentencepiece":
        corpus = []
        for rec in train_records:
            corpus.append(rec[args.src_key])
            corpus.append(rec[args.tgt_key])
        files = train_sentencepiece_tokenizer(
            corpus=corpus,
            output_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
            model_type=args.spm_model_type,
            character_coverage=args.character_coverage,
            model_prefix=args.spm_model_prefix,
            user_defined_symbols=args.extra_symbols,
        )
        tokenizer = SentencePieceTokenizer(files["model_file"])
        config_path = Path(args.tokenizer_dir) / "tokenizer_config.json"
        tokenizer.save_config(
            output_path=config_path,
            vocab_size=tokenizer.vocab_size,
            extra={"model_type": args.spm_model_type, "character_coverage": args.character_coverage},
        )
        return {
            "config_path": str(config_path),
            "files": files,
            "type": "sentencepiece",
        }

    zh_texts = [rec[args.src_key] for rec in train_records]
    en_texts = [rec[args.tgt_key] for rec in train_records]
    files = train_char_word_tokenizer(
        zh_texts=zh_texts,
        en_texts=en_texts,
        output_dir=args.tokenizer_dir,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
    )
    config = {
        "type": "char_word",
        **files,
        "extra": {
            "src_vocab_size": args.src_vocab_size,
            "tgt_vocab_size": args.tgt_vocab_size,
        },
    }
    config_path = Path(args.tokenizer_dir) / "tokenizer_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return {
        "config_path": str(config_path),
        "files": files,
        "type": "char_word",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess zh→en data")
    parser.add_argument("--raw_dir", default="data/raw/AP0004_Midterm&Final_translation_dataset_zh_en")
    parser.add_argument("--train_files", nargs="+", default=["train_100k.jsonl"])
    parser.add_argument("--valid_file", default="valid.jsonl")
    parser.add_argument("--test_file", default="test.jsonl")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--tokenizer_dir", default="data/vocab")
    parser.add_argument("--tokenizer_type", choices=["sentencepiece", "char_word"], default="sentencepiece")
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--src_vocab_size", type=int, default=8000)
    parser.add_argument("--tgt_vocab_size", type=int, default=16000)
    parser.add_argument("--spm_model_type", choices=["unigram", "bpe", "char", "word"], default="unigram")
    parser.add_argument("--spm_model_prefix", default="spm_zh_en")
    parser.add_argument("--character_coverage", type=float, default=0.9995)
    parser.add_argument("--extra_symbols", nargs="*", default=[])
    parser.add_argument("--src_key", default="zh")
    parser.add_argument("--tgt_key", default="en")
    parser.add_argument("--min_len", type=int, default=2)
    parser.add_argument("--max_src_len", type=int, default=120)
    parser.add_argument("--max_tgt_len", type=int, default=80)
    parser.add_argument("--max_length_ratio", type=float, default=4.0)
    parser.add_argument("--truncate_long", action="store_true")
    parser.add_argument("--lowercase_en", dest="lowercase_en", action="store_true", default=True)
    parser.add_argument("--no-lowercase_en", dest="lowercase_en", action="store_false")
    parser.add_argument("--deduplicate", dest="deduplicate", action="store_true", default=True)
    parser.add_argument("--no-deduplicate", dest="deduplicate", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = CleaningConfig(
        min_len=args.min_len,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        max_length_ratio=args.max_length_ratio,
        lowercase_en=args.lowercase_en,
        truncate_long=args.truncate_long,
    )

    train_paths = [raw_dir / name for name in args.train_files]
    valid_paths = [raw_dir / args.valid_file]
    test_paths = [raw_dir / args.test_file]

    train_records, train_stats = process_split(train_paths, args.src_key, args.tgt_key, cfg, args.deduplicate)
    valid_records, valid_stats = process_split(valid_paths, args.src_key, args.tgt_key, cfg, False)
    test_records, test_stats = process_split(test_paths, args.src_key, args.tgt_key, cfg, False)

    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "valid.jsonl", valid_records)
    write_jsonl(output_dir / "test.jsonl", test_records)

    tokenizer_info = train_tokenizer(train_records, args)

    summary = {
        "cleaning_config": cfg.__dict__,
        "train_stats": train_stats,
        "valid_stats": valid_stats,
        "test_stats": test_stats,
        "tokenizer": tokenizer_info,
    }
    summary_path = Path("experiments/results") / "preprocess_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved processed splits to {output_dir}")
    print(f"Tokenizer config stored at {tokenizer_info['config_path']}")
    print(f"Preprocess summary -> {summary_path}")


if __name__ == "__main__":
    main()