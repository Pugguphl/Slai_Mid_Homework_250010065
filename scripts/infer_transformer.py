#!/usr/bin/env python3
"""CLI for running Transformer inference on raw text or files."""
from __future__ import annotations

import argparse
from pathlib import Path

import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transformer.infer import build_model_from_checkpoint, translate_sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transformer inference CLI")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--input_text", help="Single sentence to translate")
    parser.add_argument("--input_file", help="File containing sentences to translate")
    parser.add_argument("--output_file", help="Where to save translations")
    parser.add_argument("--strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, tokenizer = build_model_from_checkpoint(args.checkpoint, device)

    if args.input_file:
        sentences = [line.strip() for line in Path(args.input_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    elif args.input_text:
        sentences = [args.input_text]
    else:
        sentences = [input("请输入要翻译的中文句子: ")]

    translations = translate_sentences(model, tokenizer, sentences, args.strategy, args.beam_size)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for line in translations:
                f.write(line + "\n")
    else:
        for src, tgt in zip(sentences, translations):
            print(f"SRC: {src}\nTGT: {tgt}\n")


if __name__ == "__main__":
    main()