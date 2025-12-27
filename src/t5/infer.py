#!/usr/bin/env python3
"""
T5 Inference for Chinese-English Translation.

This script performs inference using a fine-tuned T5 model on test data,
supporting both greedy decoding and beam search.

Usage:
    python src/t5/infer.py --model_path experiments/logs/t5_small \
        --input data/processed/test.jsonl \
        --output experiments/results/t5_predictions.jsonl \
        --beam_size 4
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: Path):
    """Save data to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def translate_batch(
    model,
    tokenizer,
    sources: List[str],
    device: torch.device,
    max_length: int = 128,
    beam_size: int = 1,
) -> List[str]:
    """
    Translate a batch of source sentences.

    Args:
        model: T5 model
        tokenizer: T5 tokenizer
        sources: List of source sentences
        device: Device to run on
        max_length: Maximum generation length
        beam_size: Beam size for beam search (1 = greedy)

    Returns:
        List of translated sentences
    """
    # Add task prefix
    inputs = [f"translate Chinese to English: {text}" for text in sources]

    # Tokenize
    encoded = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True,
        padding=True,
        return_tensors='pt'
    ).to(device)

    # Generate
    with torch.no_grad():
        if beam_size == 1:
            # Greedy decoding
            outputs = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=1,
                early_stopping=False,
            )
        else:
            # Beam search
            outputs = model.generate(
                **encoded,
                max_length=max_length,
                num_beams=beam_size,
                early_stopping=True,
            )

    # Decode
    translations = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return translations


def main():
    parser = argparse.ArgumentParser(description='T5 inference for Chinese-English translation')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned T5 model')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSONL file for predictions')
    parser.add_argument('--src_key', type=str, default='zh',
                        help='Source language key in JSONL')
    parser.add_argument('--tgt_key', type=str, default='en',
                        help='Target language key in JSONL (for evaluation)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--beam_size', type=int, default=4,
                        help='Beam size (1 = greedy decoding)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum generation length')
    args = parser.parse_args()

    # Resolve paths
    model_path = PROJECT_ROOT / args.model_path
    input_file = PROJECT_ROOT / args.input
    output_file = PROJECT_ROOT / args.output

    print("=" * 60)
    print("T5 Inference for Chinese-English Translation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"Beam size: {args.beam_size}")
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_path), local_files_only=True)
    model.to(device)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # Load input data
    print(f"Loading data from {input_file}...")
    data = load_jsonl(input_file)
    print(f"Total examples: {len(data)}")
    print()

    # Extract sources and references (if available)
    sources = [ex[args.src_key] for ex in data]
    has_references = args.tgt_key in data[0] if data else False
    references = [ex[args.tgt_key] for ex in data] if has_references else None

    # Translate in batches
    print("Translating...")
    predictions = []

    for i in tqdm(range(0, len(sources), args.batch_size)):
        batch_sources = sources[i:i + args.batch_size]
        batch_predictions = translate_batch(
            model, tokenizer, batch_sources, device,
            max_length=args.max_length,
            beam_size=args.beam_size
        )
        predictions.extend(batch_predictions)

    print(f"Translation complete. Generated {len(predictions)} predictions.")
    print()

    # Save predictions
    output_data = []
    for i, (src, pred) in enumerate(zip(sources, predictions)):
        item = {
            args.src_key: src,
            f'{args.tgt_key}_pred': pred,
        }
        if has_references:
            item[args.tgt_key] = references[i]
        output_data.append(item)

    save_jsonl(output_data, output_file)
    print(f"✓ Predictions saved to {output_file}")
    print()

    # Evaluate BLEU if references available
    if has_references:
        print("Calculating BLEU score...")
        bleu = corpus_bleu(predictions, [references])
        print(f"BLEU: {bleu.score:.4f}")
        print()

        # Save metrics
        metrics_file = output_file.parent / f"{output_file.stem}_metrics.json"
        metrics = {
            "bleu": bleu.score,
            "beam_size": args.beam_size,
            "num_examples": len(predictions),
        }

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        print(f"✓ Metrics saved to {metrics_file}")
        print()

    # Print sample predictions
    print("Sample predictions:")
    print("-" * 60)
    for i in range(min(5, len(sources))):
        print(f"Source:     {sources[i]}")
        print(f"Prediction: {predictions[i]}")
        if has_references:
            print(f"Reference:  {references[i]}")
        print()

    print("=" * 60)
    print("Inference complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
