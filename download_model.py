#!/usr/bin/env python3
"""
Download T5 model from HuggingFace Hub to local directory.
"""

import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PROJECT_ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='Download T5 model to local directory')
    parser.add_argument('--model', type=str, default='t5-small',
                        help='Model name from HuggingFace Hub (default: t5-small)')
    parser.add_argument('--output', type=str, default='models/t5-small',
                        help='Output directory path (default: models/t5-small)')
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading T5 Model from HuggingFace Hub")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output directory: {output_dir}")
    print()

    # Download tokenizer
    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.save_pretrained(str(output_dir))
    print(f"✓ Tokenizer saved to {output_dir}")
    print()

    # Download model
    print("Downloading model (this may take a few minutes)...")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.save_pretrained(str(output_dir))
    print(f"✓ Model saved to {output_dir}")
    print()

    # Verify
    print("Verifying downloaded model...")
    tokenizer_test = AutoTokenizer.from_pretrained(str(output_dir), local_files_only=True)
    model_test = AutoModelForSeq2SeqLM.from_pretrained(str(output_dir), local_files_only=True)
    print(f"✓ Model loaded successfully from local directory")
    print(f"  Model parameters: {sum(p.numel() for p in model_test.parameters()) / 1e6:.2f}M")
    print()

    print("=" * 60)
    print("Download completed successfully!")
    print("=" * 60)
    print()
    print("To use this local model in your training, update your config file:")
    print(f"  pretrained_name: {output_dir}")
    print()


if __name__ == '__main__':
    main()
