#!/usr/bin/env python3
"""
T5 Fine-tuning for Chinese-English Translation.

This script fine-tunes a pretrained T5 model (t5-small) on the Chinese-English
translation dataset using HuggingFace Transformers.

Key features:
- Task prefix: "translate Chinese to English: {source}"
- Seq2SeqTrainer for efficient training
- Automatic evaluation with BLEU
- Checkpoint saving
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
import numpy as np
from sacrebleu import corpus_bleu
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def preprocess_function(examples, tokenizer, src_key='zh', tgt_key='en', max_length=128):
    """
    Preprocess examples for T5 training.

    Adds task prefix: "translate Chinese to English: {source}"
    """
    # Add task prefix
    inputs = [f"translate Chinese to English: {text}" for text in examples[src_key]]
    targets = examples[tgt_key]

    # Tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding=False)

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding=False)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU score for evaluation."""
    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 in both preds and labels (used for padding)
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Ensure predictions are valid token IDs (clip to vocab size)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)

    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute BLEU
    bleu = corpus_bleu(decoded_preds, [decoded_labels])

    return {"bleu": bleu.score}


def main():
    parser = argparse.ArgumentParser(description='Fine-tune T5 for Chinese-English translation')
    parser.add_argument('--config', type=str, default='experiments/configs/t5_small.yaml',
                        help='Path to configuration file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    args = parser.parse_args()

    # Load config
    config_path = PROJECT_ROOT / args.config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("T5 Fine-tuning for Chinese-English Translation")
    print("=" * 60)
    print(f"Config: {config_path}")
    print(f"Model: {config['model']['pretrained_name']}")
    print()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()

    # Load tokenizer and model
    model_name = config['model']['pretrained_name']
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print()

    # Load data
    data_config = config['data']
    train_file = PROJECT_ROOT / data_config['train_file']
    valid_file = PROJECT_ROOT / data_config['valid_file']

    print(f"Loading training data from {train_file}")
    train_data = load_jsonl(train_file)
    print(f"Training examples: {len(train_data)}")

    print(f"Loading validation data from {valid_file}")
    valid_data = load_jsonl(valid_file)
    print(f"Validation examples: {len(valid_data)}")
    print()

    # Convert to HuggingFace Dataset
    src_key = data_config.get('src_key', 'zh')
    tgt_key = data_config.get('tgt_key', 'en')

    train_dataset = Dataset.from_dict({
        src_key: [ex[src_key] for ex in train_data],
        tgt_key: [ex[tgt_key] for ex in train_data],
    })

    valid_dataset = Dataset.from_dict({
        src_key: [ex[src_key] for ex in valid_data],
        tgt_key: [ex[tgt_key] for ex in valid_data],
    })

    # Preprocess datasets
    max_length = config['model'].get('max_length', 128)
    print(f"Preprocessing datasets (max_length={max_length})...")

    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, src_key, tgt_key, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    valid_dataset = valid_dataset.map(
        lambda x: preprocess_function(x, tokenizer, src_key, tgt_key, max_length),
        batched=True,
        remove_columns=valid_dataset.column_names,
    )
    print("Preprocessing complete.")
    print()

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # Training arguments
    training_config = config['training']
    output_dir = PROJECT_ROOT / config['evaluation']['output_dir']

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=training_config['learning_rate'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config.get('eval_batch_size', training_config['batch_size']),
        num_train_epochs=training_config['num_epochs'],
        weight_decay=training_config.get('weight_decay', 0.01),
        warmup_steps=training_config.get('warmup_steps', 500),
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        generation_max_length=max_length,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=100,
        report_to="none",  # Disable wandb/tensorboard
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )

    print("=" * 60)
    print("Starting training...")
    print("=" * 60)
    print()

    # Train
    trainer.train()

    print()
    print("=" * 60)
    print("Training complete!")
    print("=" * 60)
    print()

    # Save final model
    final_model_path = PROJECT_ROOT / config['evaluation']['model_save_path']
    final_model_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving final model to {final_model_path}")
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    print(f"✓ Model saved to {final_model_path}")
    print()

    # Final evaluation
    print("Running final evaluation...")
    metrics = trainer.evaluate()

    print("Final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Save metrics
    metrics_file = PROJECT_ROOT / config['experiments']['metrics_file']
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Metrics saved to {metrics_file}")
    print()

    print("=" * 60)
    print("T5 fine-tuning completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
