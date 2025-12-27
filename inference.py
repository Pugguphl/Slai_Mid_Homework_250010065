#!/usr/bin/env python3
"""
One-click inference script for all NMT models.

This script supports:
- RNN models (Seq2Seq with Attention)
- Transformer models (with various ablations)
- T5 pre-trained models

Usage:
    python inference.py --model_type rnn --checkpoint experiments/logs/best_rnn_additive.pt
    python inference.py --model_type transformer --checkpoint experiments/logs/transformer_abs_ln_best.pt
    python inference.py --model_type t5 --checkpoint experiments/logs/t5_small_best
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.dataset import TranslationDataset, collate_translation_batch
from src.common.tokenizer import load_tokenizer
from src.rnn.model import Seq2Seq, Seq2SeqConfig
from src.transformer.model import TransformerConfig, TransformerSeq2Seq


def corpus_bleu(predictions: List[str], references: List[str]) -> float:
    """Calculate corpus BLEU score using sacrebleu."""
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(predictions, [[ref] for ref in references]).score
    except ImportError:
        print("Warning: sacrebleu not installed, returning 0.0")
        return 0.0


def load_rnn_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, object]:
    """Load RNN model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]["model"]
    tokenizer_path = ckpt.get("tokenizer") or ckpt["config"]["data"]["tokenizer_file"]
    tokenizer = load_tokenizer(tokenizer_path)
    
    model_cfg = Seq2SeqConfig(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.get("embedding_dim", 256),
        hidden_dim=config.get("hidden_dim", 512),
        num_layers=config.get("num_layers", 2),
        dropout=config.get("dropout", 0.3),
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        rnn_type=config.get("architecture", "gru"),
        attention=config.get("attention", "dot"),
        max_decode_len=config.get("max_decode_len", 120),
    )
    model = Seq2Seq(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def load_transformer_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, object]:
    """Load Transformer model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]["model"]
    tokenizer_path = ckpt.get("tokenizer") or ckpt["config"]["data"]["tokenizer_file"]
    tokenizer = load_tokenizer(tokenizer_path)
    
    model_cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        num_heads=config.get("num_heads", 8),
        num_encoder_layers=config.get("num_encoder_layers", 6),
        num_decoder_layers=config.get("num_decoder_layers", 6),
        dim_feedforward=config.get("dim_feedforward", 2048),
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "relu"),
        max_position_embeddings=config.get("max_position_embeddings", 256),
        pad_idx=tokenizer.pad_id,
        bos_idx=tokenizer.bos_id,
        eos_idx=tokenizer.eos_id,
        share_embeddings=config.get("share_embeddings", True),
        tie_output=config.get("tie_output", True),
        norm_first=config.get("norm_first", True),
        positional_encoding=config.get("positional_encoding", "sinusoidal"),
        norm_type=config.get("norm_type", "layernorm"),
    )
    model = TransformerSeq2Seq(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def load_t5_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, object]:
    """Load T5 model from checkpoint."""
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer
    except ImportError:
        raise ImportError("transformers library is required for T5 models")
    
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    model.to(device)
    model.eval()
    tokenizer = T5Tokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def run_inference_rnn(model: torch.nn.Module, tokenizer, dataset_path: str, split: str, 
                      beam_size: int = 4, device: torch.device = torch.device("cpu")) -> Tuple[float, List[str], List[str]]:
    """Run inference for RNN model."""
    data_cfg = {}
    dataset = TranslationDataset(
        file_path=dataset_path,
        tokenizer=tokenizer,
        src_key=data_cfg.get("src_key", "zh"),
        tgt_key=data_cfg.get("tgt_key", "en"),
        add_bos_eos=True,
    )

    def collate_with_text(batch, tokenizer):
        data = collate_translation_batch(batch, tokenizer)
        data["src_text"] = [item["src_text"] for item in batch]
        data["tgt_text"] = [item["tgt_text"] for item in batch]
        return data

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        if beam_size == 1:
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=partial(collate_with_text, tokenizer=tokenizer),
            )
            for batch in tqdm(loader, desc=f"Greedy decoding on {split}"):
                src = batch["src"].to(device)
                lengths = batch["src_lengths"].to(device)
                preds = model.greedy_decode(src, lengths)
                texts = [tokenizer.decode_tgt(seq.tolist()) for seq in preds]
                predictions.extend(texts)
                references.extend(batch["tgt_text"])
        else:
            for sample in tqdm(dataset, desc=f"Beam{beam_size} decoding on {split}"):
                src_tensor = torch.tensor(sample["src_ids"], dtype=torch.long, device=device).unsqueeze(0)
                lengths = torch.tensor([len(sample["src_ids"])], dtype=torch.long, device=device)
                pred = model.beam_search(src_tensor, lengths, beam_size=beam_size)
                predictions.append(tokenizer.decode_tgt(pred.squeeze(0).tolist()))
                references.append(sample["tgt_text"])

    bleu = corpus_bleu(predictions, references) if predictions else 0.0
    return bleu, predictions, references


def run_inference_transformer(model: torch.nn.Module, tokenizer, dataset_path: str, split: str,
                              beam_size: int = 4, device: torch.device = torch.device("cpu")) -> Tuple[float, List[str], List[str]]:
    """Run inference for Transformer model."""
    data_cfg = {}
    dataset = TranslationDataset(
        file_path=dataset_path,
        tokenizer=tokenizer,
        src_key=data_cfg.get("src_key", "zh"),
        tgt_key=data_cfg.get("tgt_key", "en"),
        add_bos_eos=True,
    )

    def collate_with_text(batch, tokenizer):
        data = collate_translation_batch(batch, tokenizer)
        data["src_text"] = [item["src_text"] for item in batch]
        data["tgt_text"] = [item["tgt_text"] for item in batch]
        return data

    predictions = []
    references = []

    model.eval()
    with torch.no_grad():
        if beam_size == 1:
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                collate_fn=partial(collate_with_text, tokenizer=tokenizer),
            )
            for batch in tqdm(loader, desc=f"Greedy decoding on {split}"):
                src = batch["src"].to(device)
                lengths = batch["src_lengths"].to(device)
                preds = model.greedy_decode(src, lengths)
                texts = [tokenizer.decode_tgt(seq.tolist()) for seq in preds]
                predictions.extend(texts)
                references.extend(batch["tgt_text"])
        else:
            for sample in tqdm(dataset, desc=f"Beam{beam_size} decoding on {split}"):
                src_tensor = torch.tensor(sample["src_ids"], dtype=torch.long, device=device).unsqueeze(0)
                lengths = torch.tensor([len(sample["src_ids"])], dtype=torch.long, device=device)
                pred = model.beam_search(src_tensor, lengths, beam_size=beam_size)
                predictions.append(tokenizer.decode_tgt(pred.squeeze(0).tolist()))
                references.append(sample["tgt_text"])

    bleu = corpus_bleu(predictions, references) if predictions else 0.0
    return bleu, predictions, references


def run_inference_t5(model: torch.nn.Module, tokenizer, dataset_path: str, split: str,
                     beam_size: int = 4, device: torch.device = torch.device("cpu")) -> Tuple[float, List[str], List[str]]:
    """Run inference for T5 model."""
    import json
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for item in tqdm(data, desc=f"T5 decoding on {split}"):
            src_text = item["zh"]
            tgt_text = item["en"]
            
            inputs = tokenizer(src_text, return_tensors="pt", padding=True, truncation=True, max_length=256)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if beam_size == 1:
                outputs = model.generate(**inputs, max_length=256)
            else:
                outputs = model.generate(**inputs, max_length=256, num_beams=beam_size)
            
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(pred_text)
            references.append(tgt_text)
    
    bleu = corpus_bleu(predictions, references) if predictions else 0.0
    return bleu, predictions, references


def main():
    parser = argparse.ArgumentParser(description="One-click inference for all NMT models")
    parser.add_argument("--model_type", type=str, required=True, choices=["rnn", "transformer", "t5"],
                        help="Model type: rnn, transformer, or t5")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"], help="Dataset split")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size (1 for greedy)")
    parser.add_argument("--output_dir", type=str, default="experiments/results/inference_outputs",
                        help="Output directory for predictions")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_path = Path(args.data_dir) / f"{args.split}.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Model Type: {args.model_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Beam Size: {args.beam_size}")
    print(f"{'='*60}\n")

    if args.model_type == "rnn":
        model, tokenizer = load_rnn_model(str(checkpoint_path), device)
        bleu, predictions, references = run_inference_rnn(model, tokenizer, str(dataset_path), args.split, args.beam_size, device)
    elif args.model_type == "transformer":
        model, tokenizer = load_transformer_model(str(checkpoint_path), device)
        bleu, predictions, references = run_inference_transformer(model, tokenizer, str(dataset_path), args.split, args.beam_size, device)
    elif args.model_type == "t5":
        model, tokenizer = load_t5_model(str(checkpoint_path), device)
        bleu, predictions, references = run_inference_t5(model, tokenizer, str(dataset_path), args.split, args.beam_size, device)

    output_file = output_dir / f"{args.model_type}_{checkpoint_path.stem}_{args.split}_beam{args.beam_size}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n")

    print(f"\n{'='*60}")
    print(f"BLEU Score: {bleu:.4f}")
    print(f"Predictions saved to: {output_file}")
    print(f"{'='*60}\n")

    print("Example translations:")
    for i in range(min(3, len(predictions))):
        print(f"\n--- Example {i+1} ---")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")

    return bleu


if __name__ == "__main__":
    main()
