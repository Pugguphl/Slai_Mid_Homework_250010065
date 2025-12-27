#!/usr/bin/env python3
"""RNN inference script for analyze_rnn_results.py."""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.dataset import TranslationDataset, collate_translation_batch
from src.common.tokenizer import load_tokenizer
from src.rnn.model import Seq2Seq, Seq2SeqConfig


def corpus_bleu(predictions, references):
    """Calculate corpus BLEU score using sacrebleu."""
    try:
        import sacrebleu
        return sacrebleu.corpus_bleu(predictions, [references])
    except ImportError:
        print("Warning: sacrebleu not installed, returning 0.0")
        return type('obj', (object,), {'score': 0.0})()
    except TypeError:
        try:
            import sacrebleu
            return sacrebleu.corpus_bleu(predictions, references)
        except Exception as e:
            print(f"Warning: BLEU calculation failed: {e}")
            return type('obj', (object,), {'score': 0.0})()


def run_inference(checkpoint_path: str, split: str, beam_size: int = 4) -> float:
    """
    Run inference on a checkpoint and return BLEU score.

    Args:
        checkpoint_path: Path to model checkpoint
        split: 'valid' or 'test'
        beam_size: Beam size for beam search

    Returns:
        BLEU score
    """
    if not Path(checkpoint_path).exists():
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return 0.0

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        data_cfg = ckpt.get("config", {}).get("data", {})
        dataset_path = Path(__file__).resolve().parents[1] / f"data/processed/{split}.jsonl"
        
        if not dataset_path.exists():
            print(f"Warning: Dataset file {dataset_path} not found")
            return 0.0

        dataset = TranslationDataset(
            file_path=str(dataset_path),
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

        bleu = corpus_bleu(predictions, [[ref] for ref in references]).score if predictions else 0.0
        return bleu

    except Exception as e:
        print(f"Warning: Inference failed for {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="RNN inference for analyze_rnn_results.py")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--split", required=True, choices=["valid", "test"], help="Dataset split")
    parser.add_argument("--beam_size", type=int, default=4, help="Beam size (1 for greedy)")
    args = parser.parse_args()

    bleu = run_inference(args.checkpoint, args.split, args.beam_size)
    print(f"BLEU: {bleu:.4f}")


if __name__ == "__main__":
    main()
