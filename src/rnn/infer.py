"""Inference helper for trained RNN checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from src.common.dataset import pad_sequences
from src.common.tokenizer import load_tokenizer
from src.rnn.model import Seq2Seq, Seq2SeqConfig


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Seq2Seq, dict, object]:
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
    return model, ckpt, tokenizer


def prepare_batch(sentences: List[str], tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode_src(sent, add_special_tokens=True) for sent in sentences]
    src, lengths = pad_sequences(encoded, tokenizer.src_pad_id)
    return src, lengths


def translate_sentences(model: Seq2Seq, tokenizer, sentences: List[str], strategy: str = "greedy", beam_size: int = 4) -> List[str]:
    device = next(model.parameters()).device
    src, lengths = prepare_batch(sentences, tokenizer)
    src = src.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        if strategy == "greedy":
            preds = model.greedy_decode(src, lengths)
            return [tokenizer.decode_tgt(seq.tolist()) for seq in preds]

        outputs = []
        for i in range(src.size(0)):
            pred = model.beam_search(src[i : i + 1], lengths[i : i + 1], beam_size=beam_size)
            outputs.append(tokenizer.decode_tgt(pred.squeeze(0).tolist()))
        return outputs


def main():
    parser = argparse.ArgumentParser(description="RNN inference")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint")
    parser.add_argument("--input_text", help="Single sentence to translate")
    parser.add_argument("--input_file", help="File containing sentences to translate")
    parser.add_argument("--output_file", help="Where to save translations")
    parser.add_argument("--strategy", choices=["greedy", "beam"], default="greedy")
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()

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