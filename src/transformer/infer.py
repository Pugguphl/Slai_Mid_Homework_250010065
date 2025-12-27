"""Inference helpers for Transformer checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from src.common.dataset import pad_sequences
from src.common.tokenizer import load_tokenizer
from src.transformer.model import TransformerConfig, TransformerSeq2Seq


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[TransformerSeq2Seq, dict, object]:
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
    )
    model = TransformerSeq2Seq(model_cfg)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, ckpt, tokenizer


def prepare_batch(sentences: List[str], tokenizer) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode_src(sent, add_special_tokens=True) for sent in sentences]
    src, lengths = pad_sequences(encoded, tokenizer.src_pad_id)
    return src, lengths


def translate_sentences(
    model: TransformerSeq2Seq,
    tokenizer,
    sentences: List[str],
    strategy: str = "greedy",
    beam_size: int = 4,
) -> List[str]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Transformer inference helper")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_text", help="Single sentence to translate")
    parser.add_argument("--input_file", help="File with sentences to translate")
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