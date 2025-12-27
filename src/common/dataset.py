"""Dataset utilities for training/inference."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.common.tokenizer import BaseTokenizer, load_tokenizer


def _load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


@dataclass
class DatasetConfig:
    file_path: str
    tokenizer_config: str
    src_key: str = "zh"
    tgt_key: str = "en"
    add_bos_eos: bool = True
    max_samples: Optional[int] = None


class TranslationDataset(Dataset):
    """PyTorch dataset that tokenizes on the fly using configured tokenizer."""

    def __init__(self, file_path: str, tokenizer: BaseTokenizer, src_key: str = "zh", tgt_key: str = "en", add_bos_eos: bool = True, max_samples: Optional[int] = None) -> None:
        super().__init__()
        self.path = Path(file_path)
        self.samples = _load_jsonl(self.path)
        if max_samples:
            self.samples = self.samples[:max_samples]
        self.tokenizer = tokenizer
        self.src_key = src_key
        self.tgt_key = tgt_key
        self.add_bos_eos = add_bos_eos

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        record = self.samples[idx]
        src_ids = self.tokenizer.encode_src(record[self.src_key], add_special_tokens=self.add_bos_eos)
        tgt_ids = self.tokenizer.encode_tgt(record[self.tgt_key], add_special_tokens=self.add_bos_eos)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids,
            "src_text": record[self.src_key],
            "tgt_text": record[self.tgt_key],
        }


def pad_sequences(sequences: List[List[int]], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    max_len = lengths.max().item()
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_len), pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, lengths


def collate_translation_batch(batch: List[Dict], tokenizer: BaseTokenizer, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
    src_batch, src_lengths = pad_sequences([item["src_ids"] for item in batch], tokenizer.src_pad_id)
    tgt_batch, tgt_lengths = pad_sequences([item["tgt_ids"] for item in batch], tokenizer.tgt_pad_id)

    data = {
        "src": src_batch,
        "src_lengths": src_lengths,
        "tgt": tgt_batch,
        "tgt_lengths": tgt_lengths,
    }
    if device is not None:
        data = {k: v.to(device) for k, v in data.items()}
    return data


def load_dataset(config: DatasetConfig) -> Tuple[TranslationDataset, BaseTokenizer]:
    tokenizer = load_tokenizer(config.tokenizer_config)
    dataset = TranslationDataset(
        file_path=config.file_path,
        tokenizer=tokenizer,
        src_key=config.src_key,
        tgt_key=config.tgt_key,
        add_bos_eos=config.add_bos_eos,
        max_samples=config.max_samples,
    )
    return dataset, tokenizer