"""Tokenizer utilities used across the project."""
from __future__ import annotations

import json
import os
import tempfile
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

try:
    import sentencepiece as spm
except ImportError as exc:  # pragma: no cover - import guard for static analyzers
    raise ImportError("sentencepiece is required. Please install via requirements.txt") from exc

SPECIAL_TOKENS = {
    "pad": "<pad>",
    "unk": "<unk>",
    "bos": "<s>",
    "eos": "</s>",
}


class BaseTokenizer(ABC):
    pad_token: str = SPECIAL_TOKENS["pad"]
    unk_token: str = SPECIAL_TOKENS["unk"]
    bos_token: str = SPECIAL_TOKENS["bos"]
    eos_token: str = SPECIAL_TOKENS["eos"]

    @abstractmethod
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ...

    def encode_src(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.encode(text, add_special_tokens=add_special_tokens)

    def encode_tgt(self, text: str, add_special_tokens: bool = False) -> List[int]:
        return self.encode(text, add_special_tokens=add_special_tokens)

    @abstractmethod
    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        ...

    def decode_src(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_tgt(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @property
    @abstractmethod
    def pad_id(self) -> int:
        ...

    @property
    @abstractmethod
    def bos_id(self) -> int:
        ...

    @property
    @abstractmethod
    def eos_id(self) -> int:
        ...

    @property
    @abstractmethod
    def unk_id(self) -> int:
        ...

    @property
    def src_pad_id(self) -> int:
        return self.pad_id

    @property
    def tgt_pad_id(self) -> int:
        return self.pad_id

    @property
    def src_bos_id(self) -> int:
        return self.bos_id

    @property
    def tgt_bos_id(self) -> int:
        return self.bos_id

    @property
    def src_eos_id(self) -> int:
        return self.eos_id

    @property
    def tgt_eos_id(self) -> int:
        return self.eos_id


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: str):
        self.model_path = str(model_path)
        self.processor = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def pad_id(self) -> int:
        return self.processor.pad_id()

    @property
    def bos_id(self) -> int:
        return self.processor.bos_id()

    @property
    def eos_id(self) -> int:
        return self.processor.eos_id()

    @property
    def unk_id(self) -> int:
        return self.processor.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = self.processor.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        if skip_special_tokens:
            filtered = [tid for tid in token_ids if tid not in {self.pad_id, self.bos_id, self.eos_id}]
        else:
            filtered = list(token_ids)
        return self.processor.decode(filtered)

    def save_config(self, output_path: str, vocab_size: int, extra: Dict) -> None:
        config = {
            "type": "sentencepiece",
            "model_file": str(self.model_path),
            "vocab_size": vocab_size,
            "extra": extra,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


class CharWordTokenizer(BaseTokenizer):
    """Fallback tokenizer using char-level zh and whitespace English."""

    def __init__(self, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int]):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_rev = {idx: tok for tok, idx in src_vocab.items()}
        self.tgt_rev = {idx: tok for tok, idx in tgt_vocab.items()}

    @property
    def pad_id(self) -> int:
        return self.tgt_vocab[self.pad_token]

    @property
    def bos_id(self) -> int:
        return self.tgt_vocab[self.bos_token]

    @property
    def eos_id(self) -> int:
        return self.tgt_vocab[self.eos_token]

    @property
    def unk_id(self) -> int:
        return self.tgt_vocab[self.unk_token]

    @property
    def src_pad_id(self) -> int:  # type: ignore[override]
        return self.src_vocab[self.pad_token]

    @property
    def src_bos_id(self) -> int:  # type: ignore[override]
        return self.src_vocab[self.bos_token]

    @property
    def src_eos_id(self) -> int:  # type: ignore[override]
        return self.src_vocab[self.eos_token]

    @property
    def tgt_pad_id(self) -> int:  # type: ignore[override]
        return self.pad_id

    @property
    def tgt_bos_id(self) -> int:  # type: ignore[override]
        return self.bos_id

    @property
    def tgt_eos_id(self) -> int:  # type: ignore[override]
        return self.eos_id

    @property
    def vocab_size(self) -> int:
        return max(len(self.src_vocab), len(self.tgt_vocab))

    def _encode_tokens(self, tokens: Sequence[str], vocab: Dict[str, int]) -> List[int]:
        return [vocab.get(tok, vocab[self.unk_token]) for tok in tokens]

    def encode_src(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = [ch for ch in text if not ch.isspace()]
        ids = self._encode_tokens(tokens, self.src_vocab)
        if add_special_tokens:
            ids = [self.src_vocab[self.bos_token]] + ids + [self.src_vocab[self.eos_token]]
        return ids

    def encode_tgt(self, text: str, add_special_tokens: bool = False) -> List[int]:
        tokens = text.split()
        ids = self._encode_tokens(tokens, self.tgt_vocab)
        if add_special_tokens:
            ids = [self.tgt_vocab[self.bos_token]] + ids + [self.tgt_vocab[self.eos_token]]
        return ids

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:  # unused but keeps interface
        return self.encode_tgt(text, add_special_tokens)

    def decode_src(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.src_rev.get(tid, self.unk_token)
            if skip_special_tokens and tok in SPECIAL_TOKENS.values():
                continue
            tokens.append(tok)
        return "".join(tokens)

    def decode_tgt(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for tid in token_ids:
            tok = self.tgt_rev.get(tid, self.unk_token)
            if skip_special_tokens and tok in SPECIAL_TOKENS.values():
                continue
            tokens.append(tok)
        return " ".join(tokens)

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.decode_tgt(token_ids, skip_special_tokens)

    def save_config(self, output_path: str, extra: Dict) -> None:
        config = {
            "type": "char_word",
            "src_vocab_file": extra["src_vocab_file"],
            "tgt_vocab_file": extra["tgt_vocab_file"],
            "extra": extra,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)


def train_sentencepiece_tokenizer(
    corpus: Iterable[str],
    output_dir: str,
    vocab_size: int = 32000,
    model_type: str = "unigram",
    character_coverage: float = 0.9995,
    model_prefix: str = "spm_zh_en",
    user_defined_symbols: Sequence[str] | None = None,
) -> Dict[str, str]:
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        for line in corpus:
            if line.strip():
                tmp.write(line.strip() + "\n")
        tmp_path = tmp.name

    prefix_path = os.path.join(output_dir, model_prefix)
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=prefix_path,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=list(user_defined_symbols or []),
        normalization_rule_name="nmt_nfkc_cf",
        train_extremely_large_corpus=True,
    )

    os.unlink(tmp_path)
    return {
        "model_file": f"{prefix_path}.model",
        "vocab_file": f"{prefix_path}.vocab",
    }


def _build_vocab(tokens: Iterable[Sequence[str]], vocab_size: int) -> Dict[str, int]:
    counter = Counter()
    for seq in tokens:
        counter.update(seq)

    vocab = {
        SPECIAL_TOKENS["pad"]: 0,
        SPECIAL_TOKENS["unk"]: 1,
        SPECIAL_TOKENS["bos"]: 2,
        SPECIAL_TOKENS["eos"]: 3,
    }
    for token, _ in counter.most_common(max(vocab_size - len(vocab), 0)):
        if token in vocab:
            continue
        vocab[token] = len(vocab)
    return vocab


def train_char_word_tokenizer(
    zh_texts: Iterable[str],
    en_texts: Iterable[str],
    output_dir: str,
    src_vocab_size: int = 8000,
    tgt_vocab_size: int = 16000,
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    zh_tokens = ([ch for ch in text if not ch.isspace()] for text in zh_texts)
    en_tokens = (text.split() for text in en_texts)

    src_vocab = _build_vocab(zh_tokens, src_vocab_size)
    tgt_vocab = _build_vocab(en_tokens, tgt_vocab_size)

    src_path = output / "char_word_src_vocab.json"
    tgt_path = output / "char_word_tgt_vocab.json"
    with src_path.open("w", encoding="utf-8") as f:
        json.dump(src_vocab, f, ensure_ascii=False, indent=2)
    with tgt_path.open("w", encoding="utf-8") as f:
        json.dump(tgt_vocab, f, ensure_ascii=False, indent=2)

    return {
        "src_vocab_file": str(src_path),
        "tgt_vocab_file": str(tgt_path),
    }


def load_tokenizer(config_path: str) -> BaseTokenizer:
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_type = config.get("type")
    if tokenizer_type == "sentencepiece":
        return SentencePieceTokenizer(config["model_file"])
    if tokenizer_type == "char_word":
        with open(config["src_vocab_file"], "r", encoding="utf-8") as f_src:
            src_vocab = json.load(f_src)
        with open(config["tgt_vocab_file"], "r", encoding="utf-8") as f_tgt:
            tgt_vocab = json.load(f_tgt)
        return CharWordTokenizer(src_vocab, tgt_vocab)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")