"""Utilities for cleaning and filtering parallel text pairs."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "CleaningConfig",
    "clean_parallel_record",
]


WHITESPACE_REGEX = re.compile(r"\s+")
CJK_PUNCT_TRANSLATIONS = str.maketrans({
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
})


@dataclass
class CleaningConfig:
    min_len: int = 1
    max_src_len: int = 128
    max_tgt_len: int = 96
    max_length_ratio: float = 4.0
    lowercase_en: bool = True
    truncate_long: bool = False
    allow_empty: bool = False


def _strip_control_chars(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")


def _normalize_spaces(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = text.replace("\ufeff", "")
    return WHITESPACE_REGEX.sub(" ", text).strip()


def _clean_zh(text: str) -> str:
    text = _strip_control_chars(text)
    text = text.translate(CJK_PUNCT_TRANSLATIONS)
    # keep Chinese spacing compact but avoid double spaces
    text = text.replace("  ", " ")
    return _normalize_spaces(text)


def _clean_en(text: str, lowercase: bool) -> str:
    text = _strip_control_chars(text)
    text = text.replace("“", '"').replace("”", '"').replace("’", "'")
    text = _normalize_spaces(text)
    if lowercase:
        text = text.lower()
    return text


def _length_ok(value: int, min_len: int, max_len: int, truncate: bool) -> Tuple[bool, Optional[int]]:
    if value < min_len:
        return False, None
    if value > max_len:
        if truncate:
            return True, max_len
        return False, None
    return True, None


def clean_parallel_record(
    record: Dict[str, str],
    src_key: str,
    tgt_key: str,
    cfg: CleaningConfig,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Clean and validate a zh→en pair.

    Returns (cleaned_record, None) if valid else (None, reason).
    """
    if src_key not in record or tgt_key not in record:
        return None, "missing_field"

    src_text = _clean_zh(record[src_key])
    tgt_text = _clean_en(record[tgt_key], lowercase=cfg.lowercase_en)

    if not cfg.allow_empty and (len(src_text) == 0 or len(tgt_text) == 0):
        return None, "empty"

    src_ok, src_trunc = _length_ok(len(src_text), cfg.min_len, cfg.max_src_len, cfg.truncate_long)
    if not src_ok:
        return None, "src_length"
    if src_trunc is not None:
        src_text = src_text[:src_trunc]

    tgt_tokens = tgt_text.split()
    tgt_len = len(tgt_tokens)
    tgt_ok, tgt_trunc = _length_ok(tgt_len, cfg.min_len, cfg.max_tgt_len, cfg.truncate_long)
    if not tgt_ok:
        return None, "tgt_length"
    if tgt_trunc is not None:
        tgt_tokens = tgt_tokens[: cfg.max_tgt_len]
        tgt_text = " ".join(tgt_tokens)

    if cfg.max_length_ratio > 0:
        ratio = len(src_text) / max(tgt_len, 1)
        if ratio > cfg.max_length_ratio or ratio < 1 / cfg.max_length_ratio:
            return None, "length_ratio"

    cleaned = dict(record)
    cleaned[src_key] = src_text
    cleaned[tgt_key] = tgt_text
    return cleaned, None
