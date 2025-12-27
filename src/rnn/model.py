"""Encoder-decoder with configurable attention and decoding strategies."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


RNNType = Literal["gru", "lstm"]
AttentionType = Literal["dot", "multiplicative", "additive"]


def _get_rnn(rnn_type: RNNType):
    if rnn_type == "gru":
        return nn.GRU
    if rnn_type == "lstm":
        return nn.LSTM
    raise ValueError(f"Unsupported rnn_type={rnn_type}")


class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, pad_idx: int, rnn_type: RNNType = "gru") -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        rnn_cls = _get_rnn(rnn_type)
        self.rnn = rnn_cls(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn_type = rnn_type

    def forward(self, src: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, attn_type: AttentionType = "dot") -> None:
        super().__init__()
        self.attn_type = attn_type
        if attn_type == "multiplicative":
            self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
        elif attn_type == "additive":
            self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_state: torch.Tensor, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # decoder_state: (batch, hidden), encoder_outputs: (batch, seq, hidden)
        if self.attn_type == "dot":
            energy = torch.bmm(encoder_outputs, decoder_state.unsqueeze(2)).squeeze(2)
        elif self.attn_type == "multiplicative":
            energy = torch.bmm(self.linear(encoder_outputs), decoder_state.unsqueeze(2)).squeeze(2)
        else:  # additive
            seq_len = encoder_outputs.size(1)
            decoder_expanded = decoder_state.unsqueeze(1).expand(-1, seq_len, -1)
            combined = torch.tanh(self.linear(torch.cat([decoder_expanded, encoder_outputs], dim=2)))
            energy = self.v(combined).squeeze(2)

        energy = energy.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(energy, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_layers: int, dropout: float, attn: Attention, rnn_type: RNNType = "gru") -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        rnn_cls = _get_rnn(rnn_type)
        self.rnn = rnn_cls(
            embed_dim + hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.attn = attn
        self.fc_out = nn.Linear(hidden_dim * 2, vocab_size)
        self.rnn_type = rnn_type

    def forward(self, input_tokens: torch.Tensor, hidden, encoder_outputs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_tokens)).unsqueeze(1)
        if isinstance(hidden, tuple):
            decoder_state = hidden[0][-1]
        else:
            decoder_state = hidden[-1]
        context, attn_weights = self.attn(decoder_state, encoder_outputs, mask)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = output.squeeze(1)
        logits = self.fc_out(torch.cat([output, context], dim=1))
        return logits, hidden


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_layers: int
    dropout: float
    pad_idx: int
    bos_idx: int
    eos_idx: int
    rnn_type: RNNType = "gru"
    attention: AttentionType = "dot"
    max_decode_len: int = 100


class Seq2Seq(nn.Module):
    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            pad_idx=config.pad_idx,
            rnn_type=config.rnn_type,
        )
        attn = Attention(config.hidden_dim, config.attention)
        self.decoder = Decoder(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
            attn=attn,
            rnn_type=config.rnn_type,
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        batch_size, tgt_len = tgt.size()
        vocab_size = self.config.vocab_size
        outputs = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)

        encoder_outputs, hidden = self.encoder(src, src_lengths)
        input_tokens = tgt[:, 0]
        mask = (src != self.config.pad_idx)

        for t in range(1, tgt_len):
            logits, hidden = self.decoder(input_tokens, hidden, encoder_outputs, mask)
            outputs[:, t] = logits
            teacher_force = random.random() < teacher_forcing_ratio
            next_input = tgt[:, t] if teacher_force else logits.argmax(dim=1)
            input_tokens = next_input

        return outputs

    def greedy_decode(self, src: torch.Tensor, src_lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        batch_size = src.size(0)
        mask = (src != self.config.pad_idx)
        input_tokens = torch.full((batch_size,), self.config.bos_idx, dtype=torch.long, device=src.device)
        max_len = max_len or self.config.max_decode_len
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            logits, hidden = self.decoder(input_tokens, hidden, encoder_outputs, mask)
            next_tokens = logits.argmax(dim=1)
            outputs.append(next_tokens)
            finished = finished | (next_tokens == self.config.eos_idx)
            input_tokens = next_tokens
            if finished.all():
                break

        return torch.stack(outputs, dim=1)

    def beam_search(self, src: torch.Tensor, src_lengths: torch.Tensor, beam_size: int = 4, max_len: Optional[int] = None) -> torch.Tensor:
        if src.size(0) != 1:
            raise ValueError("Beam search currently supports batch_size=1 for simplicity")
        max_len = max_len or self.config.max_decode_len
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        mask = (src != self.config.pad_idx)

        beams = [
            {
                "tokens": [self.config.bos_idx],
                "log_prob": 0.0,
                "hidden": hidden,
            }
        ]
        completed = []

        for _ in range(max_len):
            new_beams = []
            for beam in beams:
                input_token = torch.tensor([beam["tokens"][-1]], device=src.device)
                logits, new_hidden = self.decoder(input_token, beam["hidden"], encoder_outputs, mask)
                log_probs = torch.log_softmax(logits, dim=1)
                topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=1)
                for log_prob, idx in zip(topk_log_probs[0], topk_indices[0]):
                    new_tokens = beam["tokens"] + [idx.item()]
                    new_log_prob = beam["log_prob"] + log_prob.item()
                    candidate = {
                        "tokens": new_tokens,
                        "log_prob": new_log_prob,
                        "hidden": new_hidden,
                    }
                    if idx.item() == self.config.eos_idx:
                        completed.append(candidate)
                    else:
                        new_beams.append(candidate)
            beams = sorted(new_beams, key=lambda b: b["log_prob"], reverse=True)[:beam_size]
            if not beams:
                break

        if not completed:
            completed = beams

        best = max(completed, key=lambda b: b["log_prob"] / len(b["tokens"]))
        return torch.tensor(best["tokens"][1:], device=src.device).unsqueeze(0)