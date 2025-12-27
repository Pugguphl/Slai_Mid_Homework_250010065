from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    max_position_embeddings: int = 512
    pad_idx: int = 0
    bos_idx: int = 2
    eos_idx: int = 3
    max_decode_len: int = 120
    share_embeddings: bool = True
    tie_output: bool = True
    norm_first: bool = True
    learned_positional_encoding: bool = False
    # New parameters for ablation experiments
    positional_encoding: str = "sinusoidal"  # Options: 'sinusoidal', 'learned', 'relative'
    norm_type: str = "layernorm"  # Options: 'layernorm', 'rmsnorm'


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, learned: bool = False) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.learned = learned
        if learned:
            self.embedding = nn.Embedding(max_len, d_model)
            nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            self.register_buffer("position_ids", torch.arange(max_len).unsqueeze(0), persistent=False)
        else:
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned:
            positions = self.position_ids[:, : x.size(1)].to(x.device)
            pos = self.embedding(positions)
        else:
            pos = self.pe[:, : x.size(1)].to(x.device)
        return self.dropout(x + pos)


class RelativePositionalBias(nn.Module):
    """
    T5-style relative positional bias for self-attention.

    Instead of adding positional information to embeddings, this computes a bias
    term that is added to attention scores based on the relative distance between
    query and key positions.

    Args:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance to consider (default: 128)
        bidirectional: Whether to use bidirectional relative positions (default: True)
    """

    def __init__(self, num_heads: int, max_distance: int = 128, bidirectional: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional

        # Number of relative position buckets
        # For bidirectional: buckets cover both directions
        num_buckets = max_distance * 2 if bidirectional else max_distance

        # Learnable relative position bias table: [num_buckets, num_heads]
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)

    def compute_bias(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        """
        Compute relative position bias matrix.

        Args:
            qlen: Query sequence length
            klen: Key sequence length
            device: Device to create tensor on

        Returns:
            Bias tensor of shape [num_heads, qlen, klen]
        """
        # Create position indices
        q_pos = torch.arange(qlen, dtype=torch.long, device=device).unsqueeze(1)  # [qlen, 1]
        k_pos = torch.arange(klen, dtype=torch.long, device=device).unsqueeze(0)  # [1, klen]

        # Compute relative positions: positive means key is after query
        relative_position = k_pos - q_pos  # [qlen, klen]

        # Convert to bucket indices
        if self.bidirectional:
            # Map negative distances to [0, max_distance)
            # Map positive distances to [max_distance, 2*max_distance)
            bucket_idx = torch.where(
                relative_position < 0,
                torch.clamp(-relative_position, max=self.max_distance - 1),
                torch.clamp(relative_position, max=self.max_distance - 1) + self.max_distance
            )
        else:
            # Only consider forward positions, clamp to [0, max_distance)
            bucket_idx = torch.clamp(relative_position, min=0, max=self.max_distance - 1)

        # Look up bias values: [qlen, klen, num_heads]
        bias = self.relative_attention_bias(bucket_idx)

        # Transpose to [num_heads, qlen, klen] for attention computation
        return bias.permute(2, 0, 1).contiguous()


def _get_activation_fn(activation: str):
    """Get activation function by name."""
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


def _get_norm_layer(norm_type: str, d_model: int, eps: float = 1e-6):
    """Get normalization layer by type."""
    if norm_type == "layernorm":
        return nn.LayerNorm(d_model, eps=eps)
    elif norm_type == "rmsnorm":
        from src.common.layers import RMSNorm
        return RMSNorm(d_model, eps=eps)
    else:
        raise ValueError(f"Unsupported norm_type: {norm_type}")


class CustomTransformerEncoderLayer(nn.Module):
    """
    Custom TransformerEncoderLayer with configurable normalization and relative position bias support.

    This is needed because PyTorch's built-in TransformerEncoderLayer only supports LayerNorm.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_first: bool = True,
        relative_bias: Optional[RelativePositionalBias] = None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = _get_norm_layer(norm_type, d_model)
        self.norm2 = _get_norm_layer(norm_type, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first
        self.relative_bias = relative_bias

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: Optional attention mask
            src_key_padding_mask: [batch_size, seq_len] mask for padding tokens

        Returns:
            output: [batch_size, seq_len, d_model]
        """
        # Add relative position bias to attention if enabled
        attn_mask = src_mask
        if self.relative_bias is not None:
            batch_size, seq_len = src.size(0), src.size(1)
            rel_bias = self.relative_bias.compute_bias(seq_len, seq_len, src.device)  # [num_heads, seq_len, seq_len]
            # Expand to [batch_size * num_heads, seq_len, seq_len] for MultiheadAttention
            rel_bias = rel_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch_size, num_heads, seq_len, seq_len]
            rel_bias = rel_bias.reshape(batch_size * self.relative_bias.num_heads, seq_len, seq_len)
            if attn_mask is None:
                attn_mask = rel_bias
            else:
                attn_mask = attn_mask + rel_bias

        if self.norm_first:
            # Pre-norm architecture
            src = src + self._sa_block(self.norm1(src), attn_mask, src_key_padding_mask)
            src = src + self._ff_block(self.norm2(src))
        else:
            # Post-norm architecture
            src = self.norm1(src + self._sa_block(src, attn_mask, src_key_padding_mask))
            src = self.norm2(src + self._ff_block(src))
        return src

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention block."""
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class CustomTransformerDecoderLayer(nn.Module):
    """
    Custom TransformerDecoderLayer with configurable normalization and relative position bias support.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_type: str = "layernorm",
        norm_first: bool = True,
        relative_bias_self: Optional[RelativePositionalBias] = None,
        relative_bias_cross: Optional[RelativePositionalBias] = None,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = _get_norm_layer(norm_type, d_model)
        self.norm2 = _get_norm_layer(norm_type, d_model)
        self.norm3 = _get_norm_layer(norm_type, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.norm_first = norm_first
        self.relative_bias_self = relative_bias_self
        self.relative_bias_cross = relative_bias_cross

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, tgt_len, d_model]
            memory: [batch_size, src_len, d_model]
            tgt_mask: Causal mask for target
            memory_mask: Optional mask for memory
            tgt_key_padding_mask: Padding mask for target
            memory_key_padding_mask: Padding mask for memory

        Returns:
            output: [batch_size, tgt_len, d_model]
        """
        batch_size = tgt.size(0)

        # Add relative bias to self-attention
        self_attn_mask = tgt_mask
        if self.relative_bias_self is not None:
            tgt_len = tgt.size(1)
            rel_bias = self.relative_bias_self.compute_bias(tgt_len, tgt_len, tgt.device)
            # Expand to [batch_size * num_heads, tgt_len, tgt_len]
            rel_bias = rel_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
            rel_bias = rel_bias.reshape(batch_size * self.relative_bias_self.num_heads, tgt_len, tgt_len)
            if self_attn_mask is None:
                self_attn_mask = rel_bias
            else:
                self_attn_mask = self_attn_mask + rel_bias

        # Add relative bias to cross-attention (optional, usually not used)
        cross_attn_mask = memory_mask
        if self.relative_bias_cross is not None:
            tgt_len = tgt.size(1)
            src_len = memory.size(1)
            rel_bias = self.relative_bias_cross.compute_bias(tgt_len, src_len, tgt.device)
            # Expand to [batch_size * num_heads, tgt_len, src_len]
            rel_bias = rel_bias.unsqueeze(0).expand(batch_size, -1, -1, -1)
            rel_bias = rel_bias.reshape(batch_size * self.relative_bias_cross.num_heads, tgt_len, src_len)
            if cross_attn_mask is None:
                cross_attn_mask = rel_bias
            else:
                cross_attn_mask = cross_attn_mask + rel_bias

        if self.norm_first:
            # Pre-norm architecture
            tgt = tgt + self._sa_block(self.norm1(tgt), self_attn_mask, tgt_key_padding_mask)
            tgt = tgt + self._mha_block(self.norm2(tgt), memory, cross_attn_mask, memory_key_padding_mask)
            tgt = tgt + self._ff_block(self.norm3(tgt))
        else:
            # Post-norm architecture
            tgt = self.norm1(tgt + self._sa_block(tgt, self_attn_mask, tgt_key_padding_mask))
            tgt = self.norm2(tgt + self._mha_block(tgt, memory, cross_attn_mask, memory_key_padding_mask))
            tgt = self.norm3(tgt + self._ff_block(tgt))
        return tgt

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention block."""
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Multi-head cross-attention block."""
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Feed-forward block."""
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


class TransformerSeq2Seq(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.scale = math.sqrt(config.d_model)

        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)
        if config.share_embeddings:
            self.tgt_embedding = self.src_embedding
        else:
            self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_idx)

        # Position encoding setup based on config
        use_relative_pos = (config.positional_encoding == "relative")

        if use_relative_pos:
            # Use relative position bias, no absolute positional encoding
            self.src_pos = None
            self.tgt_pos = None
            # Create relative position bias modules
            self.encoder_rel_bias = RelativePositionalBias(config.num_heads, max_distance=128, bidirectional=True)
            self.decoder_rel_bias_self = RelativePositionalBias(config.num_heads, max_distance=128, bidirectional=False)
            self.decoder_rel_bias_cross = None  # Typically not used for cross-attention
        else:
            # Use absolute positional encoding (sinusoidal or learned)
            use_learned = (config.positional_encoding == "learned") or config.learned_positional_encoding
            self.src_pos = PositionalEncoding(
                config.d_model,
                config.dropout,
                config.max_position_embeddings,
                learned=use_learned,
            )
            self.tgt_pos = PositionalEncoding(
                config.d_model,
                config.dropout,
                config.max_position_embeddings,
                learned=use_learned,
            )
            self.encoder_rel_bias = None
            self.decoder_rel_bias_self = None
            self.decoder_rel_bias_cross = None

        # Build encoder and decoder with custom layers
        encoder_layers = []
        for _ in range(config.num_encoder_layers):
            layer = CustomTransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                norm_type=config.norm_type,
                norm_first=config.norm_first,
                relative_bias=self.encoder_rel_bias,
            )
            encoder_layers.append(layer)
        self.encoder = nn.ModuleList(encoder_layers)

        decoder_layers = []
        for _ in range(config.num_decoder_layers):
            layer = CustomTransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation=config.activation,
                norm_type=config.norm_type,
                norm_first=config.norm_first,
                relative_bias_self=self.decoder_rel_bias_self,
                relative_bias_cross=self.decoder_rel_bias_cross,
            )
            decoder_layers.append(layer)
        self.decoder = nn.ModuleList(decoder_layers)

        # Final layer norm for encoder and decoder (if using pre-norm)
        if config.norm_first:
            self.encoder_norm = _get_norm_layer(config.norm_type, config.d_model)
            self.decoder_norm = _get_norm_layer(config.norm_type, config.d_model)
        else:
            self.encoder_norm = None
            self.decoder_norm = None

        self.generator = nn.Linear(config.d_model, config.vocab_size)
        if config.tie_output and config.share_embeddings:
            self.generator.weight = self.src_embedding.weight

    def forward(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor], tgt: torch.Tensor) -> torch.Tensor:
        tgt_input = tgt[:, :-1]
        memory, src_padding_mask = self.encode(src)
        logits = self.decode(memory, src_padding_mask, tgt_input)
        return logits

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src.eq(self.config.pad_idx)
        src_emb = self.src_embedding(src) * self.scale

        # Apply positional encoding if not using relative positions
        if self.src_pos is not None:
            src_emb = self.src_pos(src_emb)

        # Apply encoder layers
        memory = src_emb
        for layer in self.encoder:
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)

        # Apply final norm if using pre-norm
        if self.encoder_norm is not None:
            memory = self.encoder_norm(memory)

        return memory, src_key_padding_mask

    def decode(self, memory: torch.Tensor, src_padding_mask: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt_input) * self.scale

        # Apply positional encoding if not using relative positions
        if self.tgt_pos is not None:
            tgt_emb = self.tgt_pos(tgt_emb)

        tgt_mask = self.generate_square_subsequent_mask(tgt_input.size(1), tgt_input.device)
        tgt_key_padding_mask = tgt_input.eq(self.config.pad_idx)

        # Apply decoder layers
        output = tgt_emb
        for layer in self.decoder:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_padding_mask,
            )

        # Apply final norm if using pre-norm
        if self.decoder_norm is not None:
            output = self.decoder_norm(output)

        return self.generator(output)

    @staticmethod
    def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        mask = torch.full((size, size), float("-inf"), device=device)
        mask = mask.triu(1)
        return mask

    def greedy_decode(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor], max_len: Optional[int] = None) -> torch.Tensor:
        max_len = max_len or self.config.max_decode_len
        batch_size = src.size(0)
        with torch.no_grad():
            memory, src_padding_mask = self.encode(src)
            ys = torch.full((batch_size, 1), self.config.bos_idx, dtype=torch.long, device=src.device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=src.device)
            outputs = []
            for _ in range(max_len):
                logits = self.decode(memory, src_padding_mask, ys)
                next_token = logits[:, -1, :].argmax(dim=-1)
                outputs.append(next_token)
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                finished |= next_token.eq(self.config.eos_idx)
                if finished.all():
                    break
            if not outputs:
                return torch.zeros(batch_size, 0, dtype=torch.long, device=src.device)
            return torch.stack(outputs, dim=1)

    def beam_search(self, src: torch.Tensor, src_lengths: Optional[torch.Tensor], beam_size: int = 4, max_len: Optional[int] = None) -> torch.Tensor:
        if src.size(0) != 1:
            raise ValueError("Beam search currently supports batch_size=1")
        max_len = max_len or self.config.max_decode_len
        with torch.no_grad():
            memory, src_padding_mask = self.encode(src)
            beams = [
                {
                    "tokens": [self.config.bos_idx],
                    "log_prob": 0.0,
                }
            ]
            completed = []
            for _ in range(max_len):
                new_beams = []
                for beam in beams:
                    if beam["tokens"][-1] == self.config.eos_idx:
                        completed.append(beam)
                        continue
                    tgt_tokens = torch.tensor(beam["tokens"], dtype=torch.long, device=src.device).unsqueeze(0)
                    logits = self.decode(memory, src_padding_mask, tgt_tokens)
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)
                    for log_prob, token_id in zip(topk_log_probs[0], topk_indices[0]):
                        new_beams.append({
                            "tokens": beam["tokens"] + [token_id.item()],
                            "log_prob": beam["log_prob"] + log_prob.item(),
                        })
                beams = sorted(new_beams, key=lambda b: b["log_prob"], reverse=True)[:beam_size]
                if not beams:
                    break
            if not completed:
                completed = beams
            best = max(completed, key=lambda b: b["log_prob"] / max(len(b["tokens"]), 1))
            return torch.tensor(best["tokens"][1:], dtype=torch.long, device=src.device).unsqueeze(0)
