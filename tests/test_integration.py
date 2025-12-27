"""Integration tests for new Transformer features (RMSNorm, relative position bias)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pytest

from src.common.layers import RMSNorm
from src.transformer.model import (
    TransformerConfig,
    TransformerSeq2Seq,
    RelativePositionalBias,
    CustomTransformerEncoderLayer,
    CustomTransformerDecoderLayer,
)


def test_rmsnorm_forward():
    """Test RMSNorm forward pass."""
    batch_size, seq_len, d_model = 2, 10, 512
    rms = RMSNorm(dim=d_model)
    x = torch.randn(batch_size, seq_len, d_model)

    output = rms(x)

    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    # Check that output is normalized (approximately unit RMS)
    rms_vals = torch.sqrt(torch.mean(output ** 2, dim=-1))
    assert torch.allclose(rms_vals, torch.ones_like(rms_vals), atol=0.1), \
        f"Output not properly normalized, RMS range: [{rms_vals.min():.3f}, {rms_vals.max():.3f}]"
    print("✓ RMSNorm forward pass test passed")


def test_relative_positional_bias():
    """Test T5-style relative positional bias computation."""
    num_heads, qlen, klen = 8, 10, 15
    rel_bias = RelativePositionalBias(num_heads=num_heads, max_distance=128, bidirectional=True)

    device = torch.device('cpu')
    bias = rel_bias.compute_bias(qlen, klen, device)

    expected_shape = (num_heads, qlen, klen)
    assert bias.shape == expected_shape, f"Expected shape {expected_shape}, got {bias.shape}"
    assert not torch.isnan(bias).any(), "Bias contains NaN values"
    assert not torch.isinf(bias).any(), "Bias contains Inf values"
    print(f"✓ RelativePositionalBias test passed, bias range: [{bias.min():.3f}, {bias.max():.3f}]")


def test_custom_encoder_layer_with_layernorm():
    """Test custom encoder layer with LayerNorm."""
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8

    layer = CustomTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        norm_type='layernorm',
        norm_first=True,
    )

    src = torch.randn(batch_size, seq_len, d_model)
    output = layer(src)

    assert output.shape == src.shape, f"Expected shape {src.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ CustomTransformerEncoderLayer with LayerNorm test passed")


def test_custom_encoder_layer_with_rmsnorm():
    """Test custom encoder layer with RMSNorm."""
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8

    layer = CustomTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        norm_type='rmsnorm',
        norm_first=True,
    )

    src = torch.randn(batch_size, seq_len, d_model)
    output = layer(src)

    assert output.shape == src.shape, f"Expected shape {src.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ CustomTransformerEncoderLayer with RMSNorm test passed")


def test_custom_encoder_layer_with_relative_bias():
    """Test custom encoder layer with relative position bias."""
    batch_size, seq_len, d_model, num_heads = 2, 10, 512, 8

    rel_bias = RelativePositionalBias(num_heads=num_heads, max_distance=128, bidirectional=True)
    layer = CustomTransformerEncoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        norm_type='layernorm',
        norm_first=True,
        relative_bias=rel_bias,
    )

    src = torch.randn(batch_size, seq_len, d_model)
    output = layer(src)

    assert output.shape == src.shape, f"Expected shape {src.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ CustomTransformerEncoderLayer with relative bias test passed")


def test_custom_decoder_layer():
    """Test custom decoder layer."""
    batch_size, tgt_len, src_len, d_model, num_heads = 2, 10, 15, 512, 8

    layer = CustomTransformerDecoderLayer(
        d_model=d_model,
        nhead=num_heads,
        dim_feedforward=2048,
        dropout=0.1,
        norm_type='rmsnorm',
        norm_first=True,
    )

    tgt = torch.randn(batch_size, tgt_len, d_model)
    memory = torch.randn(batch_size, src_len, d_model)
    output = layer(tgt, memory)

    assert output.shape == tgt.shape, f"Expected shape {tgt.shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ CustomTransformerDecoderLayer test passed")


def test_transformer_with_rmsnorm():
    """Test full Transformer model with RMSNorm."""
    vocab_size = 1000
    batch_size, src_len, tgt_len = 2, 10, 15

    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        norm_type='rmsnorm',
        positional_encoding='sinusoidal',
    )

    model = TransformerSeq2Seq(config)

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(src, None, tgt)

    expected_shape = (batch_size, tgt_len - 1, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Transformer with RMSNorm test passed")


def test_transformer_with_relative_position():
    """Test full Transformer model with relative positional encoding."""
    vocab_size = 1000
    batch_size, src_len, tgt_len = 2, 10, 15

    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        norm_type='layernorm',
        positional_encoding='relative',
    )

    model = TransformerSeq2Seq(config)

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(src, None, tgt)

    expected_shape = (batch_size, tgt_len - 1, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Transformer with relative positional encoding test passed")


def test_transformer_with_learned_position():
    """Test full Transformer model with learned positional encoding."""
    vocab_size = 1000
    batch_size, src_len, tgt_len = 2, 10, 15

    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=256,
        num_heads=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
        norm_type='layernorm',
        positional_encoding='learned',
    )

    model = TransformerSeq2Seq(config)

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    output = model(src, None, tgt)

    expected_shape = (batch_size, tgt_len - 1, vocab_size)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    print("✓ Transformer with learned positional encoding test passed")


def test_transformer_ablation_matrix():
    """Test all 4 ablation configurations."""
    vocab_size = 1000
    batch_size, src_len, tgt_len = 2, 10, 15

    configs = [
        ('sinusoidal', 'layernorm'),
        ('sinusoidal', 'rmsnorm'),
        ('relative', 'layernorm'),
        ('relative', 'rmsnorm'),
    ]

    for pos_enc, norm_type in configs:
        config = TransformerConfig(
            vocab_size=vocab_size,
            d_model=128,  # Smaller for faster testing
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=512,
            dropout=0.1,
            norm_type=norm_type,
            positional_encoding=pos_enc,
        )

        model = TransformerSeq2Seq(config)

        src = torch.randint(0, vocab_size, (batch_size, src_len))
        tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

        output = model(src, None, tgt)

        expected_shape = (batch_size, tgt_len - 1, vocab_size)
        assert output.shape == expected_shape, \
            f"Config ({pos_enc}, {norm_type}): Expected shape {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), \
            f"Config ({pos_enc}, {norm_type}): Output contains NaN"

        print(f"✓ Transformer ({pos_enc}, {norm_type}) test passed")


if __name__ == "__main__":
    print("Running integration tests for new Transformer features...\n")

    test_rmsnorm_forward()
    test_relative_positional_bias()
    test_custom_encoder_layer_with_layernorm()
    test_custom_encoder_layer_with_rmsnorm()
    test_custom_encoder_layer_with_relative_bias()
    test_custom_decoder_layer()
    test_transformer_with_rmsnorm()
    test_transformer_with_relative_position()
    test_transformer_with_learned_position()
    test_transformer_ablation_matrix()

    print("\n✅ All integration tests passed!")
