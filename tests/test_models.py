import unittest

import torch

from src.rnn.model import Seq2Seq, Seq2SeqConfig
from src.transformer.model import TransformerConfig, TransformerSeq2Seq


class TestModels(unittest.TestCase):
    def test_rnn_forward(self):
        config = Seq2SeqConfig(
            vocab_size=128,
            embed_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            pad_idx=0,
            bos_idx=2,
            eos_idx=3,
            max_decode_len=20,
        )
        model = Seq2Seq(config)
        src_lengths = torch.randint(5, 12, (8,))
        max_src = int(src_lengths.max().item())
        src = torch.zeros(8, max_src, dtype=torch.long)
        for i, length in enumerate(src_lengths):
            src[i, : length] = torch.randint(4, 100, (length,))
        tgt = torch.randint(4, 100, (8, 10))
        outputs = model(src, src_lengths, tgt)
        self.assertEqual(outputs.shape[0], 8)
        self.assertEqual(outputs.shape[1], 10)

    def test_transformer_forward(self):
        config = TransformerConfig(
            vocab_size=128,
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=64,
            dropout=0.1,
            max_position_embeddings=64,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            max_decode_len=32,
        )
        model = TransformerSeq2Seq(config)
        batch = 4
        src = torch.randint(3, 120, (batch, 12))
        tgt = torch.randint(3, 120, (batch, 10))
        tgt[:, 0] = config.bos_idx
        outputs = model(src, None, tgt)
        self.assertEqual(outputs.shape, (batch, tgt.size(1) - 1, config.vocab_size))

    def test_transformer_decode(self):
        config = TransformerConfig(
            vocab_size=64,
            d_model=16,
            num_heads=2,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.1,
            max_position_embeddings=32,
            pad_idx=0,
            bos_idx=1,
            eos_idx=2,
            max_decode_len=16,
        )
        model = TransformerSeq2Seq(config)
        src = torch.randint(3, 32, (2, 6))
        decoded = model.greedy_decode(src, None, max_len=10)
        self.assertEqual(decoded.ndim, 2)


if __name__ == "__main__":
    unittest.main()