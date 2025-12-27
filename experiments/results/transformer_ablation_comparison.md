# Transformer Ablation Study Results

## 2×2 Ablation Matrix: Positional Encoding × Normalization

| experiment              | positional_encoding   | norm_type   |   best_bleu |   best_epoch |   final_bleu |   final_train_loss |   final_valid_loss |   total_epochs |   test_greedy_bleu |   test_beam4_bleu |   test_beam8_bleu |
|:------------------------|:----------------------|:------------|------------:|-------------:|-------------:|-------------------:|-------------------:|---------------:|-------------------:|------------------:|------------------:|
| Absolute PE + RMSNorm   | Absolute PE           | RMSNorm     |      4.4334 |          496 |       4.1167 |             3.7845 |             6.7215 |            500 |             3.6284 |            4.0254 |            4.0003 |
| Absolute PE + LayerNorm | Absolute PE           | LayerNorm   |      4.0041 |          472 |       3.4898 |             3.7894 |             6.6711 |            500 |             2.7940 |            3.3041 |            3.3374 |
| Relative PE + RMSNorm   | Relative PE           | RMSNorm     |      3.9200 |          500 |       3.9200 |             3.3112 |             7.3415 |            500 |             2.8151 |            2.6100 |            2.7912 |
| Relative PE + LayerNorm | Relative PE           | LayerNorm   |      3.2314 |          467 |       2.8012 |             3.3613 |             7.5133 |            500 |             2.3224 |            2.7382 |            2.5325 |

## Summary

**Best Configuration:** Absolute PE + RMSNorm

- Best Valid BLEU: 4.4334 (epoch 496)
- Final BLEU: 4.1167
- Final Valid Loss: 6.7215
- Test Set Greedy BLEU: 3.6284
- Test Set Beam4 BLEU: 4.0254
- Test Set Beam8 BLEU: 4.0003

## Key Findings

1. **Positional Encoding:**
   - Absolute (sinusoidal) average BLEU: 4.2188
   - Relative (T5-style) average BLEU: 3.5757
   - Winner: Absolute

2. **Normalization:**
   - LayerNorm average BLEU: 3.6178
   - RMSNorm average BLEU: 4.1767
   - Winner: RMSNorm

