# Hyperparameter Sensitivity Analysis Results\n\n## Experiment Comparison\n\n| experiment   |   best_bleu |   best_epoch |   final_bleu |   final_train_loss |   final_valid_loss |   total_epochs |   test_greedy_bleu |   test_beam4_bleu |   test_beam8_bleu |
|:-------------|------------:|-------------:|-------------:|-------------------:|-------------------:|---------------:|-------------------:|------------------:|------------------:|
| Batch 256    |      4.6424 |          493 |       4.3785 |             1.8993 |             6.6908 |            500 |             0.0000 |            0.0000 |            0.0000 |
| D_model 768  |      4.5429 |          404 |       4.1722 |             3.2263 |             6.7384 |            500 |             0.0000 |            0.0000 |            0.0000 |
| LR 1e-3      |      4.0723 |          468 |       3.5732 |             3.7730 |             6.6825 |            500 |             0.0000 |            0.0000 |            0.0000 |
| Baseline     |      4.0041 |          472 |       3.4898 |             3.7894 |             6.6711 |            500 |             0.0000 |            0.0000 |            0.0000 |\n\n## Summary

**Best Configuration:** Batch 256

- Best BLEU: 4.6424 (epoch 493)
- Final BLEU: 4.3785
- Final Valid Loss: 6.6908
- Test Set Greedy BLEU: 0.0000
- Test Set Beam4 BLEU: 0.0000
- Test Set Beam8 BLEU: 0.0000

## Key Findings\n\n1. **Batch Size Impact:** Increasing batch size from 128 to 256 improved BLEU by 0.6382 (+15.94%).\n\n2. **Learning Rate Impact:** Increasing learning rate from 5e-4 to 1e-3 improved BLEU by 0.0682 (+1.70%).\n\n3. **Model Dimension Impact:** Increasing d_model from 512 to 768 improved BLEU by 0.5388 (+13.46%).\n\n