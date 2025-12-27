# 分析脚本使用指南

## 更新内容

已更新 `analyze_transformer_ablations.py` 和 `analyze_hyperparam_sweeps.py`，使其支持在测试集/验证集上运行推理，类似于 `analyze_rnn_results.py`。

## 使用方法

### 1. analyze_transformer_ablations.py

**基本用法**（只分析训练指标，不运行推理）：
```bash
python scripts/analyze_transformer_ablations.py --skip_infer
```

**在验证集上运行推理**（默认）：
```bash
python scripts/analyze_transformer_ablations.py --split valid
```

**在测试集上运行推理**：
```bash
python scripts/analyze_transformer_ablations.py --split test
```

**完整参数**：
```bash
python scripts/analyze_transformer_ablations.py \
    --results_dir experiments/results \
    --output_dir experiments/results/figures \
    --split test \
    --skip_infer  # 可选，跳过推理
```

### 2. analyze_hyperparam_sweeps.py

需要手动更新该脚本，添加类似的 `--split` 和 `--skip_infer` 参数到 main 函数。

参考 `analyze_transformer_ablations.py` 的实现：
1. 在 main() 中添加参数解析
2. 定义checkpoints字典
3. 调用 generate_comparison_table 时传递额外参数
4. 更新 generate_comparison_table 函数签名，添加推理逻辑

### 3. analyze_rnn_results.py (已支持)

```bash
# 在验证集上运行（默认）
python scripts/analyze_rnn_results.py --split valid

# 在测试集上运行
python scripts/analyze_rnn_results.py --split test

# 跳过推理
python scripts/analyze_rnn_results.py --skip_infer
```

## 输出内容

运行推理后，对比表将包含额外的列：
- `{split}_greedy_bleu`: Greedy解码的BLEU分数
- `{split}_beam4_bleu`: Beam size=4的BLEU分数
- `{split}_beam8_bleu`: Beam size=8的BLEU分数

其中 `{split}` 为 `valid` 或 `test`。

## 注意事项

1. **推理时间**: 每个checkpoint的推理可能需要数分钟，4个实验 × 3种beam size = 12次推理
2. **Checkpoint路径**: 确保checkpoints存在于期望的路径
3. **内存**: Beam search需要更多内存，特别是beam size=8时

## 示例工作流

```bash
# 1. 运行消融实验
bash scripts/run_transformer_ablation_matrix.sh

# 2. 仅分析训练指标（快速）
python scripts/analyze_transformer_ablations.py --skip_infer

# 3. 在测试集上运行完整评估（慢）
python scripts/analyze_transformer_ablations.py --split test

# 4. 查看结果
cat experiments/results/transformer_ablation_comparison.md
```

## 修改 analyze_hyperparam_sweeps.py 的TODO

如需完整支持，请参考 `analyze_transformer_ablations.py` 的以下修改：

1. **Line 27**: 添加 `import subprocess`
2. **Line 40-89**: 添加 `run_inference()` 函数
3. **Line 206**: 更新函数签名为 `def generate_comparison_table(experiments: dict, checkpoints: dict, split: str, skip_infer: bool, output_csv: Path, output_md: Path):`
4. **Line 243-255**: 在 generate_comparison_table 中添加推理逻辑
5. **Line 297-300**: 在 main() 中添加 `--split` 和 `--skip_infer` 参数
6. **Line 321-327**: 在 main() 中定义 checkpoints 字典
7. **Line 335-342**: 更新 generate_comparison_table 调用，传递新参数
