# 中英翻译项目：完整实验执行指南

## 项目概览

本文档提供了完整的实验执行流程，涵盖所有未完成的实验（Step 6-9）。

---

## 实验概览

### 已完成（Steps 0-5）
- ✅ 数据处理和分词
- ✅ RNN模型实验（6个实验：GRU/LSTM × 3种attention）
- ✅ Transformer基线实现

### 待执行（Steps 6-9）
- ⏳ Transformer消融实验（4个实验）
- ⏳ 超参数敏感性分析（3个sweep实验）
- ⏳ T5微调
- ⏳ 统一报告和可视化

---

## 实验执行命令

### 1. Transformer消融实验（Step 6）

**实验矩阵**: 2(位置编码) × 2(归一化) = 4个实验

```bash
# 在大服务器上运行（推荐）
cd /path/to/zh-en-nmt-midterm

# 单GPU运行
GPUS=1 bash scripts/run_transformer_ablation_matrix.sh

# 或多GPU运行（如果有2个GPU）
GPUS=2 bash scripts/run_transformer_ablation_matrix.sh

# 后台运行并记录日志
nohup bash scripts/run_transformer_ablation_matrix.sh > experiments/logs/ablation_run.log 2>&1 &

# 实验完成后生成分析
python scripts/analyze_transformer_ablations.py
```

**预期输出**:
- `experiments/results/transformer_abs_ln_metrics.csv`
- `experiments/results/transformer_abs_rms_metrics.csv`
- `experiments/results/transformer_rel_ln_metrics.csv`
- `experiments/results/transformer_rel_rms_metrics.csv`
- `experiments/results/transformer_ablation_comparison.csv`
- `experiments/results/transformer_ablation_comparison.md`
- `experiments/results/figures/transformer_ablation_heatmap.png`
- `experiments/results/figures/transformer_ablation_curves.png`

**估计时间**: 8-12小时（4个实验 × 50 epochs）

---

### 2. 超参数敏感性分析（Step 7）

**Sweep矩阵**: 3个sweep + 1个baseline = 4个实验

```bash
# 运行超参数sweep实验
bash scripts/run_hyperparam_sweep.sh

# 或后台运行
nohup bash scripts/run_hyperparam_sweep.sh > experiments/logs/sweep_run.log 2>&1 &

# 实验完成后生成分析
python scripts/analyze_hyperparam_sweeps.py
```

**Sweep配置**:
1. Batch size sweep: 128 (baseline) vs 256
2. Learning rate sweep: 5e-4 (baseline) vs 1e-3
3. Model dimension sweep: d=512 (baseline) vs d=768

**预期输出**:
- `experiments/results/sweep_batch_256_metrics.csv`
- `experiments/results/sweep_lr_1e3_metrics.csv`
- `experiments/results/sweep_d768_metrics.csv`
- `experiments/results/hyperparam_sweep_comparison.csv`
- `experiments/results/hyperparam_sweep_comparison.md`
- `experiments/results/figures/hyperparam_sweep_combined.png`

**估计时间**: 6-9小时（3个实验，baseline复用）

---

### 3. T5微调（Step 8）

**模型**: t5-small (60M参数)

```bash
# 训练T5模型
bash scripts/train_t5.sh

# 或后台运行
nohup bash scripts/train_t5.sh > experiments/logs/t5_train.log 2>&1 &

# 训练完成后进行推理
python src/t5/infer.py \
    --model_path experiments/logs/t5_small_best \
    --input data/processed/test.jsonl \
    --output experiments/results/t5_predictions.jsonl \
    --beam_size 4
```

**预期输出**:
- `experiments/logs/t5_small_best/` (模型目录)
- `experiments/results/t5_small_metrics.json`
- `experiments/results/t5_predictions.jsonl`

**估计时间**: 2-3小时（10 epochs）

---

### 4. 统一推理（所有模型）

**使用统一推理脚本**:

```bash
# RNN推理
python inference.py \
    --model_type rnn \
    --checkpoint experiments/logs/best_rnn_model.pt \
    --input data/processed/test.jsonl \
    --output results_rnn.jsonl \
    --beam_size 4

# Transformer推理
python inference.py \
    --model_type transformer \
    --checkpoint experiments/logs/transformer_abs_ln_best.pt \
    --input data/processed/test.jsonl \
    --output results_transformer.jsonl \
    --beam_size 4

# T5推理
python inference.py \
    --model_type t5 \
    --checkpoint experiments/logs/t5_small_best \
    --input data/processed/test.jsonl \
    --output results_t5.jsonl \
    --beam_size 4
```

---

### 5. 生成统一报告和可视化（Step 9）

```bash
# 1. 生成主对比表（汇总所有实验结果）
python scripts/generate_master_comparison.py

# 2. 生成所有可视化图表
python scripts/generate_all_visualizations.py

# 3. 生成RNN分析（如果尚未运行）
python scripts/analyze_rnn_results.py

# 4. 确认所有分析脚本已运行
python scripts/analyze_transformer_ablations.py
python scripts/analyze_hyperparam_sweeps.py
```

**预期输出**:
- `experiments/results/master_comparison.csv` (主对比表)
- `experiments/results/master_comparison.md` (Markdown格式)
- `experiments/results/figures/training_curves_all.png`
- `experiments/results/figures/bleu_comparison_bar.png`
- `experiments/results/figures/category_comparison.png`
- `experiments/results/figures/param_efficiency.png`
- `experiments/results/figures/epoch_efficiency.png`

---

## 完整执行流程（推荐顺序）

### 阶段1: Transformer消融实验（必须先完成）
```bash
# 1. 运行消融实验（~8-12小时）
nohup bash scripts/run_transformer_ablation_matrix.sh > experiments/logs/ablation_run.log 2>&1 &

# 2. 监控进度
tail -f experiments/logs/ablation_run.log

# 3. 完成后生成分析
python scripts/analyze_transformer_ablations.py
```

### 阶段2: 超参数敏感性分析
```bash
# 1. 运行sweep实验（~6-9小时）
nohup bash scripts/run_hyperparam_sweep.sh > experiments/logs/sweep_run.log 2>&1 &

# 2. 监控进度
tail -f experiments/logs/sweep_run.log

# 3. 完成后生成分析
python scripts/analyze_hyperparam_sweeps.py
```

### 阶段3: T5微调
```bash
# 1. 训练T5（~2-3小时）
nohup bash scripts/train_t5.sh > experiments/logs/t5_train.log 2>&1 &

# 2. 监控进度
tail -f experiments/logs/t5_train.log

# 3. 完成后进行推理
python src/t5/infer.py \
    --model_path experiments/logs/t5_small_best \
    --input data/processed/test.jsonl \
    --output experiments/results/t5_predictions.jsonl \
    --beam_size 4
```

### 阶段4: 统一报告生成
```bash
# 1. 生成主对比表
python scripts/generate_master_comparison.py

# 2. 生成所有可视化
python scripts/generate_all_visualizations.py

# 3. 确认RNN分析已完成
python scripts/analyze_rnn_results.py
```

---

## 监控和检查

### 检查GPU使用情况
```bash
nvidia-smi
watch -n 1 nvidia-smi  # 实时监控
```

### 检查训练进程
```bash
ps aux | grep "python.*train.py"
```

### 检查生成的文件
```bash
# 检查metrics文件
ls -lh experiments/results/*.csv
ls -lh experiments/results/*.json

# 检查模型checkpoint
ls -lh experiments/logs/*.pt

# 检查可视化图表
ls -lh experiments/results/figures/*.png
```

---

## 预期总时间

- **Transformer消融实验**: 8-12小时
- **超参数sweep**: 6-9小时
- **T5微调**: 2-3小时
- **推理和分析**: 1-2小时

**总计**: 约17-26小时（顺序执行）

如果有多台服务器或GPU，可以并行运行不同阶段以节省时间。

---

## 最终检查清单

完成所有实验后，确认以下文件都已生成：

### Metrics文件
- [ ] `experiments/results/transformer_abs_ln_metrics.csv`
- [ ] `experiments/results/transformer_abs_rms_metrics.csv`
- [ ] `experiments/results/transformer_rel_ln_metrics.csv`
- [ ] `experiments/results/transformer_rel_rms_metrics.csv`
- [ ] `experiments/results/sweep_batch_256_metrics.csv`
- [ ] `experiments/results/sweep_lr_1e3_metrics.csv`
- [ ] `experiments/results/sweep_d768_metrics.csv`
- [ ] `experiments/results/t5_small_metrics.json`

### 对比表和分析
- [ ] `experiments/results/master_comparison.csv`
- [ ] `experiments/results/master_comparison.md`
- [ ] `experiments/results/transformer_ablation_comparison.csv`
- [ ] `experiments/results/transformer_ablation_comparison.md`
- [ ] `experiments/results/hyperparam_sweep_comparison.csv`
- [ ] `experiments/results/hyperparam_sweep_comparison.md`

### 可视化图表（至少11个）
- [ ] `experiments/results/figures/training_curves_all.png`
- [ ] `experiments/results/figures/bleu_comparison_bar.png`
- [ ] `experiments/results/figures/category_comparison.png`
- [ ] `experiments/results/figures/param_efficiency.png`
- [ ] `experiments/results/figures/epoch_efficiency.png`
- [ ] `experiments/results/figures/transformer_ablation_heatmap.png`
- [ ] `experiments/results/figures/transformer_ablation_curves.png`
- [ ] `experiments/results/figures/hyperparam_sweep_combined.png`
- [ ] `experiments/results/figures/rnn_*` (RNN分析图表)

---

## 故障排除

### 如果训练中断
```bash
# 检查最后的checkpoint
ls -lt experiments/logs/*.pt | head -5

# 查看日志确认错误
tail -100 experiments/logs/ablation_run.log
```

### 如果内存不足
```bash
# 减小batch size（修改config文件）
# 或使用gradient accumulation
```

### 如果GPU资源不足
```bash
# 使用单GPU模式
GPUS=1 bash scripts/run_transformer_ablation_matrix.sh

# 或逐个运行实验
python src/transformer/train.py --config experiments/configs/transformer_abs_ln.yaml
```

---

## 联系和支持

如有问题，请检查：
1. 日志文件：`experiments/logs/*.log`
2. 配置文件：`experiments/configs/*.yaml`
3. 环境依赖：确保安装了所有必需的库

---

**祝实验顺利！** 🚀
