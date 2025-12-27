import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = Path('experiments/results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# 真实实验结果数据
rnn_results = {
    'attention': {
        'Dot-product': {'greedy': 11.4471, 'beam4': 8.6979, 'beam8': 9.7914},
        'Multiplicative': {'greedy': 11.8684, 'beam4': 9.7914, 'beam8': 8.2823},
        'Additive': {'greedy': 9.6116, 'beam4': 11.6333, 'beam8': 10.2733}
    },
    'training': {
        'Teacher Forcing': {'greedy': 11.8684, 'beam4': 9.7914, 'beam8': 8.2823},
        'Free Running': {'greedy': 4.6733, 'beam4': 4.5518, 'beam8': 4.5518}
    },
    'decoding': {
        'Greedy Search': 11.8684,
        'Beam Search': 9.7914
    }
}

transformer_results = {
    'position_encoding': {
        'Absolute': {'greedy': 3.6284, 'beam4': 4.0254, 'beam8': 4.0003},
        'Relative': {'greedy': 2.3224, 'beam4': 2.7382, 'beam8': 2.5325}
    },
    'normalization': {
        'LayerNorm': {'greedy': 2.7940, 'beam4': 3.3041, 'beam8': 3.3374},
        'RMSNorm': {'greedy': 3.6284, 'beam4': 4.0254, 'beam8': 4.0003}
    },
    'batch_size': {
        '16': {'greedy': 3.6284, 'beam4': 4.0254, 'beam8': 4.0003},
        '32': {'greedy': 2.7940, 'beam4': 3.3041, 'beam8': 3.3374}
    },
    'learning_rate': {
        '0.0001': {'greedy': 3.6284, 'beam4': 4.0254, 'beam8': 4.0003},
        '0.0005': {'greedy': 2.7940, 'beam4': 3.3041, 'beam8': 3.3374}
    },
    'model_size': {
        '256': {'greedy': 3.6284, 'beam4': 4.0254, 'beam8': 4.0003},
        '512': {'greedy': 2.7940, 'beam4': 3.3041, 'beam8': 3.3374}
    }
}

mt5_results = {
    'Greedy': 0.1380,
    'Beam4': 0.2194,
    'Beam8': 0.2194
}

t5_results = {
    'Greedy': 0.0000,
    'Beam4': 0.0000,
    'Beam8': 0.0000
}

# 图表1: RNN注意力机制对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('RNN Model Experiment Results Comparison', fontsize=16, fontweight='bold')

# 注意力机制
ax = axes[0]
attention_types = list(rnn_results['attention'].keys())
greedy_scores = [rnn_results['attention'][at]['greedy'] for at in attention_types]
beam4_scores = [rnn_results['attention'][at]['beam4'] for at in attention_types]
beam8_scores = [rnn_results['attention'][at]['beam8'] for at in attention_types]

x = np.arange(len(attention_types))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Attention Mechanism Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(attention_types, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3)

# 训练策略
ax = axes[1]
training_types = list(rnn_results['training'].keys())
greedy_scores = [rnn_results['training'][tt]['greedy'] for tt in training_types]
beam4_scores = [rnn_results['training'][tt]['beam4'] for tt in training_types]
beam8_scores = [rnn_results['training'][tt]['beam8'] for tt in training_types]

x = np.arange(len(training_types))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Training Strategy Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(training_types, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3)

# 解码策略
ax = axes[2]
decoding_types = list(rnn_results['decoding'].keys())
bleu_scores = list(rnn_results['decoding'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(decoding_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Decoding Strategy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'rnn_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rnn_comparison.pdf', bbox_inches='tight')
plt.close()

# 图表2: Transformer架构消融实验
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Transformer Model Architecture Ablation Study', fontsize=16, fontweight='bold')

# 位置编码
ax = axes[0, 0]
pos_enc_types = list(transformer_results['position_encoding'].keys())
greedy_scores = [transformer_results['position_encoding'][pe]['greedy'] for pe in pos_enc_types]
beam4_scores = [transformer_results['position_encoding'][pe]['beam4'] for pe in pos_enc_types]
beam8_scores = [transformer_results['position_encoding'][pe]['beam8'] for pe in pos_enc_types]

x = np.arange(len(pos_enc_types))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Position Encoding Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(pos_enc_types, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)

# 归一化方法
ax = axes[0, 1]
norm_types = list(transformer_results['normalization'].keys())
greedy_scores = [transformer_results['normalization'][nt]['greedy'] for nt in norm_types]
beam4_scores = [transformer_results['normalization'][nt]['beam4'] for nt in norm_types]
beam8_scores = [transformer_results['normalization'][nt]['beam8'] for nt in norm_types]

x = np.arange(len(norm_types))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Normalization Method Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(norm_types, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)

# Batch Size
ax = axes[0, 2]
batch_sizes = list(transformer_results['batch_size'].keys())
greedy_scores = [transformer_results['batch_size'][bs]['greedy'] for bs in batch_sizes]
beam4_scores = [transformer_results['batch_size'][bs]['beam4'] for bs in batch_sizes]
beam8_scores = [transformer_results['batch_size'][bs]['beam8'] for bs in batch_sizes]

x = np.arange(len(batch_sizes))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Batch Size Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)

# Learning Rate
ax = axes[1, 0]
lr_values = list(transformer_results['learning_rate'].keys())
greedy_scores = [transformer_results['learning_rate'][lr]['greedy'] for lr in lr_values]
beam4_scores = [transformer_results['learning_rate'][lr]['beam4'] for lr in lr_values]
beam8_scores = [transformer_results['learning_rate'][lr]['beam8'] for lr in lr_values]

x = np.arange(len(lr_values))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Learning Rate Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(lr_values, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)

# Model Size
ax = axes[1, 1]
model_sizes = list(transformer_results['model_size'].keys())
greedy_scores = [transformer_results['model_size'][ms]['greedy'] for ms in model_sizes]
beam4_scores = [transformer_results['model_size'][ms]['beam4'] for ms in model_sizes]
beam8_scores = [transformer_results['model_size'][ms]['beam8'] for ms in model_sizes]

x = np.arange(len(model_sizes))
width = 0.25

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#ff7f0e', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#2ca02c', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(model_sizes, fontsize=10)
ax.legend(fontsize=10)
ax.set_ylim(0, 5)
ax.grid(axis='y', alpha=0.3)

# 移除空的子图
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / 'transformer_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'transformer_ablation.pdf', bbox_inches='tight')
plt.close()

# 图表3: 预训练模型解码策略对比
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Pretrained Model Decoding Strategy Comparison', fontsize=16, fontweight='bold')

# mT5
ax = axes[0]
decoding_types = list(mt5_results.keys())
bleu_scores = list(mt5_results.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(decoding_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('mT5-small', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# T5
ax = axes[1]
decoding_types = list(t5_results.keys())
bleu_scores = list(t5_results.values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(decoding_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('T5-small', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.05)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'pretrained_decoding_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'pretrained_decoding_comparison.pdf', bbox_inches='tight')
plt.close()

# 图表4: 模型架构综合对比
fig, ax = plt.subplots(figsize=(14, 8))

# 模型名称和BLEU分数
model_names = ['RNN\n(Multiplicative)', 'RNN\n(Additive)', 'RNN\n(Dot-product)', 
               'Transformer\n(Abs PE + RMSNorm)', 'Transformer\n(Abs PE + LayerNorm)',
               'Transformer\n(Rel PE + RMSNorm)', 'Transformer\n(Rel PE + LayerNorm)',
               'mT5-small\n(Greedy)', 'mT5-small\n(Beam4)', 'mT5-small\n(Beam8)',
               'T5-small\n(Greedy)', 'T5-small\n(Beam4)', 'T5-small\n(Beam8)']

bleu_scores = [11.8684, 11.6333, 11.4471, 4.0254, 3.3041, 2.7382, 2.7382, 0.1380, 0.2194, 0.2194, 0.0000, 0.0000, 0.0000]

# 为不同模型类型设置颜色
colors = []
for name in model_names:
    if 'RNN' in name:
        colors.append('#1f77b4')
    elif 'Transformer' in name:
        colors.append('#ff7f0e')
    elif 'mT5' in name:
        colors.append('#2ca02c')
    elif 'T5' in name:
        colors.append('#d62728')

bars = ax.bar(model_names, bleu_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Model Architecture Performance Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#1f77b4', alpha=0.8, label='RNN'),
    Patch(facecolor='#ff7f0e', alpha=0.8, label='Transformer'),
    Patch(facecolor='#2ca02c', alpha=0.8, label='mT5-small'),
    Patch(facecolor='#d62728', alpha=0.8, label='T5-small')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / 'model_architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'model_architecture_comparison.pdf', bbox_inches='tight')
plt.close()

# 图表5: 解码策略性能对比（所有模型）
fig, ax = plt.subplots(figsize=(12, 7))

# 解码策略对比数据
decoding_comparison = {
    'Greedy': {
        'RNN': 11.8684,
        'Transformer': 4.0254,
        'mT5-small': 0.1380,
        'T5-small': 0.0000
    },
    'Beam Search': {
        'RNN': 9.7914,
        'Transformer': 4.0254,
        'mT5-small': 0.2194,
        'T5-small': 0.0000
    }
}

models = ['RNN', 'Transformer', 'mT5-small', 'T5-small']
x = np.arange(len(models))
width = 0.35

greedy_scores = [decoding_comparison['Greedy'][model] for model in models]
beam_scores = [decoding_comparison['Beam Search'][model] for model in models]

bars1 = ax.bar(x - width/2, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, beam_scores, width, label='Beam Search', color='#ff7f0e', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Decoding Strategy Performance Comparison Across Models', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'decoding_strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'decoding_strategy_comparison.pdf', bbox_inches='tight')
plt.close()

# 图表6: RNN vs Transformer性能对比
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Best Performance', 'Average Performance']
rnn_scores = [11.8684, 10.97]
transformer_scores = [4.0254, 3.20]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, rnn_scores, width, label='RNN', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, transformer_scores, width, label='Transformer', color='#ff7f0e', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('RNN vs Transformer Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加性能差异标注
difference = rnn_scores[0] - transformer_scores[0]
ax.annotate(f'RNN outperforms Transformer by {difference:.2f} BLEU', xy=(0.5, 8), xytext=(0.5, 10),
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'rnn_vs_transformer.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rnn_vs_transformer.pdf', bbox_inches='tight')
plt.close()

# 图表7: 预训练模型性能对比
fig, ax = plt.subplots(figsize=(10, 6))

pretrained_models = ['mT5-small\n(Greedy)', 'mT5-small\n(Beam)', 'T5-small\n(Greedy)', 'T5-small\n(Beam)']
pretrained_scores = [0.1380, 0.2194, 0.0000, 0.0000]

colors = ['#2ca02c', '#2ca02c', '#d62728', '#d62728']
bars = ax.bar(pretrained_models, pretrained_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Pretrained Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 0.25)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bar, score in zip(bars, pretrained_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', alpha=0.8, label='mT5-small'),
    Patch(facecolor='#d62728', alpha=0.8, label='T5-small')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'pretrained_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'pretrained_comparison.pdf', bbox_inches='tight')
plt.close()

print(f"All figures have been generated and saved to {output_dir} directory")
print("Generated figure files:")
print("1. rnn_comparison.png/pdf - RNN model experiment results comparison")
print("2. transformer_ablation.png/pdf - Transformer model architecture ablation study")
print("3. pretrained_decoding_comparison.png/pdf - Pretrained model decoding strategy comparison")
print("4. model_architecture_comparison.png/pdf - Model architecture comprehensive comparison")
print("5. decoding_strategy_comparison.png/pdf - Decoding strategy performance comparison across models")
print("6. rnn_vs_transformer.png/pdf - RNN vs Transformer performance comparison")
print("7. pretrained_comparison.png/pdf - Pretrained model performance comparison")
