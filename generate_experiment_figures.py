import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Use default font (no Chinese font needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create output directory
output_dir = Path('experiments/results/figures')
output_dir.mkdir(parents=True, exist_ok=True)

# Experiment result data
rnn_results = {
    'attention': {
        'Dot-product': 0.2666,
        'Multiplicative': 0.2666,
        'Additive': 0.2666
    },
    'training': {
        'Teacher Forcing': 0.2666,
        'Free Running': 0.2666
    },
    'decoding': {
        'Greedy Search': 0.2666,
        'Beam Search': 0.2666
    }
}

transformer_results = {
    'position_encoding': {
        'Absolute': 0.3238,
        'Relative': 0.3238
    },
    'normalization': {
        'LayerNorm': 0.3238,
        'RMSNorm': 0.3238
    },
    'batch_size': {
        '16': 0.3238,
        '32': 0.3238
    },
    'learning_rate': {
        '0.0001': 0.3238,
        '0.0005': 0.3238
    },
    'model_size': {
        '256': 0.3238,
        '512': 0.3238
    }
}

mt5_results = {
    'Greedy': 7.6655930852934135,
    'Beam4': 8.636749482221076,
    'Beam8': 8.762931017068905
}

t5_results = {
    'Greedy': 3.0485,
    'Beam4': 3.0360,
    'Beam8': 3.0379
}

# Chart 1: RNN Attention Mechanism Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('RNN Model Experiment Results Comparison', fontsize=16, fontweight='bold')

# Attention mechanism
ax = axes[0]
attention_types = list(rnn_results['attention'].keys())
bleu_scores = list(rnn_results['attention'].values())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
bars = ax.bar(attention_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Attention Mechanism Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Training strategy
ax = axes[1]
training_types = list(rnn_results['training'].keys())
bleu_scores = list(rnn_results['training'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(training_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Training Strategy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Decoding strategy
ax = axes[2]
decoding_types = list(rnn_results['decoding'].keys())
bleu_scores = list(rnn_results['decoding'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(decoding_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Decoding Strategy Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.35)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'rnn_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rnn_comparison.pdf', bbox_inches='tight')
plt.close()

# Chart 2: Transformer Architecture Ablation Study
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Transformer Model Architecture Ablation Study', fontsize=16, fontweight='bold')

# Position encoding
ax = axes[0, 0]
pos_enc_types = list(transformer_results['position_encoding'].keys())
bleu_scores = list(transformer_results['position_encoding'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(pos_enc_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Position Encoding Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Normalization method
ax = axes[0, 1]
norm_types = list(transformer_results['normalization'].keys())
bleu_scores = list(transformer_results['normalization'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(norm_types, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Normalization Method Comparison', fontsize=14, fontweight='bold')
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Batch Size
ax = axes[0, 2]
batch_sizes = list(transformer_results['batch_size'].keys())
bleu_scores = list(transformer_results['batch_size'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(batch_sizes, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Batch Size Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Batch Size', fontsize=12)
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Learning Rate
ax = axes[1, 0]
lr_values = list(transformer_results['learning_rate'].keys())
bleu_scores = list(transformer_results['learning_rate'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(lr_values, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Learning Rate Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Learning Rate', fontsize=12)
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Model Size
ax = axes[1, 1]
model_sizes = list(transformer_results['model_size'].keys())
bleu_scores = list(transformer_results['model_size'].values())
colors = ['#1f77b4', '#ff7f0e']
bars = ax.bar(model_sizes, bleu_scores, color=colors, alpha=0.8)
ax.set_ylabel('BLEU Score', fontsize=12)
ax.set_title('Model Size Comparison', fontsize=14, fontweight='bold')
ax.set_xlabel('Hidden Size', fontsize=12)
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / 'transformer_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'transformer_ablation.pdf', bbox_inches='tight')
plt.close()

# Chart 3: Pretrained Model Decoding Strategy Comparison
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
ax.set_ylim(0, 14)
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
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig(output_dir / 'pretrained_decoding_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'pretrained_decoding_comparison.pdf', bbox_inches='tight')
plt.close()

# Chart 4: Model Architecture Comprehensive Comparison
fig, ax = plt.subplots(figsize=(14, 8))

# Model names and BLEU scores
model_names = ['RNN\n(Dot-product)', 'RNN\n(Multiplicative)', 'RNN\n(Additive)',
               'Transformer\n(Absolute)', 'Transformer\n(Relative)',
               'mT5-small\n(Greedy)', 'mT5-small\n(Beam4)', 'mT5-small\n(Beam8)',
               'T5-small\n(Greedy)', 'T5-small\n(Beam4)', 'T5-small\n(Beam8)']

bleu_scores = [0.2666, 0.2666, 0.2666, 0.3238, 0.3238, 7.6655930852934135, 8.636749482221076, 8.762931017068905, 3.0485, 3.0360, 3.0379]

# Set colors for different model types
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
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, score in zip(bars, bleu_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add legend
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

# Chart 5: Decoding Strategy Performance Comparison (All Models)
fig, ax = plt.subplots(figsize=(12, 10))

# Decoding strategy comparison data
decoding_comparison = {
    'Greedy': {
        'RNN': 11.8684,
        'Transformer': 4.0254,
        'mT5-small': 7.6655930852934135
    },
    'Beam Search': {
        'RNN': 9.7914,
        'Transformer': 4.0254,
        'mT5-small': 8.762931017068905
    }
}

models = ['RNN', 'Transformer', 'mT5-small']
x = np.arange(len(models))
width = 0.35

greedy_scores = [decoding_comparison['Greedy'][model] for model in models]
beam_scores = [decoding_comparison['Beam Search'][model] for model in models]

bars1 = ax.bar(x - width/2, greedy_scores, width, label='Greedy', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, beam_scores, width, label='Beam Search', color='#ff7f0e', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Decoding Strategy Performance Comparison by Model', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=12)
ax.set_ylim(0, 14)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'decoding_strategy_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'decoding_strategy_comparison.pdf', bbox_inches='tight')
plt.close()

# Chart 6: RNN vs Transformer Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Best Performance', 'Average Performance']
rnn_scores = [0.2666, 0.2666]
transformer_scores = [0.3238, 0.3238]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, rnn_scores, width, label='RNN', color='#1f77b4', alpha=0.8)
bars2 = ax.bar(x + width/2, transformer_scores, width, label='Transformer', color='#ff7f0e', alpha=0.8)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('RNN vs Transformer Performance Comparison', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12)
ax.legend(fontsize=12)
ax.set_ylim(0, 0.4)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add performance improvement annotation
improvement = (transformer_scores[0] - rnn_scores[0]) / rnn_scores[0] * 100
ax.annotate(f'Improvement: {improvement:.1f}%', xy=(0.5, 0.35), xytext=(0.5, 0.38),
            fontsize=12, ha='center', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3),
            arrowprops=dict(arrowstyle='->', lw=2))

plt.tight_layout()
plt.savefig(output_dir / 'rnn_vs_transformer.png', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'rnn_vs_transformer.pdf', bbox_inches='tight')
plt.close()

# Chart 7: Pretrained Model Performance Comparison
fig, ax = plt.subplots(figsize=(10, 6))

pretrained_models = ['mT5-small\n(Greedy)', 'mT5-small\n(Beam)', 'T5-small\n(Greedy)', 'T5-small\n(Beam)']
pretrained_scores = [7.6655930852934135, 8.636749482221076, 3.0485, 3.0360]

colors = ['#2ca02c', '#2ca02c', '#d62728', '#d62728']
bars = ax.bar(pretrained_models, pretrained_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)

ax.set_ylabel('BLEU Score', fontsize=14, fontweight='bold')
ax.set_title('Pretrained Model Performance Comparison', fontsize=16, fontweight='bold')
ax.set_ylim(0, 10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add value labels
for bar, score in zip(bars, pretrained_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ca02c', alpha=0.8, label='mT5-small'),
    Patch(facecolor='#d62728', alpha=0.8, label='T5-small')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

plt.tight_layout()
plt.savefig(output_dir / 'pretrained_comparison.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir / 'pretrained_comparison.pdf', bbox_inches='tight')
plt.close()

print(f"All charts have been generated and saved to {output_dir}")
print("Generated chart files:")
print("1. rnn_comparison.png/pdf - RNN Model Experiment Results Comparison")
print("2. transformer_ablation.png/pdf - Transformer Model Architecture Ablation Study")
print("3. pretrained_decoding_comparison.png/pdf - Pretrained Model Decoding Strategy Comparison")
print("4. model_architecture_comparison.png/pdf - Model Architecture Performance Comparison")
print("5. decoding_strategy_comparison.png/pdf - Decoding Strategy Performance Comparison by Model")
print("6. rnn_vs_transformer.png/pdf - RNN vs Transformer Performance Comparison")
print("7. pretrained_comparison.png/pdf - Pretrained Model Performance Comparison")
