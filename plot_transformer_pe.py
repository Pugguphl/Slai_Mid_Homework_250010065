import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

position_encodings = ['Absolute', 'Relative']
greedy_scores = [3.6284, 2.3224]
beam4_scores = [4.0254, 2.7382]
beam8_scores = [4.0003, 2.5325]

x = np.arange(len(position_encodings))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width, greedy_scores, width, label='Greedy', color='#2ecc71', alpha=0.8)
bars2 = ax.bar(x, beam4_scores, width, label='Beam4', color='#3498db', alpha=0.8)
bars3 = ax.bar(x + width, beam8_scores, width, label='Beam8', color='#9b59b6', alpha=0.8)

ax.set_xlabel('Positional Encoding', fontsize=12, fontweight='bold')
ax.set_ylabel('BLEU Score', fontsize=12, fontweight='bold')
ax.set_title('Transformer Positional Encoding Comparison', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(position_encodings, fontsize=10)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')

ax.set_ylim(0, 5)

def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=8)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

plt.tight_layout()
plt.savefig('experiments/results/figures/transformer_positional_encoding_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('experiments/results/figures/transformer_positional_encoding_comparison.pdf', bbox_inches='tight')
plt.close()

print("Transformer Positional Encoding Comparison chart saved successfully!")
print("Files saved:")
print("  - experiments/results/figures/transformer_positional_encoding_comparison.png")
print("  - experiments/results/figures/transformer_positional_encoding_comparison.pdf")
