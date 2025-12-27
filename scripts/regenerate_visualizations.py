import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_rnn_training_curves():
    """Generate RNN training curves visualization."""
    results_dir = Path("/mnt/afs/250010065/SLAI_NLP_Mid_project/zh-en-nmt-midterm/experiments/results")
    output_path = results_dir / "figures" / "rnn_training_curves.png"
    
    # RNN metrics files
    rnn_files = {
        "Additive (Teacher Forcing)": results_dir / "rnn_additive_metrics.csv",
        "Multiplicative (Teacher Forcing)": results_dir / "rnn_multiplicative_metrics.csv",
        "Additive (Free Running)": results_dir / "rnn_additive_no_teacher_metrics.csv",
        "Multiplicative (Free Running)": results_dir / "rnn_multiplicative_no_teacher_metrics.csv",
    }
    
    plt.figure(figsize=(12, 7))
    
    for label, file_path in rnn_files.items():
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            if "epoch" in df.columns and "bleu" in df.columns:
                plt.plot(df["epoch"], df["bleu"], label=label, linewidth=2, marker='o', markersize=3)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    plt.title("RNN Training Curves (BLEU over Epochs)", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ RNN training curves saved to {output_path}")

def plot_transformer_ablation_heatmap():
    """Generate Transformer ablation heatmap."""
    results_dir = Path("/mnt/afs/250010065/SLAI_NLP_Mid_project/zh-en-nmt-midterm/experiments/results")
    output_path = results_dir / "figures" / "transformer_ablation_heatmap.png"
    
    # Transformer metrics files
    transformer_files = {
        ("Absolute PE", "RMSNorm"): results_dir / "transformer_abs_rms_metrics.csv",
        ("Absolute PE", "LayerNorm"): results_dir / "transformer_abs_ln_metrics.csv",
        ("Relative PE", "RMSNorm"): results_dir / "transformer_rel_rms_metrics.csv",
        ("Relative PE", "LayerNorm"): results_dir / "transformer_rel_ln_metrics.csv",
    }
    
    # Collect best BLEU scores
    data = []
    for (pe_type, norm_type), file_path in transformer_files.items():
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            if "bleu" in df.columns:
                best_bleu = df["bleu"].max()
                data.append({
                    "Position Encoding": pe_type,
                    "Normalization": norm_type,
                    "Best BLEU": best_bleu
                })
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not data:
        print("No data found for Transformer heatmap")
        return
    
    df = pd.DataFrame(data)
    
    # Create pivot table for heatmap
    pivot = df.pivot(index="Normalization", columns="Position Encoding", values="Best BLEU")
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Best BLEU Score'}, 
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title("Transformer Ablation Study: Best BLEU Scores", fontsize=16, fontweight='bold')
    plt.xlabel("Position Encoding", fontsize=12)
    plt.ylabel("Normalization", fontsize=12)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Transformer ablation heatmap saved to {output_path}")

def plot_transformer_ablation_curves():
    """Generate Transformer ablation curves."""
    results_dir = Path("/mnt/afs/250010065/SLAI_NLP_Mid_project/zh-en-nmt-midterm/experiments/results")
    output_path = results_dir / "figures" / "transformer_ablation_curves.png"
    
    # Transformer metrics files
    transformer_files = {
        "Absolute PE + RMSNorm": results_dir / "transformer_abs_rms_metrics.csv",
        "Absolute PE + LayerNorm": results_dir / "transformer_abs_ln_metrics.csv",
        "Relative PE + RMSNorm": results_dir / "transformer_rel_rms_metrics.csv",
        "Relative PE + LayerNorm": results_dir / "transformer_rel_ln_metrics.csv",
    }
    
    plt.figure(figsize=(12, 7))
    
    for label, file_path in transformer_files.items():
        if not file_path.exists():
            continue
        try:
            df = pd.read_csv(file_path)
            if "epoch" in df.columns and "bleu" in df.columns:
                plt.plot(df["epoch"], df["bleu"], label=label, linewidth=2, marker='o', markersize=3)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    plt.title("Transformer Training Curves (BLEU over Epochs)", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("BLEU Score", fontsize=12)
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Transformer ablation curves saved to {output_path}")

if __name__ == "__main__":
    print("Generating RNN training curves...")
    plot_rnn_training_curves()
    
    print("\nGenerating Transformer ablation heatmap...")
    plot_transformer_ablation_heatmap()
    
    print("\nGenerating Transformer ablation curves...")
    plot_transformer_ablation_curves()
    
    print("\n✓ All visualizations generated successfully!")
