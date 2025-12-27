#!/usr/bin/env python3
"""
Analyze hyperparameter sensitivity sweep experiments.

This script:
1. Loads metrics from 4 experiments (baseline + 3 sweeps)
2. Optionally runs inference on test/valid set for final BLEU scores
3. Generates comparison visualizations:
   - BLEU vs hyperparameter curves
   - Training time comparison
   - Convergence speed analysis
4. Saves results to experiments/results/

Expected inputs:
- experiments/results/transformer_abs_ln_metrics.csv (baseline)
- experiments/results/sweep_batch_256_metrics.csv
- experiments/results/sweep_lr_1e3_metrics.csv
- experiments/results/sweep_d768_metrics.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_metrics(metrics_path: Path) -> pd.DataFrame:
    """Load metrics CSV with error handling."""
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found, skipping...")
        return None
    return pd.read_csv(metrics_path)


def run_inference(checkpoint_path: Path, split: str, beam_size: int = 4) -> float:
    """
    Run inference on a checkpoint and return BLEU score.

    Args:
        checkpoint_path: Path to model checkpoint
        split: 'valid' or 'test'
        beam_size: Beam size for beam search

    Returns:
        BLEU score
    """
    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint {checkpoint_path} not found")
        return 0.0

    input_file = PROJECT_ROOT / f"data/processed/{split}.jsonl"
    output_file = PROJECT_ROOT / f"experiments/results/temp_{checkpoint_path.stem}_{split}_beam{beam_size}.jsonl"

    cmd = [
        "python", str(PROJECT_ROOT / "src/transformer/infer.py"),
        "--checkpoint", str(checkpoint_path),
        "--input", str(input_file),
        "--output", str(output_file),
        "--tokenizer", str(PROJECT_ROOT / "data/vocab/tokenizer_config.json"),
        "--beam_size", str(beam_size),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        # Parse BLEU from output
        for line in result.stdout.split('\n'):
            if 'BLEU:' in line or 'bleu:' in line.lower():
                parts = line.split(':')
                if len(parts) >= 2:
                    bleu_str = parts[-1].strip()
                    try:
                        return float(bleu_str)
                    except ValueError:
                        pass

        return 0.0

    except subprocess.TimeoutExpired:
        print(f"Warning: Inference timed out for {checkpoint_path}")
        return 0.0
    except Exception as e:
        print(f"Warning: Inference failed for {checkpoint_path}: {e}")
        return 0.0



def plot_sweep_curves(experiments: dict, sweep_param: str, param_values: dict, output_path: Path):
    """
    Plot BLEU curves for a specific hyperparameter sweep.

    Args:
        experiments: Dict mapping experiment name to DataFrame
        sweep_param: Name of the sweep parameter (e.g., 'batch_size')
        param_values: Dict mapping experiment name to parameter value
        output_path: Output file path
    """
    plt.figure(figsize=(10, 6))

    for name, df in experiments.items():
        if df is None or 'bleu' not in df.columns:
            continue
        param_val = param_values.get(name, "Unknown")
        plt.plot(df['epoch'], df['bleu'], marker='o', markersize=3, linewidth=2,
                label=f'{sweep_param}={param_val}', alpha=0.8)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation BLEU', fontsize=12)
    plt.title(f'Hyperparameter Sensitivity: {sweep_param}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ {sweep_param} sweep curve saved to {output_path}")


def plot_training_time_comparison(experiments: dict, output_path: Path):
    """Plot training time comparison bar chart."""
    names = []
    times = []

    for name, df in experiments.items():
        if df is None:
            continue
        names.append(name)
        # Estimate total training time if 'time' column exists
        if 'time' in df.columns:
            times.append(df['time'].sum())
        else:
            times.append(0)  # Placeholder if time not logged

    if all(t == 0 for t in times):
        print("Warning: Training time not logged in metrics, skipping time comparison plot")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(names)), times, color='steelblue', alpha=0.8)
    plt.xticks(range(len(names)), names, rotation=15, ha='right')
    plt.ylabel('Total Training Time (seconds)', fontsize=12)
    plt.title('Training Time Comparison', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training time comparison saved to {output_path}")


def plot_all_sweeps_combined(all_experiments: dict, output_path: Path):
    """Plot all hyperparameter sweeps in a 3-panel figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sweep 1: Batch size
    ax = axes[0]
    batch_exps = {
        'batch=128 (baseline)': all_experiments.get('Baseline'),
        'batch=256': all_experiments.get('Batch 256'),
    }
    for name, df in batch_exps.items():
        if df is not None and 'bleu' in df.columns:
            ax.plot(df['epoch'], df['bleu'], marker='o', markersize=3, linewidth=2, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation BLEU', fontsize=11)
    ax.set_title('Batch Size Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sweep 2: Learning rate
    ax = axes[1]
    lr_exps = {
        'lr=5e-4 (baseline)': all_experiments.get('Baseline'),
        'lr=1e-3': all_experiments.get('LR 1e-3'),
    }
    for name, df in lr_exps.items():
        if df is not None and 'bleu' in df.columns:
            ax.plot(df['epoch'], df['bleu'], marker='s', markersize=3, linewidth=2, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation BLEU', fontsize=11)
    ax.set_title('Learning Rate Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Sweep 3: Model dimension
    ax = axes[2]
    dim_exps = {
        'd_model=512 (baseline)': all_experiments.get('Baseline'),
        'd_model=768': all_experiments.get('D_model 768'),
    }
    for name, df in dim_exps.items():
        if df is not None and 'bleu' in df.columns:
            ax.plot(df['epoch'], df['bleu'], marker='^', markersize=3, linewidth=2, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation BLEU', fontsize=11)
    ax.set_title('Model Dimension Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Combined sweep curves saved to {output_path}")


def generate_comparison_table(experiments: dict, checkpoints: dict, split: str, skip_infer: bool, output_csv: Path, output_md: Path):
    """Generate comparison table and save as CSV and Markdown."""
    rows = []

    for name, df in experiments.items():
        if df is None:
            continue

        # Extract metrics
        best_bleu = df['bleu'].max() if 'bleu' in df.columns else 0.0
        best_epoch = df.loc[df['bleu'].idxmax(), 'epoch'] if 'bleu' in df.columns else 0
        final_bleu = df['bleu'].iloc[-1] if 'bleu' in df.columns else 0.0
        final_train_loss = df['train_loss'].iloc[-1] if 'train_loss' in df.columns else 0.0
        final_valid_loss = df['valid_loss'].iloc[-1] if 'valid_loss' in df.columns else 0.0
        total_epochs = len(df)

        rows.append({
            'experiment': name,
            'best_bleu': best_bleu,
            'best_epoch': best_epoch,
            'final_bleu': final_bleu,
            'final_train_loss': final_train_loss,
            'final_valid_loss': final_valid_loss,
            'total_epochs': total_epochs,
        })

        # Run inference on test/valid set if not skipping
        if not skip_infer and name in checkpoints:
            checkpoint = checkpoints[name]
            if checkpoint and checkpoint.exists():
                print(f"Running inference for {name} on {split} set...")
                greedy_bleu = run_inference(checkpoint, split, beam_size=1)
                beam4_bleu = run_inference(checkpoint, split, beam_size=4)
                beam8_bleu = run_inference(checkpoint, split, beam_size=8)

                rows[-1][f'{split}_greedy_bleu'] = greedy_bleu
                rows[-1][f'{split}_beam4_bleu'] = beam4_bleu
                rows[-1][f'{split}_beam8_bleu'] = beam8_bleu
                print(f"  Greedy: {greedy_bleu:.4f}, Beam4: {beam4_bleu:.4f}, Beam8: {beam8_bleu:.4f}")

    # Create DataFrame
    comparison_df = pd.DataFrame(rows)
    comparison_df = comparison_df.sort_values('best_bleu', ascending=False)

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"✓ Comparison table saved to {output_csv}")

    # Generate Markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Hyperparameter Sensitivity Analysis Results\\n\\n")
        f.write("## Experiment Comparison\\n\\n")
        f.write(comparison_df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\\n\\n")

        # Summary
        best_row = comparison_df.iloc[0]
        f.write("## Summary\n\n")
        f.write(f"**Best Configuration:** {best_row['experiment']}\n\n")
        f.write(f"- Best BLEU: {best_row['best_bleu']:.4f} (epoch {best_row['best_epoch']:.0f})\n")
        f.write(f"- Final BLEU: {best_row['final_bleu']:.4f}\n")
        f.write(f"- Final Valid Loss: {best_row['final_valid_loss']:.4f}\n")

        # Add test/valid set results if available
        if not skip_infer and f'{split}_beam4_bleu' in best_row:
            f.write(f"- {split.capitalize()} Set Greedy BLEU: {best_row[f'{split}_greedy_bleu']:.4f}\n")
            f.write(f"- {split.capitalize()} Set Beam4 BLEU: {best_row[f'{split}_beam4_bleu']:.4f}\n")
            f.write(f"- {split.capitalize()} Set Beam8 BLEU: {best_row[f'{split}_beam8_bleu']:.4f}\n")

        f.write("\n")

        # Insights
        f.write("## Key Findings\\n\\n")

        # Batch size impact
        baseline_bleu = comparison_df[comparison_df['experiment'].str.contains('Baseline', na=False)]['best_bleu'].values
        batch256_bleu = comparison_df[comparison_df['experiment'].str.contains('Batch 256', na=False)]['best_bleu'].values
        if len(baseline_bleu) > 0 and len(batch256_bleu) > 0:
            delta = batch256_bleu[0] - baseline_bleu[0]
            f.write(f"1. **Batch Size Impact:** Increasing batch size from 128 to 256 ")
            f.write(f"{'improved' if delta > 0 else 'reduced'} BLEU by {abs(delta):.4f} ")
            f.write(f"({delta/baseline_bleu[0]*100:+.2f}%).\\n\\n")

        # Learning rate impact
        lr1e3_bleu = comparison_df[comparison_df['experiment'].str.contains('LR 1e-3', na=False)]['best_bleu'].values
        if len(baseline_bleu) > 0 and len(lr1e3_bleu) > 0:
            delta = lr1e3_bleu[0] - baseline_bleu[0]
            f.write(f"2. **Learning Rate Impact:** Increasing learning rate from 5e-4 to 1e-3 ")
            f.write(f"{'improved' if delta > 0 else 'reduced'} BLEU by {abs(delta):.4f} ")
            f.write(f"({delta/baseline_bleu[0]*100:+.2f}%).\\n\\n")

        # Model dimension impact
        d768_bleu = comparison_df[comparison_df['experiment'].str.contains('D_model 768', na=False)]['best_bleu'].values
        if len(baseline_bleu) > 0 and len(d768_bleu) > 0:
            delta = d768_bleu[0] - baseline_bleu[0]
            f.write(f"3. **Model Dimension Impact:** Increasing d_model from 512 to 768 ")
            f.write(f"{'improved' if delta > 0 else 'reduced'} BLEU by {abs(delta):.4f} ")
            f.write(f"({delta/baseline_bleu[0]*100:+.2f}%).\\n\\n")

    print(f"✓ Comparison Markdown saved to {output_md}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Analyze hyperparameter sensitivity experiments')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                        help='Directory containing metrics CSVs')
    parser.add_argument('--output_dir', type=str, default='experiments/results/figures',
                        help='Output directory for visualizations')
    parser.add_argument('--split', type=str, default='valid', choices=['valid', 'test'],
                        help='Dataset split for inference (valid or test)')
    parser.add_argument('--skip_infer', action='store_true',
                        help='Skip inference on split, only analyze training metrics')
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir

    print("=" * 60)
    print("Hyperparameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Skip inference: {args.skip_infer}")
    print()

    # Load all metrics
    all_experiments = {
        'Baseline': load_metrics(results_dir / 'transformer_abs_ln_metrics.csv'),
        'Batch 256': load_metrics(results_dir / 'sweep_batch_256_metrics.csv'),
        'LR 1e-3': load_metrics(results_dir / 'sweep_lr_1e3_metrics.csv'),
        'D_model 768': load_metrics(results_dir / 'sweep_d768_metrics.csv'),
    }

    # Checkpoint paths
    checkpoints = {
        'Baseline': PROJECT_ROOT / 'experiments/logs/transformer_abs_ln_best.pt',
        'Batch 256': PROJECT_ROOT / 'experiments/logs/sweep_batch_256_best.pt',
        'LR 1e-3': PROJECT_ROOT / 'experiments/logs/sweep_lr_1e3_best.pt',
        'D_model 768': PROJECT_ROOT / 'experiments/logs/sweep_d768_best.pt',
    }

    # Check if we have any data
    if all(df is None for df in all_experiments.values()):
        print("Error: No metrics files found. Please run experiments first.")
        return

    # Generate comparison table (with optional inference)
    comparison_df = generate_comparison_table(
        all_experiments,
        checkpoints,
        args.split,
        args.skip_infer,
        results_dir / 'hyperparam_sweep_comparison.csv',
        results_dir / 'hyperparam_sweep_comparison.md'
    )

    # Generate combined visualization
    plot_all_sweeps_combined(all_experiments, output_dir / 'hyperparam_sweep_combined.png')

    # Generate training time comparison (if available)
    plot_training_time_comparison(all_experiments, output_dir / 'hyperparam_training_time.png')

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - {results_dir / 'hyperparam_sweep_comparison.csv'}")
    print(f"  - {results_dir / 'hyperparam_sweep_comparison.md'}")
    print(f"  - {output_dir / 'hyperparam_sweep_combined.png'}")
    print(f"  - {output_dir / 'hyperparam_training_time.png'} (if time data available)")
    print()


if __name__ == '__main__':
    main()
