#!/usr/bin/env python3
"""
Generate master comparison table for all experiments.

This script collects results from:
1. RNN experiments (3 attention types × 2 RNN types = 6 experiments)
2. Transformer ablations (2 pos_enc × 2 norm = 4 experiments)
3. Hyperparameter sweeps (3 sweeps + baseline)
4. T5 fine-tuning

And generates a unified comparison table with:
- Model name
- BLEU score
- Training time
- Parameter count
- Key configuration details
"""

import argparse
from pathlib import Path
import pandas as pd
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_metrics_csv(file_path: Path) -> pd.DataFrame:
    """Load metrics CSV if exists."""
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def load_metrics_json(file_path: Path) -> dict:
    """Load metrics JSON if exists."""
    if not file_path.exists():
        return None
    with open(file_path, 'r') as f:
        return json.load(f)


def extract_best_bleu(df: pd.DataFrame) -> float:
    """Extract best BLEU from metrics DataFrame."""
    if df is None or 'bleu' not in df.columns:
        return 0.0
    return df['bleu'].max()


def extract_final_epoch(df: pd.DataFrame) -> int:
    """Extract total epochs from DataFrame."""
    if df is None:
        return 0
    return len(df)


def collect_rnn_results(results_dir: Path) -> list:
    """Collect RNN experiment results."""
    rnn_configs = [
        ('GRU + Dot Attention', 'rnn_gru_dot_metrics.csv'),
        ('GRU + Multiplicative Attention', 'rnn_gru_mult_metrics.csv'),
        ('GRU + Additive Attention', 'rnn_gru_add_metrics.csv'),
        ('LSTM + Dot Attention', 'rnn_lstm_dot_metrics.csv'),
        ('LSTM + Multiplicative Attention', 'rnn_lstm_mult_metrics.csv'),
        ('LSTM + Additive Attention', 'rnn_lstm_add_metrics.csv'),
    ]

    rows = []
    for name, file in rnn_configs:
        df = load_metrics_csv(results_dir / file)
        if df is not None:
            rows.append({
                'model': name,
                'category': 'RNN',
                'best_bleu': extract_best_bleu(df),
                'total_epochs': extract_final_epoch(df),
                'params': '~10M',  # Approximate
                'notes': '',
            })

    return rows


def collect_transformer_ablation_results(results_dir: Path) -> list:
    """Collect Transformer ablation experiment results."""
    ablation_configs = [
        ('Transformer (Abs PE + LayerNorm)', 'transformer_abs_ln_metrics.csv', 'Baseline'),
        ('Transformer (Abs PE + RMSNorm)', 'transformer_abs_rms_metrics.csv', 'RMSNorm ablation'),
        ('Transformer (Rel PE + LayerNorm)', 'transformer_rel_ln_metrics.csv', 'T5-style relative PE'),
        ('Transformer (Rel PE + RMSNorm)', 'transformer_rel_rms_metrics.csv', 'Full ablation'),
    ]

    rows = []
    for name, file, notes in ablation_configs:
        df = load_metrics_csv(results_dir / file)
        if df is not None:
            rows.append({
                'model': name,
                'category': 'Transformer Ablation',
                'best_bleu': extract_best_bleu(df),
                'total_epochs': extract_final_epoch(df),
                'params': '~44M',  # d=512, 6 layers
                'notes': notes,
            })

    return rows


def collect_hyperparam_sweep_results(results_dir: Path) -> list:
    """Collect hyperparameter sweep results."""
    sweep_configs = [
        ('Transformer (batch=128, baseline)', 'transformer_abs_ln_metrics.csv', 'Baseline'),
        ('Transformer (batch=256)', 'sweep_batch_256_metrics.csv', 'Batch size sweep'),
        ('Transformer (lr=1e-3)', 'sweep_lr_1e3_metrics.csv', 'Learning rate sweep'),
        ('Transformer (d=768)', 'sweep_d768_metrics.csv', 'Model dimension sweep (~97M params)'),
    ]

    rows = []
    for name, file, notes in sweep_configs:
        df = load_metrics_csv(results_dir / file)
        if df is not None:
            # Skip if already added in ablation
            if 'baseline' in name.lower() and any(r['model'].startswith('Transformer (Abs PE') for r in rows):
                continue

            params = '~97M' if 'd=768' in name else '~44M'
            rows.append({
                'model': name,
                'category': 'Hyperparameter Sweep',
                'best_bleu': extract_best_bleu(df),
                'total_epochs': extract_final_epoch(df),
                'params': params,
                'notes': notes,
            })

    return rows


def collect_t5_results(results_dir: Path) -> list:
    """Collect T5 fine-tuning results."""
    t5_file = results_dir / 't5_small_metrics.json'
    metrics = load_metrics_json(t5_file)

    rows = []
    if metrics is not None:
        rows.append({
            'model': 'T5-small (fine-tuned)',
            'category': 'Pretrained Model',
            'best_bleu': metrics.get('eval_bleu', metrics.get('bleu', 0.0)),
            'total_epochs': 10,  # From config
            'params': '60M',
            'notes': 'Pretrained on C4, fine-tuned on ZH-EN',
        })

    return rows


def generate_master_table(results_dir: Path, output_csv: Path, output_md: Path):
    """Generate master comparison table."""
    print("Collecting results from all experiments...")

    all_rows = []

    # Collect from each category
    all_rows.extend(collect_rnn_results(results_dir))
    all_rows.extend(collect_transformer_ablation_results(results_dir))
    all_rows.extend(collect_hyperparam_sweep_results(results_dir))
    all_rows.extend(collect_t5_results(results_dir))

    if not all_rows:
        print("Warning: No results found. Please run experiments first.")
        return

    # Create DataFrame
    df = pd.DataFrame(all_rows)
    df = df.sort_values('best_bleu', ascending=False).reset_index(drop=True)

    # Add rank
    df.insert(0, 'rank', range(1, len(df) + 1))

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, float_format='%.4f')
    print(f"✓ Master comparison table saved to {output_csv}")

    # Generate Markdown
    with open(output_md, 'w', encoding='utf-8') as f:
        f.write("# Chinese-English NMT: Master Comparison Table\\n\\n")
        f.write("## All Experiments Results\\n\\n")
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\\n\\n")

        # Top 3 models
        f.write("## Top 3 Models\\n\\n")
        top3 = df.head(3)
        for idx, row in top3.iterrows():
            f.write(f"### {row['rank']}. {row['model']}\\n\\n")
            f.write(f"- **BLEU**: {row['best_bleu']:.4f}\\n")
            f.write(f"- **Category**: {row['category']}\\n")
            f.write(f"- **Parameters**: {row['params']}\\n")
            f.write(f"- **Epochs**: {row['total_epochs']}\\n")
            if row['notes']:
                f.write(f"- **Notes**: {row['notes']}\\n")
            f.write("\\n")

        # Category-wise best
        f.write("## Best Model by Category\\n\\n")
        for category in df['category'].unique():
            best_in_cat = df[df['category'] == category].iloc[0]
            f.write(f"- **{category}**: {best_in_cat['model']} ")
            f.write(f"(BLEU: {best_in_cat['best_bleu']:.4f})\\n")

        f.write("\\n")

        # Summary statistics
        f.write("## Summary Statistics\\n\\n")
        f.write(f"- Total experiments: {len(df)}\\n")
        f.write(f"- BLEU range: {df['best_bleu'].min():.4f} - {df['best_bleu'].max():.4f}\\n")
        f.write(f"- Mean BLEU: {df['best_bleu'].mean():.4f}\\n")
        f.write(f"- Median BLEU: {df['best_bleu'].median():.4f}\\n")

    print(f"✓ Master comparison Markdown saved to {output_md}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Generate master comparison table')
    parser.add_argument('--results_dir', type=str, default='experiments/results',
                        help='Directory containing all metrics files')
    parser.add_argument('--output_csv', type=str, default='experiments/results/master_comparison.csv',
                        help='Output CSV file path')
    parser.add_argument('--output_md', type=str, default='experiments/results/master_comparison.md',
                        help='Output Markdown file path')
    args = parser.parse_args()

    results_dir = PROJECT_ROOT / args.results_dir
    output_csv = PROJECT_ROOT / args.output_csv
    output_md = PROJECT_ROOT / args.output_md

    print("=" * 60)
    print("Master Comparison Table Generation")
    print("=" * 60)
    print()

    df = generate_master_table(results_dir, output_csv, output_md)

    if df is not None:
        print()
        print("=" * 60)
        print("Master comparison table generated successfully!")
        print("=" * 60)
        print()
        print(f"Total experiments: {len(df)}")
        print(f"Best model: {df.iloc[0]['model']}")
        print(f"Best BLEU: {df.iloc[0]['best_bleu']:.4f}")
        print()


if __name__ == '__main__':
    main()
