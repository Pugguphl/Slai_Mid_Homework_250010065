#!/usr/bin/env python3
"""
Analyze Transformer ablation experiments (2x2 matrix: positional_encoding × norm_type).

This script:
1. Loads metrics CSVs from 4 ablation experiments
2. Optionally runs inference on test/valid set for final BLEU scores
3. Generates comparison visualizations:
   - 2×2 heatmap of final BLEU scores
   - Training curves for all 4 variants
   - Ablation comparison table
4. Saves results to experiments/results/

Expected inputs:
- experiments/results/transformer_abs_ln_metrics.csv
- experiments/results/transformer_abs_rms_metrics.csv
- experiments/results/transformer_rel_ln_metrics.csv
- experiments/results/transformer_rel_rms_metrics.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import subprocess
import json
import sys
import time
from functools import partial

import torch
from sacrebleu import corpus_bleu
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

    try:
        from src.transformer.model import TransformerConfig, TransformerSeq2Seq
        from src.common.tokenizer import load_tokenizer
        from src.common.dataset import TranslationDataset, collate_translation_batch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(str(checkpoint_path), map_location=device)
        
        config = ckpt["config"]["model"]
        tokenizer_path = ckpt.get("tokenizer") or ckpt["config"]["data"]["tokenizer_file"]
        tokenizer = load_tokenizer(tokenizer_path)
        
        model_cfg = TransformerConfig(
            vocab_size=tokenizer.vocab_size,
            d_model=config.get("d_model", 512),
            num_heads=config.get("num_heads", 8),
            num_encoder_layers=config.get("num_encoder_layers", 6),
            num_decoder_layers=config.get("num_decoder_layers", 6),
            dim_feedforward=config.get("dim_feedforward", 2048),
            dropout=config.get("dropout", 0.1),
            activation=config.get("activation", "relu"),
            max_position_embeddings=config.get("max_position_embeddings", 256),
            pad_idx=tokenizer.pad_id,
            bos_idx=tokenizer.bos_id,
            eos_idx=tokenizer.eos_id,
            share_embeddings=config.get("share_embeddings", True),
            tie_output=config.get("tie_output", True),
            norm_first=config.get("norm_first", True),
            positional_encoding=config.get("positional_encoding", "sinusoidal"),
            norm_type=config.get("norm_type", "layernorm"),
        )
        model = TransformerSeq2Seq(model_cfg)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()

        data_cfg = ckpt.get("config", {}).get("data", {})
        dataset_path = PROJECT_ROOT / f"data/processed/{split}.jsonl"
        
        if not dataset_path.exists():
            print(f"Warning: Dataset file {dataset_path} not found")
            return 0.0

        dataset = TranslationDataset(
            file_path=str(dataset_path),
            tokenizer=tokenizer,
            src_key=data_cfg.get("src_key", "zh"),
            tgt_key=data_cfg.get("tgt_key", "en"),
            add_bos_eos=True,
        )

        def collate_with_text(batch, tokenizer):
            data = collate_translation_batch(batch, tokenizer)
            data["src_text"] = [item["src_text"] for item in batch]
            data["tgt_text"] = [item["tgt_text"] for item in batch]
            return data

        predictions = []
        references = []

        model.eval()
        with torch.no_grad():
            if beam_size == 1:
                loader = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=False,
                    collate_fn=partial(collate_with_text, tokenizer=tokenizer),
                )
                for batch in tqdm(loader, desc=f"Greedy decoding on {split}"):
                    src = batch["src"].to(device)
                    lengths = batch["src_lengths"].to(device)
                    preds = model.greedy_decode(src, lengths)
                    texts = [tokenizer.decode_tgt(seq.tolist()) for seq in preds]
                    predictions.extend(texts)
                    references.extend(batch["tgt_text"])
            else:
                for sample in tqdm(dataset, desc=f"Beam{beam_size} decoding on {split}"):
                    src_tensor = torch.tensor(sample["src_ids"], dtype=torch.long, device=device).unsqueeze(0)
                    lengths = torch.tensor([len(sample["src_ids"])], dtype=torch.long, device=device)
                    pred = model.beam_search(src_tensor, lengths, beam_size=beam_size)
                    predictions.append(tokenizer.decode_tgt(pred.squeeze(0).tolist()))
                    references.append(sample["tgt_text"])

        bleu = corpus_bleu(predictions, [references]).score if predictions else 0.0
        return bleu

    except Exception as e:
        print(f"Warning: Inference failed for {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0



def plot_heatmap(comparison_df: pd.DataFrame, output_path: Path):
    """Create 2×2 heatmap of BLEU scores (positional_encoding × norm_type)."""
    # Prepare data for heatmap
    pivot = comparison_df.pivot_table(
        values='best_bleu',
        index='norm_type',
        columns='positional_encoding',
        aggfunc='mean'
    )

    # Reorder to consistent format
    pivot = pivot.reindex(index=['layernorm', 'rmsnorm'],
                          columns=['sinusoidal', 'relative'])

    # Create heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt='.4f',
        cmap='YlGnBu',
        cbar_kws={'label': 'BLEU Score'},
        linewidths=1,
        linecolor='gray'
    )

    # Labels
    ax.set_xlabel('Positional Encoding', fontsize=12, fontweight='bold')
    ax.set_ylabel('Normalization Type', fontsize=12, fontweight='bold')
    ax.set_title('Transformer Ablation: BLEU Scores\n(Positional Encoding × Normalization)',
                 fontsize=14, fontweight='bold', pad=20)

    # Better tick labels
    ax.set_xticklabels(['Absolute (Sinusoidal)', 'Relative (T5-style)'], rotation=0)
    ax.set_yticklabels(['LayerNorm', 'RMSNorm'], rotation=0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Heatmap saved to {output_path}")


def plot_training_curves(experiments: dict, output_path: Path):
    """Plot training curves for all 4 ablation experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot 1: BLEU over epochs
    ax = axes[0]
    for name, df in experiments.items():
        if df is not None and 'bleu' in df.columns:
            ax.plot(df['epoch'], df['bleu'], marker='o', markersize=3, linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation BLEU', fontsize=11)
    ax.set_title('Validation BLEU over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)

    # Plot 2: Training loss over epochs
    ax = axes[1]
    for name, df in experiments.items():
        if df is not None and 'train_loss' in df.columns:
            ax.plot(df['epoch'], df['train_loss'], marker='s', markersize=3, linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title('Training Loss over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Validation loss over epochs
    ax = axes[2]
    for name, df in experiments.items():
        if df is not None and 'valid_loss' in df.columns:
            ax.plot(df['epoch'], df['valid_loss'], marker='^', markersize=3, linewidth=1.5, label=name, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=11)
    ax.set_title('Validation Loss over Training', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 4: Learning rate schedule (if available)
    ax = axes[3]
    has_lr = False
    for name, df in experiments.items():
        if df is not None and 'learning_rate' in df.columns:
            ax.plot(df['epoch'], df['learning_rate'], marker='d', markersize=3, linewidth=1.5, label=name, alpha=0.8)
            has_lr = True
    if has_lr:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Learning rate not logged', ha='center', va='center', fontsize=12)
        ax.axis('off')

    plt.suptitle('Transformer Ablation: Training Dynamics', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to {output_path}")


def generate_comparison_table(experiments: dict, checkpoints: dict, split: str, skip_infer: bool, output_csv: Path, output_md: Path):
    """Generate comparison table and save as CSV and Markdown."""
    rows = []

    for name, df in experiments.items():
        if df is None:
            continue

        # Parse experiment name
        parts = name.split(' + ')
        if len(parts) == 2:
            pos_enc = parts[0].strip()
            norm_type = parts[1].strip()
        else:
            pos_enc = 'unknown'
            norm_type = 'unknown'

        # Extract metrics
        best_bleu = df['bleu'].max() if 'bleu' in df.columns else 0.0
        best_epoch = df.loc[df['bleu'].idxmax(), 'epoch'] if 'bleu' in df.columns else 0
        final_bleu = df['bleu'].iloc[-1] if 'bleu' in df.columns else 0.0
        final_train_loss = df['train_loss'].iloc[-1] if 'train_loss' in df.columns else 0.0
        final_valid_loss = df['valid_loss'].iloc[-1] if 'valid_loss' in df.columns else 0.0
        total_epochs = len(df)

        rows.append({
            'experiment': name,
            'positional_encoding': pos_enc,
            'norm_type': norm_type,
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
        f.write("# Transformer Ablation Study Results\n\n")
        f.write("## 2×2 Ablation Matrix: Positional Encoding × Normalization\n\n")
        f.write(comparison_df.to_markdown(index=False, floatfmt='.4f'))
        f.write("\n\n")

        # Summary
        best_row = comparison_df.iloc[0]
        f.write("## Summary\n\n")
        f.write(f"**Best Configuration:** {best_row['experiment']}\n\n")
        f.write(f"- Best Valid BLEU: {best_row['best_bleu']:.4f} (epoch {best_row['best_epoch']:.0f})\n")
        f.write(f"- Final BLEU: {best_row['final_bleu']:.4f}\n")
        f.write(f"- Final Valid Loss: {best_row['final_valid_loss']:.4f}\n")

        # Add test/valid set results if available
        if not skip_infer and f'{split}_beam4_bleu' in best_row:
            f.write(f"- {split.capitalize()} Set Greedy BLEU: {best_row[f'{split}_greedy_bleu']:.4f}\n")
            f.write(f"- {split.capitalize()} Set Beam4 BLEU: {best_row[f'{split}_beam4_bleu']:.4f}\n")
            f.write(f"- {split.capitalize()} Set Beam8 BLEU: {best_row[f'{split}_beam8_bleu']:.4f}\n")

        f.write("\n")

        # Insights
        f.write("## Key Findings\n\n")

        # Compare positional encoding
        abs_mean = comparison_df[comparison_df['positional_encoding'].str.contains('Absolute', na=False)]['best_bleu'].mean()
        rel_mean = comparison_df[comparison_df['positional_encoding'].str.contains('Relative', na=False)]['best_bleu'].mean()
        f.write(f"1. **Positional Encoding:**\n")
        f.write(f"   - Absolute (sinusoidal) average BLEU: {abs_mean:.4f}\n")
        f.write(f"   - Relative (T5-style) average BLEU: {rel_mean:.4f}\n")
        f.write(f"   - Winner: {'Absolute' if abs_mean > rel_mean else 'Relative'}\n\n")

        # Compare normalization
        ln_mean = comparison_df[comparison_df['norm_type'].str.contains('LayerNorm', na=False)]['best_bleu'].mean()
        rms_mean = comparison_df[comparison_df['norm_type'].str.contains('RMSNorm', na=False)]['best_bleu'].mean()
        f.write(f"2. **Normalization:**\n")
        f.write(f"   - LayerNorm average BLEU: {ln_mean:.4f}\n")
        f.write(f"   - RMSNorm average BLEU: {rms_mean:.4f}\n")
        f.write(f"   - Winner: {'LayerNorm' if ln_mean > rms_mean else 'RMSNorm'}\n\n")

    print(f"✓ Comparison Markdown saved to {output_md}")

    return comparison_df


def main():
    parser = argparse.ArgumentParser(description='Analyze Transformer ablation experiments')
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
    print("Transformer Ablation Analysis")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Skip inference: {args.skip_infer}")
    print()

    # Load all metrics
    experiments = {
        'Absolute PE + LayerNorm': load_metrics(results_dir / 'transformer_abs_ln_metrics.csv'),
        'Absolute PE + RMSNorm': load_metrics(results_dir / 'transformer_abs_rms_metrics.csv'),
        'Relative PE + LayerNorm': load_metrics(results_dir / 'transformer_rel_ln_metrics.csv'),
        'Relative PE + RMSNorm': load_metrics(results_dir / 'transformer_rel_rms_metrics.csv'),
    }

    # Checkpoint paths
    checkpoints = {
        'Absolute PE + LayerNorm': PROJECT_ROOT / 'experiments/logs/transformer_abs_ln_best.pt',
        'Absolute PE + RMSNorm': PROJECT_ROOT / 'experiments/logs/transformer_abs_rms_best.pt',
        'Relative PE + LayerNorm': PROJECT_ROOT / 'experiments/logs/transformer_rel_ln_best.pt',
        'Relative PE + RMSNorm': PROJECT_ROOT / 'experiments/logs/transformer_rel_rms_best.pt',
    }

    # Check if we have any data
    if all(df is None for df in experiments.values()):
        print("Error: No metrics files found. Please run experiments first.")
        return

    # Generate comparison table (with optional inference)
    comparison_df = generate_comparison_table(
        experiments,
        checkpoints,
        args.split,
        args.skip_infer,
        results_dir / 'transformer_ablation_comparison.csv',
        results_dir / 'transformer_ablation_comparison.md'
    )

    # Generate visualizations
    plot_heatmap(comparison_df, output_dir / 'transformer_ablation_heatmap.png')
    plot_training_curves(experiments, output_dir / 'transformer_ablation_curves.png')

    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"  - {results_dir / 'transformer_ablation_comparison.csv'}")
    print(f"  - {results_dir / 'transformer_ablation_comparison.md'}")
    print(f"  - {output_dir / 'transformer_ablation_heatmap.png'}")
    print(f"  - {output_dir / 'transformer_ablation_curves.png'}")
    print()


if __name__ == '__main__':
    main()
