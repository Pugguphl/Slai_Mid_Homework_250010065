#!/usr/bin/env python3
"""
Analyze RNN experiment results.

This script:
1) scans experiments/logs for RNN runs, loads their metrics.csv (train/valid curves if present)
2) runs decoding on valid/test sets (greedy + beam(4,8)) using existing inference script
3) aggregates into a master comparison CSV
4) generates basic figures (training curves + BLEU barplot)

Assumptions (project conventions):
- RNN training produces a folder under experiments/logs/ containing:
  - config.yaml (or the original config copied)
  - metrics.csv (or train_metrics.csv/valid_metrics.csv)
  - checkpoints (*.pt) or a best.pt
- scripts/infer_rnn.py supports:
    python scripts/infer_rnn.py --checkpoint <path> --split valid --beam_size <k>
  and prints BLEU to stdout OR writes a result file.
Because implementations vary, this script is defensive and will:
- prefer to parse a metrics CSV if it includes columns like bleu_valid / bleu
- otherwise run infer_rnn.py and parse "BLEU" from stdout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402


RNN_CONFIG_PAT = re.compile(r"rnn_(dot|multiplicative|additive)(?:_no_teacher)?\.ya?ml$")
BLEU_RE = re.compile(r"(?:BLEU|bleu)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class Run:
    name: str
    log_dir: Path
    checkpoint: Optional[Path]
    metrics_csv: Optional[Path]


def _find_metrics_csv(log_dir: Path) -> Optional[Path]:
    candidates = [
        log_dir / "metrics.csv",
        log_dir / "train_metrics.csv",
        log_dir / "valid_metrics.csv",
        log_dir / "metrics_valid.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: first csv in folder
    csvs = sorted(log_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def _find_checkpoint(log_dir: Path) -> Optional[Path]:
    candidates = [
        log_dir / "best.pt",
        log_dir / "best_model.pt",
        log_dir / "checkpoint_best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    pts = sorted(log_dir.glob("*.pt"))
    if pts:
        # prefer something like epochXX or last
        return pts[-1]
    # some trainings store under checkpoints/
    ckpt_dir = log_dir / "checkpoints"
    if ckpt_dir.exists():
        pts = sorted(ckpt_dir.glob("*.pt"))
        return pts[-1] if pts else None
    return None


def discover_runs(root: Path) -> List[Run]:
    runs: List[Run] = []
    logs_dir = root / "experiments" / "logs"
    results_dir = root / "experiments" / "results"
    if not logs_dir.exists():
        return runs

    for f in sorted(logs_dir.glob("*.pt")):
        if "rnn" not in f.name.lower():
            continue
        name = f.stem
        metrics_csv = None
        if results_dir.exists():
            candidates = [
                results_dir / f"{name}_metrics.csv",
                results_dir / f"{name}.csv",
            ]
            for p in candidates:
                if p.exists():
                    metrics_csv = p
                    break
        runs.append(Run(name=name, log_dir=logs_dir, checkpoint=f, metrics_csv=metrics_csv))
    return runs


def run_infer(root: Path, checkpoint: Path, split: str, beam_size: int) -> float:
    cmd = [
        "python",
        str(root / "scripts" / "infer_rnn.py"),
        "--checkpoint",
        str(checkpoint),
        "--split",
        split,
        "--beam_size",
        str(beam_size),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    proc = subprocess.run(cmd, cwd=root, stdout=subprocess.PIPE, stderr=None, text=True, env=env)
    out = proc.stdout or ""
    m = BLEU_RE.search(out)
    if m:
        return float(m.group(1))
    raise RuntimeError(f"Failed to parse BLEU from infer output for {checkpoint} (beam={beam_size}). Output:\n{out}")


def parse_best_bleu_from_metrics(metrics_csv: Path) -> Dict[str, float]:
    """
    Try to parse best BLEU from metrics CSV.
    Returns keys:
      - bleu_valid_best
      - bleu_train_best (optional)
    """
    df = pd.read_csv(metrics_csv)
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    result: Dict[str, float] = {}

    # common patterns
    for key in ["bleu_valid", "valid_bleu", "bleu_val", "val_bleu", "bleu"]:
        if key in df.columns:
            result["bleu_best"] = float(pd.to_numeric(df[key], errors="coerce").max())
            break
    for key in ["loss_train", "train_loss", "loss"]:
        if key in df.columns:
            result["loss_min"] = float(pd.to_numeric(df[key], errors="coerce").min())
            break
    return result


def plot_training_curves(runs: List[Run], out_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plotted = 0
    for r in runs:
        if not r.metrics_csv or not r.metrics_csv.exists():
            continue
        try:
            df = pd.read_csv(r.metrics_csv)
        except Exception:
            continue
        df.columns = [c.lower() for c in df.columns]
        if "epoch" not in df.columns:
            continue
        ycol = None
        for c in ["bleu_valid", "valid_bleu", "bleu_val", "val_bleu", "bleu"]:
            if c in df.columns:
                ycol = c
                break
        if not ycol:
            continue
        plt.plot(df["epoch"], df[ycol], label=r.name)
        plotted += 1

    plt.title("RNN Training Curves (BLEU over epochs)")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU")
    if plotted:
        plt.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_bleu_bar(df_master: pd.DataFrame, out_path: Path, metric_col: str = "valid_beam4_bleu") -> None:
    if metric_col not in df_master.columns:
        return
    plot_df = df_master.sort_values(metric_col, ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["run"], plot_df[metric_col])
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel(metric_col)
    plt.title(f"RNN Comparison ({metric_col})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=str(Path(__file__).resolve().parents[1]), help="project root")
    ap.add_argument("--split", type=str, default="valid", choices=["valid", "test"], help="decode split")
    ap.add_argument("--output_csv", type=str, default="experiments/results/rnn_comparison_master.csv")
    ap.add_argument("--fig_dir", type=str, default="experiments/results/figures")
    ap.add_argument("--skip_infer", action="store_true", help="only parse metrics CSV, do not run decoding")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"No RNN runs found under {root/'experiments/logs'}")

    rows: List[Dict[str, object]] = []
    for r in tqdm(runs, desc="Analyzing RNN runs"):
        row: Dict[str, object] = {
            "run": r.name,
            "log_dir": str(r.log_dir),
            "checkpoint": str(r.checkpoint) if r.checkpoint else "",
            "metrics_csv": str(r.metrics_csv) if r.metrics_csv else "",
        }
        if r.metrics_csv and r.metrics_csv.exists():
            try:
                row.update(parse_best_bleu_from_metrics(r.metrics_csv))
            except Exception as e:
                row["metrics_parse_error"] = str(e)

        if not args.skip_infer and r.checkpoint and r.checkpoint.exists():
            try:
                row[f"{args.split}_greedy_bleu"] = run_infer(root, r.checkpoint, args.split, beam_size=1)
                row[f"{args.split}_beam4_bleu"] = run_infer(root, r.checkpoint, args.split, beam_size=4)
                row[f"{args.split}_beam8_bleu"] = run_infer(root, r.checkpoint, args.split, beam_size=8)
            except Exception as e:
                row["infer_error"] = str(e)

        rows.append(row)

    out_csv = root / args.output_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_master = pd.DataFrame(rows)
    df_master.to_csv(out_csv, index=False)

    fig_dir = root / args.fig_dir
    plot_training_curves(runs, fig_dir / "rnn_training_curves.png")
    # prefer valid beam4 if present
    metric_col = f"{args.split}_beam4_bleu"
    plot_bleu_bar(df_master, fig_dir / "rnn_attention_comparison.png", metric_col=metric_col)

    print(f"[analyze_rnn_results] Wrote: {out_csv}")
    print(f"[analyze_rnn_results] Figures under: {fig_dir}")


if __name__ == "__main__":
    main()
