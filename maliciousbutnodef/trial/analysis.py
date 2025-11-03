"""Analysis and visualization for federated learning experiments.

Usage:
  python -m trial.analysis --logdir logs/ --outdir figs/ --runs baseline:0.0,mal10:0.1,mal20:0.2,mal30:0.3

This script expects for each run (identified by a name and malicious fraction):
  logs/<run_name>/global_metrics.csv
  logs/<run_name>/client_metrics.jsonl (optional extended per-client data)
Future extensions: confusion matrices, per-class accuracy, update norms.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})

@dataclass
class RunSpec:
    name: str
    fraction: float
    path: Path


def load_global_metrics(path: Path) -> pd.DataFrame:
    csv_path = path / "global_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def panel_training_dynamics(runs: List[RunSpec], outdir: Path):
    # Panel A: Accuracy
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.0))
    ax_acc, ax_loss = axes
    for r in runs:
        df = load_global_metrics(r.path)
        ax_acc.plot(df['round'], df['accuracy'], label=f"{r.name} ({int(r.fraction*100)}% mal)")
        # Placeholder for train accuracy if available
        ax_loss.plot(df['round'], 1-df['accuracy'], label=f"{r.name}")  # stand-in for loss
    ax_acc.set_ylabel("Test Accuracy")
    ax_acc.set_xlabel("Round")
    ax_acc.legend(frameon=False)
    ax_loss.set_ylabel("Proxy Loss (1-acc)")
    ax_loss.set_xlabel("Round")
    fig.suptitle("Training Dynamics (Placeholder)")
    fig.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(outdir / "fig1_training_dynamics.png")
    plt.close(fig)


def figure_attack_impact(runs: List[RunSpec], outdir: Path):
    records = []
    for r in runs:
        df = load_global_metrics(r.path)
        final_acc = df['accuracy'].iloc[-1]
        records.append({"fraction": r.fraction, "final_acc": final_acc})
    imp = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(data=imp, x='fraction', y='final_acc', ax=ax)
    ax.set_xlabel("Malicious Fraction")
    ax.set_ylabel("Final Accuracy")
    fig.tight_layout()
    outdir.mkdir(exist_ok=True, parents=True)
    fig.savefig(outdir / "fig2_attack_impact.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logdir', type=Path, default=Path('logs'))
    ap.add_argument('--outdir', type=Path, default=Path('figs'))
    ap.add_argument('--runs', type=str, required=True, help='Comma separated name:fraction list, e.g. baseline:0.0,mal10:0.1')
    args = ap.parse_args()

    runs: List[RunSpec] = []
    for token in args.runs.split(','):
        name, frac = token.split(':')
        runs.append(RunSpec(name=name, fraction=float(frac), path=args.logdir / name))

    panel_training_dynamics(runs, args.outdir)
    figure_attack_impact(runs, args.outdir)

    print(f"Saved figures to {args.outdir}")

if __name__ == '__main__':
    main()
