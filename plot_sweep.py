"""
Bar chart of exact-match accuracy for baseline + selected LoRA ranks.
One figure per dataset.

Usage:
    python plot_sweep.py --output_dir output/sweep --save_dir output/sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PLOT_RANKS = [4, 8, 16, 32]


def load_results(output_dir: str):
    lora_results = {}   # {dataset: {rank: exact_match}}
    baseline_results = {}  # {dataset: exact_match}

    for result_file in sorted(Path(output_dir).glob("*/eval_result.json")):
        with open(result_file) as f:
            data = json.load(f)

        dataset = data.get("dataset")
        rank = data.get("rank")
        exact_match = data.get("exact_match")
        is_baseline = data.get("baseline", False)

        if dataset is None or exact_match is None:
            print(f"Skipping {result_file} (missing fields)")
            continue

        if is_baseline or rank is None:
            baseline_results[dataset] = exact_match
            print(f"Loaded baseline: dataset={dataset} exact_match={exact_match:.2f}%")
        else:
            lora_results.setdefault(dataset, {})[rank] = exact_match
            print(f"Loaded LoRA: dataset={dataset} rank={rank} exact_match={exact_match:.2f}%")

    return lora_results, baseline_results


def plot_dataset(dataset, rank_scores, baseline_score, save_path):
    labels = ["Baseline"] + [f"r={r}" for r in PLOT_RANKS]
    values = []

    if baseline_score is not None:
        values.append(baseline_score)
    else:
        values.append(0.0)
        print(f"Warning: no baseline for {dataset}, using 0")

    for r in PLOT_RANKS:
        values.append(rank_scores.get(r, 0.0))
        if r not in rank_scores:
            print(f"Warning: no result for {dataset} rank={r}, using 0")

    colors = ["#d62728"] + ["#1f77b4"] * len(PLOT_RANKS)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors, width=0.55, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    title = dataset.replace("_", " ").title()
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Exact Match (%)", fontsize=12)
    ax.set_title(f"DiffuGPT LoRA — {title}", fontsize=13)
    ax.set_ylim(0, max(values) * 1.25 + 1)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/sweep")
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    save_dir = Path(args.save_dir or args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    lora_results, baseline_results = load_results(args.output_dir)

    if not lora_results:
        print("No LoRA eval_result.json files found. Run eval_sweep_all.sh first.")
    else:
        for dataset, rank_scores in sorted(lora_results.items()):
            baseline = baseline_results.get(dataset)
            save_path = save_dir / f"sweep_accuracy_{dataset}.png"
            plot_dataset(dataset, rank_scores, baseline, save_path)
