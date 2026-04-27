"""
Plot exact-match accuracy vs LoRA rank for each dataset.

Usage:
    python plot_sweep.py --output_dir output/sweep --save_path sweep_accuracy.png

Reads eval_result.json files written by eval_sweep_all.sh.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_results(output_dir: str):
    results = {}
    for result_file in sorted(Path(output_dir).glob("*/eval_result.json")):
        with open(result_file) as f:
            data = json.load(f)
        dataset = data.get("dataset")
        rank = data.get("rank")
        exact_match = data.get("exact_match")
        if dataset is None or rank is None or exact_match is None:
            print(f"Skipping {result_file} (missing fields: {data})")
            continue
        results.setdefault(dataset, {})[rank] = exact_match
        print(f"Loaded: dataset={dataset} rank={rank} exact_match={exact_match:.2f}%")
    return results


def plot_results(results: dict, save_path: str):
    fig, ax = plt.subplots(figsize=(7, 5))

    markers = ["o", "s", "^", "D", "v", "P"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, (dataset, rank_scores) in enumerate(sorted(results.items())):
        ranks = sorted(rank_scores.keys())
        scores = [rank_scores[r] for r in ranks]
        label = dataset.replace("_", " ").title()
        ax.plot(ranks, scores, marker=markers[i % len(markers)],
                color=colors[i % len(colors)], label=label, linewidth=2, markersize=8)
        for r, s in zip(ranks, scores):
            ax.annotate(f"{s:.1f}", (r, s), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Exact Match (%)", fontsize=12)
    ax.set_title("DiffuGPT LoRA: Exact Match vs Rank", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/sweep")
    parser.add_argument("--save_path", type=str, default="sweep_accuracy.png")
    args = parser.parse_args()

    results = load_results(args.output_dir)
    if not results:
        print("No eval_result.json files found. Run eval_sweep_all.sh first.")
    else:
        plot_results(results, args.save_path)
