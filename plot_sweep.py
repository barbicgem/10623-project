"""
Plot exact-match accuracy vs LoRA rank, one figure per dataset.
Baseline (no LoRA) shown as a horizontal dashed line.

Usage:
    python plot_sweep.py --output_dir output/sweep --save_dir output/sweep
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


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
    fig, ax = plt.subplots(figsize=(7, 5))

    ranks = sorted(rank_scores.keys())
    scores = [rank_scores[r] for r in ranks]

    ax.plot(ranks, scores, marker="o", color="#1f77b4", linewidth=2,
            markersize=8, label="LoRA fine-tuned")
    for r, s in zip(ranks, scores):
        ax.annotate(f"{s:.1f}", (r, s), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)

    if baseline_score is not None:
        ax.axhline(baseline_score, color="#d62728", linestyle="--", linewidth=1.5,
                   label=f"Baseline (no LoRA): {baseline_score:.1f}%")

    title = dataset.replace("_", " ").title()
    ax.set_xlabel("LoRA Rank", fontsize=12)
    ax.set_ylabel("Exact Match (%)", fontsize=12)
    ax.set_title(f"DiffuGPT LoRA — {title}", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/sweep")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save plots (defaults to output_dir)")
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
