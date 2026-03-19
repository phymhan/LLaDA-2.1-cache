#!/usr/bin/env python3
"""
Plot AR-ness results from analyze_arness.py.

Produces two figures:
  1. Local AR-ness vs block_length (curves for k=1,2,3,4)
  2. Global AR-ness vs block_length (curves for k=1,2,3,4)
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# FIGSIZE = (5, 4)
FIGSIZE = (3, 2.5)

def plot_arness(data, metric, title, ax, max_k):
    block_lengths = data["block_lengths"]
    results = data["results"]

    for k in range(1, max_k + 1):
        means = [results[str(bl)][f"{metric}_mean"][str(k)] for bl in block_lengths]
        stds = [results[str(bl)][f"{metric}_std"][str(k)] for bl in block_lengths]
        means = np.array(means)
        stds = np.array(stds)
        ax.errorbar(block_lengths, means, yerr=stds, marker="o", label=f"@k={k}",
                     capsize=3, linewidth=1.5, markersize=5)

    ax.set_xlabel("Block Length")
    ax.set_ylabel(f"{title}")
    # ax.set_title(title)
    ax.set_xticks(block_lengths)
    ax.set_xticklabels([str(b) for b in block_lengths])
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(
        description="Plot AR-ness results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="arness_results.json")
    parser.add_argument("--output_prefix", type=str, default="arness",
                        help="Output prefix: produces {prefix}_local.pdf and {prefix}_global.pdf")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--max_k", type=int, default=None, help="Override max_k from data (default: use all)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    max_k = args.max_k or data["max_k"]
    model_name = data.get("model_dir", "").split("/")[-1]
    task = data.get("task", "")
    n = data.get("n_samples", "?")
    gl = data.get("gen_length", "?")
    subtitle = f"{model_name}, {task}, n={n}, gen_length={gl}"

    # Local AR-ness
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    plot_arness(data, "local", "Local AR-ness", ax, max_k)
    # ax.set_title(f"Local AR-ness\n{subtitle}", fontsize=10)
    fig.tight_layout()
    local_path = f"{args.output_prefix}_local.{args.format}"
    fig.savefig(local_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {local_path}")
    plt.close(fig)

    # Global AR-ness
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    plot_arness(data, "global", "Global AR-ness", ax, max_k)
    # ax.set_title(f"Global AR-ness\n{subtitle}", fontsize=10)
    fig.tight_layout()
    global_path = f"{args.output_prefix}_global.{args.format}"
    fig.savefig(global_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {global_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
