#!/usr/bin/env python3
"""
Plot per-step decoding confidence from analyze_confidence.py.

Produces 4 figures per task:
  1. Static,  x = step index,        y = mean confidence
  2. Static,  x = normalized progress, y = mean confidence
  3. Dynamic, x = step index,        y = mean confidence
  4. Dynamic, x = normalized progress, y = mean confidence

Each figure has one curve per block_length.
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# FIGSIZE = (5, 4)
FIGSIZE = (3, 2.5)

def aggregate_by_step(all_samples):
    """
    Aggregate confidence across samples by step index.

    all_samples: list of samples, each is list of steps, each step is list of floats.
    Returns: (step_indices, mean_conf, std_conf, counts) arrays.
    """
    max_steps = max(len(s) for s in all_samples) if all_samples else 0
    step_means = []  # per-step mean confidence per sample
    for step_idx in range(max_steps):
        vals = []
        for sample in all_samples:
            if step_idx < len(sample) and sample[step_idx]:
                vals.append(np.mean(sample[step_idx]))
        if vals:
            step_means.append((step_idx, np.mean(vals), np.std(vals), len(vals)))
        else:
            step_means.append((step_idx, np.nan, np.nan, 0))
    if not step_means:
        return np.array([]), np.array([]), np.array([]), np.array([])
    arr = np.array(step_means)
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def aggregate_ntokens_by_step(all_samples):
    """Aggregate number of decoded tokens per step across samples."""
    max_steps = max(len(s) for s in all_samples) if all_samples else 0
    out = []
    for step_idx in range(max_steps):
        vals = []
        for sample in all_samples:
            if step_idx < len(sample) and sample[step_idx]:
                vals.append(len(sample[step_idx]))
        if vals:
            out.append((step_idx, np.mean(vals), np.std(vals)))
        else:
            out.append((step_idx, np.nan, np.nan))
    if not out:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(out)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def aggregate_ntokens_by_normalized(all_samples, n_bins=100):
    """Aggregate number of decoded tokens by normalized progress."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_vals = [[] for _ in range(n_bins)]
    for sample in all_samples:
        n_steps = len(sample)
        if n_steps == 0:
            continue
        for step_idx, step_confs in enumerate(sample):
            if not step_confs:
                continue
            progress = step_idx / n_steps
            bin_idx = min(int(progress * n_bins), n_bins - 1)
            bin_vals[bin_idx].append(len(step_confs))
    mean_n = np.array([np.mean(v) if v else np.nan for v in bin_vals])
    std_n = np.array([np.std(v) if v else np.nan for v in bin_vals])
    return bin_centers, mean_n, std_n


def aggregate_by_normalized(all_samples, n_bins=100):
    """
    Aggregate confidence by normalized progress (step / total_steps).

    Returns: (bin_centers, mean_conf, std_conf) arrays.
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_vals = [[] for _ in range(n_bins)]

    for sample in all_samples:
        n_steps = len(sample)
        if n_steps == 0:
            continue
        for step_idx, step_confs in enumerate(sample):
            if not step_confs:
                continue
            progress = step_idx / n_steps
            bin_idx = min(int(progress * n_bins), n_bins - 1)
            bin_vals[bin_idx].append(np.mean(step_confs))

    mean_conf = np.array([np.mean(v) if v else np.nan for v in bin_vals])
    std_conf = np.array([np.std(v) if v else np.nan for v in bin_vals])
    return bin_centers, mean_conf, std_conf


def plot_by_step(data, strategy, ax, block_lengths, smooth_window=0):
    """Plot confidence vs step index for one strategy."""
    results = data["results"]
    for bl in block_lengths:
        entry = results[str(bl)].get(strategy, [])
        samples = entry.get("confidences", entry) if isinstance(entry, dict) else entry
        if not samples:
            continue
        steps, means, stds, counts = aggregate_by_step(samples)
        if len(steps) == 0:
            continue
        # Optional smoothing
        if smooth_window > 1:
            kernel = np.ones(smooth_window) / smooth_window
            means = np.convolve(means, kernel, mode="same")
            stds = np.convolve(stds, kernel, mode="same")
        valid = ~np.isnan(means)
        ax.plot(steps[valid], means[valid], label=f"B={bl}", linewidth=1.2, alpha=0.85)
        ax.fill_between(steps[valid], (means - stds)[valid], (means + stds)[valid], alpha=0.1)


def plot_by_normalized(data, strategy, ax, block_lengths, n_bins=100):
    """Plot confidence vs normalized progress for one strategy."""
    results = data["results"]
    for bl in block_lengths:
        entry = results[str(bl)].get(strategy, [])
        samples = entry.get("confidences", entry) if isinstance(entry, dict) else entry
        if not samples:
            continue
        centers, means, stds = aggregate_by_normalized(samples, n_bins=n_bins)
        valid = ~np.isnan(means)
        ax.plot(centers[valid], means[valid], label=f"B={bl}", linewidth=1.2, alpha=0.85)
        ax.fill_between(centers[valid], (means - stds)[valid], (means + stds)[valid], alpha=0.1)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-step decoding confidence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=str, default="confidence_results.json")
    parser.add_argument("--output_prefix", type=str, default="confidence",
                        help="Prefix for output files")
    parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png", "svg"])
    parser.add_argument("--n_bins", type=int, default=100,
                        help="Number of bins for normalized progress plots")
    parser.add_argument("--smooth", type=int, default=0,
                        help="Smoothing window for step-based plots (0 = no smoothing)")
    parser.add_argument("--block_lengths", type=str, default=None,
                        help="Comma-separated block lengths to plot (default: all from data)")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    if args.block_lengths is not None:
        block_lengths = [int(x) for x in args.block_lengths.split(",")]
    else:
        block_lengths = data["block_lengths"]
    model_name = data.get("model_dir", "").split("/")[-1]
    task = data.get("task", "?")
    n = data.get("n_samples", "?")
    gl = data.get("gen_length", "?")
    subtitle = f"{model_name}, {task}, n={n}, gen_length={gl}"

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # --- Static plots (single y-axis) ---
    for mode, xlabel, title in [
        ("step",       "Step Index",          "Confidence (static, by step)"),
        ("normalized", "Normalized Progress", "Confidence (static, normalized)"),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
        if mode == "step":
            plot_by_step(data, "static", ax, block_lengths, smooth_window=args.smooth)
        else:
            plot_by_normalized(data, "static", ax, block_lengths, n_bins=args.n_bins)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean Confidence")
        # ax.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fname = f"{args.output_prefix}_static_{mode}.{args.format}"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
        plt.close(fig)

    # --- Dynamic plots (twin y-axis: confidence + n_tokens) ---
    results = data["results"]
    for mode, xlabel, title in [
        ("step",       "Step Index",          "Confidence (dynamic, by step)"),
        ("normalized", "Normalized Progress", "Confidence (dynamic, normalized)"),
    ]:
        fig, ax_conf = plt.subplots(1, 1, figsize=FIGSIZE)
        ax_ntok = ax_conf.twinx()

        conf_handles = []
        ntok_handles = []
        for ci, bl in enumerate(block_lengths):
            color = colors[ci % len(colors)]
            entry = results[str(bl)].get("dynamic", [])
            samples = entry.get("confidences", entry) if isinstance(entry, dict) else entry
            if not samples:
                continue

            if mode == "step":
                steps, means, stds, _ = aggregate_by_step(samples)
                nsteps, nmeans, nstds = aggregate_ntokens_by_step(samples)
            else:
                steps, means, stds = aggregate_by_normalized(samples, n_bins=args.n_bins)
                nsteps, nmeans, nstds = aggregate_ntokens_by_normalized(samples, n_bins=args.n_bins)

            if len(steps) == 0:
                continue

            if mode == "step" and args.smooth > 1:
                kernel = np.ones(args.smooth) / args.smooth
                means = np.convolve(means, kernel, mode="same")
                stds = np.convolve(stds, kernel, mode="same")
                nmeans = np.convolve(nmeans, kernel, mode="same")

            valid_c = ~np.isnan(means)
            valid_n = ~np.isnan(nmeans)

            h1, = ax_conf.plot(steps[valid_c], means[valid_c],
                               color=color, linewidth=1.2, alpha=0.85, label=f"B={bl}")
            ax_conf.fill_between(steps[valid_c], (means - stds)[valid_c], (means + stds)[valid_c],
                                 color=color, alpha=0.08)
            conf_handles.append(h1)

            h2, = ax_ntok.plot(nsteps[valid_n], nmeans[valid_n],
                               color=color, linewidth=1.0, alpha=0.5, linestyle="--")
            ntok_handles.append(h2)

        ax_conf.set_xlabel(xlabel)
        ax_conf.set_ylabel("Mean Confidence")
        ax_ntok.set_ylabel("Tokens Decoded / Step", alpha=0.6)
        # ax_conf.set_title(f"{title}\n{subtitle}", fontsize=10)
        ax_conf.set_ylim(0, 1.05)
        ax_conf.grid(True, alpha=0.3)

        # Combined legend
        if conf_handles:
            extra = plt.Line2D([], [], color="gray", linestyle="--", linewidth=1.0, alpha=0.5, label="n_tokens (--)")
            ax_conf.legend(handles=conf_handles + [extra], loc="best")

        fig.tight_layout()
        fname = f"{args.output_prefix}_dynamic_{mode}.{args.format}"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved: {fname}")
        plt.close(fig)


if __name__ == "__main__":
    main()
