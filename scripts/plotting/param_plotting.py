#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def plot_hohd_params(csv_path: str, out_dir: str = "outputs/graphs"):
    """Plot θ and τ evolution per detector from a logged CSV file."""
    os.makedirs(out_dir, exist_ok=True)
    data = defaultdict(lambda: {"epoch": [], "theta": [], "tau": []})

    title_fs = 18
    label_fs = 15
    legend_fs = 11
    tick_fs = 13

    # --- Load data ---
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            k = int(row["k"])
            data[k]["epoch"].append(int(row["epoch"]))
            data[k]["theta"].append(float(row["theta"]))
            data[k]["tau"].append(float(row["tau"]))

    # --- Sort by epoch ---
    for k in data:
        zipped = sorted(zip(data[k]["epoch"], data[k]["theta"], data[k]["tau"]))
        e, th, ta = zip(*zipped)
        data[k]["epoch"], data[k]["theta"], data[k]["tau"] = list(e), list(th), list(ta)

    # --- Create one window with two subplots ---
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    # --- Fix colors once so both subplots match exactly ---
    sorted_keys = sorted(data.keys())
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_map = {k: default_colors[i % len(default_colors)] for i, k in enumerate(sorted_keys)}

    handles = []
    labels = []

    # --- Plot θ ---
    for k in sorted_keys:
        d = data[k]
        line, = axes[0].plot(d["epoch"], d["theta"], color=color_map[k], label=f"k={k}")
        handles.append(line)
        labels.append(f"k={k}")

    axes[0].set_title("θ (theta)", fontsize=title_fs)
    axes[0].set_ylabel("θ value", fontsize=label_fs)
    axes[0].set_ylim(0.0, 1.0)

    # --- Plot τ ---
    for k in sorted_keys:
        d = data[k]
        axes[1].plot(d["epoch"], d["tau"], color=color_map[k], label=f"k={k}")

    axes[1].set_title("τ (tau)", fontsize=title_fs)
    axes[1].set_xlabel("Epoch", fontsize=label_fs)
    axes[1].set_ylabel("τ value", fontsize=label_fs)
    axes[1].set_ylim(0.01, 1)
    axes[1].set_yscale("log")  # Log scale for better visibility of small values

    for ax in axes:
        ax.grid(True, which="major", alpha=0.3, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=tick_fs)

    # --- Common legend outside the plots ---
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, -0.02),
        frameon=True,
        fontsize=legend_fs,
    )

    # leave space at top for the shared legend
    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # save combined figure
    save_path = os.path.join(out_dir, "theta_tau_per_k.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")

    # show on screen if running interactively
    plt.show()

    plt.close(fig)
    print(f"[hohd] Saved plot to: {save_path}")


# --- CLI entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot θ/τ evolution per detector from hohd CSV log.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file (e.g., outputs/hohd_params.csv)")
    parser.add_argument("--out", type=str, default="outputs/graphs", help="Output directory for saved plots")
    args = parser.parse_args()

    plot_hohd_params(args.csv, args.out)