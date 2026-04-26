#!/usr/bin/env python3
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_all_outputs(pred_file: str, targ_file: str, out_file: str):
    preds_hz = np.load(pred_file)
    targs_hz = np.load(targ_file)

    if preds_hz.shape != targs_hz.shape:
        raise ValueError(
            f"Shape mismatch: preds shape={preds_hz.shape}, targets shape={targs_hz.shape}"
        )

    if preds_hz.ndim != 2:
        raise ValueError(
            f"Expected 2D arrays of shape (N, D), got preds.ndim={preds_hz.ndim}"
        )

    num_outputs = preds_hz.shape[1]

    # choose subplot layout automatically
    ncols = math.ceil(math.sqrt(num_outputs))
    nrows = math.ceil(num_outputs / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    for i in range(num_outputs):
        ax = axes[i]
        x = np.clip(targs_hz[:, i], 1e-12, None)
        y = np.clip(preds_hz[:, i], 1e-12, None)

        # scatter
        ax.scatter(x, y, s=5, alpha=0.5, rasterized=True)

        # diagonal
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e2, 1e6)
        ax.set_ylim(1e2, 1e6)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"Output {i}")
        ax.set_xlabel("GT (Hz)")
        ax.set_ylabel("Pred (Hz)")
        ax.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)

    # hide unused axes
    for j in range(num_outputs, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_file, dpi=150)
    plt.close(fig)

    print(f"Saved figure to: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot prediction vs target scatter plots for all outputs."
    )
    parser.add_argument(
        "--preds",
        type=str,
        required=True,
        help="Path to predictions .npy file"
    )
    parser.add_argument(
        "--targs",
        type=str,
        required=True,
        help="Path to targets .npy file"
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output image path, e.g. outputs/scatter_all_outputs.png"
    )

    args = parser.parse_args()
    plot_all_outputs(args.preds, args.targs, args.out)