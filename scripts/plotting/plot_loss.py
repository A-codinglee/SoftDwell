#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import numpy as np

def plot_loss(metrics_csv: str, out_path: str = "outputs/graphs/loss_curve.png"):
    """Plot training and validation loss curves from a CSV file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    epochs, train_losses, val_losses = [], [], []

    # --- Load data ---
    with open(metrics_csv, "r") as f:
        first_line = f.readline().strip()

        # Check if first line is header
        if not any(c.isalpha() for c in first_line):
            epoch, train, val = first_line.split(",")
            epochs.append(int(epoch))
            train_losses.append(float(train))
            val_losses.append(float(val))

        for line in f:
            if not line.strip():
                continue
            epoch, train, val = line.strip().split(",")
            epochs.append(int(epoch))
            train_losses.append(float(train))
            val_losses.append(float(val))

    # --- Convert to arrays ---
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # --- Omit first epoch(s) for readability ---
    mask = epochs > 1

    plt.figure(figsize=(7, 4))
    plt.plot(epochs[mask], train_losses[mask], marker="o", markersize=4, label="Train")
    plt.plot(epochs[mask], val_losses[mask], marker="s", label="Validation")

    plt.xlabel("Epoch", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.title("Training and Validation Loss", fontsize=18)
    plt.legend(fontsize=13)
    plt.grid(True, which="major", alpha=0.3, linewidth=0.5)
    # plt.yscale("log")
    ax = plt.gca()
    ax.tick_params(axis="both", labelsize=13)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plot] Saved loss curve to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to metrics CSV file (e.g., outputs/metrics.csv)")
    parser.add_argument("--out", type=str, default="outputs/graphs/loss_curve.png",
                        help="Output path for the saved plot image")
    args = parser.parse_args()

    plot_loss(args.csv, args.out)