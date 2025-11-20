#!/usr/bin/env python3
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse

def plot_loss(metrics_csv: str, out_path: str = "outputs/graphs/loss_curve.png"):
    """Plot training and validation loss curves from a CSV file."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    epochs, train_losses, val_losses = [], [], []

    # --- Load data ---
    with open(metrics_csv, "r") as f:
        first_line = f.readline().strip()
        # Check if first line is header (contains non-numeric chars)
        if any(c.isalpha() for c in first_line):
            pass  # header line already skipped
        else:
            # If no header, treat the first line as data
            epoch, train, val = first_line.split(",")
            epochs.append(int(epoch))
            train_losses.append(float(train))
            val_losses.append(float(val))
        # Continue reading remaining lines
        for line in f:
            if not line.strip():
                continue
            epoch, train, val = line.strip().split(",")
            epochs.append(int(epoch))
            train_losses.append(float(train))
            val_losses.append(float(val))

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.yscale("log")  # log scale for stability
    ax = plt.gca()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] Saved loss curve to {out_path}")


# --- CLI Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training and validation loss curves.")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to metrics CSV file (e.g., outputs/metrics.csv)")
    parser.add_argument("--out", type=str, default="outputs/graphs/loss_curve.png",
                        help="Output path for the saved plot image")
    args = parser.parse_args()

    plot_loss(args.csv, args.out)
