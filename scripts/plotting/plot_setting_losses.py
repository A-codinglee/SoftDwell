import argparse
import csv
import math
import os
import os
from pathlib import Path

import matplotlib.pyplot as plt


def mean(values):
    return sum(values) / len(values) if values else 0.0


def sample_sd(values):
    n = len(values)
    if n <= 1:
        return 0.0
    m = mean(values)
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return math.sqrt(var)


def find_run_files(root: Path, k: int, j: int):
    pattern = f"{k}Det_{j}Res_*/metrics.csv"
    return sorted(root.glob(pattern))


def read_metrics_csv(path: Path):
    rows = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "train_loss", "val_loss"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{path} must contain columns: epoch, train_loss, val_loss"
            )

        for row in reader:
            epoch = int(row["epoch"])
            train_loss = float(row["train_loss"])
            val_loss = float(row["val_loss"])
            rows[epoch] = {
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
    return rows


def aggregate_runs(run_dicts):
    if not run_dicts:
        raise ValueError("No runs provided for aggregation.")

    common_epochs = set(run_dicts[0].keys())
    for run in run_dicts[1:]:
        common_epochs &= set(run.keys())

    common_epochs = sorted(common_epochs)

    aggregated = []
    for epoch in common_epochs:
        train_vals = [run[epoch]["train_loss"] for run in run_dicts]
        val_vals = [run[epoch]["val_loss"] for run in run_dicts]

        aggregated.append({
            "epoch": epoch,
            "train_mean": mean(train_vals),
            "train_sd": sample_sd(train_vals),
            "val_mean": mean(val_vals),
            "val_sd": sample_sd(val_vals),
        })

    return aggregated


def save_aggregated_csv(rows, out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mean", "train_sd", "val_mean", "val_sd"])
        for row in rows:
            writer.writerow([
                row["epoch"],
                row["train_mean"],
                row["train_sd"],
                row["val_mean"],
                row["val_sd"],
            ])


def plot_aggregated(rows, k, j, out_png=None):
    filtered_rows = [row for row in rows if row["epoch"] >= 2]

    epochs = [row["epoch"] for row in filtered_rows]

    train_mean = [row["train_mean"] for row in filtered_rows]
    train_low = [row["train_mean"] - row["train_sd"] for row in filtered_rows]
    train_high = [row["train_mean"] + row["train_sd"] for row in filtered_rows]

    val_mean = [row["val_mean"] for row in filtered_rows]
    val_low = [row["val_mean"] - row["val_sd"] for row in filtered_rows]
    val_high = [row["val_mean"] + row["val_sd"] for row in filtered_rows]

    plt.figure(figsize=(7, 5))

    plt.plot(epochs, train_mean, label="Train loss (mean)", marker="o", markersize=3)
    plt.fill_between(epochs, train_low, train_high, alpha=0.2)

    plt.plot(epochs, val_mean, label="Validation loss (mean)", marker="o", markersize=3)
    plt.fill_between(epochs, val_low, val_high, alpha=0.2)

    plt.yscale("log")
    plt.ylim(0.22, 0.62)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate metrics.csv files for a given K/J setting and plot mean ± SD loss curves."
    )
    parser.add_argument("--root", type=str, default=".", help="Root directory containing run folders")
    parser.add_argument("--k", type=int, required=True, help="Detector count, e.g. 4")
    parser.add_argument("--j", type=int, required=True, help="Resolution, e.g. 11")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Output prefix. Saves both <prefix>_loss.png and <prefix>_aggregated.csv",
    )

    args = parser.parse_args()

    root = Path(args.root)
    files = find_run_files(root, args.k, args.j)

    if not files:
        raise FileNotFoundError(
            f"No matching files found for pattern: {args.k}Det_{args.j}Res_*/metrics.csv under {root}"
        )

    print("Found files:")
    for f in files:
        print(f"  {f}")

    run_dicts = [read_metrics_csv(f) for f in files]
    aggregated = aggregate_runs(run_dicts)

    out_png = None
    out_csv = None
    if args.save is not None:
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        prefix = Path(args.save)
        out_png = prefix / f"{prefix.name}_loss.png"
        out_csv = prefix / f"{prefix.name}_aggregated.csv"

        save_aggregated_csv(aggregated, out_csv)
        print(f"Saved aggregated CSV: {out_csv}")

    plot_aggregated(aggregated, args.k, args.j, out_png=out_png)

    if out_png is not None:
        print(f"Saved plot: {out_png}")


if __name__ == "__main__":
    main()