import csv
import matplotlib.pyplot as plt
from collections import defaultdict

csv_path = "outputs/hohd_params.csv"  # <-- your file

# Load rows
data = defaultdict(lambda: {"epoch": [], "theta": [], "tau": []})
with open(csv_path, newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        k = int(row["k"])
        data[k]["epoch"].append(int(row["epoch"]))
        data[k]["theta"].append(float(row["theta"]))
        data[k]["tau"].append(float(row["tau"]))

# Sort by epoch for each k
for k in data:
    zipped = sorted(zip(data[k]["epoch"], data[k]["theta"], data[k]["tau"]))
    e, th, ta = zip(*zipped)
    data[k]["epoch"], data[k]["theta"], data[k]["tau"] = list(e), list(th), list(ta)

# Plot theta
plt.figure()#!/usr/bin/env python3
import os
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

def plot_hohd_params(csv_path: str, out_dir: str = "outputs/graphs"):
    """Plot θ and τ evolution per detector from a logged CSV file."""
    os.makedirs(out_dir, exist_ok=True)
    data = defaultdict(lambda: {"epoch": [], "theta": [], "tau": []})

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

    # --- Plot θ ---
    plt.figure()
    for k, d in sorted(data.items()):
        plt.plot(d["epoch"], d["theta"], label=f"k={k}")
    plt.title("θ (theta) per detector")
    plt.xlabel("Epoch")
    plt.ylabel("θ value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "theta_per_k.png"), dpi=200)

    # --- Plot τ ---
    plt.figure()
    for k, d in sorted(data.items()):
        plt.plot(d["epoch"], d["tau"], label=f"k={k}")
    plt.title("τ (tau) per detector")
    plt.xlabel("Epoch")
    plt.ylabel("τ value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tau_per_k.png"), dpi=200)

    plt.close("all")
    print(f"[hohd] Saved plots to: {out_dir}")

# --- CLI entry ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot θ/τ evolution per detector from hohd CSV log.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file (e.g., outputs/hohd_params.csv)")
    parser.add_argument("--out", type=str, default="outputs/graphs", help="Output directory for saved plots")
    args = parser.parse_args()

    plot_hohd_params(args.csv, args.out)

for k, d in sorted(data.items()):
    plt.plot(d["epoch"], d["theta"], label=f"k={k}")
plt.title("theta per detector")
plt.xlabel("epoch"); plt.ylabel("theta")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/graphs/theta_per_k.png", dpi=200)

# Plot tau
plt.figure()
for k, d in sorted(data.items()):
    plt.plot(d["epoch"], d["tau"], label=f"k={k}")
plt.title("tau per detector")
plt.xlabel("epoch"); plt.ylabel("tau")
plt.legend(); plt.tight_layout()
plt.savefig("outputs/graphs/tau_per_k.png", dpi=200)

plt.show()
