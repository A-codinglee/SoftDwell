#!/usr/bin/env python3
# scripts/plotting/plot_speed.py
"""
Training speed of several sets of noise cross-validation runs of the same cases (e.g. versions of the
pipeline), from each run's train_val_metrics.csv. Writes speed.png and speed.csv to --out-dir:
time per epoch (training + validation) and total training time, per case and per set.

Usage:
  python3 -m scripts.plotting.plot_speed --dirs <cv-dir> <cv-dir> ... --labels A B ... --out-dir <dir>
"""
import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_online_results import BLUE_RAMP, GRID, INK, INK2, MUTED, SURFACE


def load(cv_dir):
    """{case: (mean epoch seconds over epochs 3.., epochs, total training hours)}"""
    out = {}
    for run in sorted(glob.glob(os.path.join(cv_dir, "*", "seed1"))):
        m = re.search(r"(holdout|only)-.*$", os.path.basename(os.path.dirname(run)))
        path = os.path.join(run, "train_val_metrics.csv")
        if not m or not os.path.isfile(path):
            continue
        t = np.array([float(r["epoch_time_s"]) for r in csv.DictReader(open(path))])
        if t.size > 3:
            out[m.group(0)] = (t[2:].mean(), t.size, t.sum() / 3600)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    sets = [load(d) for d in a.dirs]
    cases = sorted(set.intersection(*[set(s) for s in sets]), key=lambda c: (not c.startswith("holdout"), c))
    os.makedirs(a.out_dir, exist_ok=True)
    cols = [BLUE_RAMP[i] for i in np.linspace(3, 12, len(sets)).round().astype(int)]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [2.6, 1]})
    w = 0.8 / len(sets)
    x = np.arange(len(cases))
    for j, (s, lab, col) in enumerate(zip(sets, a.labels, cols)):
        ax.bar(x + (j - (len(sets) - 1) / 2) * w, [s[c][0] for c in cases], width=w * 0.92, color=col, label=lab,
               edgecolor=SURFACE, linewidth=0.5)
    ax.set_xticks(x, [c.replace("holdout-", "held out\n").replace("only-", "only\n") for c in cases], fontsize=8.5)
    ax.set_ylabel("seconds per epoch (epochs 3 to end)")
    ax.set_title("Time per epoch, per case")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.legend(fontsize=9, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0.0, 1.07), ncol=len(sets), frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)

    means = [np.mean([s[c][0] for c in cases]) for s in sets]
    hours = [np.mean([s[c][2] for c in cases]) for s in sets]
    bars = bx.bar(range(len(sets)), means, color=cols, edgecolor=SURFACE, width=0.7)
    for i, (b, m, h) in enumerate(zip(bars, means, hours)):
        bx.text(b.get_x() + b.get_width() / 2, m, f"{m:.1f} s\n{means[0] / m:.2f}x\n{h:.1f} h/run",
                ha="center", va="bottom", fontsize=9, color=INK)
    bx.set_xticks(range(len(sets)), [l.replace(" + ", "\n+ ") for l in a.labels], fontsize=8.5)
    bx.set_ylim(0, max(means) * 1.35)
    bx.set_title("Average over the cases")
    bx.yaxis.grid(True); bx.set_axisbelow(True)
    fig.suptitle("Training speed of the noise cross-validation runs", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.01, 0.905, "Same ten cases and settings in every set; speed-up relative to the first set; "
             "h/run = training time per run (epochs differ a little between runs).", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(os.path.join(a.out_dir, "speed.png"), dpi=150); plt.close(fig)

    with open(os.path.join(a.out_dir, "speed.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["case"] + [f"{l}: {k}" for l in a.labels for k in ("s/epoch", "epochs", "hours")])
        for c in cases:
            wr.writerow([c] + [f"{v:.3f}" if isinstance(v, float) else v for s in sets for v in s[c]])
    print("average s/epoch: " + ", ".join(f"{l} {m:.1f}" for l, m in zip(a.labels, means)))


if __name__ == "__main__":
    main()
