#!/usr/bin/env python3
# scripts/plotting/plot_sets_by_recording.py
"""
The right panel of plot_noise_cv's noise_cv.png ("Does training on 4 recordings help?") for several sets
of runs of the same cases, e.g. versions of the pipeline: one block per unseen recording, one line per set.
Reads each set's comparison/*.csv (made by plot_noise_cv.py). Writes by_recording.png to --out-dir.

Usage:
  python3 -m scripts.plotting.plot_sets_by_recording --dirs <cv-dir> ... --labels A ... --out-dir <dir>
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from scripts.plotting.plot_arch_compare import MULTI, load_set
from scripts.plotting.plot_online_results import BLUE_RAMP, GRID, INK2, MUTED, SURFACE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dirs", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    sets = [load_set(d)[0] for d in a.dirs]
    keys = set.intersection(*[set(s) for s in sets])
    tests = sorted({k[1] for k in keys})
    n = len(sets)
    cols = [BLUE_RAMP[i] for i in np.linspace(4, 12, n).round().astype(int)]
    gap = 0.8 / n

    fig, ax = plt.subplots(figsize=(11, 1.15 * len(tests) * max(1, n / 2) + 2.6))
    yt = []
    for i, te in enumerate(tests):
        y0 = (len(tests) - 1 - i) * (n * gap + 0.5)
        yt.append(y0 + (n - 1) * gap / 2)
        for j, (S, col) in enumerate(zip(sets, cols)):
            y = y0 + (n - 1 - j) * gap
            ones = [S[k]["rae"] for k in keys if k[1] == te and k[0] != MULTI]
            if ones:
                ax.plot([min(ones), max(ones)], [y, y], color=GRID, lw=2, zorder=1)
                ax.scatter(ones, [y] * len(ones), s=34, facecolor=SURFACE, edgecolor=col, linewidth=1.5, zorder=2)
            if (MULTI, te) in S:
                m = S[(MULTI, te)]
                ax.scatter([m["val"]], [y], s=44, marker="D", facecolor=SURFACE, edgecolor=MUTED, linewidth=1.2,
                           zorder=2)
                ax.scatter([m["rae"]], [y], s=110, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
                ax.text(m["rae"], y + gap * 0.42, f"{m['rae']:.3f}", ha="center", va="bottom", fontsize=7.5,
                        color=INK2)
    ax.set_yticks(yt, tests)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("test RAE on the recording (lower is better)"); ax.set_ylabel("tested on")
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=c, mec=SURFACE, label=f"{l} (top to bottom: line {j + 1})")
               for j, (l, c) in enumerate(zip(a.labels, cols))]
    handles += [Line2D([], [], marker="o", ls="", ms=10, color=MUTED, mec=SURFACE,
                       label="large dot: trained on the 4 other recordings (value printed)"),
                Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=MUTED, mew=1.5,
                       label="small circles: trained on 1 other recording, one per model"),
                Line2D([], [], marker="D", ls="", ms=6, mfc=SURFACE, mec=MUTED, mew=1.2,
                       label="diamond: the 4-recording model on its own training noise (validation)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=1, fontsize=8.8,
              labelcolor=INK2)
    fig.suptitle("Does training on 4 recordings help? Each version of the pipeline, per unseen recording",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(a.out_dir, exist_ok=True)
    fig.savefig(os.path.join(a.out_dir, "by_recording.png"), dpi=150); plt.close(fig)
    print(f"wrote {os.path.join(a.out_dir, 'by_recording.png')}")


if __name__ == "__main__":
    main()
