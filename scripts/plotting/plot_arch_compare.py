#!/usr/bin/env python3
# scripts/plotting/plot_arch_compare.py
"""
Compare two sets of noise cross-validation runs of the same cases (e.g. two architectures), each already
summarized by plot_noise_cv.py (reads <dir>/comparison/*.csv and the runs' train_val_metrics.csv).
Writes into --out-dir:

  arch_scatter.png         every (model, unseen recording): set A against set B for the rate error and the
                           three recording conditions; below the diagonal = set B better
  arch_per_rate.png        average test error per rate, both sets, for 4-recording and 1-recording models
  arch_val_vs_test.png     per model: error on its own training noise (validation) and on unseen noise
  arch_training_curves.png validation rate error over the epochs, both sets, one panel per case
  arch_by_recording.png    per unseen recording: 4-recording model vs single-recording models, both sets
                           (the right panel of plot_noise_cv's noise_cv.png, for two sets)
  arch_summary.csv         the averages behind the figures

Usage:
  python3 -m scripts.plotting.plot_arch_compare --a <cv-dir A> --b <cv-dir B> --labels "A" "B" --out-dir <dir>
"""
import argparse
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_online_results import BLUE, GRID, INK, INK2, MUTED, ORANGE, SURFACE

MULTI = "4 other recordings"
RATES = ["k01", "k10", "k12", "k21", "k23", "k32"]
CONDS = [("baseline", "Baseline (pA)"), ("amplitude", "Amplitude (pA)"), ("sigma", "Noise sigma (pA)")]


def model_key(trained_on):
    return MULTI if ("+" in trained_on or trained_on.startswith("the 4")) else trained_on


def load_set(cv_dir):
    """{(model, tested_on): {"rae", "val", "k01".., "baseline", "amplitude", "sigma"}} and {case: metrics rows}."""
    comp = os.path.join(cv_dir, "comparison")
    d = {}
    for r in csv.DictReader(open(os.path.join(comp, "noise_cv.csv"))):
        d.setdefault((model_key(r["trained_on"]), r["tested_on"]), {}).update(
            rae=float(r["test_rae"]), val=float(r["val_rae_on_training_noise"]), run_dir=r["run_dir"])
    for r in csv.DictReader(open(os.path.join(comp, "noise_cv_per_rate.csv"))):
        d[(model_key(r["trained_on"]), r["tested_on"])].update({k: float(r[f"rae_{k}"]) for k in RATES})
    for r in csv.DictReader(open(os.path.join(comp, "noise_cv_conditions.csv"))):
        e = d[(model_key(r["trained_on"]), r["tested_on"])]
        for c, _ in CONDS:
            e[c] = float(r[f"mean_abs_error_{c}_pA"])
    curves = {}
    for run in sorted(glob.glob(os.path.join(cv_dir, "*", "seed*"))):
        case = re.search(r"(holdout|only)-.*$", os.path.basename(os.path.dirname(run))).group(0)
        rows = list(csv.DictReader(open(os.path.join(run, "train_val_metrics.csv"))))
        curves[case] = (np.array([int(x["epoch"]) for x in rows]),
                        np.array([np.mean([float(x[f"val_rae_out_{i}"]) for i in range(1, 7)]) for x in rows]))
    return d, curves


def style(ax):
    ax.yaxis.grid(True); ax.set_axisbelow(True)


def fig_scatter(A, B, keys, labels, path):
    metrics = [("rae", "Rate error (RAE)")] + CONDS
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.6))
    for ax, (m, title) in zip(axes, metrics):
        x = np.array([A[k][m] for k in keys]); y = np.array([B[k][m] for k in keys])
        multi = np.array([k[0] == MULTI for k in keys])
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = 0.06 * (hi - lo); lo, hi = lo - pad, hi + pad
        ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=GRID, alpha=0.45, lw=0)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=0.9)
        ax.scatter(x[~multi], y[~multi], s=42, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5,
                   label="trained on 1 recording", zorder=3)
        ax.scatter(x[multi], y[multi], s=60, color=INK, edgecolor=SURFACE, linewidth=1.2,
                   label="trained on the 4 others", zorder=4)
        for k, xi, yi in zip(keys, x, y):
            if k[1] == "2023-02-01":
                ax.annotate("2023", (xi, yi), xytext=(4, -3), textcoords="offset points", fontsize=7.5, color=INK2)
        better = int((y < x).sum())
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1])
        ax.set_title(f"{title}\n{labels[1]} better in {better} of {len(keys)}", fontsize=10.5)
        style(ax)
    axes[0].legend(loc="lower right", fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f"Every model on every unseen noise recording: {labels[0]} vs {labels[1]}", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.9, "Each point is one (model, unseen recording) pair, same test samples in both. Grey area below "
             "the diagonal: the second set is better. Labelled points are tests on the 2023-02-01 recording.",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.02, 1, 0.88))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_per_rate(A, B, keys, labels, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    x = np.arange(len(RATES))
    for ax, (title, sel) in zip(axes, [("Trained on the 4 other recordings", lambda k: k[0] == MULTI),
                                       ("Trained on 1 recording", lambda k: k[0] != MULTI)]):
        ks = [k for k in keys if sel(k)]
        a = [np.mean([A[k][r] for k in ks]) for r in RATES]
        b = [np.mean([B[k][r] for k in ks]) for r in RATES]
        for xi, ai, bi in zip(x, a, b):
            ax.plot([xi, xi], [ai, bi], color=GRID, lw=2, zorder=1)
        ax.scatter(x, a, s=60, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=labels[0])
        ax.scatter(x, b, s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=labels[1])
        ax.set_xticks(x, ["k01  C→O", "k10  O→C", "k12  O→C", "k21  C→O", "k23  C→O", "k32  O→C"], fontsize=9)
        ax.set_title(f"{title} (average of {len(ks)} tests)")
        style(ax)
    axes[0].set_ylabel("test RAE on unseen noise")
    axes[0].legend(fontsize=9, labelcolor=INK2)
    fig.suptitle("Error per transition rate on unseen noise recordings", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_val_vs_test(A, B, keys, labels, path):
    models = sorted({k[0] for k in keys if k[0] != MULTI}) + [MULTI]
    fig, ax = plt.subplots(figsize=(11, 6.0))
    w = 0.18
    for i, m in enumerate(models):
        ks = [k for k in keys if k[0] == m]
        for j, (S, col, lab) in enumerate([(A, BLUE, labels[0]), (B, ORANGE, labels[1])]):
            xv = i + (j - 0.5) * 2.2 * w
            val = np.mean([S[k]["val"] for k in ks]); test = np.mean([S[k]["rae"] for k in ks])
            ax.plot([xv, xv], [val, test], color=col, lw=2, alpha=0.5)
            ax.scatter([xv], [val], s=48, facecolor=SURFACE, edgecolor=col, linewidth=1.6, zorder=3,
                       label=f"{lab}: own training noise" if i == 0 else None)
            ax.scatter([xv], [test], s=58, color=col, edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=f"{lab}: unseen noise" if i == 0 else None)
    ax.set_xticks(range(len(models)), [f"only {m}" if m != MULTI else "the 4 others\n(5 models)" for m in models],
                  fontsize=9)
    ax.set_ylabel("rate RAE")
    ax.set_title("Familiar vs unseen noise, per model (unseen: average over its test recordings)")
    style(ax)
    ax.legend(ncol=2, fontsize=8.5, labelcolor=INK2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("How much is lost on unseen noise", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_by_recording(A, B, keys, labels, path):
    """Like the right panel of noise_cv.png, with both sets: one row per unseen recording, set A above set B."""
    tests = sorted({k[1] for k in keys})
    fig, ax = plt.subplots(figsize=(10.5, 0.95 * len(tests) + 2.4))
    yt = []
    for i, te in enumerate(tests):
        y0 = len(tests) - 1 - i
        yt.append(y0)
        for S, col, dy in [(A, BLUE, 0.17), (B, ORANGE, -0.17)]:
            y = y0 + dy
            ones = [S[k]["rae"] for k in keys if k[1] == te and k[0] != MULTI]
            if ones:
                ax.plot([min(ones), max(ones)], [y, y], color=GRID, lw=2, zorder=1)
                ax.scatter(ones, [y] * len(ones), s=34, facecolor=SURFACE, edgecolor=col, linewidth=1.4, zorder=2)
            if (MULTI, te) in S:
                m = S[(MULTI, te)]
                ax.scatter([m["val"]], [y], s=46, marker="D", facecolor=SURFACE, edgecolor=MUTED, linewidth=1.3,
                           zorder=2)
                ax.scatter([m["rae"]], [y], s=110, color=col, edgecolor=SURFACE, linewidth=1.4, zorder=3)
    ax.set_yticks(yt, tests)
    ax.set_ylim(-0.6, len(tests) - 0.4)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("test RAE on the recording (lower is better)"); ax.set_ylabel("tested on")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=10, color=BLUE, mec=SURFACE, label=f"{labels[0]}: trained on the 4 other recordings"),
               Line2D([], [], marker="o", ls="", ms=10, color=ORANGE, mec=SURFACE, label=f"{labels[1]}: trained on the 4 other recordings"),
               Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=BLUE, mew=1.4, label=f"{labels[0]}: trained on 1 other recording (one per model)"),
               Line2D([], [], marker="o", ls="", ms=6, mfc=SURFACE, mec=ORANGE, mew=1.4, label=f"{labels[1]}: trained on 1 other recording (one per model)"),
               Line2D([], [], marker="D", ls="", ms=6, mfc=SURFACE, mec=MUTED, mew=1.3, label="the 4-recording model on its own training noise (validation)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.13), ncol=1, fontsize=8.8, labelcolor=INK2)
    fig.suptitle(f"Does training on 4 recordings help? {labels[0]} (upper) vs {labels[1]} (lower)", x=0.01, ha="left",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_curves(CA, CB, labels, path):
    cases = sorted(set(CA) & set(CB), key=lambda c: (not c.startswith("holdout"), c))
    n = len(cases); ncol = 5; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.9 * nrow), sharey=True, squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax, c in zip(axes.ravel(), cases):
        for (C, col, lab) in [(CA, BLUE, labels[0]), (CB, ORANGE, labels[1])]:
            e, v = C[c]
            ax.plot(e, v, color=col, lw=1.1, label=lab)
            ax.scatter([e[np.argmin(v)]], [v.min()], s=22, color=col, zorder=3)
        ax.set_title(c.replace("holdout-", "held out: ").replace("only-", "only: "), fontsize=9.5, loc="left")
        ax.set_xlabel("epoch", fontsize=8.5)
        style(ax)
    for ax in axes[:, 0]:
        ax.set_ylabel("val RAE (rates)")
    axes[0, 0].set_ylim(0.55, 0.85)
    axes[0, 0].legend(fontsize=8, labelcolor=INK2)
    fig.suptitle("Validation error during training (dot: best epoch)", x=0.01, ha="left", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", required=True, help="cv-dir of the first set (e.g. COCO_noise5)")
    p.add_argument("--b", required=True, help="cv-dir of the second set")
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    A, CA = load_set(args.a)
    B, CB = load_set(args.b)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0] == MULTI, k))
    fig_scatter(A, B, keys, args.labels, os.path.join(args.out_dir, "arch_scatter.png"))
    fig_per_rate(A, B, keys, args.labels, os.path.join(args.out_dir, "arch_per_rate.png"))
    fig_val_vs_test(A, B, keys, args.labels, os.path.join(args.out_dir, "arch_val_vs_test.png"))
    fig_curves(CA, CB, args.labels, os.path.join(args.out_dir, "arch_training_curves.png"))
    fig_by_recording(A, B, keys, args.labels, os.path.join(args.out_dir, "arch_by_recording.png"))
    with open(os.path.join(args.out_dir, "arch_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        cols = ["rae", "val"] + RATES + [c for c, _ in CONDS]
        w.writerow(["group", "n_pairs"] + [f"{lab}_{c}" for c in cols for lab in ("A", "B")])
        for g, sel in [("trained on 4 others", lambda k: k[0] == MULTI), ("trained on 1", lambda k: k[0] != MULTI),
                       ("all", lambda k: True)]:
            ks = [k for k in keys if sel(k)]
            w.writerow([g, len(ks)] + [f"{np.mean([S[k][c] for k in ks]):.4f}" for c in cols for S in (A, B)])
    print(f"wrote 5 figures and arch_summary.csv to {args.out_dir} ({len(keys)} pairs; A = {args.labels[0]}, "
          f"B = {args.labels[1]})")


if __name__ == "__main__":
    main()
