#!/usr/bin/env python3
# scripts/plotting/plot_noise_cv.py
"""
Compare the noise cross-validation runs made by submit_noise_cv.sh (reads every <cv-dir>/*/seed<N>,
writes into <cv-dir>/comparison):

  noise_cv.png   left:  test RAE of every model on every noise recording it did not train on
                 right: per recording, the model trained on the other recordings against the
                        models trained on a single one
  noise_cv.csv   the numbers, one row per (model, test recording)
  noise_cv_pred_vs_true.png   predicted vs true rate (all rates pooled) as a density map, one per
                              model and unseen recording, in the same layout as the grid above
  noise_cv_pred_vs_true_k01.png, ...   the same for each rate separately
  noise_cv_per_rate.csv       test RAE per rate, one row per (model, test recording)
  noise_cv_conditions_baseline.png, _amplitude.png, _sigma.png   predicted vs true recording
                              condition, same layout (runs trained with --predict-conditions)
  noise_cv_conditions.csv     mean absolute error of the conditions per (model, test recording)
  noise_cv_vs.png             with --compare-dir: the same cases in two sets of runs, side by side
  noise_cv_error_dist.png     distribution of log10(predicted / true) per unseen recording

Usage:
  python3 -m scripts.plotting.plot_noise_cv --cv-dir <...>/hist_train_online/COCO_noise5
"""
import argparse
import csv
import glob
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_online_results import (BLUE, GRID, INK, INK2, LOG2, MUTED, ORANGE, SEQ, SEQ_FROM_SURFACE,
                                                  SURFACE)

# bottom row of the grids: a different model in every column, the one that held that column's recording out
MULTI_ROW = "the 4 other\nrecordings\n(5 models, one\nper column)"

def short(path):
    """2026_02_20_0003_start0_100s.bin -> 2026-02-20, Patch_Noise_Gain_100_... -> Patch"""
    name = os.path.basename(path)
    m = re.match(r"(\d{4})_(\d{2})_(\d{2})", name)
    return "-".join(m.groups()) if m else name.split("_")[0]


def load_runs(cv_dir, seed):
    """Finished runs: dicts with train (labels), test {label: RAE}, val (rates RAE at the best epoch)."""
    runs = []
    for d in sorted(glob.glob(os.path.join(cv_dir, "*", f"seed{seed}"))):
        res = os.path.join(d, "test_by_noise_file.json")
        if not os.path.isfile(res):
            continue
        with open(os.path.join(d, "train_config.json")) as f:
            cfg = json.load(f)
        with open(res) as f:
            test = {short(r["noise_file"]): r["rae"] for r in json.load(f)}
        with open(os.path.join(d, "train_val_metrics.csv")) as f:
            rows = list(csv.DictReader(f))
        best = min(rows, key=lambda r: float(r["val_loss"]))
        n_rates = cfg["model"]["num_rates"]
        val = float(np.mean([float(best[f"val_rae_out_{j + 1}"]) for j in range(n_rates)]))
        # log10(predicted / true) of every rate, per test recording (rows of noise_source_test.npy
        # index the training files followed by the test files)
        names = [short(p) for p in cfg["data"]["noise_files_train_val"] + cfg["data"]["noise_files_test"]]
        src = np.load(os.path.join(d, "noise_source_test.npy"))
        lt = np.log10(np.load(os.path.join(d, "targs_test.npy")))
        lp = np.log10(np.maximum(np.load(os.path.join(d, "preds_test.npy")), 1e-300))
        pairs = {names[k]: (lt[src == k], lp[src == k]) for k in np.unique(src)}      # (samples, rates)
        with open(os.path.join(d, "sim_config.json")) as f:
            labels = [st.get("label", "?") for st in json.load(f)["model"]["states"]]
        rates = []                                  # same order as the network outputs
        for i in range(len(labels) - 1):
            rates += [f"k{i}{i + 1}  {labels[i]}\u2192{labels[i + 1]}", f"k{i + 1}{i}  {labels[i + 1]}\u2192{labels[i]}"]
        # recording conditions [baseline, amplitude, noise sigma] in pA (runs trained with --predict-conditions)
        cond_pairs, unit = {}, ""
        if os.path.isfile(os.path.join(d, "preds_aux_test.npy")):
            with open(os.path.join(d, "conditions.json")) as f:
                cj = json.load(f)
            unit = cj["unit"]
            ta = np.load(os.path.join(d, "aux_test.npy")) / cj["counts_per_unit"]
            pa = np.load(os.path.join(d, "preds_aux_test.npy")) / cj["counts_per_unit"]
            cond_pairs = {names[k]: (ta[src == k], pa[src == k]) for k in np.unique(src)}
        runs.append({"dir": d, "cond_pairs": cond_pairs, "unit": unit, "train": [short(p) for p in cfg["data"]["noise_files_train_val"]],
                     "test": test, "val": val, "best_epoch": int(best["epoch"]), "pairs": pairs,
                     "rates": rates})
    return runs


def fig_pred_vs_true(files, single, multi, path, rate=None, rate_name="all rates pooled"):
    """Density maps in the layout of the RAE grid: rows = trained on, columns = tested on. rate: column, None = all."""
    sel = slice(None) if rate is None else rate
    n = len(files)
    lo, hi = 1.5, 6.5
    edges = np.linspace(lo, hi, 51)
    cells = {}
    for i, tr in enumerate(files + [None]):
        for j, te in enumerate(files):
            run = multi.get(te) if tr is None else single.get(tr)
            if run and te in run["pairs"]:
                lt, lp = (a[:, sel].ravel() for a in run["pairs"][te])
                h = np.histogram2d(lt, np.clip(lp, lo, hi), bins=[edges, edges])[0]
                # share of each true-rate column, so cells with different sample counts compare
                cells[i, j] = (h / np.maximum(h.sum(axis=1, keepdims=True), 1), np.abs(lp - lt))
    vmax = max(np.percentile(h, 99.5) for h, _ in cells.values())

    fig, axes = plt.subplots(n + 1, n, figsize=(2.35 * n + 1.3, 2.35 * (n + 1) + 1.0), squeeze=False)
    rows = files + [MULTI_ROW]
    ticks = [2, 4, 6]
    for i in range(n + 1):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels([f"$10^{t}$" for t in ticks] if i == n else [])
            ax.set_yticklabels([f"$10^{t}$" for t in ticks] if j == 0 else [])
            if i == 0:
                ax.set_title(files[j], loc="center", fontsize=10)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=10, color=INK)
            if (i, j) not in cells:
                ax.text(0.5, 0.5, "trained\non it", transform=ax.transAxes, ha="center", va="center", color=MUTED,
                        fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            h, err = cells[i, j]
            im = ax.pcolormesh(edges, edges, h.T, cmap=SEQ_FROM_SURFACE, vmin=0, vmax=vmax, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
            ax.text(0.04, 0.96, f"within \u00d72: {np.mean(err < LOG2) * 100:.0f}%", transform=ax.transAxes, va="top",
                    ha="left", fontsize=8.5, color=INK2)
    fig.suptitle(f"Predicted vs true rate on unseen noise recordings ({rate_name})", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.02, 0.955, "Rows: trained on. Columns: tested on. x: true rate (Hz), y: predicted rate (Hz). "
             "Black line: perfect prediction.\nColour: share of the samples with that true rate.",
             color=INK2, fontsize=9.5, va="top")
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.925), h_pad=0.6, w_pad=0.4)
    cax = fig.add_axes([0.945, 0.35, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.set_label("share of samples", color=INK2)
    fig.savefig(path, dpi=140); plt.close(fig)


def fig_condition(files, single, multi, path, col, label, unit):
    """Predicted vs true recording condition `col`, in the layout of the RAE grid. Returns {(row, column): mean |error|}."""
    n = len(files)
    cells, lo, hi = {}, np.inf, -np.inf
    for i, tr in enumerate(files + [None]):
        for j, te in enumerate(files):
            run = multi.get(te) if tr is None else single.get(tr)
            if run and te in run["cond_pairs"] and np.isfinite(run["cond_pairs"][te][1][:, col]).any():
                t, pr = (a[:, col] for a in run["cond_pairs"][te])
                cells[i, j] = (t, pr)
                lo, hi = min(lo, t.min()), max(hi, t.max())
    if not cells:
        return {}
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    edges = np.linspace(lo, hi, 41)
    hists = {}
    for key, (t, pr) in cells.items():
        h = np.histogram2d(t, np.clip(pr, lo, hi), bins=[edges, edges])[0]
        hists[key] = h / np.maximum(h.sum(axis=1, keepdims=True), 1)     # share of each true-value column
    vmax = max(np.percentile(h, 99.5) for h in hists.values())

    fig, axes = plt.subplots(n + 1, n, figsize=(2.35 * n + 1.3, 2.35 * (n + 1) + 1.0), squeeze=False)
    rows = files + [MULTI_ROW]
    mae = {}
    for i in range(n + 1):
        for j in range(n):
            ax = axes[i, j]
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            ax.locator_params(nbins=4)
            if i < n:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])
            if i == 0:
                ax.set_title(files[j], loc="center", fontsize=10)
            if j == 0:
                ax.set_ylabel(rows[i], fontsize=10, color=INK)
            if (i, j) not in cells:
                ax.text(0.5, 0.5, "trained\non it", transform=ax.transAxes, ha="center", va="center", color=MUTED,
                        fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            t, pr = cells[i, j]
            mae[files[i] if i < n else "the 4 other recordings", files[j]] = float(np.abs(pr - t).mean())
            im = ax.pcolormesh(edges, edges, hists[i, j].T, cmap=SEQ_FROM_SURFACE, vmin=0, vmax=vmax, rasterized=True)
            ax.plot([lo, hi], [lo, hi], color=INK, lw=0.8)
            ax.text(0.04, 0.96, f"mean |error|: {np.abs(pr - t).mean():.2g} {unit}", transform=ax.transAxes, va="top",
                    ha="left", fontsize=8.5, color=INK2)
    fig.suptitle(f"Predicted vs true {label.lower()} on unseen noise recordings", x=0.02, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.02, 0.955, f"Rows: trained on. Columns: tested on. x: true {label.lower()} ({unit}), y: predicted ({unit}). "
             "Black line: perfect prediction.\nColour: share of the samples with that true value.",
             color=INK2, fontsize=9.5, va="top")
    fig.tight_layout(rect=(0.0, 0.0, 0.93, 0.925), h_pad=0.6, w_pad=0.4)
    cax = fig.add_axes([0.945, 0.35, 0.012, 0.3])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.set_label("share of samples", color=INK2)
    fig.savefig(path, dpi=140); plt.close(fig)
    return mae


def fig_two_sets(files, sets, path):
    """
    Test RAE of the same cases in two sets of runs (e.g. no reuse / reuse), one row per
    (trained on, tested on). sets: [(label, single, multi), (label, single, multi)].
    """
    rows = []                                   # (group label, row label, value in set 0, value in set 1)
    for tr in files + [None]:
        for te in files:
            vals = []
            for _, single, multi in sets:
                run = multi.get(te) if tr is None else single.get(tr)
                vals.append(run["test"].get(te) if run else None)
            if None not in vals:
                rows.append(("the 4 other recordings" if tr is None else tr, te, vals[0], vals[1]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(9.5, 0.3 * len(rows) + 2.6))
    y, yt, ylab, last = 0.0, [], [], None
    for g, te, a, b in rows:
        if g != last:
            y -= 0.9
            ax.text(-0.02, y, f"trained on {g}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=9.5, fontweight="bold", color=INK)
            last = g
            y -= 1.0
        ax.plot([a, b], [y, y], color=GRID, lw=2, zorder=1)
        ax.scatter([a], [y], s=55, color=BLUE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=sets[0][0] if not yt else None)
        ax.scatter([b], [y], s=55, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=3, label=sets[1][0] if not yt else None)
        yt.append(y); ylab.append(f"tested on {te}")
        y -= 1.0
    ax.set_yticks(yt, ylab, fontsize=9)
    ax.set_ylim(y, 0.2)
    ax.grid(axis="x"); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("test RAE on the recording (lower is better)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9.5, labelcolor=INK2)
    fig.suptitle(f"Rate error on unseen noise recordings:\n{sets[0][0]} vs {sets[1][0]}", x=0.02, ha="left", fontsize=12.5,
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_error_dist(files, single, multi, path):
    """Per unseen recording: log10(predicted / true), 4-recording model against the single-recording models."""
    n, lim = len(files), 2.5
    edges = np.linspace(-lim, lim, 76)
    mid = 0.5 * (edges[:-1] + edges[1:])
    dens = lambda e: np.histogram(np.clip(e, -lim, lim), edges, density=True)[0]
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.9), sharey=True, squeeze=False)
    for j, (ax, te) in enumerate(zip(axes[0], files)):
        ax.axvspan(-LOG2, LOG2, color=GRID, lw=0)
        ax.axvline(0, color=INK, lw=0.8)
        first = True
        for tr in files:
            if tr in single and te in single[tr]["pairs"]:
                lt, lp = (a.ravel() for a in single[tr]["pairs"][te])
                ax.plot(mid, dens(lp - lt), color=ORANGE, lw=1.2, alpha=0.9,
                        label="trained on 1 other recording (one line per model)" if first and j == 0 else None)
                first = False
        if te in multi:
            lt, lp = (a.ravel() for a in multi[te]["pairs"][te])
            ax.plot(mid, dens(lp - lt), color=BLUE, lw=2.2, label="trained on the 4 other recordings" if j == 0 else None)
            ax.text(0.97, 0.96, f"4 recordings:\nwithin \u00d72: {np.mean(np.abs(lp - lt) < LOG2) * 100:.0f}%",
                    transform=ax.transAxes, va="top", ha="right", fontsize=8.5, color=INK2)
        ax.set_title(f"tested on {te}", fontsize=10)
        ax.set_xlim(-lim, lim)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xticklabels(["100\u00d7\ntoo low", "10\u00d7", "exact", "10\u00d7", "100\u00d7\ntoo high"], fontsize=8.5)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes[0, 0].set_ylabel("density of test rates")
    axes[0, 0].set_ylim(0)
    fig.suptitle("How far off are the predicted rates on unseen noise recordings?", x=0.012, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.012, 0.9, "Predicted / true rate on a log scale, all rates pooled. Grey band: within a factor of 2. "
             "Errors beyond 300\u00d7 are counted at the outer edges.",
             color=INK2, fontsize=9.5)
    fig.legend(loc="lower left", bbox_to_anchor=(0.012, 0.0), ncol=2, fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cv-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--compare-dir", default=None,
                   help="a second folder of the same cases (e.g. the no-reuse runs): also writes noise_cv_vs.png")
    p.add_argument("--labels", nargs=2, default=["other runs", "these runs"], metavar=("COMPARE", "THIS"),
                   help="names of --compare-dir and --cv-dir in noise_cv_vs.png")
    args = p.parse_args()

    runs = load_runs(args.cv_dir, args.seed)
    single = {r["train"][0]: r for r in runs if len(r["train"]) == 1}
    multi = {next(iter(r["test"])): r for r in runs if len(r["train"]) > 1 and len(r["test"]) == 1}
    files = sorted(set(single) | set(multi))
    if not files:
        raise SystemExit(f"no finished runs in {args.cv_dir}")
    n = len(files)
    out_dir = os.path.join(args.cv_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    # rows: one model per single training recording, then "all other recordings" (a different model per column)
    grid = np.full((n + 1, n), np.nan)
    for i, tr in enumerate(files):
        for j, te in enumerate(files):
            grid[i, j] = single.get(tr, {"test": {}})["test"].get(te, np.nan)
    for j, te in enumerate(files):
        grid[n, j] = multi[te]["test"][te] if te in multi else np.nan

    with open(os.path.join(out_dir, "noise_cv.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trained_on", "tested_on", "test_rae", "val_rae_on_training_noise", "best_epoch", "run_dir"])
        for r in runs:
            for te, v in r["test"].items():
                w.writerow([" + ".join(r["train"]), te, f"{v:.4f}", f"{r['val']:.4f}", r["best_epoch"], r["dir"]])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.32})
    fig.suptitle("Rate error (RAE) on noise recordings the model did not train on   ·   lower is better",
                 x=0.012, ha="left", fontsize=12, fontweight="bold")

    # ---- left: every model x every test recording ----
    vmin, vmax = np.nanmin(grid), np.nanmax(grid)
    ax.imshow(np.ma.masked_invalid(grid), cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(n + 1):
        for j in range(n):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "trained\non it" if i < n and files[i] == files[j] else "–", ha="center", va="center",
                        color=MUTED, fontsize=8)
                continue
            dark = (v - vmin) / max(vmax - vmin, 1e-12) > 0.45
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if dark else INK,
                    fontweight="bold" if v == np.nanmin(grid[:, j]) else "normal")
    ax.set_xticks(range(n), files)
    ax.set_yticks(range(n + 1), files + [MULTI_ROW])
    ax.set_xticks(np.arange(-0.5, n), minor=True)
    ax.set_yticks(np.arange(-0.5, n + 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    ax.axhline(n - 0.5, color=SURFACE, linewidth=7)
    ax.set_frame_on(False)
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.set_title("Every model on every unseen recording (bold = best in column)")

    # ---- right: per test recording, 4 recordings vs 1 ----
    y = np.arange(n)[::-1]
    for j, te in enumerate(files):
        ones = grid[:n, j][~np.isnan(grid[:n, j])]
        if ones.size:
            bx.plot([ones.min(), ones.max()], [y[j], y[j]], color=GRID, linewidth=2, zorder=1)
            bx.scatter(ones, np.full(ones.size, y[j]), s=60, color=ORANGE, edgecolor=SURFACE, linewidth=1.5, zorder=2,
                       label="trained on 1 other recording (one dot per model)" if j == 0 else None)
        if te in multi:
            bx.scatter([multi[te]["val"]], [y[j]], s=60, facecolor=SURFACE, edgecolor=MUTED, linewidth=1.5, zorder=2,
                       label="same model on its own training noise (validation)" if j == 0 else None)
            bx.scatter([grid[n, j]], [y[j]], s=110, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       label="trained on the 4 other recordings" if j == 0 else None)
    bx.set_yticks(y, files)
    bx.set_ylim(-0.6, n - 0.4)
    bx.grid(axis="x")
    bx.tick_params(axis="y", length=0)
    bx.spines["left"].set_visible(False)
    bx.set_xlabel("test RAE on the recording")
    bx.set_ylabel("tested on")
    bx.set_title("Does training on 4 recordings help?")
    h, lab = bx.get_legend_handles_labels()
    order = sorted(range(len(lab)), key=lambda k: ("4 other" not in lab[k], "1 other" not in lab[k]))
    bx.legend([h[k] for k in order], [lab[k] for k in order], loc="upper center", bbox_to_anchor=(0.45, -0.16),
              ncol=1, fontsize=9, labelcolor=INK2)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.30)
    path = os.path.join(out_dir, "noise_cv.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    fig_pred_vs_true(files, single, multi, os.path.join(out_dir, "noise_cv_pred_vs_true.png"))
    if args.compare_dir:
        other = load_runs(args.compare_dir, args.seed)
        o_single = {r["train"][0]: r for r in other if len(r["train"]) == 1}
        o_multi = {next(iter(r["test"])): r for r in other if len(r["train"]) > 1 and len(r["test"]) == 1}
        fig_two_sets(files, [(args.labels[0], o_single, o_multi), (args.labels[1], single, multi)],
                     os.path.join(out_dir, "noise_cv_vs.png"))

    cond_rows = {}
    for col, (key, label) in enumerate([("baseline", "Baseline"), ("amplitude", "Amplitude"), ("sigma", "Noise sigma")]):
        mae = fig_condition(files, single, multi, os.path.join(out_dir, f"noise_cv_conditions_{key}.png"), col, label,
                            runs[0]["unit"])
        for k, v in mae.items():
            cond_rows.setdefault(k, {})[key] = v
    if cond_rows:
        with open(os.path.join(out_dir, "noise_cv_conditions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            keys = sorted({k for v in cond_rows.values() for k in v})
            w.writerow(["trained_on", "tested_on"] + [f"mean_abs_error_{k}_{runs[0]['unit']}" for k in keys])
            for (tr, te), v in cond_rows.items():
                w.writerow([tr, te] + [f"{v[k]:.4f}" if k in v else "" for k in keys])
    rates = runs[0]["rates"]
    for k, name in enumerate(rates):
        fig_pred_vs_true(files, single, multi, os.path.join(out_dir, f"noise_cv_pred_vs_true_{name.split()[0]}.png"),
                         rate=k, rate_name=f"rate {name}")
    with open(os.path.join(out_dir, "noise_cv_per_rate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trained_on", "tested_on"] + [f"rae_{nm.split()[0]}" for nm in rates])
        for r in runs:
            for te, (lt, lp) in r["pairs"].items():
                w.writerow([" + ".join(r["train"]), te]
                           + [f"{v:.4f}" for v in np.sqrt(np.abs(lp - lt) + 1e-12).mean(axis=0)])
    fig_error_dist(files, single, multi, os.path.join(out_dir, "noise_cv_error_dist.png"))
    print(f"wrote {path}, noise_cv_pred_vs_true.png, noise_cv_error_dist.png and noise_cv.csv ({len(single)} single-recording and {len(multi)} hold-out runs)")


if __name__ == "__main__":
    main()
