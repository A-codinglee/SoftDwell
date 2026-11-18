#!/usr/bin/env python3
# scripts/plotting/plot_online_results.py
"""
Figures for one train_hist_online run (reads the run folder, writes PNGs into <run>/plots):

  1_pred_vs_true.png        predicted vs true rate, one density map per rate
  2_error_distribution.png  histogram of log10(predicted / true) per rate
  3_error_by_condition.png  typical error by amplitude x noise, and by size of the true rate
  4_training_curves.png     train / validation loss per epoch
  5_error_vs_conditions.png typical error against baseline, amplitude, noise sigma and their ratio
  error_vs_conditions.csv   the numbers behind figure 5 (median error in decades, and mean RAE)
  6_conditions_pred_vs_true.png  predicted vs true baseline / amplitude / noise sigma
                            (only for runs trained with --predict-conditions)

Usage:
  python3 -m scripts.plotting.plot_online_results --run-dir outputs/hist_train_online/COCO/seed1
  # with a larger test set made by scripts.hist_pipeline.eval_hist_online:
  python3 -m scripts.plotting.plot_online_results --run-dir <run> --eval-dir <run>/eval_test10000
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---- style: light surface, recessive axes, one blue ramp for magnitude, blue/orange for two series ----
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
SEQ_FROM_SURFACE = LinearSegmentedColormap.from_list("seq_blue0", [SURFACE] + BLUE_RAMP)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.grid": False, "legend.frameon": False,
})

LOG2 = np.log10(2.0)


def load_run(run_dir, eval_dir=None):
    """Test predictions come from eval_dir if given (see eval_hist_online.py), otherwise from run_dir."""
    test_dir = eval_dir or run_dir
    preds = np.load(os.path.join(test_dir, "preds_test.npy"))
    targs = np.load(os.path.join(test_dir, "targs_test.npy"))
    aux = np.load(os.path.join(test_dir, "aux_test.npy"))
    with open(os.path.join(run_dir, "sim_config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(run_dir, "train_val_metrics.csv")) as f:
        metrics = list(csv.DictReader(f))

    labels = [st.get("label", "?") for st in cfg["model"]["states"]]
    names = []
    for i in range(len(labels) - 1):          # same order as the network outputs
        names.append(f"k{i}{i + 1}  {labels[i]}→{labels[i + 1]}")
        names.append(f"k{i + 1}{i}  {labels[i + 1]}→{labels[i]}")

    unit, scale = "counts", 1.0
    amp = cfg.get("amplifier")
    if amp and amp.get("profile") == "axopatch_200b":     # 16 bit, +/-100 pA at gain 100
        unit, scale = "pA", 2.0 ** 15 / (100.0 * 100.0 / float(amp["gain"]))
    preds_aux = None                                       # only for runs trained with --predict-conditions
    aux_path = os.path.join(test_dir, "preds_aux_test.npy")
    if os.path.isfile(aux_path):
        preds_aux = np.load(aux_path) / scale
    return preds, targs, aux / scale, preds_aux, unit, names, "".join(labels), metrics


def grid_axes(n, width=3.6, height=3.5):
    ncols = 3 if n > 4 else 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(width * ncols, height * nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    return fig, axes.ravel()[:n]


def fig_pred_vs_true(preds, targs, names, title, path):
    lt, lp = np.log10(targs), np.log10(preds)
    lo, hi = np.floor(lt.min() * 2) / 2 - 0.5, np.ceil(lt.max() * 2) / 2 + 0.5
    edges = np.linspace(lo, hi, int(round((hi - lo) / 0.2)) + 1)
    hists = [np.histogram2d(lt[:, j], np.clip(lp[:, j], lo, hi), bins=[edges, edges])[0] for j in range(len(names))]
    vmax = max(h.max() for h in hists)

    fig, axes = grid_axes(len(names))
    for j, ax in enumerate(axes):
        im = ax.pcolormesh(edges, edges, hists[j].T, cmap=SEQ_FROM_SURFACE, vmin=0, vmax=vmax, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0)
        for off in (-1.0, 1.0):
            ax.plot([lo, hi], [lo + off, hi + off], color=MUTED, lw=0.8)
        err = np.abs(lp[:, j] - lt[:, j])
        ax.set_title(names[j])
        ax.text(0.03, 0.97, f"within ×2: {np.mean(err < LOG2) * 100:.0f}%\nwithin ×10: {np.mean(err < 1) * 100:.0f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ticks = np.arange(np.ceil(lo), np.floor(hi) + 1)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ticks]); ax.set_yticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
    for ax in axes[-3:]:
        ax.set_xlabel("true rate (Hz)")
    for ax in axes[::3]:
        ax.set_ylabel("predicted rate (Hz)")
    fig.suptitle(f"{title}: predicted vs true rate on the test set", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.925, "Black line: perfect prediction. Grey lines: a factor of 10 off. Colour: test samples per cell.",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.91))
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.45])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.set_label("samples per cell", color=INK2)
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_error_distribution(preds, targs, names, title, path):
    err = np.log10(preds) - np.log10(targs)
    lim = 3.0
    edges = np.linspace(-lim, lim, 61)
    fig, axes = grid_axes(len(names), height=2.9)
    ymax = max(np.histogram(np.clip(err[:, j], -lim, lim), edges)[0].max() for j in range(len(names))) * 1.12
    for j, ax in enumerate(axes):
        ax.axvspan(-LOG2, LOG2, color=GRID, lw=0)
        ax.hist(np.clip(err[:, j], -lim, lim), bins=edges, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(0, color=INK, lw=1.0)
        ax.set_title(names[j])
        ax.text(0.97, 0.95, f"median |error|\n{np.median(np.abs(err[:, j])):.2f} decades",
                transform=ax.transAxes, va="top", ha="right", fontsize=9, color=INK2)
        ax.set_xlim(-lim, lim); ax.set_ylim(0, ymax)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_xticklabels(["", "100\u00d7\ntoo low", "10\u00d7", "exact", "10\u00d7", "100\u00d7\ntoo high", ""], fontsize=8.5)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    for ax in axes[::3]:
        ax.set_ylabel("test samples")
    fig.suptitle(f"{title}: how far off is each predicted rate?", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.915, "Predicted / true rate on a log scale. Grey band: within a factor of 2. Errors beyond 1000\u00d7 are counted in the outermost bars.", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150); plt.close(fig)


def annotate_cells(ax, values, vmin, vmax, counts=None):
    for (r, c), v in np.ndenumerate(values):
        if not np.isfinite(v):
            continue
        dark = (v - vmin) / max(vmax - vmin, 1e-12) > 0.5
        txt = f"{v:.2f}" if counts is None else f"{v:.2f}\nn={int(counts[r, c])}"
        ax.text(c, r, txt, ha="center", va="center", fontsize=8.5, color="#ffffff" if dark else INK, linespacing=1.3)


def fig_error_by_condition(preds, targs, aux, unit, names, title, path):
    err = np.abs(np.log10(preds) - np.log10(targs))
    amp, sigma = aux[:, 1], aux[:, 2]

    # left: amplitude x sigma (error averaged over the rates of each sample, then median per cell)
    fixed = np.ptp(amp) < 1e-9 * max(1.0, abs(amp).max()) or np.ptp(sigma) < 1e-9 * max(1.0, abs(sigma).max())
    n_amp, n_sig = 6, 5
    per_sample = err.mean(axis=1)
    left = counts = None
    if not fixed:
        a_edges = np.linspace(amp.min(), amp.max(), n_amp + 1)
        s_edges = np.linspace(sigma.min(), sigma.max(), n_sig + 1)
        ai = np.clip(np.digitize(amp, a_edges) - 1, 0, n_amp - 1)
        si = np.clip(np.digitize(sigma, s_edges) - 1, 0, n_sig - 1)
        left = np.full((n_sig, n_amp), np.nan); counts = np.zeros((n_sig, n_amp))
        for r in range(n_sig):
            for c in range(n_amp):
                m = (si == r) & (ai == c)
                counts[r, c] = m.sum()
                if m.sum() >= 5:
                    left[r, c] = np.median(per_sample[m])

    # right: rate x decade of the true value
    lt = np.log10(targs)
    d_lo, d_hi = int(np.floor(lt.min() + 1e-9)), int(np.ceil(lt.max() - 1e-9))
    decades = list(range(d_lo, d_hi))
    right = np.full((len(names), len(decades)), np.nan)
    for j in range(len(names)):
        for c, d in enumerate(decades):
            m = (lt[:, j] >= d) & (lt[:, j] < d + 1)
            if m.sum() >= 5:
                right[j, c] = np.median(err[m, j])

    both = [x for x in (left, right) if x is not None]
    vmin = min(np.nanmin(x) for x in both); vmax = max(np.nanmax(x) for x in both)

    fig, axs = plt.subplots(1, 2 if left is not None else 1, figsize=(12.5 if left is not None else 6.5, 5.0), squeeze=False)
    axs = axs.ravel()
    k = 0
    if left is not None:
        ax = axs[0]; k = 1
        im = ax.imshow(left, origin="lower", cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
        annotate_cells(ax, left, vmin, vmax, counts)
        ax.set_xticks(np.arange(n_amp + 1) - 0.5); ax.set_xticklabels([f"{v:.0f}" for v in a_edges])
        ax.set_yticks(np.arange(n_sig + 1) - 0.5); ax.set_yticklabels([f"{v:.2f}" for v in s_edges])
        ax.set_xlabel(f"amplitude: open minus closed level ({unit})"); ax.set_ylabel(f"noise sigma ({unit})")
        ax.set_title("By amplitude and noise (all rates averaged)")
    ax = axs[k]
    im = ax.imshow(right, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto")
    annotate_cells(ax, right, vmin, vmax)
    ax.set_xticks(range(len(decades)))
    ax.set_xticklabels([f"$10^{{{d}}}$–$10^{{{d + 1}}}$" for d in decades])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("true value of that rate (Hz)")
    ax.set_title("By rate and by how fast the true rate is")
    for a in axs:
        a.tick_params(length=0)
        for s in a.spines.values():
            s.set_visible(False)
    fig.suptitle(f"{title}: where are the predictions good or bad?", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, "Each cell: median |log10(predicted / true)| in decades. 0.30 = typically a factor of 2 off, "
                          "1.00 = a factor of 10. Darker = worse.", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.89))
    cax = fig.add_axes([0.945, 0.2, 0.012, 0.55])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.set_label("median error (decades)", color=INK2)
    fig.savefig(path, dpi=150); plt.close(fig)


def binned_error(x, per_sample_err, per_sample_rae, edges):
    """Per bin of x: (centre, n, median error, 25%, 75%, mean RAE). Bins with < 10 samples are skipped."""
    rows = []
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 10:
            continue
        q25, med, q75 = np.percentile(per_sample_err[m], [25, 50, 75])
        rows.append((edges[b], edges[b + 1], int(m.sum()), med, q25, q75, float(per_sample_rae[m].mean())))
    return rows


def fig_error_vs_conditions(preds, targs, aux, unit, title, path, csv_path, n_bins=10):
    dlog = np.abs(np.log10(preds) - np.log10(targs))
    err = dlog.mean(axis=1)                    # decades, averaged over the rates of one sample
    rae = np.sqrt(dlog).mean(axis=1)           # the RAE the training log reports, per sample
    base, amp, sigma = aux[:, 0], aux[:, 1], aux[:, 2]

    def varies(v):
        return np.ptp(v) > 1e-9 * max(1.0, np.abs(v).max())

    panels = []
    if varies(base):
        panels.append((f"baseline ({unit})", "Baseline", base, np.linspace(base.min(), base.max(), n_bins + 1), False))
    if varies(amp):
        panels.append((f"amplitude ({unit})", "Amplitude", amp, np.linspace(amp.min(), amp.max(), n_bins + 1), False))
    if varies(sigma):
        panels.append((f"noise sigma ({unit})", "Noise sigma", sigma, np.linspace(sigma.min(), sigma.max(), n_bins + 1), False))
    if varies(amp) or varies(sigma):
        snr = amp / np.maximum(sigma, 1e-30)
        panels.append(("amplitude / sigma", "Signal-to-noise ratio", snr,
                       np.geomspace(snr.min(), snr.max(), n_bins + 1), True))
    if not panels:
        return False

    results = [(name, binned_error(x, err, rae, edges)) for _, name, x, edges, _ in panels]
    y_hi = max(r[5] for _, rows in results for r in rows) * 1.08

    fig, axes = plt.subplots(1, len(panels), figsize=(4.1 * len(panels), 4.3), squeeze=False, sharey=True)
    for ax, (xlabel, name, x, edges, logx), (_, rows) in zip(axes.ravel(), panels, results):
        c = np.array([np.sqrt(r[0] * r[1]) if logx else 0.5 * (r[0] + r[1]) for r in rows])
        med, q25, q75 = (np.array([r[k] for r in rows]) for k in (3, 4, 5))
        ax.fill_between(c, q25, q75, color=BLUE, alpha=0.12, lw=0)
        ax.plot(c, med, color=BLUE, lw=2, solid_capstyle="round", marker="o", ms=6, mec=SURFACE, mew=1.5)
        for k in (0, len(c) - 1):
            ax.annotate(f"{med[k]:.2f}", (c[k], med[k]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=9, color=INK2)
        if logx:
            ax.set_xscale("log")
            ticks = [t for t in (1, 2, 5, 10, 20, 50, 100, 200, 500) if edges[0] * 0.9 <= t <= edges[-1] * 1.1]
            ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks]); ax.minorticks_off()
        ax.set_ylim(0, y_hi)
        ax.set_title(name); ax.set_xlabel(xlabel)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    axes.ravel()[0].set_ylabel("error (decades)")
    fig.suptitle(f"{title}: error against the recording conditions", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.9, "Line: median over test samples of |log10(predicted / true)|, averaged over all rates. "
                        "Band: middle half of the samples. 0.30 = a factor of 2 off, 1.00 = a factor of 10.",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150); plt.close(fig)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "bin_low", "bin_high", "n_samples", "median_error_decades",
                    "error_q25", "error_q75", "mean_rae"])
        for name, rows in results:
            for r in rows:
                w.writerow([name] + [f"{v:.6g}" if isinstance(v, float) else v for v in r])
    return True


def fig_conditions_pred_vs_true(preds_aux, aux, unit, title, path, n_bins=40):
    cols = [j for j in range(aux.shape[1]) if np.isfinite(preds_aux[:, j]).any()]   # NaN column = not predicted
    labels = ("Baseline", "Amplitude", "Noise sigma")
    panels = []
    for j in cols:
        t, p = aux[:, j], preds_aux[:, j]
        pad = 0.08 * (t.max() - t.min())
        lo, hi = t.min() - pad, t.max() + pad
        edges = np.linspace(lo, hi, n_bins + 1)
        panels.append((j, lo, hi, edges, np.histogram2d(t, np.clip(p, lo, hi), bins=[edges, edges])[0]))
    vmax = max(h.max() for *_, h in panels)

    fig, axes = plt.subplots(1, len(cols), figsize=(4.1 * len(cols), 4.9), squeeze=False)
    for ax, (j, lo, hi, edges, h) in zip(axes.ravel(), panels):
        im = ax.pcolormesh(edges, edges, h.T, cmap=SEQ_FROM_SURFACE, vmin=0, vmax=vmax, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK, lw=1.0)
        err = np.abs(preds_aux[:, j] - aux[:, j])
        span = aux[:, j].max() - aux[:, j].min()
        ax.set_title(labels[j])
        ax.text(0.03, 0.97, f"mean |error|: {err.mean():.3g} {unit}\n= {100 * err.mean() / span:.1f}% of the range",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK2)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"true ({unit})")
    axes.ravel()[0].set_ylabel(f"predicted ({unit})")
    fig.suptitle(f"{title}: predicted vs true recording conditions", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.9, "Black line: perfect prediction. Colour: test samples per cell.", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 0.93, 0.88))
    cax = fig.add_axes([0.945, 0.25, 0.012, 0.45])
    cb = fig.colorbar(im, cax=cax); cb.outline.set_visible(False); cb.set_label("samples per cell", color=INK2)
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_training_curves(metrics, title, path):
    ep = np.array([int(r["epoch"]) for r in metrics])
    tr = np.array([float(r["train_loss"]) for r in metrics])
    va = np.array([float(r["val_loss"]) for r in metrics])
    lr = np.array([float(r["lr"]) for r in metrics])
    best = int(np.argmin(va))

    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(ep, tr, color=BLUE, lw=2, solid_capstyle="round", label="train (refreshing buffer)")
    ax.plot(ep, va, color=ORANGE, lw=2, solid_capstyle="round", label="validation (fixed set)")
    ax.plot(ep[best], va[best], "o", ms=8, color=ORANGE, mec=SURFACE, mew=2)
    y_lo = min(tr.min(), va.min()); y_hi = np.percentile(np.concatenate([tr, va]), 99)
    pad = 0.08 * (y_hi - y_lo)
    ax.set_ylim(y_lo - pad, y_hi + pad)
    ax.annotate(f"best validation {va[best]:.4f}\n(epoch {ep[best]})", (ep[best], va[best]),
                xytext=(0, 26), textcoords="offset points", ha="right", fontsize=9, color=INK2)
    for i in np.nonzero(lr[1:] != lr[:-1])[0] + 1:
        ax.axvline(ep[i], color=AXIS, lw=0.8)
        ax.text(ep[i], ax.get_ylim()[1], f" lr {lr[i]:.0e}", va="top", ha="left", fontsize=8.5, color=MUTED)
    ax.text(ep[-1], tr[-1], " train", va="center", fontsize=9, color=INK2)
    ax.text(ep[-1], va[-1], " validation", va="center", fontsize=9, color=INK2)
    ax.set_xlim(ep[0], ep[-1] + 0.09 * (ep[-1] - ep[0] + 1))
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_xlabel("epoch"); ax.set_ylabel("log-cosh loss")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))
    fig.suptitle(f"{title}: training and validation loss", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="e.g. outputs/hist_train_online/COCO/seed1")
    p.add_argument("--eval-dir", default=None,
                   help="folder with test predictions from eval_hist_online.py (default: the run's own test set)")
    p.add_argument("--out-dir", default=None, help="default: <eval-dir or run-dir>/plots")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(args.eval_dir or args.run_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    preds, targs, aux, preds_aux, unit, names, topology, metrics = load_run(args.run_dir, args.eval_dir)
    topology = f"{topology} ({len(preds):,} test samples)"

    fig_pred_vs_true(preds, targs, names, topology, os.path.join(out_dir, "1_pred_vs_true.png"))
    fig_error_distribution(preds, targs, names, topology, os.path.join(out_dir, "2_error_distribution.png"))
    fig_error_by_condition(preds, targs, aux, unit, names, topology, os.path.join(out_dir, "3_error_by_condition.png"))
    fig_training_curves(metrics, topology, os.path.join(out_dir, "4_training_curves.png"))
    made = fig_error_vs_conditions(preds, targs, aux, unit, topology,
                                   os.path.join(out_dir, "5_error_vs_conditions.png"),
                                   os.path.join(out_dir, "error_vs_conditions.csv"))
    n_figs = 5 if made else 4
    if preds_aux is not None:
        fig_conditions_pred_vs_true(preds_aux, aux, unit, topology, os.path.join(out_dir, "6_conditions_pred_vs_true.png"))
        n_figs += 1
    print(f"wrote {n_figs} figures to {out_dir}")


if __name__ == "__main__":
    main()
