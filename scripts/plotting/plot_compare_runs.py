#!/usr/bin/env python3
# scripts/plotting/plot_compare_runs.py
"""
Compare two or three online runs that were tested on the same simulated test set (same --sim-config and
--seed, e.g. train_hist_online.py against softdwell_pipeline/train_softdwell_online.py). The first run is
the reference the others are measured against. Writes PNGs into --out-dir:

  1_test_rae.png              test RAE per rate and overall, with a paired bootstrap of each difference
  2_learning_curves.png       validation loss against training samples seen and against hours of training
  3_error_vs_true_rate.png    median error per rate, binned by the true value of that rate
  4_paired_error.png          per-sample error of each run against the first
  5_softdwell_detectors_<n>.png  learned SoftDwell thresholds / softness per epoch, for run n if it has
                              softdwell_params.csv
  compare_summary.json        the numbers behind figure 1

Usage:
  python3 -m scripts.plotting.plot_compare_runs \
      --run outputs/hist_train_online/COC_noreuse_cmp/seed1 --label "Hist (24 fixed thresholds)" \
      --run outputs/softdwell_online/COC_cmp/4Det_24Res/seed1 --label "SoftDwell (4 learned thresholds)" \
      --out-dir outputs/compare/COC_hist_vs_softdwell
  # a third run, e.g. the ConvNeXt V2 SoftDwell:
  #   ... --run outputs/softdwell_online/COC_cmp/convnext_v2/4Det_24Res/seed1 --label "SoftDwell + ConvNeXt V2"
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plotting.plot_online_results import (AXIS, BLUE, GRID, INK, INK2, MUTED, ORANGE, SEQ_FROM_SURFACE,
                                                  SURFACE, grid_axes, load_run)

COLORS = [BLUE, ORANGE, "#1baf7a"]               # categorical slots 1-3 (all pairs distinguishable): runs 1-3
DETECTOR_COLORS = [BLUE, ORANGE, "#1baf7a", "#eda100"]  # slots 1-4, direct-labelled (slots 3-4 are below 3:1)


def rae(preds, targs):
    """Per sample and rate, the RAE the training logs report: sqrt(|log10 pred - log10 true|)."""
    return np.sqrt(np.abs(np.log10(preds) - np.log10(targs)) + 1e-12)


def load(run_dir, label):
    preds, targs, aux, _, unit, names, topology, metrics = load_run(run_dir)
    with open(os.path.join(run_dir, "train_config.json")) as f:
        tc = json.load(f)
    ep = np.array([int(r["epoch"]) for r in metrics])
    return dict(label=label, dir=run_dir, preds=preds, targs=targs, names=names, topology=topology,
                epoch=ep, val=np.array([float(r["val_loss"]) for r in metrics]),
                # validation RAE over the rate outputs only: comparable between models, unlike val_loss,
                # which for some models (selfcal) also averages over baseline / amplitude / sigma outputs
                val_rae=np.array([np.mean([float(r[f"val_rae_out_{j + 1}"]) for j in range(len(names))])
                                  for r in metrics]),
                hours=np.cumsum([float(r["epoch_time_s"]) for r in metrics]) / 3600,
                samples=ep * tc["samples_per_epoch"], sim=tc["data"]["sim_config"], seed=tc["seed"],
                norm_range=tc["model"].get("norm_range"))


def paired_bootstrap(a, b, n=2000, seed=0):
    """95% interval of mean(b) - mean(a) over test samples resampled together."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(b.mean() - a.mean()), np.percentile(d, [2.5, 97.5]).tolist()


def fig_test_rae(runs, path):
    names = runs[0]["names"] + ["overall"]
    per = [np.append(rae(r["preds"], r["targs"]).mean(axis=0), rae(r["preds"], r["targs"]).mean()) for r in runs]
    ref = rae(runs[0]["preds"], runs[0]["targs"])
    diffs = []                                  # per later run: [(mean, ci95) per rate..., overall]
    for r in runs[1:]:
        b = rae(r["preds"], r["targs"])
        d = [paired_bootstrap(ref[:, j], b[:, j], seed=j) for j in range(ref.shape[1])]
        diffs.append(d + [paired_bootstrap(ref.mean(axis=1), b.mean(axis=1), seed=99)])

    n = len(runs)
    fig, ax = plt.subplots(figsize=(10.5 if n > 2 else 9.5, 5.0))
    x, w = np.arange(len(names)), 0.8 / n - 0.02
    for i, (r, v) in enumerate(zip(runs, per)):
        bars = ax.bar(x + (i - (n - 1) / 2) * (w + 0.02), v, width=w, color=COLORS[i], linewidth=0, label=r["label"])
        for bx, bv in zip(bars, v):
            ax.text(bx.get_x() + bx.get_width() / 2, bv + 0.01, f"{bv:.3f}", ha="center", va="bottom",
                    fontsize=8 if n > 2 else 8.5, color=INK2)
    for j in range(len(names)):
        if n == 2:
            d, (lo, hi) = diffs[0][j]
            txt = f"Δ {d:+.3f}{'' if lo <= 0 <= hi else ' *'}\n[{lo:+.3f}, {hi:+.3f}]"
        else:
            txt = "\n".join(f"{i + 2}: {d:+.3f}{'' if lo <= 0 <= hi else ' *'}"
                            for i, (d, (lo, hi)) in enumerate(dd[j] for dd in diffs))
        # x in data, y in axes fraction: just below the tick labels
        ax.text(x[j], -0.09, txt, ha="center", va="top", fontsize=8, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.tick_params(axis="x", pad=21 + 13 * max(2, n - 1))  # room for the lines of differences (2 with 2 runs)
    ax.set_ylim(0, max(v.max() for v in per) * 1.18)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_ylabel("test RAE (lower is better)")
    ax.legend(loc="upper left", ncols=n)
    fig.suptitle(f"{runs[0]['topology']}: test RAE on the same {len(ref):,} test samples", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    if n == 2:
        note = (f"Under each pair: {runs[1]['label']} minus {runs[0]['label']}, with a 95% paired bootstrap "
                "interval; * = the interval excludes 0.")
    else:
        note = (f"Under each group: run 2, 3 (legend order) minus {runs[0]['label']}; * = its 95% paired bootstrap "
                "interval excludes 0 (intervals in compare_summary.json).")
    fig.text(0.02, 0.905, note, color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150); plt.close(fig)
    return {"rates": names, "rae": {r["label"]: v.tolist() for r, v in zip(runs, per)},
            "difference_vs_run1": {r["label"]: [{"mean": d, "ci95": ci} for d, ci in dd]
                                   for r, dd in zip(runs[1:], diffs)}}


def fig_learning_curves(runs, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key, xlabel in zip(axes, ["samples", "hours"], ["training samples seen", "hours of training"]):
        for i, r in enumerate(runs):
            x = r[key] / (1e6 if key == "samples" else 1)
            b = int(np.argmin(r["val"]))              # the checkpoint that was tested (lowest val loss)
            ax.plot(x, r["val_rae"], color=COLORS[i], lw=2, solid_capstyle="round",
                    label=f"{r['label']}: {r['val_rae'][b]:.4f} at the tested epoch")
            ax.plot(x[b], r["val_rae"][b], "o", ms=8, color=COLORS[i], mec=SURFACE, mew=2)
        ax.set_xlabel(xlabel + (" (millions)" if key == "samples" else ""))
        ax.yaxis.grid(True); ax.set_axisbelow(True)
    lo = min(r["val_rae"].min() for r in runs)
    hi = max(np.percentile(r["val_rae"], 97) for r in runs)
    axes[0].set_ylim(lo - 0.05 * (hi - lo), hi)
    axes[0].set_ylabel("validation RAE, mean over rates")
    axes[0].legend(loc="upper right")
    fig.suptitle(f"{runs[0]['topology']}: validation RAE during training", x=0.02, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.02, 0.905, "Every sample is used once, so samples seen = fresh simulated traces. Dot: epoch with the lowest validation loss "
             "(the model that was tested).\nRAE over the rate outputs only, so models with extra outputs compare fairly. The validation sets differ in size: the first N samples of the same "
             "stream.", color=INK2, fontsize=9.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_error_vs_true_rate(runs, path, n_bins=12):
    lt = np.log10(runs[0]["targs"])
    edges = np.linspace(np.floor(lt.min()), np.ceil(lt.max()), n_bins + 1)
    names = runs[0]["names"]
    fig, axes = grid_axes(len(names), width=4.6, height=3.3)
    for j, ax in enumerate(axes):
        idx = np.clip(np.digitize(lt[:, j], edges) - 1, 0, n_bins - 1)
        ctr = 0.5 * (edges[:-1] + edges[1:])
        for i, r in enumerate(runs):
            err = np.abs(np.log10(r["preds"][:, j]) - lt[:, j])
            q = np.array([np.percentile(err[idx == b], [25, 50, 75]) if (idx == b).sum() >= 10 else [np.nan] * 3
                          for b in range(n_bins)])
            ax.fill_between(ctr, q[:, 0], q[:, 2], color=COLORS[i], alpha=0.14, lw=0)
            ax.plot(ctr, q[:, 1], color=COLORS[i], lw=2, marker="o", ms=4, mec=SURFACE, mew=1, label=r["label"])
        ax.set_title(names[j])
        ticks = np.arange(edges[0], edges[-1] + 1)
        ax.set_xticks(ticks); ax.set_xticklabels([f"$10^{{{int(t)}}}$" for t in ticks])
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    for ax in axes[-2:]:
        ax.set_xlabel("true rate (Hz)")
    for ax in axes[::2]:
        ax.set_ylabel("|error| (decades)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"{runs[0]['topology']}: error against the true rate", x=0.02, ha="left", fontsize=13,
                 fontweight="bold")
    fig.text(0.02, 0.925, "Line: median |log10(predicted / true)| per bin of the true rate. Band: 25-75% of the "
             "samples. 0.3 decades = a factor of 2.", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_paired_error(runs, path):
    e = [np.abs(np.log10(r["preds"]) - np.log10(r["targs"])).mean(axis=1) for r in runs]
    hi = np.percentile(np.concatenate(e), 99.5)
    edges = np.linspace(0, hi, 61)
    hs = [np.histogram2d(np.clip(e[0], 0, hi), np.clip(ei, 0, hi), bins=[edges, edges])[0] for ei in e[1:]]
    vmax = max(h.max() for h in hs)
    fig, axes = plt.subplots(1, len(hs), figsize=(6.4 * len(hs), 5.8), squeeze=False)
    for ax, h, ei, r in zip(axes[0], hs, e[1:], runs[1:]):
        im = ax.pcolormesh(edges, edges, h.T, cmap=SEQ_FROM_SURFACE, vmin=0, vmax=vmax, rasterized=True)
        ax.plot([0, hi], [0, hi], color=INK, lw=1.0)
        better = np.mean(ei < e[0]) * 100
        ax.text(0.03, 0.97, f"{r['label']} better on {better:.0f}% of samples\n"
                f"{runs[0]['label']} better on {100 - better:.0f}%\n"
                f"correlation of errors: {np.corrcoef(e[0], ei)[0, 1]:.2f}",
                transform=ax.transAxes, va="top", fontsize=9, color=INK2)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"{runs[0]['label']}: mean |error| (decades)")
        ax.set_ylabel(f"{r['label']}: mean |error| (decades)")
    fig.suptitle("Do the models fail on the same samples?", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.92, "Each test sample's error, averaged over its rates. Black line: equal error.\n"
             f"Below the line, the model on the y axis is better than {runs[0]['label']}.", color=INK2, fontsize=9,
             va="top")
    fig.tight_layout(rect=(0, 0, 0.9, 0.86))
    cb = fig.colorbar(im, cax=fig.add_axes([0.92, 0.2, 0.012, 0.5])); cb.outline.set_visible(False)
    cb.set_label("test samples per cell", color=INK2)
    fig.savefig(path, dpi=150); plt.close(fig)


def fig_softdwell_detectors(run, levels, path):
    p = np.genfromtxt(os.path.join(run["dir"], "softdwell_params.csv"), delimiter=",", names=True)
    lo, hi = run["norm_range"]
    span = hi - lo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for k in np.unique(p["k"]).astype(int):
        m = p["k"] == k
        c = DETECTOR_COLORS[k % len(DETECTOR_COLORS)]
        for ax, col, scale, off in [(axes[0], "theta", span, lo), (axes[1], "tau", span, 0.0)]:
            y = p[col][m] * scale + off
            ax.plot(p["epoch"][m], y, color=c, lw=2, solid_capstyle="round")
            ax.text(p["epoch"][m][-1], y[-1], f"  detector {k + 1}: {y[-1]:,.0f}", va="center", fontsize=8.5,
                    color=INK2)
    for lab, v in levels.items():
        axes[0].hlines(v, 0, p["epoch"].max(), color=MUTED, lw=1, ls=(0, (4, 3)))   # stops before the labels
        axes[0].text(4, v, f"{lab} level {v:,.0f}", ha="left", va="bottom", fontsize=8.5, color=MUTED)
    axes[0].set_ylabel("threshold θ (raw counts)")
    axes[1].set_ylabel("softness τ (raw counts)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xlim(right=p["epoch"].max() * 1.32)
    fig.suptitle(f"{run['label']}: learned detectors", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.905, f"θ and τ converted from the layer's 0..1 scale to counts (input range {lo:,.0f}"
             f"..{hi:,.0f}). Dashed: the simulated current levels. Larger τ = softer threshold.",
             color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(path, dpi=150); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True,
                   help="run folder (two or three; the first is the reference)")
    p.add_argument("--label", action="append", required=True, help="name of each run, same order as --run")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    if not 2 <= len(args.run) <= len(COLORS) or len(args.label) != len(args.run):
        raise SystemExit(f"give 2 to {len(COLORS)} --run, and one --label for each")
    runs = [load(d, lab) for d, lab in zip(args.run, args.label)]
    for r in runs[1:]:
        if not np.array_equal(runs[0]["targs"], r["targs"]):
            raise SystemExit(f"{r['dir']} was not tested on the same samples as {runs[0]['dir']} (test targets differ)")
    os.makedirs(args.out_dir, exist_ok=True)

    summary = fig_test_rae(runs, os.path.join(args.out_dir, "1_test_rae.png"))
    fig_learning_curves(runs, os.path.join(args.out_dir, "2_learning_curves.png"))
    fig_error_vs_true_rate(runs, os.path.join(args.out_dir, "3_error_vs_true_rate.png"))
    fig_paired_error(runs, os.path.join(args.out_dir, "4_paired_error.png"))
    for n, r in enumerate(runs, start=1):
        if os.path.isfile(os.path.join(r["dir"], "softdwell_params.csv")) and r["norm_range"]:
            with open(os.path.join(r["dir"], "sim_config.json")) as f:
                groups = json.load(f)["model"].get("level_groups", {})
            levels = {lab: g["level"] for lab, g in groups.items() if isinstance(g.get("level"), (int, float))}
            fig_softdwell_detectors(r, levels, os.path.join(args.out_dir, f"5_softdwell_detectors_{n}.png"))
    summary.update({"runs": [{"label": r["label"], "dir": r["dir"], "best_val_loss": float(r["val"].min()),
                              "val_rae_at_tested_epoch": float(r["val_rae"][int(np.argmin(r["val"]))]),
                              "samples_seen": int(r["samples"][-1]), "hours": float(r["hours"][-1])} for r in runs],
                    "sim_config": runs[0]["sim"], "seed": runs[0]["seed"], "n_test": len(runs[0]["targs"])})
    with open(os.path.join(args.out_dir, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
