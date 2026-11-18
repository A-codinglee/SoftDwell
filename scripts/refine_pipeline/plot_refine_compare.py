#!/usr/bin/env python3
# scripts/refine_pipeline/plot_refine_compare.py
"""
Comparison plots of a finished refinement run (train_refine_online.py): original stage 1 vs A (phase-A refiner on the
original stage 1) vs B (fine-tuned stage 1 + refiner), all on the run's test set.

  1_rae_per_output.png     RAE per output (rates, baseline, amplitude, sigma) for the three, A and B at both stop rules
  2_iterations.png         rate RAE and mean discrepancy per refinement iteration, A and B (original stage 1 as a line)
  3_paired_error.png       per test trace: B's rate error minus original stage 1's (stage-1 vs B, same trace)
  4_error_vs_true_rate.png per rate: median |log10 error| vs true rate, original stage 1 vs B
  5_rae_by_noise_file.png  rate RAE per noise recording, original stage 1 vs B
  6_test_rae.png           test RAE per rate and overall with paired-bootstrap differences (plot_compare_runs style)
  7_pred_vs_true_<x>.png   predicted vs true rate, density scatter, for stage 1 / A / B (plot_online_results style)
  8_paired_error_scatter.png  per trace: error of A / B against the original stage 1's (do they fail on the same traces?)

Per-trace plots (3-8) need every iterate: B's are in refine_test.npz; A's in refine_test_phaseA.npz, which runs
started after 2026-11-15 save (older runs show A only in 1 and 2, from refine_test_summary.json). For a run without --finetune-stage1 there is no B: "B" is then the refined run itself.

Usage:  python3 -m scripts.refine_pipeline.plot_refine_compare --run-dir outputs/refine_online/<run>/seed1
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# reference palette (dataviz skill): text / surface tokens, categorical slots 1-2; stage 1 is the neutral baseline
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C_S1, C_A, C_B = "#8f8e88", "#2a78d6", "#eb6834"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def fig_output_grid(models, ytrue, true_c, rate_names, title, path):
    """
    Predicted vs true for every output, in the style of plot_noise_cv.py: rows = outputs (rates, then baseline /
    amplitude / sigma), columns = models; colour = share of the samples with that true value. models: list of
    (label, log10 rates (N, nr), conditions in pA (N, 3) or None).
    """
    from scripts.plotting.plot_online_results import INK as K, INK2 as K2, LOG2, MUTED, SEQ_FROM_SURFACE
    nr = ytrue.shape[1]
    cond_names = [("baseline", "pA"), ("amplitude", "pA"), ("sigma", "pA")]
    rows = [(nm, "rate") for nm in rate_names] + cond_names
    nm_, nc = len(rows), len(models)
    r_edges = np.linspace(1.5, 6.5, 51)
    cells, c_edges = {}, {}
    for i, (rname, kind) in enumerate(rows):
        if kind == "rate":
            t = ytrue[:, i]
            edges = r_edges
        else:
            k = i - nr
            t = true_c[:, k]
            lo, hi = t.min(), t.max()
            pad = 0.08 * (hi - lo)
            edges = c_edges[k] = np.linspace(lo - pad, hi + pad, 41)
        for j, (lab, pr_r, pr_c) in enumerate(models):
            if kind == "rate":
                p = pr_r[:, i]
            elif pr_c is None:
                continue
            else:
                p = pr_c[:, i - nr]
            h = np.histogram2d(t, np.clip(p, edges[0], edges[-1]), bins=[edges, edges])[0]
            cells[i, j] = (h / np.maximum(h.sum(axis=1, keepdims=True), 1), np.abs(p - t), edges)
    vmax_r = max(np.percentile(cells[i, j][0], 99.5) for (i, j) in cells if i < nr)
    vmax_c = max([np.percentile(cells[i, j][0], 99.5) for (i, j) in cells if i >= nr] or [1])

    # layout: rows = models, columns = outputs (wide, as the noise-CV grids)
    fig, axes = plt.subplots(nc, nm_, figsize=(2.3 * nm_ + 1.0, 2.6 * nc + 2.4), squeeze=False)
    im_r = im_c = None
    for i, (rname, kind) in enumerate(rows):
        for j, (lab, _, _) in enumerate(models):
            ax = axes[j, i]
            if j == 0:
                ax.set_title(rname if kind == "rate" else f"{rname} (pA)", fontsize=10.5, fontweight="bold")
            if i == 0:
                ax.set_ylabel(lab, fontsize=10.5, color=K)
            if (i, j) not in cells:
                ax.text(0.5, 0.5, "not saved\nper trace", transform=ax.transAxes, ha="center", va="center",
                        color=MUTED, fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                continue
            h, err, edges = cells[i, j]
            lo, hi = edges[0], edges[-1]
            im = ax.pcolormesh(edges, edges, h.T, cmap=SEQ_FROM_SURFACE, vmin=0,
                               vmax=vmax_r if kind == "rate" else vmax_c, rasterized=True)
            if kind == "rate":
                im_r = im
            else:
                im_c = im
            ax.plot([lo, hi], [lo, hi], color=K, lw=0.8)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
            if kind == "rate":
                ticks = [2, 4, 6]
                ax.set_xticks(ticks); ax.set_yticks(ticks)
                ax.set_xticklabels([f"$10^{t}$" for t in ticks]); ax.set_yticklabels([f"$10^{t}$" for t in ticks])
                txt = f"within ×2: {np.mean(err < LOG2) * 100:.0f}%\nRAE {np.sqrt(err + 1e-12).mean():.3f}"
            else:
                ax.locator_params(nbins=4)
                txt = f"mean |error|:\n{err.mean():.3g} pA"
            if j < nc - 1:
                ax.set_xticklabels([])
            ax.tick_params(labelsize=8.5)
            ax.text(0.04, 0.96, txt, transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color=K2)
    fig.suptitle(f"{title}: predicted vs true on the same test traces, every output", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 1 - 0.62 / fig.get_figheight(), "Rows: models. Columns: outputs. x: true value, y: predicted "
             "(rates in Hz, levels in pA). Black line: perfect prediction. Colour: share of the samples with that "
             "true value (rates and levels on separate scales).", color=K2, fontsize=9.5, va="top")
    fig.tight_layout(rect=(0.0, 1.2 / fig.get_figheight(), 1.0, 1 - 0.95 / fig.get_figheight()), h_pad=2.2,
                     w_pad=0.5)
    for im, cols, lab in ((im_r, range(nr), "share of samples (rates)"),
                          (im_c, range(nr, nm_), "share of samples (levels)")):
        if im is None:
            continue
        x0 = axes[-1, cols[0]].get_position().x0
        x1 = axes[-1, cols[-1]].get_position().x1
        cax = fig.add_axes([x0 + 0.2 * (x1 - x0), 0.6 / fig.get_figheight(), 0.6 * (x1 - x0), 0.1 / fig.get_figheight()])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal"); cb.outline.set_visible(False)
        cb.set_label(lab, color=K2, fontsize=9); cb.ax.tick_params(labelsize=8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None, help="default <run-dir>/plots")
    p.add_argument("--title", default=None)
    a = p.parse_args()
    run = a.run_dir
    out = a.out or os.path.join(run, "plots")
    os.makedirs(out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelcolor": INK, "text.color": INK,
                         "figure.facecolor": SURF, "savefig.facecolor": SURF})

    S = json.load(open(os.path.join(run, "refine_test_summary.json")))
    tc = json.load(open(os.path.join(run, "train_config.json")))
    sc = json.load(open(os.path.join(run, "sim_config.json")))
    nr = tc["model"]["num_rates"]
    rate_names = [nm for i in range(nr // 2) for nm in (f"k{i}→{i + 1}", f"k{i + 1}→{i}")]
    out_names = rate_names + ["baseline", "amplitude", "sigma"]
    has_a = "phase_A" in S
    title = a.title or os.path.basename(os.path.dirname(os.path.abspath(run)))
    s1 = S.get("stage1_original", {"rae_per_out": S["by_iteration"][0]["rae_per_out"],
                                   "rae_rates": S["by_iteration"][0]["rae_rates"]})

    # ---------- 1: RAE per output ----------
    series = [("original stage 1", s1["rae_per_out"], s1["rae_rates"], C_S1, None)]
    if has_a:
        series.append(("A best-fit", S["phase_A"]["best-fit"]["rae_per_out"], S["phase_A"]["best-fit"]["rae_rates"],
                       C_A, None))
    k_tr = tc["refinement"]["iters"]
    series += [("B best-fit", S["best-fit"]["rae_per_out"], S["best-fit"]["rae_rates"], C_B, None),
               (f"B fixed {k_tr} it.", S["by_iteration"][k_tr]["rae_per_out"], S["by_iteration"][k_tr]["rae_rates"],
                C_B, "////")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    style(ax)
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(out_names))
    for i, (lab, v, vr, col, hatch) in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, v, w * 0.9, color=SURF if hatch else col, edgecolor=col, hatch=hatch,
               lw=1.2 if hatch else 0, label=f"{lab} (rates {vr:.4f})")
    ax.set_xticks(x, out_names)
    ax.set_ylabel("RAE  (mean √|log10 error|)")
    ax.set_title(f"{title}: test RAE per output, lower is better")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(0, max(max(s[1]) for s in series) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_rae_per_output.png"), dpi=150)
    plt.close(fig)

    # ---------- 2: per iteration (two panels, one axis each) ----------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, lab in ((a1, "rae_rates", "rate RAE"), (a2, "discrepancy", "mean discrepancy d")):
        style(ax)
        its = np.arange(len(S["by_iteration"]))
        if has_a:
            va = [r[key] for r in S["phase_A"]["by_iteration"]]
            ax.plot(its, va, "-o", color=C_A, lw=2, ms=5, label="A")
        vb = [r[key] for r in S["by_iteration"]]
        ax.plot(its, vb, "-o", color=C_B, lw=2, ms=5, label="B")
        if key == "rae_rates":
            ax.axhline(s1["rae_rates"], color=C_S1, lw=1.5, ls="--", label="original stage 1")
        ax.axvline(k_tr, color=GRID, lw=1, zorder=0)
        ax.text(k_tr, ax.get_ylim()[1], f" trained count ({k_tr})", color=INK2, fontsize=8, va="top")
        ax.set_xlabel("refinement iteration (0 = stage 1's estimate)")
        ax.set_ylabel(lab)
        ax.legend(frameon=False, fontsize=9)
    a1.set_title("rate error per iteration")
    a2.set_title("fit to the observed trace per iteration")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_iterations.png"), dpi=150)
    plt.close(fig)

    # ---------- per-trace data (B) ----------
    z = np.load(os.path.join(run, "refine_test.npz"))
    th, d, tgt, src = z["theta"], z["discrepancy"], z["target"], z["noise_source"]
    ytrue = tgt[:, :nr]
    p0_path = os.path.join(run, "preds_test_stage1_orig.npy")
    p0 = np.log10(np.load(p0_path)) if os.path.isfile(p0_path) else th[:, 0, :nr]
    kb = d.argmin(1)
    pb = th[np.arange(len(th)), kb, :nr]
    e0, eb = np.abs(p0 - ytrue), np.abs(pb - ytrue)
    r0, rb = np.sqrt(e0 + 1e-12).mean(1), np.sqrt(eb + 1e-12).mean(1)          # per-trace RAE over the rates

    # ---------- 3: paired per-trace difference ----------
    diff = rb - r0
    fig, ax = plt.subplots(figsize=(7, 3.8))
    style(ax)
    lim = np.quantile(np.abs(diff), 0.99)
    ax.hist(np.clip(diff, -lim, lim), bins=81, range=(-lim, lim), color=C_B, alpha=0.9)
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(diff.mean(), color=INK2, lw=1.2, ls="--")
    better = float((diff < 0).mean() * 100)
    worse = float((diff > 0).mean() * 100)
    ax.set_xlabel("per-trace rate RAE: B best-fit − original stage 1   (left = B better; outer bins hold the 1% tails)")
    ax.set_ylabel("test traces")
    ax.set_title(f"{title}: B better on {better:.0f}%, worse on {worse:.0f}%, mean change {diff.mean():+.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "3_paired_error.png"), dpi=150)
    plt.close(fig)

    # ---------- 4: error vs true rate ----------
    cols = min(nr, 3) if nr > 4 else nr
    rows = int(np.ceil(nr / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.0 * rows), squeeze=False)
    edges = np.linspace(2, 6, 17)
    mid = 0.5 * (edges[1:] + edges[:-1])
    for j in range(nr):
        ax = axs.flat[j]
        style(ax)
        b = np.digitize(ytrue[:, j], edges) - 1
        for e, col, lab in ((e0, C_S1, "original stage 1"), (eb, C_B, "B best-fit")):
            med = [np.median(e[b == i, j]) if (b == i).sum() >= 20 else np.nan for i in range(len(mid))]
            ax.plot(mid, med, "-o", color=col, lw=2, ms=4, label=lab)
        ax.set_title(rate_names[j])
        ax.set_xlabel("true log10 rate (Hz)")
        if j % cols == 0:
            ax.set_ylabel("median |log10 error|")
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    for k in range(nr, rows * cols):
        axs.flat[k].set_visible(False)
    fig.suptitle(f"{title}: error vs true rate", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "4_error_vs_true_rate.png"), dpi=150)
    plt.close(fig)

    # ---------- 5: by noise file ----------
    paths = list(sc["noise"]["source_paths"]) + list(sc["noise"].get("test_source_paths", []))
    ids = sorted(set(src.astype(int).tolist()))
    names = [os.path.basename(paths[i]).split("_start")[0].split("_Noise_")[0].replace("_Gain_100", "").replace(".bin", "") for i in ids]
    v0 = [r0[src == i].mean() for i in ids]
    vb = [rb[src == i].mean() for i in ids]
    fig, ax = plt.subplots(figsize=(max(5, 1.8 * len(ids) + 2), 3.8))
    style(ax)
    x = np.arange(len(ids))
    ax.bar(x - 0.2, v0, 0.38, color=C_S1, label="original stage 1")
    ax.bar(x + 0.2, vb, 0.38, color=C_B, label="B best-fit")
    for xi, a0, ab in zip(x, v0, vb):
        ax.text(xi + 0.2, ab, f"{ab - a0:+.4f}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylabel("rate RAE")
    ax.set_ylim(0, max(max(v0), max(vb)) * 1.3)
    ax.set_title(f"{title}: rate RAE per noise recording (label: B − stage 1)")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "5_rae_by_noise_file.png"), dpi=150)
    plt.close(fig)

    # ---------- 6-8: in the style of plot_compare_runs / plot_online_results (same functions) ----------
    from scripts.plotting import plot_compare_runs as pcr
    from scripts.plotting import plot_online_results as por
    labels = [st.get("label", "?") for st in sc["model"]["states"]]
    pnames = [nm for i in range(len(labels) - 1) for nm in (f"k{i}{i + 1}  {labels[i]}→{labels[i + 1]}",
                                                            f"k{i + 1}{i}  {labels[i + 1]}→{labels[i]}")]
    topo = "".join(labels)
    targ_hz = 10.0 ** ytrue
    runs = [dict(label="Original stage 1", preds=10.0 ** p0)]
    pa_path = os.path.join(run, "refine_test_phaseA.npz")
    if os.path.isfile(pa_path):                                   # saved by runs after 2026-11-15
        za = np.load(pa_path)
        ka = za["discrepancy"].argmin(1)
        runs.append(dict(label="A best-fit", preds=10.0 ** za["theta"][np.arange(len(ka)), ka, :nr]))
    runs.append(dict(label="B best-fit", preds=10.0 ** pb))
    for r in runs:
        r.update(targs=targ_hz, names=pnames, topology=f"{topo} ({title})")
    test_rae = pcr.fig_test_rae(runs, os.path.join(out, "6_test_rae.png"))
    json.dump(test_rae, open(os.path.join(out, "6_test_rae.json"), "w"), indent=2)
    for r in runs:
        tag = r["label"].split()[0] if r["label"] != "Original stage 1" else "stage1"
        por.fig_pred_vs_true(r["preds"], targ_hz, pnames, f"{topo} {r['label']}",
                             os.path.join(out, f"7_pred_vs_true_{tag}.png"))
    pcr.fig_paired_error(runs, os.path.join(out, "8_paired_error_scatter.png"))

    # ---------- 9: predicted vs true, every output (rates + conditions), one column per model ----------
    c = float(json.load(open(os.path.join(run, "conditions.json"))).get("counts_per_unit", 327.68))
    cnd = json.load(open(os.path.join(run, "conditions.json")))
    cm, cs = np.asarray(cnd["mean"]), np.asarray(cnd["std"])
    to_pa = lambda th: (th * cs + cm) / c                                  # normalized theta -> pA
    true_c = to_pa(tgt[:, nr:nr + 3])
    p0_aux_path = os.path.join(run, "preds_aux_test_stage1_orig.npy")     # runs after 2026-11-15
    s1_aux_path = os.path.join(tc["stage1_run"], "preds_aux_test.npy")    # else the stage-1 run's own (same traces)
    if os.path.isfile(p0_aux_path):
        p0_c = np.load(p0_aux_path) / c
    elif os.path.isfile(s1_aux_path) and len(np.load(s1_aux_path)) == len(tgt) and np.allclose(
            np.load(os.path.join(tc["stage1_run"], "aux_test.npy")), z["aux"]):
        p0_c = np.load(s1_aux_path) / c
    else:
        p0_c = None
    models = [("Original stage 1", p0, p0_c)]
    if os.path.isfile(pa_path):
        models.append(("A best-fit", za["theta"][np.arange(len(ka)), ka, :nr],
                       to_pa(za["theta"][np.arange(len(ka)), ka, nr:nr + 3])))
    models.append(("B best-fit", pb, to_pa(th[np.arange(len(th)), kb, nr:nr + 3])))
    fig_output_grid(models, ytrue, true_c, pnames, f"{topo} ({title})",
                    os.path.join(out, "9_pred_vs_true_all_outputs.png"))

    summary = {"rae_rates": {"stage1_original": float(r0.mean()), "B_bestfit": float(rb.mean())},
               "B_better_pct": better, "B_worse_pct": worse,
               "by_noise_file": {nm: {"stage1": float(x0), "B": float(x1)} for nm, x0, x1 in zip(names, v0, vb)}}
    json.dump(summary, open(os.path.join(out, "compare_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=1))
    print(f"plots -> {out}")


if __name__ == "__main__":
    main()
