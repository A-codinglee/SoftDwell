#!/usr/bin/env python3
# scripts/refine_pipeline/plot_recording_resim.py
"""
A recording next to traces re-simulated with each model's predictions (stage 1, A, B from predict_refine.py), in
the layout of the stage-1 resim_<recording>.png figures: whole trace (min-max band per 1 ms), 200 ms and 20 ms
zooms at fixed times, and amplitude histograms (linear and log) with summary statistics.

Every re-simulated trace uses the same Gillespie seed and the same noise draw, so the columns differ only by the
predicted parameters. Defaults (noise row 4 = the 2026_02_25 noise recording, rng 7, seed 11) reproduce the
stage-1 figure's simulated trace.

Usage:
  python3 -m scripts.refine_pipeline.plot_recording_resim --refine-run <run>/seed1 \
      --predict <run>/seed1/predict_<x>/predict_refine.json [--rule bestfit|fixed] [--out fig.png]
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.refine_pipeline.resim import ThetaToSim
from scripts.simulation.noise import SpectralNoise
from scripts.simulation.simulator import SampleSpec, TraceSimulator

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"recording": "#2a78d6", "stage 1": "#8f8e88", "A": "#1baf7a", "B": "#eb6834"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refine-run", required=True)
    p.add_argument("--predict", required=True, help="predict_refine.json")
    p.add_argument("--shift", type=float, default=0.0, help="which --shift-pA entry of predict_refine.json")
    p.add_argument("--rule", default="bestfit", choices=["bestfit", "fixed"])
    p.add_argument("--models", nargs="+", default=["stage 1", "B"],
                   help="trace columns; every model is listed in the statistics table")
    p.add_argument("--noise-row", type=int, default=4)
    p.add_argument("--noise-rng", type=int, default=7)
    p.add_argument("--sim-seed", type=int, default=11)
    p.add_argument("--zoom200", type=float, default=4.00, help="start (s) of the 200 ms window")
    p.add_argument("--zoom20", type=float, default=4.10, help="start (s) of the 20 ms window")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    run = a.refine_run
    pr = json.load(open(a.predict))
    entry = [e for e in pr["shifts"] if abs(e["shift_pA"] - a.shift) < 1e-9][0]
    tc = json.load(open(os.path.join(run, "train_config.json")))
    cond = json.load(open(os.path.join(run, "conditions.json")))
    nr = tc["model"]["num_rates"]
    sim = TraceSimulator(os.path.join(run, "sim_config.json"))
    spec = SampleSpec.from_simulator(sim)
    c, T = spec.counts_per_pA, sim.trace_len
    fs = float(sim.config["sampling"]["rate_hz"])
    to_sim = ThetaToSim(spec, cond, nr)
    noise = SpectralNoise(np.load(os.path.join(run, "noise_spectrum.npy")), T).synthesize(
        np.random.default_rng(a.noise_rng), a.noise_row)
    noise_name = spec.noise_sources(test=False)
    s1tc = json.load(open(os.path.join(tc["stage1_run"], "train_config.json")))
    noise_files = s1tc.get("data", {}).get("noise_files_train_val") or []

    thetas = {"stage 1": np.asarray(entry["stage1"]["theta"])}
    key = "bestfit_theta_median" if a.rule == "bestfit" else "fixed_theta_median"
    for v in ("A", "B"):
        if v in entry["variants"]:
            thetas[v] = np.asarray(entry["variants"][v][key])
    models = [m for m in a.models if m in thetas]
    table_models = list(thetas)

    traces, params = {}, {}
    for m in table_models:
        Q, lv, sg = to_sim(thetas[m][None])
        traces[m] = sim.simulate_trace(Q[0], lv[0], seed=a.sim_seed, noise_index=0, sigma=float(sg[0]),
                                       noise=noise) / c
        aux = (thetas[m][nr:nr + 3] * np.asarray(cond["std"]) + np.asarray(cond["mean"])) / c
        params[m] = (10 ** np.clip(thetas[m][:nr], to_sim.log_lo, to_sim.log_hi), aux)
    sim.close()
    rec = np.fromfile(pr["trace"], dtype="<i2")[:T].astype(np.float64) / c - a.shift
    t = np.arange(T) / fs
    base, amp = params["stage 1"][1][0], params["stage 1"][1][1]
    half = base + amp / 2

    series = [("Recording", rec, COLORS["recording"])] + \
             [(f"{m}" if m == "stage 1" else f"Refined {m}", traces[m], COLORS[m]) for m in models]
    nc = len(series)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "figure.facecolor": SURF, "axes.facecolor": SURF})
    fig = plt.figure(figsize=(5.6 * nc, 15))
    gs = fig.add_gridspec(4, nc, hspace=0.5, wspace=0.2, top=0.95, bottom=0.07)
    ylim = (min(x.min() for _, x, _ in series) - 0.5, max(x.max() for _, x, _ in series) + 0.5)
    ylim = (max(ylim[0], base - 6), min(ylim[1], base + amp + 14))

    def levels(ax, b, A):
        ax.axhline(b, color=INK2, ls="--", lw=1)
        ax.axhline(b + A, color=INK2, ls=":", lw=1)

    for j, (name, x, col) in enumerate(series):
        b, A = (base, amp) if j == 0 else params[models[j - 1]][1][:2]
        lab = name if j == 0 else f"Re-simulated, {name}"
        # whole trace
        ax = fig.add_subplot(gs[0, j])
        blk = int(fs * 1e-3)
        x2 = x[:T // blk * blk].reshape(-1, blk)
        tb = t[:T // blk * blk:blk]
        ax.fill_between(tb, x2.min(1), x2.max(1), color=col, lw=0, alpha=0.4)
        ax.plot(tb, x2.mean(1), color=col, lw=0.6)
        levels(ax, b, A)
        ax.set(xlim=(0, tb[-1]), ylim=ylim, xlabel="time (s)")
        ax.set_title(f"{lab}: whole {T / fs:.0f} s", loc="left", color=INK, fontsize=10)
        # zooms
        for row, (start, dur, unit) in ((1, (a.zoom200, 0.2, "ms")), (2, (a.zoom20, 0.02, "ms"))):
            ax = fig.add_subplot(gs[row, j])
            s, n = int(start * fs), int(dur * fs)
            ax.plot((t[s:s + n] - t[s]) * 1e3, x[s:s + n], color=col, lw=0.7)
            levels(ax, b, A)
            ax.set(xlim=(0, dur * 1e3), ylim=ylim, xlabel=f"time from {start:.2f} s (ms)")
            ax.set_title(f"{lab}: {dur * 1e3:.0f} ms", loc="left", color=INK, fontsize=10)
        for row in range(3):
            if j == 0:
                fig.axes[-3 + row].set_ylabel("current (pA)")

    # histograms + statistics
    edges = np.linspace(ylim[0], ylim[1], 200)
    for k, scale in enumerate(("linear", "log")):
        ax = fig.add_subplot(gs[3, k])
        for name, x, col in series:
            h, _ = np.histogram(x, edges, density=True)
            ax.step(edges[:-1], h, where="post", color=col, lw=1.8 if name == "Recording" else 1.4, label=name)
        ax.axvline(base, color=INK2, ls="--", lw=1)
        ax.axvline(base + amp, color=INK2, ls=":", lw=1)
        if scale == "log":
            ax.set(yscale="log", ylim=(1e-5, None))
        ax.set(xlabel="current (pA)", ylabel="density (1/pA)" if k == 0 else None)
        ax.set_title(f"Amplitude histogram, all {T / fs:.0f} s ({scale} scale)", loc="left", color=INK, fontsize=10)
        if k == 0:
            ax.legend(frameon=False, loc="upper left")
    ax = fig.add_subplot(gs[3, 2 if nc > 2 else 1])
    ax.axis("off")
    stats = [("mean", np.mean), ("std", np.std)] + \
            [({1: "1st pct"}.get(q, f"{q}th pct"), (lambda q: lambda x: np.percentile(x, q))(q))
             for q in (1, 5, 50, 95, 99, 99.9)] + [("% above half-open", lambda x: 100 * np.mean(x > half))]
    heads = ["rec"] + table_models
    cols_x = [rec] + [traces[m] for m in table_models]
    lines = [f"{'pA':>18}" + "".join(f"{h:>10}" for h in heads)]
    for nm, f in stats:
        nm = "median" if nm == "50th pct" else nm
        lines.append(f"{nm:>18}" + "".join(f"{f(x):10.2f}" for x in cols_x))
    lines.append("")
    lines.append(f"{'rates (Hz)':>18}")
    rate_lab = [r.replace("->", "→") for r in pr["rate_names"]]
    for i, rl in enumerate(rate_lab):
        lines.append(f"{rl:>18}" + f"{'':>10}" + "".join(f"{params[m][0][i]:10,.0f}" for m in table_models))
    for i, nm in enumerate(("baseline", "amplitude", "sigma")):
        lines.append(f"{nm + ' (pA)':>18}" + f"{'':>10}" + "".join(f"{params[m][1][i]:10.3f}" for m in table_models))
    dfresh = [entry["stage1"]["d_fresh"]] + [entry["variants"][m]["d_bestfit_fresh" if a.rule == "bestfit"
                                                                   else "d_fixed_fresh"] for m in table_models[1:]]
    lines.append(f"{'d (fresh resims)':>18}" + f"{'':>10}" + "".join(f"{d:10.4f}" for d in dfresh))
    ax.text(0.0, 1.0, "\n".join(lines), family="monospace", fontsize=9, va="top", ha="left", color=INK,
            transform=ax.transAxes)
    ax.set_title("Statistics and parameters", loc="left", color=INK, fontsize=10)

    rule = "best-fit iterate" if a.rule == "bestfit" else f"after {pr['iters']} iterations"
    rec_name = os.path.basename(pr["trace"])
    fig.suptitle(f"{rec_name} vs traces re-simulated with each model's predictions (refined: {rule}, median of "
                 f"{pr['reps']} loop copies)", x=0.01, ha="left", fontsize=12, color=INK, y=0.985)
    nf = os.path.basename(noise_files[a.noise_row]) if a.noise_row < len(noise_files) else f"row {a.noise_row}"
    fig.text(0.01, 0.005, f"All re-simulated traces use the same Gillespie seed ({a.sim_seed}) and the same noise draw "
             f"(spectrum of {nf}),\nso they differ only by the parameters. Dashed: closed level, dotted: open level "
             f"(each column its own model's). Half-open = {half:.2f} pA (stage 1). Zoom windows are at fixed times.\n"
             f"d: mean |H_obs − H_sim| over {pr['reps']} fresh re-simulations (lower = better fit).",
             color=INK2, fontsize=9, va="bottom")
    out = a.out or os.path.join(os.path.dirname(a.predict),
                                f"resim_{os.path.splitext(rec_name)[0]}_{a.rule}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("written", out)


if __name__ == "__main__":
    main()
