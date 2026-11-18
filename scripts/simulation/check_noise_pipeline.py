#!/usr/bin/env python3
# scripts/simulation/check_noise_pipeline.py
"""
Check the "spectral_resample" noise path end to end, for every noise file of a sim-config:

  recording --welch_magnitude/target_magnitude--> spectrum --SpectralNoise--> int16 noise
            --simulator (scaled to sigma, added to the trace)--> final int16 trace

For each recording the power spectrum is estimated three times with scipy.signal.welch (independent
of the code under test): of the recording itself, of the synthesized noise, and of final traces
simulated with all levels equal (so the trace is baseline + noise only). The three should have the
same shape. Also checked: standard deviation and mean of the noise in the final trace, how Gaussian
it is, and how much the recording's own spectrum changes between its first and second half.

Writes noise_check.png and noise_check.csv to --out-dir.

Usage:
  python3 -m scripts.simulation.check_noise_pipeline --sim-config configs/sim/COCO_noise5.json --out-dir <dir>
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from scripts.plotting.plot_online_results import BLUE, GRID, INK, INK2, MUTED, ORANGE
from scripts.simulation.noise import SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config

NPERSEG = 8192


def band_psd(x, fs, edges):
    """Welch power spectral density of x (float array or list of arrays), averaged into the bands of edges."""
    xs = x if isinstance(x, list) else [x]
    acc = 0.0
    for a in xs:
        f, p = scipy.signal.welch(np.asarray(a, dtype=np.float64), fs=fs, nperseg=NPERSEG, detrend="constant")
        acc = acc + p
    p = acc / len(xs)
    return np.array([p[(f >= lo) & (f < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])


def kurt(a):
    a = np.asarray(a, dtype=np.float64)
    a = a - a.mean()
    return float((a ** 4).mean() / (a ** 2).mean() ** 2 - 3.0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-traces", type=int, default=12, help="synthesized noises / simulated traces per recording")
    p.add_argument("--sigma-pA", type=float, default=1.0)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    config_path = prepare_sim_config(args.sim_config, os.path.join(args.out_dir, "sim_config.json"), seed=1)
    sim = TraceSimulator(config_path)
    spec = SampleSpec.from_simulator(sim)
    fs = float(sim.config["sampling"]["rate_hz"])
    N, K = sim.trace_len, spec.num_states
    paths = spec.all_noise_paths
    sigma = args.sigma_pA * spec.counts_per_pA                      # raw counts
    level = 1000.0                                                  # all states at the same level: no gating signal
    Q = np.where(spec.q_hi > 0, 1000.0, 0.0)
    synth = SpectralNoise(source_spectra(paths, N, spec.noise_avg_seg, spec.noise_avg_overlap,
                                         spec.noise_generate_factor), N)
    print(f"fs={fs:g} Hz trace_len={N} generate_factor={spec.noise_generate_factor} synthesis FFT length={synth.nfft} "
          f"sigma={args.sigma_pA} pA = {sigma:.1f} counts")

    df = fs / NPERSEG                                               # every band holds at least one Welch bin
    edges = np.unique(np.round(np.geomspace(2 * df, fs / 2, 41) / df)) * df
    mid = np.sqrt(edges[:-1] * edges[1:])
    rng = np.random.default_rng(0)
    shape = lambda psd: psd / psd.sum()                             # spectrum shape, independent of level
    db = lambda a, b: 10 * np.log10(a / b)

    rows = []
    fig, axes = plt.subplots(2, len(paths), figsize=(3.5 * len(paths), 6.2), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for k, path in enumerate(paths):
        name = os.path.basename(path)
        rec = np.memmap(path, dtype=np.int16, mode="r")
        half = rec.size // 2
        P_rec = band_psd(rec, fs, edges)
        P_h1, P_h2 = band_psd(rec[:half], fs, edges), band_psd(rec[half:], fs, edges)

        syn, traces, stds, means, clipped = [], [], [], [], 0.0
        for t in range(args.n_traces):
            noise, unit = synth.synthesize(rng, k)
            syn.append(noise.astype(np.float32))                    # copy: the buffer is reused
            clipped = max(clipped, float(np.mean(np.abs(noise) >= 32767)))
            tr = sim.simulate_trace(Q, np.full(K, level), seed=t, sigma=sigma, noise=(noise, unit)).astype(np.float64)
            # the simulator must add exactly sigma / unit * noise to the constant level
            expect = np.rint(level + sigma / unit * noise.astype(np.float64))
            exact = float(np.mean(np.abs(tr - expect) <= 1.0))
            traces.append(tr - level)
            stds.append(traces[-1].std()); means.append(traces[-1].mean())
        P_syn, P_tr = band_psd(syn, fs, edges), band_psd(traces, fs, edges)

        d_syn, d_tr = db(shape(P_syn), shape(P_rec)), db(shape(P_tr), shape(P_rec))
        d_half = db(shape(P_h2), shape(P_h1))
        rows.append({
            "recording": name, "seconds": rec.size / fs,
            "synth_vs_recording_max_dB": np.abs(d_syn).max(), "synth_vs_recording_rms_dB": np.sqrt((d_syn ** 2).mean()),
            "trace_vs_recording_max_dB": np.abs(d_tr).max(), "trace_vs_recording_rms_dB": np.sqrt((d_tr ** 2).mean()),
            "recording_2nd_vs_1st_half_max_dB": np.abs(d_half).max(),
            "trace_std_over_sigma": np.mean(stds) / sigma, "trace_mean_counts": np.mean(means),
            "trace_matches_scaled_noise": exact, "synth_clipped_fraction": clipped,
            "excess_kurtosis_recording": kurt(rec[: min(rec.size, 20_000_000)]), "excess_kurtosis_synth": kurt(syn[0]),
            "recording_largest_excursion_in_std": float(np.abs(rec[:20_000_000] - np.mean(rec[:20_000_000])).max()
                                                        / np.std(rec[:20_000_000])),
        })

        ax, bx = axes[0, k], axes[1, k]
        ax.loglog(mid, shape(P_rec), color=INK, lw=3.0, alpha=0.35, label="recording")
        ax.loglog(mid, shape(P_syn), color=BLUE, lw=1.6, label="synthesized noise")
        ax.loglog(mid, shape(P_tr), color=ORANGE, lw=1.2, ls=(0, (4, 2)), label="noise in the final trace")
        ax.set_title(name[:26], fontsize=9.5)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        bx.axhline(0, color=INK, lw=0.8)
        bx.axhspan(-1, 1, color=GRID, lw=0)
        bx.semilogx(mid, d_half, color=MUTED, lw=1.2, label="recording: 2nd half vs 1st half")
        bx.semilogx(mid, d_syn, color=BLUE, lw=1.6, label="synthesized vs recording")
        bx.semilogx(mid, d_tr, color=ORANGE, lw=1.2, ls=(0, (4, 2)), label="final trace vs recording")
        bx.set_ylim(-6, 6)
        bx.set_xlabel("frequency (Hz)")
        bx.yaxis.grid(True); bx.set_axisbelow(True)
    axes[0, 0].set_ylabel("share of noise power per band")
    axes[1, 0].set_ylabel("difference (dB)")
    for ax in axes[:, 1:].ravel():
        ax.tick_params(labelleft=False)
    for ax in axes[0, 1:]:
        ax.sharey(axes[0, 0])
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2[:1], l1 + l2[:1], loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=4, fontsize=9, labelcolor=INK2)
    fig.suptitle("Noise spectrum: recording vs synthesized noise vs noise in the final simulated trace", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.text(0.01, 0.925, "Top: spectrum shape (each curve sums to 1). Bottom: difference from the recording in dB; "
             "grey band = within 1 dB (26%).", color=INK2, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91))
    fig.savefig(os.path.join(args.out_dir, "noise_check.png"), dpi=150); plt.close(fig)

    with open(os.path.join(args.out_dir, "noise_check.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})
    for r in rows:
        print(" | ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{v[:14]}" for k, v in r.items()))
    sim.close()


if __name__ == "__main__":
    main()
