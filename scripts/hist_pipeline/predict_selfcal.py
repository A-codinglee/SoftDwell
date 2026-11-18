#!/usr/bin/env python3
# scripts/hist_pipeline/predict_selfcal.py
"""
Rates of recorded traces from a model trained by train_selfcal_online.py (two-look network).

Each trace file is raw int16 at the training sampling rate and gain (as the noise .bin files), cut into
consecutive pieces of the training trace length (10 s at 100 kHz = 1,000,000 points); a shorter rest is
left out. Both sets of histograms are built on the GPU from the raw trace, as in training (the fixed
ladder there came from the simulator's CPU code, which gives the same histograms).

--check N first re-simulates the run's first N test traces (same seed and stream, so the same traces),
predicts them the same way and compares with the run's preds_test.npy: this tests the whole path a
recorded trace takes here.

--shift-pA S subtracts S pA from the traces before predicting (the rates should not change when only the
baseline moves; the network was trained with baselines of -1..1 pA).

Rates are in Hz, in the order of the chain (0,1),(1,0),(1,2),(2,1),...; for a symmetric topology (COC,
COCOC) the targets were canonicalized (the vector or its reverse, whichever is lexicographically smaller),
so state 0 is the end state with the smaller rate out of it.

Usage (on a GPU node):
  python3 -m scripts.hist_pipeline.predict_selfcal --run-dir <run>/seed1 --trace <file.bin> [--check 256]
"""
import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch

from scripts.hist_pipeline import train_selfcal_online as tso
from scripts.hist_pipeline.train_hist_online import (CONDITIONS, STREAM_TEST, _pool_init, simulate_fixed_set,
                                                     split_outputs)
from scripts.simulation.simulator import SampleSpec, TraceSimulator


def run_args(run_dir):
    """The trainer's arguments (its defaults) with the network and histogram settings the run saved."""
    tc = json.load(open(os.path.join(run_dir, "train_config.json")))
    argv, sys.argv = sys.argv, ["train_selfcal_online", "--sim-config", os.path.join(run_dir, "sim_config.json"),
                                "--output-root", tempfile.gettempdir()]
    try:
        a = tso.parse_args()
    finally:
        sys.argv = argv
    m, h = tc["model"], tc["histograms"]
    a.backbone, a.hidden, a.depth, a.kernel_size = m["backbone"], m["hidden"], m["depth"], m["kernel_size"]
    a.base_scale_pA, a.amp_min_pA = m["base_scale_pA"], m["amp_min_pA"]
    a.fixed_k, a.fixed_j, a.fixed_thr_pA = h["fixed"]["thresholds"], h["fixed"]["bins"], h["fixed"]["range_pA"]
    a.selfcal_k, a.selfcal_j, a.margin_pA = h["selfcal"]["thresholds"], h["selfcal"]["bins"], h["selfcal"]["margin_pA"]
    a.hist_impl = tc.get("histogram_code", a.hist_impl)
    return a, tc


@torch.no_grad()
def predict(core, hists, X, batch_size):
    """X (N, T) int16 on the CPU -> (outputs (N, y_dim), first-look baseline (N,) pA, amplitude (N,) pA)."""
    core.eval()
    out, base, amp = [], [], []
    for s in range(0, X.shape[0], batch_size):
        p, b, a = tso.forward(core, hists, X[s:s + batch_size])
        out.append(p.float().cpu()), base.append(b.float().cpu()), amp.append(a.float().cpu())
    return torch.cat(out).numpy().astype(np.float64), torch.cat(base).numpy(), torch.cat(amp).numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="the run's seed folder (best_model.pt, train_config.json, ...)")
    p.add_argument("--trace", nargs="*", default=[], help="int16 .bin trace files")
    p.add_argument("--shift-pA", type=float, nargs="*", default=[0.0],
                   help="baseline shifts to try (pA subtracted from the traces); default 0")
    p.add_argument("--check", type=int, default=0, metavar="N", help="first compare on N of the run's test traces")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out", default=None, help="JSON with the predictions (default: <run-dir>/predictions_<name>.json)")
    cli = p.parse_args()

    device = torch.device("cuda")
    args, tc = run_args(cli.run_dir)
    cond = json.load(open(os.path.join(cli.run_dir, "conditions.json")))
    config_path = os.path.join(cli.run_dir, "sim_config.json")
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, fs = probe.trace_len, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    c, n_rates, y_dim = sim_cfg.counts_per_pA, tc["model"]["num_rates"], tc["model"]["num_outputs"]
    rate_names = [nm for i in range(n_rates // 2) for nm in (f"k{i}->{i + 1}", f"k{i + 1}->{i}")]

    hists = tso.Hists(args, trace_len, fs, c, device)
    core = tso.build_core(args, y_dim, device)
    core.load_state_dict(torch.load(os.path.join(cli.run_dir, "best_model.pt"), map_location=device))
    print(f"[Predict] {cli.run_dir}: topology {sim_cfg.topology}, {n_rates} rates ({' '.join(rate_names)}), "
          f"trace length {trace_len} at {fs:g} Hz, {c:g} counts/pA, histograms on {len(hists.devices)} GPU(s)",
          flush=True)

    if cli.check:
        ctx = torch.multiprocessing.get_context("spawn")
        n_cpus = len(os.sched_getaffinity(0))
        spectrum = os.path.join(cli.run_dir, "noise_spectrum.npy")
        with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                      initargs=(config_path, sim_cfg, tc["seed"], spectrum, cond, True)) as pool:
            X, y = simulate_fixed_set(pool, STREAM_TEST, cli.check, (trace_len,), y_dim, torch.int16)[:2]
        out, _, _ = predict(core, hists, X, cli.batch_size)
        pred_hz, targ_hz, _ = split_outputs(out, y.numpy().astype(np.float64), cond)
        saved_p = np.load(os.path.join(cli.run_dir, "preds_test.npy"))[:cli.check]
        saved_t = np.load(os.path.join(cli.run_dir, "targs_test.npy"))[:cli.check]
        d_t = np.abs(np.log10(targ_hz) - np.log10(saved_t)).max()
        d_p = np.abs(np.log10(pred_hz) - np.log10(saved_p))
        rae = lambda ph, th: float(np.sqrt(np.abs(np.log10(ph) - np.log10(th)) + 1e-12).mean())
        print(f"[Check] {cli.check} test traces: true rates match the run's: max |dlog10| {d_t:.2e}; predictions "
              f"vs the run's: max |dlog10| {d_p.max():.2e}, mean {d_p.mean():.2e}; RAE here {rae(pred_hz, targ_hz):.4f}"
              f", in the run {rae(saved_p, saved_t):.4f}", flush=True)

    results = []
    for path in cli.trace:
        raw = np.fromfile(path, dtype="<i2")
        n = raw.size // trace_len
        if n == 0:
            raise SystemExit(f"{path}: {raw.size} points, shorter than one training trace ({trace_len})")
        print(f"\n[Predict] {os.path.basename(path)}: {raw.size} points -> {n} piece(s) of {trace_len / fs:g} s"
              + (f" ({raw.size - n * trace_len} points left out)" if raw.size % trace_len else ""), flush=True)
        for shift in cli.shift_pA:
            x = raw[:n * trace_len].astype(np.int32).reshape(n, trace_len) - int(round(shift * c))
            X = torch.from_numpy(np.clip(x, -32768, 32767).astype(np.int16))
            out, base, amp = predict(core, hists, X, cli.batch_size)
            rates_hz, _, aux = split_outputs(out, out, cond)
            for k in range(n):
                r = {"file": os.path.basename(path), "piece": k, "start_s": k * trace_len / fs, "shift_pA": shift,
                     "rates_hz": dict(zip(rate_names, rates_hz[k].tolist())),
                     **{f"{nm}_pA": float(aux[k, j] / c) for j, nm in enumerate(CONDITIONS)},
                     "first_look_baseline_pA": float(base[k]), "first_look_amplitude_pA": float(amp[k])}
                results.append(r)
                print(f"  piece {k} shift {shift:+.2f} pA | " + "  ".join(f"{nm} {v:10.1f} Hz" for nm, v in
                                                                         r["rates_hz"].items())
                      + f" | baseline {r['baseline_pA']:+.3f} amplitude {r['amplitude_pA']:.3f} sigma "
                        f"{r['sigma_pA']:.3f} pA (first look: baseline {base[k]:+.3f}, amplitude {amp[k]:.3f})",
                      flush=True)
    if results:
        name = os.path.splitext(os.path.basename(cli.trace[0]))[0] if len(cli.trace) == 1 else "traces"
        out_path = cli.out or os.path.join(cli.run_dir, f"predictions_{name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Predict] written {out_path}", flush=True)


if __name__ == "__main__":
    main()
