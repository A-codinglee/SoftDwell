#!/usr/bin/env python3
# scripts/refine_pipeline/diag_discrepancy.py
"""
Can the refinement loop's signal tell the true parameters from stage 1's estimate at all?

For every test sample (the same test set as train_refine_online.py with the same seed) this re-simulates at
several parameter points with common random numbers (replicate r uses the same random numbers at every point)
and measures the discrepancy d = mean |H_obs - H_sim| that the refiner sees:

  truth          y (the parameters the observed trace was simulated with)
  stage1         theta_0
  rates0+cond*   stage-1 rates, true conditions        (which part of theta_0's error shows up in d)
  rates*+cond0   true rates, stage-1 conditions
  y+rates d      y with the rates moved by d (log10) along a random unit direction, d in --deltas

Reported per point: mean d, the paired difference to the truth (same replicate), the fraction of samples
where the truth fits better, and that difference in units of the replicate-to-replicate spread of d(truth)
(the noise floor). "avgR" rows compare histograms averaged over the R replicates first (what several
simulations per iteration would give).

Usage:
  python3 -m scripts.refine_pipeline.diag_discrepancy --stage1-run <selfcal run>/seed1 --out <dir> --seed 101
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline import train_selfcal_online as tso
from scripts.hist_pipeline.train_hist_online import STREAM_TEST, _pool_init, simulate_fixed_set, write_noise_spectra
from scripts.refine_pipeline.resim import STREAM_RESIM, Resimulator, ThetaToSim
from scripts.refine_pipeline.train_refine_online import Stage1, precompute
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1-run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n-test", type=int, default=2048)
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.5, 1.0])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = os.path.abspath(args.stage1_run)
    config_path = prepare_sim_config(os.path.join(run, "sim_config.json"), os.path.join(args.out, "sim_config.json"),
                                     args.seed)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c, fs = probe.trace_len, sim_cfg.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    cond = json.load(open(os.path.join(run, "conditions.json")))
    stage1 = Stage1(run, trace_len, fs, c, device)
    y_dim, nr = stage1.y_dim, stage1.n_rates
    to_sim = ThetaToSim(sim_cfg, cond, nr)
    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.out)
    n_cpus = len(os.sched_getaffinity(0))
    workers = args.workers or max(1, n_cpus - 2)
    ctx = mp.get_context("spawn")

    t0 = time.time()
    with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                  initargs=(config_path, sim_cfg, args.seed, spectrum_path, cond, True)) as pool:
        X, y, _, _ = simulate_fixed_set(pool, STREAM_TEST, args.n_test, (trace_len,), y_dim, torch.int16)
    test = precompute(stage1, X, y, args.batch_size)
    del X
    print(f"[Diag] {args.n_test} test samples through stage 1 in {time.time() - t0:.1f}s | {workers} workers",
          flush=True)

    theta0, y = test["theta0"], test["y"].float()
    g = torch.Generator().manual_seed(args.seed)
    u = torch.randn(args.n_test, nr, generator=g)
    u = u / u.norm(dim=1, keepdim=True)
    points = {"truth": y, "stage1": theta0,
              "rates0+cond*": torch.cat([theta0[:, :nr], y[:, nr:]], 1),
              "rates*+cond0": torch.cat([y[:, :nr], theta0[:, nr:]], 1)}
    for d in args.deltas:
        points[f"y+rates {d:g}"] = torch.cat([y[:, :nr] + d * u, y[:, nr:]], 1)

    resim = Resimulator(ctx, workers, config_path, sim_cfg, spectrum_path, args.seed, to_sim, args.batch_size,
                        trace_len, pin=tso.pin_host)
    R, N, B = args.reps, args.n_test, args.batch_size
    disc = {k: torch.zeros(R, N) for k in points}
    disc_avg = {k: torch.zeros(N) for k in points}
    t0 = time.time()
    try:
        for s in range(0, N, B):
            idx = torch.arange(s, min(s + B, N))
            h_obs = test["h_obs"][idx].to(device)
            base, amp = test["base"][idx].to(device), test["amp"][idx].to(device)
            for k, th in points.items():
                acc = 0
                for r in range(R):
                    stream = STREAM_RESIM["test"] if r == 0 else 100 + r
                    h_sim = stage1.sim_hists(resim(th[idx], stream, idx), base, amp)
                    disc[k][r, idx] = (h_obs - h_sim).abs().mean(dim=(1, 2, 3)).float().cpu()
                    acc = acc + h_sim
                disc_avg[k][idx] = (h_obs - acc / R).abs().mean(dim=(1, 2, 3)).float().cpu()
            if s == 0:
                print(f"[Diag] first batch {time.time() - t0:.1f}s", flush=True)
    finally:
        resim.close()
    print(f"[Diag] {resim.n_traces} re-simulations in {time.time() - t0:.1f}s", flush=True)

    err = (theta0[:, :nr] - y[:, :nr]).norm(dim=1)
    print(f"[Diag] stage-1 rate error |theta0 - y| (log10, {nr} rates): median {err.median():.3f}, "
          f"mean {err.mean():.3f}, 90% {err.quantile(0.9):.3f}")
    floor = disc["truth"].std(dim=0)                       # per-sample replicate spread at the truth
    print(f"[Diag] noise floor: d(truth) = {disc['truth'].mean():.5f}, replicate std per sample "
          f"{floor.mean():.5f} (median {floor.median():.5f})\n")
    hdr = f"{'point':<16}{'mean d':>10}{'d - d*':>11}{'std(d-d*)':>11}{'P(d*<d)':>9}{'(d-d*)/std_rep':>16}" \
          f"{'| avgR: d - d*':>16}{'P(d*<d)':>9}"
    print(hdr)
    rows = {}
    for k in points:
        diff = disc[k] - disc["truth"]                    # paired by replicate
        dm = diff.mean(dim=0)
        da = disc_avg[k] - disc_avg["truth"]
        rows[k] = {"mean_d": float(disc[k].mean()), "diff_mean": float(diff.mean()), "diff_std": float(dm.std()),
                   "p_truth_better": float((diff > 0).float().mean()),
                   "diff_over_floor": float((dm / floor.clamp_min(1e-9)).median()),
                   "avg_diff_mean": float(da.mean()), "avg_p_truth_better": float((da > 0).float().mean())}
        r = rows[k]
        print(f"{k:<16}{r['mean_d']:>10.5f}{r['diff_mean']:>11.5f}{r['diff_std']:>11.5f}"
              f"{r['p_truth_better']:>9.3f}{r['diff_over_floor']:>16.2f}{r['avg_diff_mean']:>16.5f}"
              f"{r['avg_p_truth_better']:>9.3f}")
    # does the discrepancy rank stage-1's errors? (Spearman, per sample)
    gap = (disc["stage1"] - disc["truth"]).mean(dim=0)
    rk = lambda v: v.argsort().argsort().float()
    rho = float(np.corrcoef(rk(gap).numpy(), rk(err).numpy())[0, 1])
    print(f"\n[Diag] Spearman(d(theta0) - d(truth), |theta0 - y|_rates) = {rho:.3f}")
    json.dump({"rows": rows, "spearman_gap_vs_err": rho, "noise_floor_mean": float(floor.mean()),
               "n_test": N, "reps": R, "deltas": args.deltas}, open(os.path.join(args.out, "diag.json"), "w"),
              indent=2)
    np.savez(os.path.join(args.out, "diag.npz"), err=err.numpy(), **{k: v.numpy() for k, v in disc.items()})


if __name__ == "__main__":
    main()
