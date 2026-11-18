#!/usr/bin/env python3
# scripts/refine_pipeline/floor_search.py
"""
How well can the simulator fit a recording at all? Direct search for the parameters with the smallest
discrepancy d = mean |H_obs - H_sim| (the refinement loop's signal, same histograms and ladders), no network.

  1. Nelder-Mead on d, in the stage-1 output space (log10 rates, standardized baseline / amplitude / sigma), from
     each start point (stage 1 and the refined A / B estimates of predict_refine.py). During the search every
     evaluation re-simulates --reps traces with fixed random numbers, so d is a deterministic function.
  2. The best point of each search and every start point are re-scored with --fresh new re-simulations (random
     numbers the search never saw), so the search cannot win by fitting its own noise.
  3. Reference: d of the same parameters when the model is exactly right, i.e. a trace SIMULATED at the found
     parameters scored against fresh re-simulations (--n-ref such pseudo-recordings). The gap between the
     recording's floor and this reference is what the simulator cannot reproduce (noise, model, ranges).

Usage (CPU):
  python3 -m scripts.refine_pipeline.floor_search --refine-run <run>/seed1 \
      --predict <run>/seed1/predict_<x>/predict_refine.json --out <dir> [--noise-source 4]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.optimize import minimize

from scripts.hist_pipeline.train_hist_online import write_noise_spectra
from scripts.refine_pipeline.resim import Resimulator, ThetaToSim
from scripts.refine_pipeline.train_refine_online import Stage1
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config

STREAM_SEARCH, STREAM_FRESH, STREAM_REF = 500, 600, 700


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refine-run", required=True)
    p.add_argument("--predict", required=True, help="predict_refine.json (start points, recording path)")
    p.add_argument("--out", required=True)
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--noise-source", type=int, default=None, help="force this noise file (row); default: random")
    p.add_argument("--reps", type=int, default=8, help="re-simulations per evaluation in the search")
    p.add_argument("--fresh", type=int, default=32, help="re-simulations for the final scores")
    p.add_argument("--n-ref", type=int, default=8, help="pseudo-recordings for the reference")
    p.add_argument("--maxfev", type=int, default=400, help="evaluations per start point")
    p.add_argument("--workers", type=int, default=0)
    a = p.parse_args()

    os.makedirs(a.out, exist_ok=True)
    run = os.path.abspath(a.refine_run)
    tc = json.load(open(os.path.join(run, "train_config.json")))
    pr = json.load(open(a.predict))
    entry = [e for e in pr["shifts"] if abs(e["shift_pA"] - a.shift) < 1e-9][0]
    seed = int(tc.get("seed", 1))
    device = torch.device("cpu")
    config_path = prepare_sim_config(os.path.join(tc["stage1_run"], "sim_config.json"),
                                     os.path.join(a.out, "sim_config.json"), seed)
    probe = TraceSimulator(config_path)
    spec = SampleSpec.from_simulator(probe)
    T, c, fs = probe.trace_len, spec.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    cond = json.load(open(os.path.join(tc["stage1_run"], "conditions.json")))
    stage1 = Stage1(tc["stage1_run"], T, fs, c, device)
    nr = stage1.n_rates
    to_sim = ThetaToSim(spec, cond, nr)
    mean, std = np.asarray(cond["mean"]), np.asarray(cond["std"])
    spectrum_path = write_noise_spectra(spec, T, a.out)
    workers = a.workers or max(1, min(48, len(os.sched_getaffinity(0)) - 2))
    nmax = max(a.reps, a.fresh, a.n_ref)
    resim = Resimulator(mp.get_context("spawn"), workers, config_path, spec, spectrum_path, seed, to_sim, nmax, T,
                        noise_source=a.noise_source)

    raw = np.fromfile(pr["trace"], dtype="<i2")[:T]
    x = np.clip(raw.astype(np.int32) - int(round(a.shift * c)), -32768, 32767).astype(np.int16)
    with torch.no_grad():
        th0, _, base, amp, h_obs = stage1(torch.from_numpy(x)[None])
    names = [nm.replace("->", "→") for nm in pr["rate_names"]]

    def to_phys(th):
        aux = (np.asarray(th[nr:nr + 3]) * std + mean) / c
        return {**{n: float(10 ** np.clip(v, to_sim.log_lo, to_sim.log_hi)) for n, v in zip(names, th[:nr])},
                "baseline_pA": float(aux[0]), "amplitude_pA": float(aux[1]), "sigma_pA": float(aux[2])}

    def hists(th, n, stream):
        tht = torch.as_tensor(np.asarray(th), dtype=torch.float32)[None].expand(n, -1)
        return stage1.sim_hists(resim(tht, stream, torch.arange(n)), base.expand(n), amp.expand(n))

    n_eval = [0]

    def d_search(th):
        n_eval[0] += 1
        return float((h_obs - hists(th, a.reps, STREAM_SEARCH)).abs().mean())

    def d_fresh(th):
        d = (h_obs - hists(th, a.fresh, STREAM_FRESH)).abs().mean(dim=(1, 2, 3))
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    def d_reference(th):
        """d when the model is right: pseudo-recordings simulated at th vs fresh re-simulations at th."""
        h_sim = hists(th, a.fresh, STREAM_FRESH).clone()
        h_ref = hists(th, a.n_ref, STREAM_REF)
        d = torch.stack([(h_ref[j] - h_sim).abs().mean(dim=(1, 2, 3)).mean() for j in range(a.n_ref)])
        return float(d.mean()), float(d.std() / np.sqrt(len(d)))

    # search steps: 0.1 decade for rates; 0.2 / 0.3 / 0.05 pA for baseline / amplitude / sigma (standardized)
    step = np.concatenate([np.full(nr, 0.1), np.array([0.2, 0.3, 0.05]) * c / std])
    starts = {"stage 1": np.asarray(entry["stage1"]["theta"], dtype=np.float64)}
    for v in ("A", "B"):
        if v in entry["variants"]:
            starts[f"{v} best-fit"] = np.asarray(entry["variants"][v]["bestfit_theta_median"], dtype=np.float64)

    report = {"recording": pr["trace"], "noise_source": a.noise_source, "reps": a.reps, "fresh": a.fresh,
              "n_ref": a.n_ref, "maxfev": a.maxfev, "results": {}}
    t0 = time.time()
    print(f"[Floor] {os.path.basename(pr['trace'])} | noise: "
          f"{'file row ' + str(a.noise_source) if a.noise_source is not None else 'random of the training files'} | "
          f"{a.reps} re-sims per evaluation, {a.fresh} fresh for the scores, {workers} workers", flush=True)
    try:
        for name, th_start in starts.items():
            n_eval[0] = 0
            best = [np.inf, th_start]

            def f(z):
                th = th_start + z * step
                v = d_search(th)
                if v < best[0]:
                    best[0], best[1] = v, th
                if n_eval[0] % 50 == 0:
                    print(f"    {name}: {n_eval[0]} evaluations, best search d {best[0]:.5f} "
                          f"({time.time() - t0:.0f}s)", flush=True)
                return v

            d_start_search = f(np.zeros_like(th_start))
            simplex = np.vstack([np.zeros(len(th_start)), np.eye(len(th_start))])
            minimize(f, np.zeros(len(th_start)), method="Nelder-Mead",
                     options={"maxfev": a.maxfev, "initial_simplex": simplex, "xatol": 0.05, "fatol": 2e-5})
            th_best = best[1]
            ds, dss = d_fresh(th_start)
            db, dbs = d_fresh(th_best)
            rs, rss = d_reference(th_start)
            rb, rbs = d_reference(th_best)
            res = {"start": to_phys(th_start), "best": to_phys(th_best), "theta_start": th_start.tolist(),
                   "theta_best": th_best.tolist(), "evaluations": n_eval[0],
                   "d_search_start": d_start_search, "d_search_best": best[0],
                   "d_fresh_start": ds, "d_fresh_start_se": dss, "d_fresh_best": db, "d_fresh_best_se": dbs,
                   "d_reference_start": rs, "d_reference_start_se": rss,
                   "d_reference_best": rb, "d_reference_best_se": rbs,
                   "change_x": {n: 10 ** (b - s) for n, b, s in zip(names, th_best[:nr], th_start[:nr])}}
            report["results"][name] = res
            print(f"\n  == from {name} ({n_eval[0]} evaluations)\n"
                  f"     fresh d: start {ds:.5f} ± {dss:.5f} -> best {db:.5f} ± {dbs:.5f}   "
                  f"(if the model were right: {rs:.5f} at the start, {rb:.5f} at the best)\n"
                  f"     start: " + "  ".join(f"{k} {v:,.3f}" if "pA" in k else f"{k} {v:,.0f}"
                                            for k, v in res["start"].items()) + "\n"
                  f"     best:  " + "  ".join(f"{k} {v:,.3f}" if "pA" in k else f"{k} {v:,.0f}"
                                            for k, v in res["best"].items()) + "\n", flush=True)
            json.dump(report, open(os.path.join(a.out, "floor_search.json"), "w"), indent=2)
    finally:
        resim.close()
    print(f"[Floor] {resim.n_traces} re-simulations, {time.time() - t0:.0f}s -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
