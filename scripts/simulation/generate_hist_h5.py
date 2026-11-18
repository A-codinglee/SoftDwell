#!/usr/bin/env python3
# scripts/simulation/generate_hist_h5.py
"""
Simulate N samples with the C++ simulator and write them in the layout
scripts.hist_pipeline.train_hist expects:

  histograms/grids     (N, D, B, B) float32
  histograms/N_events  (N, D)       int64
  labels/Q             (N, S, S)    float32
  labels/levels        (N, S)       float32
  labels/noise_sigma   (N,)         float32

Sample i depends only on (--seed, i), not on the number of workers.
"""
import argparse
import multiprocessing as mp
import os
import time

import h5py
import numpy as np

from scripts.simulation.simulator import SimConfig, TraceSimulator, sample_chain_Q

_SIM = None
_CFG = None
_SEED = None


def _init_worker(config_path: str, cfg: SimConfig, seed: int):
    global _SIM, _CFG, _SEED
    _SIM = TraceSimulator(config_path)
    _CFG = cfg
    _SEED = seed


def _simulate_one(i: int):
    rng = np.random.default_rng(np.random.SeedSequence([_SEED, i]))
    Q = sample_chain_Q(rng, _CFG.num_states, _CFG.k_min, _CFG.k_max)
    sim_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    grids, n_events = _SIM.simulate_hist(Q, _CFG.state_levels(), seed=sim_seed, noise_index=i)
    return i, grids, n_events, Q.astype(np.float32), _SIM.last_sigma


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output .h5 path")
    p.add_argument("--n", type=int, required=True, help="number of samples")
    p.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--topology", default="COC")
    p.add_argument("--level-c", type=float, default=22000.0)
    p.add_argument("--level-o", type=float, default=20000.0)
    p.add_argument("--sigma", type=float, default=2000.0, help="noise sigma, raw ADC counts")
    p.add_argument("--t-max-s", type=float, default=10.0)
    p.add_argument("--k-min", type=float, default=10.0)
    p.add_argument("--k-max", type=float, default=1e6)
    p.add_argument("--bins", type=int, default=24)
    p.add_argument("--tmin-s", type=float, default=1e-5)
    p.add_argument("--tmax-s", type=float, default=0.1)
    p.add_argument("--grind-min", type=float, default=18000.0)
    p.add_argument("--grind-max", type=float, default=24000.0)
    p.add_argument("--grind-count", type=int, default=24)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = SimConfig(
        topology=args.topology.upper(),
        levels={"C": args.level_c, "O": args.level_o},
        noise_sigma=args.sigma,
        t_max_s=args.t_max_s,
        seed=args.seed,
        hist_bins=args.bins, hist_tmin_s=args.tmin_s, hist_tmax_s=args.tmax_s,
        grind_min=args.grind_min, grind_max=args.grind_max, grind_count=args.grind_count,
        k_min=args.k_min, k_max=args.k_max,
    )
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    config_path = cfg.write_json(os.path.splitext(out)[0] + ".sim_config.json")

    N, S, D, B = args.n, cfg.num_states, cfg.grind_count, cfg.hist_bins
    levels = cfg.state_levels().astype(np.float32)
    workers = max(1, min(args.workers, N))
    print(f"[gen] N={N} topology={cfg.topology} grids=({D},{B},{B}) workers={workers}", flush=True)
    print(f"[gen] simulator config -> {config_path}", flush=True)

    t0 = time.perf_counter()
    with h5py.File(out, "w") as f:
        f.attrs["content"] = "hist"
        f.attrs["dtype"] = "hist_f32"
        f.attrs["sample_rate_hz"] = cfg.sampling_rate_hz
        f.attrs["generator"] = "scripts.simulation.generate_hist_h5"
        gh = f.create_group("histograms")
        gh.attrs["D"], gh.attrs["B"] = np.int32(D), np.int32(B)
        gh.attrs["tmin_s"], gh.attrs["tmax_s"] = cfg.hist_tmin_s, cfg.hist_tmax_s
        gh.attrs["grind_min"], gh.attrs["grind_max"] = cfg.grind_min, cfg.grind_max
        d_grids = gh.create_dataset("grids", (N, D, B, B), dtype=np.float32)
        d_nev = gh.create_dataset("N_events", (N, D), dtype=np.int64)
        gl = f.create_group("labels")
        d_Q = gl.create_dataset("Q", (N, S, S), dtype=np.float32)
        d_lv = gl.create_dataset("levels", (N, S), dtype=np.float32)
        d_sig = gl.create_dataset("noise_sigma", (N,), dtype=np.float32)

        ctx = mp.get_context("spawn")
        done, last = 0, t0
        with ctx.Pool(workers, initializer=_init_worker, initargs=(config_path, cfg, args.seed)) as pool:
            for i, grids, n_events, Q, sigma in pool.imap_unordered(_simulate_one, range(N), chunksize=8):
                d_grids[i], d_nev[i], d_Q[i], d_lv[i], d_sig[i] = grids, n_events, Q, levels, sigma
                done += 1
                now = time.perf_counter()
                if now - last >= 10.0 or done == N:
                    last = now
                    print(f"[gen] {done}/{N} ({done / (now - t0):.1f} samples/s)", flush=True)

    dt = time.perf_counter() - t0
    print(f"[gen] wrote {out} in {dt:.1f}s ({N / dt:.1f} samples/s, {N / dt / workers:.1f} per worker)", flush=True)


if __name__ == "__main__":
    main()
