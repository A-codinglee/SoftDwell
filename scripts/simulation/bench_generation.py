#!/usr/bin/env python3
# scripts/simulation/bench_generation.py
"""
Where does the time go when simulating training samples, and how does it scale with the
number of worker processes?

For each worker count it runs that many processes in parallel, each simulating --n samples,
and reports samples/s plus the mean time per sample spent in each stage:

  noise     spectral noise synthesis in Python (0 for recorded noise)
  trace     C++: Gillespie + step-response filtering + adding the noise
  hist      C++: dwell-time histograms (measured as full call minus the trace-only call)

Usage:
  python3 -m scripts.simulation.bench_generation --sim-config configs/sim/COCO.json --workers 1 8 16 32 62
"""
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from scripts.simulation.noise import SpectralNoise, target_magnitude, welch_magnitude
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config


def _work(job):
    config_path, spectrum_path, wid, n = job
    sim = TraceSimulator(config_path)
    spec = SampleSpec.from_simulator(sim)
    synth = SpectralNoise(np.load(spectrum_path), sim.trace_len) if spectrum_path else None
    rng = np.random.default_rng(1000 + wid)
    t_noise = t_trace = t_full = 0.0
    for i in range(n):
        Q = spec.sample_Q(rng)
        levels, sigma, _ = spec.sample_levels_sigma(rng)
        t0 = time.perf_counter()
        noise = synth.synthesize(rng) if synth is not None else None
        t1 = time.perf_counter()
        sim.simulate_trace(Q, levels, seed=i, noise_index=i, sigma=sigma, noise=noise)
        t2 = time.perf_counter()
        sim.simulate_hist(Q, levels, seed=i, noise_index=i, sigma=sigma, noise=noise)
        t3 = time.perf_counter()
        t_noise += t1 - t0
        t_trace += t2 - t1
        t_full += t3 - t2
    return t_noise / n, t_trace / n, t_full / n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[1, 8, 16, 32, 62])
    p.add_argument("--n", type=int, default=60, help="samples per worker")
    p.add_argument("--generate-factor", type=int, nargs="+", default=None,
                   help="spectral noise only: values of noise.generate_factor to compare (default: the config's)")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/tmp"))
    args = p.parse_args()

    config_path = prepare_sim_config(args.sim_config, os.path.join(args.tmp_dir, f"bench_sim_config_{os.getpid()}.json"), 1)
    probe = TraceSimulator(config_path)
    spec = SampleSpec.from_simulator(probe)
    trace_len = probe.trace_len
    probe.close()

    variants = [("", None)]
    if spec.noise_mode == "spectral_resample":
        mag = welch_magnitude(spec.noise_path, spec.noise_avg_seg, spec.noise_avg_overlap)
        variants = []
        for gf in (args.generate_factor or [spec.noise_generate_factor]):
            path = os.path.join(args.tmp_dir, f"bench_spectrum_{os.getpid()}_{gf}.npy")
            np.save(path, target_magnitude(mag, trace_len, gf))
            variants.append((path, gf))

    cpus = len(os.sched_getaffinity(0))
    print(f"config={args.sim_config} topology={spec.topology} noise={spec.noise_mode} cpus available={cpus}")
    print("Times are means per sample in ms. 'per-core total' = noise + full C++ call, i.e. what one training "
          "sample costs one worker.\n")
    print(f"{'gen_factor':>10} {'workers':>8} {'samples/s':>10} {'per worker':>11} | {'noise':>7} {'trace':>7} {'hist':>7} "
          f"{'per-core total':>15}")
    ctx = mp.get_context("spawn")
    for spectrum_path, gf in variants:
        for w in args.workers:
            with ctx.Pool(w) as pool:
                res = np.array(pool.map(_work, [(config_path, spectrum_path, k, args.n) for k in range(w)], chunksize=1))
            noise, trace, full = res.mean(axis=0) * 1e3
            total = noise + full                      # one training sample = synth + one full call
            rate = w * 1e3 / total
            print(f"{str(gf or '-'):>10} {w:>8} {rate:>10.1f} {rate / w:>11.2f} | {noise:>7.1f} {trace:>7.1f} "
                  f"{full - trace:>7.1f} {total:>15.1f}", flush=True)


if __name__ == "__main__":
    main()
