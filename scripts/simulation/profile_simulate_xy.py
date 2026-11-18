#!/usr/bin/env python3
# scripts/simulation/profile_simulate_xy.py
"""
Profile simulate_xy() from scripts.hist_pipeline.train_hist_online: where does the time of one
training sample go?

It runs the same steps as simulate_xy with a timer around each one, on the same samples the
training would draw (stream / index), and checks once that its output equals simulate_xy's.
The C++ call is split further with the timers inside libsimdwell.so.

  draw        Python: random generator, rates, levels, sigma
  noise       Python: spectral noise synthesis (0 for recorded noise), split into its steps
  gillespie   C++: state sequence
  synth       C++: step-response filtering + adding the noise + rounding to int16
  hist        C++: dwell-time histograms
  call        ctypes overhead and copies around the C++ call
  x / y       Python: log10 of the histograms, building the targets

--two-look profiles the samples of train_selfcal_online.py instead: simulate_xy(trace="both"), i.e. the
trace plus the network's fixed-ladder histograms built by the simulator (the run's sim_config gets that
ladder as its analysis block), noise drawn from the training recordings of --holdout-noise.

Usage (inside the container, library built with scripts/simulation/build.sh):
  python3 -m scripts.simulation.profile_simulate_xy --sim-config configs/sim/COCO.json --workers 1 62
  python3 -m scripts.simulation.profile_simulate_xy --sim-config configs/sim/COCO_noise5.json --two-look \
      --holdout-noise 2026_02_20 --predict-conditions --workers 1 62
"""
import json
import argparse
import multiprocessing as mp
import os
import time

import numpy as np

from scripts.common.canonicalize import choose_canonicalizer, extract_chain_rates_from_Q
from scripts.hist_pipeline.train_hist_online import STREAM_TRAIN, condition_norm, simulate_xy
from scripts.simulation.noise import NOISE_I16_STD, SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config

STAGES = ["draw", "noise.phases", "noise.spectrum", "noise.fft", "noise.finish",
          "gillespie", "synth", "hist", "call", "x", "y"]


def synthesize_timed(synth: SpectralNoise, rng, t, source=0):
    """SpectralNoise.synthesize() step by step (keep in sync with scripts/simulation/noise.py)."""
    c = time.perf_counter
    t0 = c()
    phi, Y = synth._phi, synth._Y
    rng.random(out=phi, dtype=np.float32)
    phi *= np.float32(2.0 * np.pi)
    t1 = c()
    np.cos(phi, out=Y.real)
    np.sin(phi, out=Y.imag)
    Y *= synth.mag[source]
    Y.imag[-1] = 0.0
    t2 = c()
    np.fft.irfft(Y, n=synth.nfft, out=synth._x)
    t3 = c()
    noise = synth._x[: synth.trace_len]
    noise -= noise.mean()
    noise *= np.float32(NOISE_I16_STD / max(float(noise.std()), 1e-30))
    np.clip(noise, -32767.0, 32767.0, out=noise)
    np.rint(noise, out=noise)
    np.copyto(synth._q, noise, casting="unsafe")
    t4 = c()
    t["noise.phases"] += t1 - t0
    t["noise.spectrum"] += t2 - t1
    t["noise.fft"] += t3 - t2
    t["noise.finish"] += t4 - t3
    return synth._q, NOISE_I16_STD


def simulate_xy_timed(sim, sim_cfg, canon, seed, stream, i, synth, cond, t, two_look=False):
    """simulate_xy() with a timer around every step. Returns (x, y, events in the trace)."""
    c = time.perf_counter
    t0 = c()
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, i]))
    Q = sim_cfg.sample_Q(rng)
    sim_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    noise_index = int(rng.integers(0, 2**32, dtype=np.uint64))
    levels, sigma, aux = sim_cfg.sample_levels_sigma(rng)
    k = 0
    if synth is not None:                       # same draw as simulate_xy
        src = sim_cfg.noise_sources(test=False)
        k = src[int(rng.integers(len(src)))] if len(src) > 1 else src[0]
    t1 = c()
    t["draw"] += t1 - t0
    noise = synthesize_timed(synth, rng, t, k) if synth is not None else None
    t2 = c()
    if two_look:
        trace, grids, n_events = sim.simulate_trace_and_hist(Q, levels, seed=sim_seed, noise_index=noise_index,
                                                             sigma=sigma, noise=noise)
    else:
        grids, n_events = sim.simulate_hist(Q, levels, seed=sim_seed, noise_index=noise_index, sigma=sigma,
                                            noise=noise)
    t3 = c()
    g, s, h = sim.last_timings()
    t["gillespie"] += g
    t["synth"] += s
    t["hist"] += h
    t["call"] += (t3 - t2) - (g + s + h)
    x = np.log10(grids + 1.0).astype(np.float32)
    if two_look:
        x = (trace, x)
    t4 = c()
    y = extract_chain_rates_from_Q(Q).astype(np.float32)
    if canon is not None:
        y = canon(y)
    y = np.log10(np.maximum(y, 1e-9)).astype(np.float32)
    if cond is not None:
        z = (aux[cond["index"]] - np.asarray(cond["mean"])) / np.asarray(cond["std"])
        y = np.concatenate([y, z.astype(np.float32)])
    t5 = c()
    t["x"] += t4 - t3
    t["y"] += t5 - t4
    return x, y, float(Q.sum())


def _work(job):
    config_path, spectrum_path, cond, seed, wid, n, two_look = job
    sim = TraceSimulator(config_path)
    if sim.last_timings() is None:
        raise RuntimeError("libsimdwell.so has no stage timers; rebuild it with scripts/simulation/build.sh")
    spec = SampleSpec.from_simulator(sim)
    canon = choose_canonicalizer(spec.topology, "auto")
    synth = SpectralNoise(np.load(spectrum_path), sim.trace_len) if spectrum_path else None

    # same result as the real function? (also warms up the FFT plan and the buffers)
    i0 = wid * 1_000_000
    x_ref, y_ref, _ = simulate_xy(sim, spec, canon, seed, STREAM_TRAIN, i0, synth, cond, "both" if two_look else False)
    x_ref = tuple(np.copy(a) for a in x_ref) if two_look else x_ref
    x_chk, y_chk, _ = simulate_xy_timed(sim, spec, canon, seed, STREAM_TRAIN, i0, synth, cond,
                                        dict.fromkeys(STAGES, 0.0), two_look)
    pairs = zip(x_ref, x_chk) if two_look else [(x_ref, x_chk)]
    same = bool(all(np.array_equal(a, b) for a, b in pairs) and np.array_equal(y_ref, y_chk))

    per_sample = np.zeros((n, len(STAGES)))
    rate_sum = np.zeros(n)
    t_wall = time.perf_counter()
    for k in range(n):
        t = dict.fromkeys(STAGES, 0.0)
        _, _, rate_sum[k] = simulate_xy_timed(sim, spec, canon, seed, STREAM_TRAIN, i0 + 1 + k, synth, cond, t,
                                              two_look)
        per_sample[k] = [t[s] for s in STAGES]
    return same, per_sample, rate_sum, time.perf_counter() - t_wall


def report(w, results):
    per = np.concatenate([r[1] for r in results]) * 1e3            # ms, (samples, stages)
    rate_sum = np.concatenate([r[2] for r in results])
    total = per.sum(axis=1)
    wall = np.mean([r[3] for r in results])
    n = len(total)
    print(f"\n=== {w} worker(s) in parallel, {n} samples | {w * (n / w) / wall:.1f} samples/s "
          f"({(n / w) / wall:.2f} per worker) | profiled path == simulate_xy: {all(r[0] for r in results)}")
    print(f"{'stage':<16} {'mean ms':>9} {'share':>7} {'median ms':>10} {'95% ms':>9}")
    groups = [("draw", ["draw"]),
              ("noise (total)", [s for s in STAGES if s.startswith("noise.")]),
              ("  phases", ["noise.phases"]), ("  spectrum", ["noise.spectrum"]),
              ("  inverse FFT", ["noise.fft"]), ("  finish", ["noise.finish"]),
              ("C++ (total)", ["gillespie", "synth", "hist", "call"]),
              ("  gillespie", ["gillespie"]), ("  synth", ["synth"]), ("  hist", ["hist"]), ("  call", ["call"]),
              ("x  log10", ["x"]), ("y  targets", ["y"])]
    for name, members in groups:
        v = per[:, [STAGES.index(m) for m in members]].sum(axis=1)
        print(f"{name:<16} {v.mean():>9.2f} {100 * v.mean() / total.mean():>6.1f}% {np.median(v):>10.2f} {np.percentile(v, 95):>9.2f}")
    print(f"{'TOTAL':<16} {total.mean():>9.2f} {'100.0%':>7} {np.median(total):>10.2f} {np.percentile(total, 95):>9.2f}")

    # how unevenly is the cost spread over samples? (fast rates = many events = slow samples)
    order = np.argsort(rate_sum)
    print("by sum of the sample's rates (slowest gating -> fastest), mean ms per sample:")
    print(f"  {'quarter':<10} {'sum of rates (Hz)':>22} {'gillespie':>10} {'synth':>8} {'hist':>8} {'total':>8} {'share of all time':>18}")
    for q, idx in enumerate(np.array_split(order, 4)):
        g, s, h = (per[idx, STAGES.index(k)].mean() for k in ("gillespie", "synth", "hist"))
        print(f"  {q + 1:<10} {rate_sum[idx].min():>10.3g} - {rate_sum[idx].max():<9.3g} {g:>10.2f} {s:>8.2f} {h:>8.2f} "
              f"{total[idx].mean():>8.2f} {100 * total[idx].sum() / total.sum():>17.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[1])
    p.add_argument("--n", type=int, default=200, help="samples per worker")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--predict-conditions", action="store_true", help="as in training: extra targets")
    p.add_argument("--tmp-dir", default=os.environ.get("TMPDIR", "/tmp"))
    p.add_argument("--two-look", action="store_true", help="profile the samples of train_selfcal_online.py")
    p.add_argument("--holdout-noise", default=None)
    p.add_argument("--fixed-ladder", type=float, nargs=4, default=[-1.67, 21.67, 24, 12],
                   metavar=("LO_PA", "HI_PA", "K", "J"), help="--two-look: the network's fixed ladder")
    args = p.parse_args()

    config_path = prepare_sim_config(args.sim_config, os.path.join(args.tmp_dir, f"profile_sim_config_{os.getpid()}.json"),
                                     args.seed, holdout_noise=args.holdout_noise)
    if args.two_look:
        lo, hi, k, j = args.fixed_ladder
        with open(config_path) as f:
            jcfg = json.load(f)
        jcfg["analysis"] = {"hist_backend": "grind", "hist2d": {"tmin_s": 1e-5, "tmax_s": 0.1, "bins": int(j)},
                            "grind": {"min_pA": lo, "max_pA": hi, "count": int(k)}}
        with open(config_path, "w") as f:
            json.dump(jcfg, f)
    probe = TraceSimulator(config_path)
    spec = SampleSpec.from_simulator(probe)
    trace_len = probe.trace_len
    probe.close()
    cond = condition_norm(spec) if args.predict_conditions else None

    spectrum_path = ""
    if spec.noise_mode == "spectral_resample":
        spectrum_path = os.path.join(args.tmp_dir, f"profile_spectrum_{os.getpid()}.npy")
        np.save(spectrum_path, source_spectra(spec.all_noise_paths, trace_len, spec.noise_avg_seg,
                                              spec.noise_avg_overlap, spec.noise_generate_factor))

    print(f"config={args.sim_config} topology={spec.topology} noise={spec.noise_mode}"
          + (f" generate_factor={spec.noise_generate_factor}" if spectrum_path else "")
          + f" trace_len={trace_len} cpus available={len(os.sched_getaffinity(0))}"
          + (" | two-look samples: trace + fixed-ladder histograms" if args.two_look else ""))
    ctx = mp.get_context("spawn")
    for w in args.workers:
        with ctx.Pool(w) as pool:
            results = pool.map(_work, [(config_path, spectrum_path, cond, args.seed, k, args.n, args.two_look)
                                       for k in range(w)],
                               chunksize=1)
        report(w, results)


if __name__ == "__main__":
    main()
