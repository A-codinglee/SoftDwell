#!/usr/bin/env python3
# scripts/hist_pipeline/bench_gpu_hist.py
"""
GPU speed and exactness of grind_hist_fast and the one-pass CUDA kernel (gpu_hist_cuda.py) against the
reference grind_hist_gpu, on real simulated traces, for the network's fixed ladder and for per-trace ladders
like its second look; plus edge cases for the CUDA kernel (crossings at every sample, flat traces, short traces).
  python3 -m scripts.hist_pipeline.bench_gpu_hist --tmp-dir <dir>
"""
import argparse
import os
import time

import numpy as np
import torch

from scripts.hist_pipeline.gpu_hist_cuda import grind_hist_cuda
from scripts.hist_pipeline.gpu_hist_fast import grind_hist_fast
from scripts.scripts.hardthresh_pipeline.gpu_hist import grind_hist_gpu, make_dwell_lut, make_thresholds
from scripts.simulation.noise import SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config


def timed(fn, reps):
    torch.cuda.synchronize()
    fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / reps, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tmp-dir", required=True)
    p.add_argument("--rows", type=int, default=32, help="traces per call (32 = one GPU's share of a batch of 128)")
    p.add_argument("--reps", type=int, default=10)
    a = p.parse_args()
    cfg = prepare_sim_config("configs/sim/COCO_noise5.json", os.path.join(a.tmp_dir, "sim_config.json"), 1)
    sim = TraceSimulator(cfg)
    spec = SampleSpec.from_simulator(sim)
    c, N = spec.counts_per_pA, sim.trace_len
    synth = SpectralNoise(source_spectra(spec.all_noise_paths, N, spec.noise_avg_seg, spec.noise_avg_overlap,
                                         spec.noise_generate_factor), N)
    traces, levels = [], []
    for i in range(a.rows):
        rng = np.random.default_rng(np.random.SeedSequence([1, 0, i]))
        Q = spec.sample_Q(rng)
        lv, sg, _ = spec.sample_levels_sigma(rng)
        nz = synth.synthesize(rng, i % len(spec.all_noise_paths))
        traces.append(sim.simulate_trace(Q, lv, seed=i, sigma=sg, noise=(nz[0].copy(), nz[1])))
        levels.append(lv)
    dev = torch.device("cuda:0")
    x = torch.from_numpy(np.stack(traces)).to(dev)
    lut = torch.from_numpy(make_dwell_lut(N, 1e5, 1e-5, 0.1, 12)).to(dev)
    fixed = torch.from_numpy(make_thresholds(-1.67 * c, 21.67 * c, 24)).to(dev)
    lo = torch.tensor([min(l) / c - 1 for l in levels]); hi = torch.tensor([max(l) / c + 1 for l in levels])
    t = torch.linspace(0, 1, 24)[None]
    percall = torch.round((lo[:, None] + (hi - lo)[:, None] * t) * c).to(torch.int16).to(dev)
    print(f"{a.rows} real traces of {N} points on {torch.cuda.get_device_name(0)}")
    for name, thr in (("fixed ladder", fixed), ("per-trace ladder (second look)", percall)):
        ref_ms, ref = timed(lambda: grind_hist_gpu(x, thr, lut, 12, k_group=8)[0], a.reps)
        print(f"--- {name}: reference {ref_ms:.1f} ms")
        for acc in ("scatter",):
            for kg in (24,):
                ms, out = timed(lambda: grind_hist_fast(x, thr, lut, 12, k_group=kg, accumulate=acc), a.reps)
                print(f"    fast, {acc:8s} k_group {kg:2d}: {ms:6.1f} ms  ({ref_ms / ms:.2f}x)  identical: {torch.equal(out, ref)}")
        ms, out = timed(lambda: grind_hist_cuda(x, thr, lut, 12), a.reps)
        print(f"    one-pass CUDA kernel:          {ms:6.1f} ms  ({ref_ms / ms:.2f}x)  identical: {torch.equal(out, ref)}")
        torch.cuda.reset_peak_memory_stats()
        grind_hist_fast(x, thr, lut, 12, k_group=24)
        print(f"    peak memory, fast k_group 24: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    # a full batch of 128 (the traces repeated 4x, ladders as for the second look)
    xb, tb = x.repeat(128 // a.rows, 1), percall.repeat(128 // a.rows, 1)
    f_ms, ref = timed(lambda: torch.cat([grind_hist_fast(xb[s:s + 32], tb[s:s + 32], lut, 12, k_group=24, accumulate="scatter")
                                      for s in range(0, 128, 32)]), a.reps)
    c_ms, out = timed(lambda: grind_hist_cuda(xb, tb, lut, 12), a.reps)
    print(f"--- batch of 128, second-look ladders: fast (as in training, 4 calls of 32) {f_ms:.1f} ms, "
          f"CUDA {c_ms:.1f} ms ({f_ms / c_ms:.1f}x)  identical: {torch.equal(out, ref)}")
    edge_cases(dev, lut, c)


def edge_cases(dev, lut, c):
    """The CUDA kernel against grind_hist_fast on traces that stress the chunk joins."""
    g = torch.Generator().manual_seed(0)
    N = lut.numel() - 1
    cases = {
        "crossing at every sample": (torch.arange(N) % 2 * 200 - 100).to(torch.int16).repeat(4, 1),
        "random +-1 (crossings almost everywhere)": (torch.randint(0, 2, (4, N), generator=g) * 2 - 1).to(torch.int16),
        "flat": torch.zeros((4, N), dtype=torch.int16),
        "few long dwells": ((torch.arange(N) // 77777) % 2 * 500).to(torch.int16).repeat(4, 1),
        "sparse random steps": torch.cumsum(torch.randint(-1, 2, (4, N), generator=g) *
                                            (torch.rand(4, N, generator=g) < 1e-4), 1).to(torch.int16) * 50,
    }
    thr = torch.tensor([-200, -1, 0, 1, 25, 49, 50, 99, 100, 101, 30000] * 2 + [5, 7], dtype=torch.int16)
    ok = True
    for name, xs in cases.items():
        xs = xs.to(dev)
        for th in (thr.to(dev), thr.to(dev).repeat(xs.shape[0], 1)):
            same = torch.equal(grind_hist_cuda(xs, th, lut, 12), grind_hist_fast(xs, th, lut, 12, k_group=24))
            ok &= same
            if not same:
                break
        print(f"    edge case {name:42s} identical: {same}")
    short = torch.randint(-3, 4, (3, 4096), generator=g).to(torch.int16).to(dev)     # one chunk per warp or less
    lut_s = lut[:4097]
    same = torch.equal(grind_hist_cuda(short, thr.to(dev), lut_s, 12), grind_hist_fast(short, thr.to(dev), lut_s, 12))
    print(f"    edge case {'short traces (4096 points)':42s} identical: {same}")
    print("--- all edge cases identical:", bool(ok and same))


if __name__ == "__main__":
    main()
