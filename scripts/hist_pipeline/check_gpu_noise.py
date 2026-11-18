#!/usr/bin/env python3
# scripts/hist_pipeline/check_gpu_noise.py
"""
Checks of gpu_noise.GPUNoiseProducer against the CPU noise (scripts.simulation.noise.SpectralNoise):
level, mean, spectrum shape per recording, reproducibility, which recordings samples get, and the
hand-over to simulator processes. Runs on a GPU if there is one, otherwise on the CPU.
  python3 -m scripts.hist_pipeline.check_gpu_noise --tmp-dir <dir>
"""
import argparse
import os

import numpy as np
import scipy.signal
import torch

from scripts.hist_pipeline.gpu_noise import GPUNoiseProducer, release_noise, take_noise
from scripts.simulation.noise import NOISE_I16_STD, SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config


def band_shape(x, fs=1e5):
    f, p = scipy.signal.welch(np.asarray(x, dtype=np.float64), fs=fs, nperseg=8192, axis=-1)
    p = p.mean(axis=0) if p.ndim > 1 else p
    df = fs / 8192
    e = np.unique(np.round(np.geomspace(2 * df, fs / 2, 41) / df)) * df
    b = np.array([p[(f >= lo) & (f < hi)].mean() for lo, hi in zip(e[:-1], e[1:])])
    return b / b.sum()


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--tmp-dir", required=True)
    a.add_argument("--n", type=int, default=12, help="noises per recording for the spectrum check")
    args = a.parse_args()
    cfg = prepare_sim_config("configs/sim/COCO_noise5.json", os.path.join(args.tmp_dir, "sim_config.json"), 1,
                             holdout_noise="2026_02_20")
    sim = TraceSimulator(cfg)
    spec = SampleSpec.from_simulator(sim)
    N = sim.trace_len
    mag = source_spectra(spec.all_noise_paths, N, spec.noise_avg_seg, spec.noise_avg_overlap,
                         spec.noise_generate_factor)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    train_src = list(spec.noise_sources(test=False))
    prod = GPUNoiseProducer(mag, N, [dev], train_src, seed=1, stream=0, ring=64, chunk=16)
    prod._mag = {dev: torch.from_numpy(prod.mag).to(dev)}
    print(f"device {dev} | training recordings (rows) {train_src} of {len(spec.all_noise_paths)}")

    # 1. which recordings the samples get: only training rows, about evenly
    ks = np.array([prod.source_of(i) for i in range(4000)])
    print("recording rows over 4000 samples:", {int(k): int((ks == k).sum()) for k in np.unique(ks)},
          "| only training rows:", set(ks.tolist()) <= set(train_src))

    # 2. reproducible: the same samples twice -> identical noise
    x1, k1 = prod.make(list(range(16)), dev)
    x2, k2 = prod.make(list(range(16)), dev)
    print("same sample indices give identical noise:", bool(torch.equal(x1, x2) and torch.equal(k1, k2)))

    # 3. level and mean, and spectrum shape against the CPU version, per recording
    cpu = SpectralNoise(mag, N)
    rng = np.random.default_rng(0)
    for k in train_src:
        idx = [i for i in range(400) if prod.source_of(i) == k][:args.n]
        g, kk = prod.make(idx, dev)
        g = g.cpu().numpy().astype(np.float64)
        c = np.stack([cpu.synthesize(rng, k)[0].astype(np.float64) for _ in range(len(idx))])
        d = 10 * np.log10(band_shape(g) / band_shape(c))
        print(f"row {k} {os.path.basename(spec.all_noise_paths[k])[:24]:24s} n={len(idx)} | std/5000 "
              f"{g.std(axis=1).mean() / NOISE_I16_STD:.5f} (CPU {c.std(axis=1).mean() / NOISE_I16_STD:.5f}) | "
              f"|mean| {np.abs(g.mean(axis=1)).max():.3f} | GPU vs CPU spectrum shape: max {np.abs(d).max():.3f} dB, "
              f"rms {np.sqrt((d ** 2).mean()):.3f} dB | rows as requested: {set(kk.tolist()) == {k}}")

    # 4. hand-over through the shared ring, as the simulator processes do it
    prod.start()
    got = [take_noise(prod.handles(), i) for i in range(40)]
    # the producer always makes noise in the same chunks of 16 consecutive samples, so compare with that chunk
    chunk0 = prod.make(list(range(16)), dev)[0].cpu().numpy()
    ok = all(np.array_equal(g[0], chunk0[i]) and g[2] == prod.source_of(i) for i, g in enumerate(got[:16]))
    single = np.abs(got[3][0].astype(int) - prod.make([3], dev)[0][0].cpu().numpy().astype(int)).max()
    print(f"made alone instead of in its chunk of 16: largest difference {single} (in steps of the int16 scale "
          f"of {NOISE_I16_STD:.0f})")
    for g in got:
        release_noise(prod.handles(), g[3])
    more = []
    for i in range(40, 200):                         # as a simulator does: take, use, release; needs refills
        g = take_noise(prod.handles(), i, timeout_s=120)
        more.append(g[2] == prod.source_of(i))
        release_noise(prod.handles(), g[3])
    prod.close()
    print("hand-over: taken noise equals the noise for that sample index:", ok,
          "| ring refilled after release (160 more samples):", len(more) == 160 and all(more))


if __name__ == "__main__":
    main()
