#!/usr/bin/env python3
# scripts/hist_pipeline/profile_selfcal.py
"""
Profile the two-look network of train_selfcal_online.py (no real training run):

  1. CPU, per sample: simulating a trace, and what building its fixed-ladder histograms on the CPU
     would add (the simulator's own histogram code, same ladder as the network's first look).
  2. GPU, per training step and per validation step: time of every stage, for 1, 2, ... GPUs building
     the histograms, plus peak GPU memory and the epoch time this implies.
  3. The GPU operations that dominate one histogram call (torch.profiler).

Uses real simulated traces from the training stream, the same settings as train_selfcal_online.py
(its options can be passed after --, e.g. -- --batch-size 64).

Usage:
  python3 -m scripts.hist_pipeline.profile_selfcal --sim-config configs/sim/COCO_noise5.json --tmp-dir <dir>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline import train_selfcal_online as T
from scripts.hist_pipeline.train_hist_online import STREAM_TRAIN, _pool_init, condition_norm, simulate_fixed_set, write_noise_spectra
from scripts.scripts.softdwell_pipeline.model import HardSelfCalibratingFusionNet
from scripts.simulation.noise import SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.metrics import LogCoshLoss


def sync_all():
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)


class Timer:
    def __init__(self):
        self.t = {}

    def run(self, name, fn):
        sync_all()
        t0 = time.perf_counter()
        out = fn()
        sync_all()
        self.t.setdefault(name, []).append(1000 * (time.perf_counter() - t0))
        return out


def table(timer, title, skip=3):
    rows = {k: np.array(v[skip:]) for k, v in timer.t.items()}
    total = sum(r.mean() for r in rows.values())
    print(f"--- {title}\n{'stage':34s}{'mean ms':>10s}{'share':>9s}{'median':>9s}{'max':>9s}")
    for k, r in rows.items():
        print(f"{k:34s}{r.mean():10.1f}{100 * r.mean() / total:8.1f}%{np.median(r):9.1f}{r.max():9.1f}")
    print(f"{'TOTAL':34s}{total:10.1f}")
    return total


def cpu_costs(args, cfg_path, n):
    """ms per sample: simulate_trace alone vs simulate_hist with the network's fixed ladder."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    a = args
    cfg["analysis"] = {"hist_backend": "grind",
                       "hist2d": {"tmin_s": T.DWELL_MIN_S, "tmax_s": T.DWELL_MAX_S, "bins": a.fixed_j},
                       "grind": {"min_pA": a.fixed_thr_pA[0], "max_pA": a.fixed_thr_pA[1], "count": a.fixed_k}}
    p2 = cfg_path.replace(".json", "_fixedladder.json")
    with open(p2, "w") as f:
        json.dump(cfg, f)
    sim = TraceSimulator(p2)
    spec = SampleSpec.from_simulator(sim)
    synth = SpectralNoise(source_spectra(spec.all_noise_paths, sim.trace_len, spec.noise_avg_seg,
                                         spec.noise_avg_overlap, spec.noise_generate_factor), sim.trace_len)
    t_tr, t_hi = [], []
    for i in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([1, STREAM_TRAIN, i]))
        Q = spec.sample_Q(rng)
        levels, sigma, _ = spec.sample_levels_sigma(rng)
        noise = synth.synthesize(rng, int(rng.integers(len(spec.noise_paths))))
        noise = (noise[0].copy(), noise[1])
        t0 = time.perf_counter(); sim.simulate_trace(Q, levels, seed=i, sigma=sigma, noise=noise)
        t_tr.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); sim.simulate_hist(Q, levels, seed=i, sigma=sigma, noise=noise)
        t_hi.append(time.perf_counter() - t0)
    sim.close()
    tr, hi = 1000 * np.mean(t_tr), 1000 * np.mean(t_hi)
    print(f"--- CPU, one core, {n} samples (noise synthesis not included)\n"
          f"simulate trace only:                        {tr:7.1f} ms\n"
          f"simulate trace + fixed-ladder histograms:    {hi:7.1f} ms  -> histograms add {hi - tr:.1f} ms per sample")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", default="configs/sim/COCO_noise5.json")
    p.add_argument("--holdout-noise", default="2026_02_20")
    p.add_argument("--tmp-dir", required=True)
    p.add_argument("--n-traces", type=int, default=512, help="real training traces simulated for the GPU part")
    p.add_argument("--steps", type=int, default=23, help="steps per setting (the first 3 are not counted)")
    p.add_argument("--cpu-samples", type=int, default=40)
    p.add_argument("--gpu-counts", type=int, nargs="+", default=None, help="default: 1 .. all GPUs")
    own, rest = p.parse_known_args()
    rest = [r for r in rest if r != "--"]
    sys.argv = [sys.argv[0], "--sim-config", own.sim_config, "--output-root", own.tmp_dir,
                "--holdout-noise", own.holdout_noise] + rest
    args = T.parse_args()
    os.makedirs(own.tmp_dir, exist_ok=True)

    cfg_path = prepare_sim_config(own.sim_config, os.path.join(own.tmp_dir, "sim_config.json"), args.seed,
                                  holdout_noise=own.holdout_noise)
    probe = TraceSimulator(cfg_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c, fs = probe.trace_len, sim_cfg.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    cond = condition_norm(sim_cfg)
    cond["std"] = [s / args.target_scale for s in cond["std"]]
    n_rates, y_dim = 2 * (sim_cfg.num_states - 1), 2 * (sim_cfg.num_states - 1) + 3
    n_gpu = torch.cuda.device_count()
    print(f"node: {os.uname().nodename} | GPUs: {n_gpu} | CPUs: {len(os.sched_getaffinity(0))} | "
          f"batch {args.batch_size} | backbone {args.backbone} | hist_rows {args.hist_rows} k_group {args.k_group}",
          flush=True)

    cpu_costs(args, cfg_path, own.cpu_samples)

    spectrum_path = write_noise_spectra(sim_cfg, trace_len, own.tmp_dir)
    t0 = time.time()
    with mp.get_context("spawn").Pool(max(1, len(os.sched_getaffinity(0)) - 1), initializer=_pool_init,
                                      initargs=(cfg_path, sim_cfg, args.seed, spectrum_path, cond, True)) as pool:
        X, Y, _, _ = simulate_fixed_set(pool, STREAM_TRAIN, own.n_traces, (trace_len,), y_dim, torch.int16)
    print(f"--- simulated {own.n_traces} real training traces in {time.time() - t0:.1f}s", flush=True)
    if n_gpu == 0:
        print("no GPU: stopping after the CPU part")
        return

    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    lvl_mean = torch.tensor(cond["mean"][:2], device=device, dtype=torch.float32)
    lvl_std = torch.tensor(cond["std"][:2], device=device, dtype=torch.float32)
    B = args.batch_size
    summary = []
    for g in (own.gpu_counts or range(1, n_gpu + 1)):
        args.hist_gpus = g
        hists = T.Hists(args, trace_len, fs, c, device)
        core = HardSelfCalibratingFusionNet(args.fixed_k, args.fixed_j, args.selfcal_k, args.selfcal_j,
                                            num_outputs=y_dim, base_scale_pA=args.base_scale_pA,
                                            amp_min_pA=args.amp_min_pA, backbone=args.backbone,
                                            convnext_hidden=args.hidden, convnext_depth=args.depth,
                                            convnext_kernel_size=args.kernel_size).to(device)
        opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        crit, level_crit = LogCoshLoss(), LogCoshLoss()
        for i in range(n_gpu):
            torch.cuda.reset_peak_memory_stats(i)

        tm = Timer()
        core.train()
        for s in range(own.steps):
            a = (s * B) % (own.n_traces - B + 1)
            # as in training with buffer % batch == 0: a view into the buffer, copied straight to the GPUs
            xb = tm.run("take batch (view, no copy)", lambda: X[a:a + B])
            yb = Y[a:a + B].to(device)
            parts = tm.run("traces to GPU(s)", lambda: hists.load(xb))
            fixed = tm.run("fixed-ladder histograms", lambda: hists.fixed(parts))
            f_fixed, base_hat, amp_hat = tm.run("1st backbone + level estimate", lambda: core.fixed_stage(fixed))
            selfcal = tm.run("self-calibrated histograms",
                             lambda: hists.selfcal(parts, base_hat.detach(), amp_hat.detach()))
            loss = tm.run("2nd backbone + head + loss", lambda: (
                crit(core.finish(f_fixed, selfcal, base_hat, amp_hat), yb)
                + args.level_aux_weight * level_crit(torch.stack([base_hat, amp_hat], 1),
                                                     (yb[:, n_rates:n_rates + 2] * lvl_std + lvl_mean) / c)))
            opt.zero_grad(set_to_none=True)
            tm.run("backward", lambda: loss.backward())
            tm.run("clip + optimizer step", lambda: (torch.nn.utils.clip_grad_norm_(core.parameters(),
                                                                                    args.clip_grad_norm), opt.step()))
        step_ms = table(tm, f"training step, batch {B}, histograms on {g} GPU(s)")

        te = Timer()
        core.eval()
        with torch.no_grad():
            for s in range(own.steps):
                a = (s * B) % (own.n_traces - B + 1)
                xb = X[a:a + B]
                fixed = hists.fixed(hists.load(xb))           # precomputed once in training, not timed
                parts = te.run("traces to GPU(s)", lambda: hists.load(xb))
                f_fixed, base_hat, amp_hat = te.run("1st backbone + level estimate", lambda: core.fixed_stage(fixed))
                selfcal = te.run("self-calibrated histograms", lambda: hists.selfcal(parts, base_hat, amp_hat))
                te.run("2nd backbone + head", lambda: core.finish(f_fixed, selfcal, base_hat, amp_hat))
        eval_ms = table(te, f"validation step, batch {B}, histograms on {g} GPU(s)")
        mem = [torch.cuda.max_memory_allocated(i) / 2 ** 30 for i in range(g)]
        print(f"peak GPU memory (GiB): " + " ".join(f"{m:.1f}" for m in mem), flush=True)
        ep = 156 * step_ms / 1000 + int(np.ceil(10000 / B)) * eval_ms / 1000
        summary.append((g, step_ms, eval_ms, ep))
        hists.pool.shutdown()
        del core, opt, hists
        torch.cuda.empty_cache()

    print("\n=== summary: epoch = 156 training steps + 10,000 validation samples (data supply not included)")
    print(f"{'GPUs for histograms':>20s}{'train step ms':>15s}{'val step ms':>13s}{'epoch s':>10s}{'speed-up':>10s}")
    for g, st, ev, ep in summary:
        print(f"{g:>20d}{st:15.1f}{ev:13.1f}{ep:10.1f}{summary[0][3] / ep:9.2f}x")

    # which GPU operations dominate one fixed-ladder histogram call
    args.hist_gpus = 1
    hists = T.Hists(args, trace_len, fs, c, device)
    parts = hists.load(X[:B])
    hists.fixed(parts)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        hists.fixed(parts)
        sync_all()
    print(f"\n--- GPU operations in one fixed-ladder histogram call ({B} traces, 1 GPU)")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=12, max_name_column_width=40))


if __name__ == "__main__":
    main()
