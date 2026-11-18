#!/usr/bin/env python3
# scripts/hist_pipeline/eval_hist_online.py
"""
Evaluate a finished train_hist_online run on a freshly simulated test set of any size.
No training: loads <run-dir>/best_model.pt and simulates with <run-dir>/sim_config.json.

The test samples come from the same random stream as the run's own test set, so the
first n are identical to it and a larger --n-test only adds new samples behind them.

Writes preds_test.npy, targs_test.npy and aux_test.npy to --out-dir
(default <run-dir>/eval_test<N>), ready for scripts.plotting.plot_online_results --eval-dir.
For a run trained with --predict-conditions also preds_aux_test.npy.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline.train_hist_online import (STREAM_TEST, _pool_init, evaluate, simulate_fixed_set,
                                                     rae_by_noise_file, split_outputs, write_noise_spectra)
from scripts.simulation.simulator import SampleSpec, TraceSimulator
from scripts.softdwell_pipeline.metrics import LogCoshLoss
from scripts.softdwell_pipeline.model import TFExactInceptionResNet


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="e.g. outputs/hist_train_online/COCO/seed1")
    p.add_argument("--n-test", type=int, default=10000)
    p.add_argument("--out-dir", default=None, help="default: <run-dir>/eval_test<N>")
    p.add_argument("--workers", type=int, default=0, help="simulator processes (0 = all CPUs but one)")
    p.add_argument("--batch-size", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or os.path.join(args.run_dir, f"eval_test{args.n_test}")
    os.makedirs(out_dir, exist_ok=True)

    config_path = os.path.join(args.run_dir, "sim_config.json")
    best_path = os.path.join(args.run_dir, "best_model.pt")
    with open(config_path) as f:
        seed = int(json.load(f)["seeding"]["seed"])     # the run's seed

    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    x_shape = (probe.hist_detectors, probe.hist_bins, probe.hist_bins)
    trace_len = probe.trace_len
    probe.close()
    cond = None                                         # set if the run was trained with --predict-conditions
    cond_path = os.path.join(args.run_dir, "conditions.json")
    if os.path.isfile(cond_path):
        with open(cond_path) as f:
            cond = json.load(f)
    n_rates = 2 * (sim_cfg.num_states - 1)
    y_dim = n_rates + (len(cond["index"]) if cond else 0)

    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.run_dir, reuse=True)

    workers = args.workers if args.workers > 0 else max(1, len(os.sched_getaffinity(0)) - 1)
    print(f"[eval] run={args.run_dir} topology={sim_cfg.topology} seed={seed} noise={sim_cfg.noise_mode}\n"
          f"[eval] simulating {args.n_test} test samples with {workers} workers...", flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_pool_init,
                                      initargs=(config_path, sim_cfg, seed, spectrum_path, cond)) as pool:
        X_test, y_test, aux_test, noise_src = simulate_fixed_set(pool, STREAM_TEST, args.n_test, x_shape, y_dim)
    dt = time.time() - t0
    print(f"[eval] simulated in {dt:.1f}s ({args.n_test / max(dt, 1e-9):.1f} samples/s)", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TFExactInceptionResNet(
        in_channels=x_shape[0],
        num_outputs=y_dim,
        use_bn=False,
        separate_heads=False,
        input_hw=x_shape[1:],
    ).to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    _, rae_sum, count, preds = evaluate(
        model, X_test, y_test, torch.arange(args.n_test), args.batch_size, device, LogCoshLoss(), y_dim)

    preds_hz, targs_hz, preds_aux = split_outputs(
        torch.cat(preds).numpy().astype(np.float64), y_test.numpy().astype(np.float64), cond)
    np.save(os.path.join(out_dir, "preds_test.npy"), preds_hz)
    np.save(os.path.join(out_dir, "targs_test.npy"), targs_hz)
    np.save(os.path.join(out_dir, "aux_test.npy"), aux_test)
    np.save(os.path.join(out_dir, "noise_source_test.npy"), noise_src)
    written = "preds_test.npy, targs_test.npy, aux_test.npy, noise_source_test.npy"

    rae_per = (rae_sum / count.clamp_min(1).float()).cpu().tolist()
    print(f"[eval] device={device} | RAE per out: {' '.join(f'{r:.4f}' for r in rae_per)} | "
          f"Overall: {float(np.mean(rae_per[:n_rates])):.4f}", flush=True)
    rae_by_noise_file(preds_hz, targs_hz, noise_src, sim_cfg, out_dir, "eval")
    if cond:
        np.save(os.path.join(out_dir, "preds_aux_test.npy"), preds_aux)
        written += ", preds_aux_test.npy"
        mae = np.abs(preds_aux - aux_test)[:, cond["index"]].mean(axis=0) / cond["counts_per_unit"]
        print(f"[eval] Mean abs error ({cond['unit']}): "
              + " | ".join(f"{nm} {e:.4g}" for nm, e in zip(cond["names"], mae)), flush=True)
    print(f"[eval] wrote {written} to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
