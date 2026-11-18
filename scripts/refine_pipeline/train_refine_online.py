#!/usr/bin/env python3
# scripts/refine_pipeline/train_refine_online.py
"""
Second-stage refinement by re-simulation (RAFT-style), on top of a network trained by
scripts/hist_pipeline/train_selfcal_online.py (HardSelfCalibratingFusionNet), which stays frozen.

  theta_0 = stage-1 estimate (log10 rates + standardized baseline / amplitude / sigma)
  loop k = 0 .. N-1:
     simulate a trace at theta_k (CPU pool, scripts/refine_pipeline/resim.py)
     histograms of the simulated trace, on the SAME ladders as the observed one (stage-1's fixed ladder and its
     self-calibrated ladder, placed once from stage-1's level estimate)
     theta_{k+1} = theta_k + SimRefineNet([H_obs, H_sim, H_obs - H_sim], theta_k, GRU state)
  loss = sum_k gamma^(N-k) LogCosh(theta_k, y) / sum_k gamma^(N-k)               (RAFT's sequence loss)

Training is closed-loop: every iteration of every training sample re-simulates at the refiner's own current
estimate, so the refiner sees the errors it will see at inference. That costs N simulations per sample on top of
the observed trace, so this trainer is bound by the CPUs (scripts/refine_pipeline/resim.py prints the split).

--finetune-stage1 runs two phases: A, stage 1 frozen, until the refiner stalls (or --stage1-freeze-epochs); then B,
the best phase-A refiner restored (and kept as best_refiner_phaseA.pt) and stage 1 trained as well: its own training loss
(LogCosh on theta_0 + level_aux_weight * LogCosh on the level estimate, as in train_selfcal_online.py) is added to
the sequence loss with --stage1-loss-weight, and the refiner's loss reaches it through the context features f (theta_0
itself enters the loop detached, and the histograms have no gradient). It has its own learning rate (--stage1-lr) and
its own gradient clip. Its outputs then change, so validation / test recompute them from the stored traces every time
(the original stage 1's test estimate is kept for comparison: preds_test_stage1_orig.npy); best_stage1.pt is saved
together with best_refiner.pt.

At test time the loop runs --test-iters iterations (more than in training, as RAFT does) plus one more
re-simulation for the last estimate. For each sample it records the estimate and the discrepancy
mean |H_obs - H_sim| (log10 histograms) of every iterate, and reports three ways to stop:
  fixed:      after k iterations (every k)
  converged:  at the first iteration that moves no rate by more than --tol (log10), otherwise the last
  best-fit:   the iterate whose re-simulation matches the observed histograms best (smallest discrepancy)

Data: same simulator and sim-config as the stage-1 run (its sim_config.json, so the same noise split), but a
different seed (default stage-1 seed + 100), so neither training nor test samples were seen by stage 1.

Writes into <output-root>/seed<seed>/: best_refiner.pt, train_config.json, refine_metrics.csv,
preds_test.npy / preds_aux_test.npy (after --iters iterations, the count trained for), preds_test_stage1.npy,
targs_test.npy, aux_test.npy, noise_source_test.npy, conditions.json (so plot_online_results.py works),
refine_test.npz (every iterate and its discrepancy) and refine_test_summary.json.

Usage:
  python3 -m scripts.refine_pipeline.train_refine_online \
      --stage1-run <selfcal output>/seed1 --output-root <dir> [--iters 4 --test-iters 8]
"""
import argparse
import csv
import json
import os
import shutil
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline import train_selfcal_online as tso
from scripts.hist_pipeline.predict_selfcal import run_args
from scripts.hist_pipeline.train_hist_online import (LR_FACTOR, LR_PATIENCE, MIN_LR, STREAM_TEST, STREAM_VAL,
                                                     _pool_init, generator_loop, rae_by_noise_file,
                                                     simulate_fixed_set, split_outputs, take_batch,
                                                     write_noise_spectra)
from scripts.refine_pipeline.model import SimRefineNet
from scripts.refine_pipeline.resim import STREAM_RESIM, Resimulator, ThetaToSim
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.metrics import LogCoshLoss, rae_log


class Stage1:
    """The stage-1 network (frozen, or fine-tuned with --finetune-stage1) and its two histogram ladders."""

    def __init__(self, run_dir, trace_len, fs, counts_per_pA, device, hist_impl=None):
        a, tc = run_args(run_dir)
        if tc["model"]["name"] != "HardSelfCalibratingFusionNet":
            raise SystemExit(f"{run_dir}: stage 1 must be a train_selfcal_online.py run, got {tc['model']['name']}")
        if hist_impl or device.type != "cuda":
            a.hist_impl = hist_impl or "fast"
        self.args, self.tc = a, tc
        self.y_dim, self.n_rates = tc["model"]["num_outputs"], tc["model"]["num_rates"]
        self.hists = tso.Hists(a, trace_len, fs, counts_per_pA, device)
        self.core = tso.build_core(a, self.y_dim, device)
        self.core.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
        self.core.eval().requires_grad_(False)
        if a.fixed_j != a.selfcal_j:
            raise SystemExit(f"the refiner stacks both ladders: needs fixed_j == selfcal_j, got {a.fixed_j}, "
                             f"{a.selfcal_j}")
        self.hist_channels = a.fixed_k + a.selfcal_k
        self.hist_hw = (a.fixed_j, a.fixed_j)
        self.ctx_dim = self.core.fixed_backbone.out_features + self.core.selfcal_backbone.out_features + 2
        # stage 1's own loss (for fine-tuning): LogCosh on its outputs + level_aux_weight * LogCosh on the levels
        cond = json.load(open(os.path.join(run_dir, "conditions.json")))
        self.lvl_mean = torch.tensor(cond["mean"][:2], device=device, dtype=torch.float32)
        self.lvl_std = torch.tensor(cond["std"][:2], device=device, dtype=torch.float32)
        self.counts_per_pA = float(counts_per_pA)
        self.level_aux_weight = float(tc.get("level_aux_weight", 0.3))

    def set_trainable(self, on: bool):
        self.core.train(on).requires_grad_(on)

    def __call__(self, x, grad=False):
        """
        x (B, T) int16 on the CPU -> theta_0, context features, baseline / amplitude estimate (pA), H_obs.
        grad=True keeps the graph (fine-tuning); the histograms never have one, and the self-calibrated ladder is
        placed from the detached level estimate, exactly as in stage-1 training.
        """
        core = self.core
        with torch.set_grad_enabled(grad):
            parts = self.hists.load(x)
            h_fixed = self.hists.fixed(parts)
            f_fixed, base, amp = core.fixed_stage(h_fixed)
            h_sc = self.hists.selfcal(parts, base.detach(), amp.detach())
            f = torch.cat([f_fixed, core.selfcal_backbone(h_sc), torch.stack([base, amp], dim=1)], dim=1)
            theta0 = torch.cat([h(f) for h in core.heads], dim=1) if core.separate_heads else core.head(f)
        return theta0.float(), f.float(), base, amp, torch.cat([h_fixed, h_sc], dim=1)

    def own_loss(self, crit, theta0, base, amp, y):
        """The stage-1 training loss (train_selfcal_online.py) on one batch."""
        level_true = (y[:, self.n_rates:self.n_rates + 2] * self.lvl_std + self.lvl_mean) / self.counts_per_pA
        return crit(theta0, y) + self.level_aux_weight * crit(torch.stack([base, amp], dim=1), level_true)

    @torch.no_grad()
    def sim_hists(self, x, base, amp):
        """Histograms of re-simulated traces on the observed traces' ladders."""
        parts = self.hists.load(x)
        return torch.cat([self.hists.fixed(parts), self.hists.selfcal(parts, base, amp)], dim=1)


def unroll(refiner, stage1, resim, theta0, ctx, base, amp, h_obs, iters, stream, idx, extra_sim=False):
    """
    The refinement loop. Returns (thetas [theta_0 .. theta_iters], discrepancies [d_0 .. ]), d_k = mean
    |H_obs - H_sim(theta_k)| per sample; with extra_sim also d_iters (one more re-simulation).
    """
    h, inp = refiner.init_state(ctx)
    theta, thetas, disc = theta0, [theta0], []
    for k in range(iters + int(extra_sim)):
        h_sim = stage1.sim_hists(resim(theta, stream, idx), base, amp)
        disc.append((h_obs - h_sim).abs().mean(dim=(1, 2, 3)).detach())
        if k == iters:
            break
        h, theta = refiner.step(h, inp, theta.detach(), h_obs, h_sim)
        thetas.append(theta)
    return thetas, disc


def sequence_loss(crit, thetas, y, gamma):
    """RAFT's weighted sum over the refined iterates (theta_0 is stage 1's and has no gradient), normalized."""
    n = len(thetas) - 1
    w = [gamma ** (n - k) for k in range(1, n + 1)]
    return sum(wk * crit(t, y) for wk, t in zip(w, thetas[1:])) / sum(w)


@torch.no_grad()
def precompute(stage1, X, y, batch_size):
    """Stage-1 outputs of a fixed set, kept on the CPU: the refiner's starting point for val / test."""
    out = {k: [] for k in ("theta0", "ctx", "base", "amp", "h_obs")}
    for s in range(0, X.shape[0], batch_size):
        for k, v in zip(out, stage1(X[s:s + batch_size])):
            out[k].append(v.cpu())
    out = {k: torch.cat(v) for k, v in out.items()}
    out["y"] = y
    return out


@torch.no_grad()
def evaluate(refiner, stage1, resim, data, iters, stream, batch_size, device, extra_sim=False):
    """
    Returns (trajectory (N, iters + 1, y_dim), discrepancies (N, iters + extra_sim)) on the CPU.
    data holds either stage 1's cached outputs (frozen stage 1) or the raw traces "x" (fine-tuned stage 1:
    its outputs are recomputed with its current weights).
    """
    refiner.eval()
    traj, disc = [], []
    n = data["y"].shape[0]
    keys = ("theta0", "ctx", "base", "amp", "h_obs")
    for s in range(0, n, batch_size):
        if "x" in data:
            g = dict(zip(keys, stage1(data["x"][s:s + batch_size])))
        else:
            g = {k: data[k][s:s + batch_size].to(device, non_blocking=True) for k in keys}
        idx = torch.arange(s, min(s + batch_size, n))
        th, d = unroll(refiner, stage1, resim, g["theta0"], g["ctx"], g["base"], g["amp"], g["h_obs"], iters,
                       stream, idx, extra_sim)
        traj.append(torch.stack(th, dim=1).float().cpu())
        disc.append(torch.stack(d, dim=1).float().cpu())
    return torch.cat(traj), torch.cat(disc)


def per_iter_metrics(traj, y, crit, n_rates):
    """[(loss, RAE per output, mean RAE over the rates)] for every iterate."""
    out = []
    for k in range(traj.shape[1]):
        r = rae_log(traj[:, k], y).mean(dim=0)
        out.append((float(crit(traj[:, k], y)), r.tolist(), float(r[:n_rates].mean())))
    return out


def stop_rules(traj, disc, tol, n_rates):
    """Per sample, the iterate picked by the 'converged' and the 'best-fit' rule (see the module doc)."""
    N, K1, _ = traj.shape
    step = (traj[:, 1:, :n_rates] - traj[:, :-1, :n_rates]).abs().amax(dim=2)          # (N, K)
    small = step < tol
    k_conv = torch.where(small.any(dim=1), small.float().argmax(dim=1) + 1, torch.full((N,), K1 - 1))
    k_best = disc[:, :K1].argmin(dim=1)
    pick = lambda k: traj[torch.arange(N), k]
    return {"converged": (pick(k_conv), k_conv), "best-fit": (pick(k_best), k_best)}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-run", required=True,
                   help="seed folder of a train_selfcal_online.py run (best_model.pt, train_config.json, ...)")
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=None, help="data seed (default: the stage-1 run's seed + 100)")

    # refinement loop
    p.add_argument("--iters", type=int, default=4, help="refinement iterations in training (each one re-simulates)")
    p.add_argument("--test-iters", type=int, default=8, help="iterations at test time")
    p.add_argument("--gamma", type=float, default=0.8, help="sequence-loss weight decay (RAFT)")
    p.add_argument("--tol", type=float, default=0.02,
                   help="'converged' stopping rule: largest change of a log10 rate in one iteration")

    # refiner network
    p.add_argument("--hidden", type=int, default=128, help="GRU state size")
    p.add_argument("--enc-hidden", type=int, default=64, help="ConvNeXt width of the observed-vs-simulated encoder")
    p.add_argument("--enc-depth", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--max-step", type=float, default=1.0, help="soft limit of one update per theta component")

    # stage 1
    p.add_argument("--finetune-stage1", action="store_true",
                   help="train stage 1 too: its own loss (on theta_0 and the level estimate) is added to the "
                        "sequence loss, and the refiner's gradient reaches it through the context features")
    p.add_argument("--stage1-lr", type=float, default=3e-5,
                   help="stage-1 learning rate when fine-tuning (its own run ended at about 1.2e-4)")
    p.add_argument("--stage1-loss-weight", type=float, default=1.0, help="weight of stage 1's own loss")
    p.add_argument("--stage1-freeze-epochs", type=int, default=-1,
                   help="phase A (stage 1 frozen) length; -1 = until the refiner stalls (--unfreeze-patience) or half "
                        "of --max-hours is used")
    p.add_argument("--unfreeze-patience", type=int, default=10,
                   help="with --stage1-freeze-epochs -1: epochs without improvement that end phase A")

    # data / CPUs
    p.add_argument("--buffer-size", type=int, default=512, help="observed traces waiting for the trainer")
    p.add_argument("--gen-workers", type=int, default=0,
                   help="processes simulating observed traces (0 = (CPUs - 2) / (iters + 1))")
    p.add_argument("--resim-workers", type=int, default=0,
                   help="processes re-simulating at the current estimates (0 = the remaining CPUs - 2)")
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-test", type=int, default=5000)
    p.add_argument("--hist-impl", default=None, choices=["fast", "cuda", "reference"],
                   help="GPU histogram code (default: the stage-1 run's; 'fast' on a CPU)")

    # training
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps-per-epoch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=1.0)
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop once another epoch would pass this many hours since start-up (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()
    stage1_run = os.path.abspath(args.stage1_run)
    tc1 = json.load(open(os.path.join(stage1_run, "train_config.json")))
    seed = args.seed if args.seed is not None else int(tc1["seed"]) + 100
    if seed == int(tc1["seed"]):
        print(f"[Parent] WARNING: seed {seed} is the stage-1 run's: its training samples are stage 1's", flush=True)
    out = os.path.join(args.output_root, f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # the stage-1 run's sim-config (its noise split included), with this run's seed
    config_path = prepare_sim_config(os.path.join(stage1_run, "sim_config.json"), os.path.join(out, "sim_config.json"),
                                     seed)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c, fs = probe.trace_len, sim_cfg.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    cond = json.load(open(os.path.join(stage1_run, "conditions.json")))     # stage 1's target scaling
    shutil.copy(os.path.join(stage1_run, "conditions.json"), os.path.join(out, "conditions.json"))

    stage1 = Stage1(stage1_run, trace_len, fs, c, device, args.hist_impl)
    y_dim, n_rates = stage1.y_dim, stage1.n_rates
    to_sim = ThetaToSim(sim_cfg, cond, n_rates)
    center, scale = to_sim.theta_center_scale(y_dim)
    refiner = SimRefineNet(stage1.hist_channels, stage1.hist_hw, y_dim, stage1.ctx_dim, center, scale,
                           hidden=args.hidden, enc_hidden=args.enc_hidden, enc_depth=args.enc_depth,
                           kernel_size=args.kernel_size, max_step=args.max_step).to(device)
    n_params = sum(p.numel() for p in refiner.parameters())

    n_cpus = len(os.sched_getaffinity(0))
    if args.gen_workers <= 0:
        args.gen_workers = max(1, (n_cpus - 2) // (args.iters + 1))
    if args.resim_workers <= 0:
        args.resim_workers = max(1, n_cpus - 2 - args.gen_workers)
    if args.buffer_size < args.batch_size:
        raise SystemExit("--buffer-size must be at least the batch size")
    spectrum_path = write_noise_spectra(sim_cfg, trace_len, out)
    print(f"[Parent] stage 1: {stage1_run} ({tc1['model']['name']}, {n_rates} rates + {y_dim - n_rates} conditions)\n"
          f"[Parent] refiner: SimRefineNet {n_params:,} parameters | {args.iters} iterations in training, "
          f"{args.test_iters} at test | seed {seed} | device {device}, histograms on {len(stage1.hists.devices)} "
          f"device(s) ({stage1.args.hist_impl})\n"
          f"[Parent] cpus={n_cpus}: {args.gen_workers} simulating observed traces, {args.resim_workers} re-simulating "
          f"| noise {sim_cfg.noise_mode}", flush=True)
    if args.finetune_stage1:
        when = (f"after {args.stage1_freeze_epochs} frozen epochs" if args.stage1_freeze_epochs >= 0 else
                f"once the refiner stalls for {args.unfreeze_patience} epochs (or half of --max-hours)")
        print(f"[Parent] two phases: A refiner only, B stage 1 fine-tuned too, {when}: lr {args.stage1_lr:g}, "
              f"own loss weight {args.stage1_loss_weight:g} (level aux {stage1.level_aux_weight:g})", flush=True)

    train_config = {
        "model": {"name": SimRefineNet.__name__, "hidden": args.hidden, "enc_hidden": args.enc_hidden,
                  "enc_depth": args.enc_depth, "kernel_size": args.kernel_size, "max_step": args.max_step,
                  "parameters": n_params, "hist_channels": stage1.hist_channels, "hist_hw": list(stage1.hist_hw),
                  "theta_dim": y_dim, "num_rates": n_rates, "ctx_dim": stage1.ctx_dim},
        "stage1_run": stage1_run, "stage1_model": tc1["model"],
        "stage1_finetune": ({"lr": args.stage1_lr, "loss_weight": args.stage1_loss_weight,
                             "freeze_epochs": args.stage1_freeze_epochs, "unfreeze_patience": args.unfreeze_patience,
                             "level_aux_weight": stage1.level_aux_weight,
                             "parameters": sum(p.numel() for p in stage1.core.parameters())}
                            if args.finetune_stage1 else None),
        "refinement": {"iters": args.iters, "test_iters": args.test_iters, "gamma": args.gamma, "tol": args.tol,
                       "discrepancy": "mean |H_obs - H_sim|, log10 histograms of both stage-1 ladders",
                       "resim_noise": "training noise files, common random numbers per sample"},
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "loss": "LogCoshLoss (RAFT sequence loss)", "grad_clip_norm": args.clip_grad_norm,
        "lr_scheduler": {"name": "ReduceLROnPlateau", "factor": LR_FACTOR, "patience": LR_PATIENCE, "min_lr": MIN_LR},
        "early_stopping": {"monitor": "val_loss (last iterate)", "patience": args.patience,
                           "min_delta": args.min_delta},
        "batch_size": args.batch_size, "steps_per_epoch": args.steps_per_epoch, "max_epochs": args.epochs,
        "seed": seed,
        "data": {"sim_config": config_path, "no_reuse": True, "buffer_size": args.buffer_size,
                 "gen_workers": args.gen_workers, "resim_workers": args.resim_workers, "n_val": args.n_val,
                 "n_test": args.n_test, "noise_mode": sim_cfg.noise_mode},
    }
    with open(os.path.join(out, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    ctx = mp.get_context("spawn")

    # 1. Fixed val / test sets -> stage-1 estimates (the traces themselves are not needed afterwards)
    t0 = time.time()
    with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                  initargs=(config_path, sim_cfg, seed, spectrum_path, cond, True)) as pool:
        X, y, _, _ = simulate_fixed_set(pool, STREAM_VAL, args.n_val, (trace_len,), y_dim, torch.int16)
        # a fine-tuned stage 1 changes, so its outputs are recomputed from the traces at every evaluation
        val = {"x": X, "y": y} if args.finetune_stage1 else precompute(stage1, X, y, args.batch_size)
        X, y, aux_test, noise_src_test = simulate_fixed_set(pool, STREAM_TEST, args.n_test, (trace_len,), y_dim,
                                                            torch.int16)
        test = precompute(stage1, X, y, args.batch_size)
        if args.finetune_stage1:            # keep the original stage 1's estimate for the comparison at the end
            test = {"x": X, "y": y, "theta0_orig": test["theta0"]}
        del X
    np.save(os.path.join(out, "aux_test.npy"), aux_test)
    np.save(os.path.join(out, "noise_source_test.npy"), noise_src_test)
    print(f"[Parent] val={args.n_val} test={args.n_test} simulated and run through stage 1 in "
          f"{time.time() - t0:.1f}s", flush=True)

    # 2. Re-simulation pool, queue of fresh observed traces + background simulators
    resim = Resimulator(ctx, args.resim_workers, config_path, sim_cfg, spectrum_path, seed, to_sim, args.batch_size,
                        trace_len, pin=tso.pin_host)
    X_buf = torch.zeros((args.buffer_size, trace_len), dtype=torch.int16).share_memory_()
    y_buf = torch.zeros((args.buffer_size, y_dim), dtype=torch.float32).share_memory_()
    data_lock = ctx.Lock()
    next_index = ctx.Value("q", 0)
    n_written = ctx.Value("q", 0, lock=False)
    n_read = ctx.Value("q", 0, lock=False)
    stop = ctx.Value("b", 0, lock=False)
    gens = [ctx.Process(target=generator_loop,
                        args=(config_path, sim_cfg, seed, spectrum_path, X_buf, y_buf, data_lock, next_index,
                              n_written, stop, n_read, cond, True), daemon=True)
            for _ in range(args.gen_workers)]
    for g in gens:
        g.start()

    try:
        train(args, out, stage1, refiner, resim, val, test, aux_test, noise_src_test, sim_cfg, cond, n_rates, device,
              (X_buf, y_buf, data_lock, n_written, n_read), t_start)
    finally:
        stop.value = 1
        resim.close()
        for g in gens:
            g.join(timeout=30)
            if g.is_alive():
                g.terminate()


def test_report(args, out, stage1, refiner, resim, test, n_rates, device, crit, label):
    """Every iterate + the stop rules on the test set, with the current weights. Returns (summary, traj, disc)."""
    hdr = f"Test {label}".strip()
    t0 = time.time()
    traj, disc = evaluate(refiner, stage1, resim, test, args.test_iters, STREAM_RESIM["test"], args.batch_size, device,
                          extra_sim=True)
    y = test["y"]
    m = per_iter_metrics(traj, y, crit, n_rates)
    print(f"[{hdr}] {time.time() - t0:.1f}s | re-simulations so far: {resim.n_traces} "
          f"({resim.n_traces / max(resim.cpu_s, 1e-9):.1f}/s per worker)", flush=True)
    print(f"[{hdr}] iteration | loss | RAE rates | RAE per out | mean discrepancy", flush=True)
    for k, (l, r, rr) in enumerate(m):
        tag = " (stage 1)" if k == 0 else (" (trained count)" if k == args.iters else "")
        print(f"[{hdr}]   {k:2d}{tag:16s} | {l:.4f} | {rr:.4f} | {' '.join(f'{v:.4f}' for v in r)} | "
              f"{float(disc[:, k].mean()):.4f}", flush=True)
    summary = {"by_iteration": [{"iter": k, "loss": l, "rae_per_out": r, "rae_rates": rr,
                                 "discrepancy": float(disc[:, k].mean())} for k, (l, r, rr) in enumerate(m)]}
    for name, (th, kk) in stop_rules(traj, disc, args.tol, n_rates).items():
        r = rae_log(th, y).mean(dim=0)
        summary[name] = {"rae_per_out": r.tolist(), "rae_rates": float(r[:n_rates].mean()),
                         "mean_iter": float(kk.float().mean())}
        print(f"[{hdr}] stop rule '{name}': RAE rates {float(r[:n_rates].mean()):.4f} | per out "
              f"{' '.join(f'{v:.4f}' for v in r.tolist())} | mean iteration {float(kk.float().mean()):.2f}",
              flush=True)
    return summary, traj, disc


def train(args, out, stage1, refiner, resim, val, test, aux_test, noise_src_test, sim_cfg, cond, n_rates, device,
          queue, t_start):
    """
    Phase A: stage 1 frozen, only the refiner trains. Without --finetune-stage1 that is the whole run.
    With it, phase B follows once phase A has stalled (--stage1-freeze-epochs -1: no improvement for
    --unfreeze-patience epochs, or half of --max-hours used; N >= 0: after N epochs): the best phase-A refiner is
    restored and saved as best_refiner_phaseA.pt, stage 1 is unfrozen, and both train together, with a fresh
    learning-rate schedule and early-stopping count. best_refiner.pt / best_stage1.pt are the best of both phases.
    """
    X_buf, y_buf, data_lock, n_written, n_read = queue
    crit = LogCoshLoss()
    ft = args.finetune_stage1
    groups = [{"params": list(refiner.parameters()), "lr": args.lr}]
    if ft:
        groups.append({"params": list(stage1.core.parameters()), "lr": args.stage1_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    make_sched = lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR,
                                                                    patience=LR_PATIENCE, min_lr=MIN_LR)
    scheduler = make_sched()
    best_val, best_path = float("inf"), os.path.join(out, "best_refiner.pt")
    best_s1_path = os.path.join(out, "best_stage1.pt")
    phase_a_path = os.path.join(out, "best_refiner_phaseA.pt")
    stage1_orig = {k: v.detach().clone() for k, v in stage1.core.state_dict().items()}

    def save_best():
        torch.save(refiner.state_dict(), best_path)
        if ft:
            torch.save(stage1.core.state_dict(), best_s1_path)

    save_best()                 # identity refiner (+ original stage 1): test still runs if nothing improves
    csv_path = os.path.join(out, "refine_metrics.csv")
    K = args.iters
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "phase", "epoch_time_s", "train_loss", "val_loss", "lr", "obs_wait_s",
                                "resim_s", *[f"val_loss_it{k}" for k in range(K + 1)],
                                *[f"val_rae_rates_it{k}" for k in range(K + 1)], "train_stage1_loss", "stage1_lr"])
    stream_train = STREAM_RESIM["train"]
    n_used, since_best, s1_on, phase_a = 0, 0, False, None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        resim_s0 = resim.wall_s
        refiner.train()
        stage1.set_trainable(s1_on)
        loss_sum, s1_sum, waited = 0.0, 0.0, 0.0
        for _ in range(args.steps_per_epoch):
            x, y, w = take_batch(X_buf, y_buf, data_lock, n_written, n_read, args.batch_size)
            waited += w
            y = y.to(device)
            idx = torch.arange(n_used, n_used + args.batch_size)
            n_used += args.batch_size
            theta0, ctx_f, base, amp, h_obs = stage1(x, grad=s1_on)
            thetas, _ = unroll(refiner, stage1, resim, theta0, ctx_f, base, amp, h_obs, K, stream_train, idx)
            loss = sequence_loss(crit, thetas, y, args.gamma)
            total = loss
            if s1_on:
                s1_loss = stage1.own_loss(crit, theta0, base, amp, y)
                total = loss + args.stage1_loss_weight * s1_loss
                s1_sum += float(s1_loss)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=args.clip_grad_norm)
            if s1_on:           # clipped on its own, so one network's large gradient does not shrink the other's
                torch.nn.utils.clip_grad_norm_(stage1.core.parameters(), max_norm=args.clip_grad_norm)
            opt.step()
            loss_sum += float(loss)
        train_loss = loss_sum / args.steps_per_epoch
        s1_train = s1_sum / args.steps_per_epoch if s1_on else float("nan")
        t_train, resim_train = time.time() - t0, resim.wall_s - resim_s0

        stage1.set_trainable(False)
        traj, _ = evaluate(refiner, stage1, resim, val, K, STREAM_RESIM["val"], args.batch_size, device)
        m = per_iter_metrics(traj, val["y"], crit, n_rates)
        val_loss = m[-1][0]
        scheduler.step(val_loss)
        dt = time.time() - t0
        lr = opt.param_groups[0]["lr"]
        phase = "B" if s1_on else "A"
        print(f"[Epoch {epoch:03d}{' B' if s1_on else ''}] {dt:.1f}s (train {t_train:.1f}s: re-simulating "
              f"{resim_train:.1f}s, waiting for observed traces {waited:.1f}s) | Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | lr {lr:g}\n"
              f"            val loss by iteration: {' '.join(f'{x[0]:.4f}' for x in m)} | RAE (rates) by iteration: "
              f"{' '.join(f'{x[2]:.4f}' for x in m)}\n"
              f"            val RAE per out (it {K}): {' '.join(f'{r:.4f}' for r in m[-1][1])}"
              + (f"\n            stage 1 own train loss {s1_train:.6f} | stage-1 lr {opt.param_groups[1]['lr']:g}"
                 if s1_on else ""), flush=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, phase, f"{dt:.3f}", train_loss, val_loss, lr, f"{waited:.3f}",
                                    f"{resim.wall_s - resim_s0:.3f}", *[x[0] for x in m], *[x[2] for x in m],
                                    s1_train, opt.param_groups[1]["lr"] if ft else ""])

        if val_loss < best_val - args.min_delta:
            best_val, since_best = val_loss, 0
            save_best()
            print(f"  -> Saved best refiner{' + stage 1' if ft else ''} (val={best_val:.6f})", flush=True)
        else:
            since_best += 1
        elapsed = time.time() - t_start
        if args.max_hours > 0 and elapsed + dt > args.max_hours * 3600:
            print(f"[Time limit] another epoch would pass --max-hours {args.max_hours:g}. Stopping.", flush=True)
            break

        if ft and not s1_on:
            if args.stage1_freeze_epochs >= 0:
                why = f"{args.stage1_freeze_epochs} frozen epochs done" if epoch >= args.stage1_freeze_epochs else None
            elif since_best >= args.unfreeze_patience:
                why = f"no improvement for {since_best} epochs"
            elif args.max_hours > 0 and elapsed > 0.5 * args.max_hours * 3600:
                why = "half of --max-hours used"
            else:
                why = None
            if why is None and since_best >= args.patience:
                why = f"phase A would stop early"
            if why:
                refiner.load_state_dict(torch.load(best_path, map_location=device))
                shutil.copy(best_path, phase_a_path)
                phase_a = {"epochs": epoch, "best_val": best_val}
                # the refiner keeps the rate it had reached; stage 1's group (lowered along with it while frozen)
                # starts at --stage1-lr
                opt.param_groups[1]["lr"] = args.stage1_lr
                opt.state.clear()                       # fresh Adam moments: the loss landscape changes now
                scheduler = make_sched()
                since_best, s1_on = 0, True
                print(f"\n[Phase B] {why}: best phase-A refiner (val={best_val:.6f}) restored and saved as "
                      f"{os.path.basename(phase_a_path)}; stage 1 unfrozen (lr {args.stage1_lr:g}, refiner lr "
                      f"{lr:g})\n", flush=True)
                continue
        if since_best >= args.patience:
            print(f"[Early Stopping] No improvement for {args.patience} epochs. Stopping.", flush=True)
            break

    # ----------------- Test -----------------
    y = test["y"]
    summary = {}
    if phase_a is not None:     # phase A on its own: its best refiner on the original stage 1
        print(f"\n[Test A] phase A ({phase_a['epochs']} epochs, stage 1 frozen): {args.test_iters} iterations on "
              f"{y.shape[0]} test samples...", flush=True)
        refiner.load_state_dict(torch.load(phase_a_path, map_location=device))
        stage1.core.load_state_dict(stage1_orig)
        stage1.set_trainable(False)
        summary["phase_A"], traj_a, disc_a = test_report(args, out, stage1, refiner, resim, test, n_rates, device,
                                                         crit, "A")
        summary["phase_A"].update(phase_a)
        # every iterate of A, for per-trace comparisons (plot_refine_compare.py); refine_test.npz holds B's
        np.savez_compressed(os.path.join(out, "refine_test_phaseA.npz"), theta=traj_a.numpy(),
                            discrepancy=disc_a.numpy())

    label = "B" if phase_a is not None else ""
    print(f"\n[Test{' ' + label if label else ''}] best {'of both phases' if phase_a is not None else 'refiner'}: "
          f"{args.test_iters} iterations on {y.shape[0]} test samples...", flush=True)
    refiner.load_state_dict(torch.load(best_path, map_location=device))
    if ft:
        stage1.core.load_state_dict(torch.load(best_s1_path, map_location=device))
        stage1.set_trainable(False)
    s, traj, disc = test_report(args, out, stage1, refiner, resim, test, n_rates, device, crit, label or "")
    summary.update(s)
    summary["best_val"] = best_val
    if "theta0_orig" in test:
        l0 = float(crit(test["theta0_orig"], y))
        r0 = rae_log(test["theta0_orig"], y).mean(dim=0)
        summary["stage1_original"] = {"loss": l0, "rae_per_out": r0.tolist(), "rae_rates": float(r0[:n_rates].mean())}
        print(f"[Test] original stage 1 (before any fine-tuning): loss {l0:.4f} | RAE rates "
              f"{float(r0[:n_rates].mean()):.4f} | per out {' '.join(f'{v:.4f}' for v in r0.tolist())}", flush=True)
        p0_hz, _, p0_aux = split_outputs(test["theta0_orig"].numpy().astype(np.float64),
                                         y.numpy().astype(np.float64), cond)
        np.save(os.path.join(out, "preds_test_stage1_orig.npy"), p0_hz)
        np.save(os.path.join(out, "preds_aux_test_stage1_orig.npy"), p0_aux)

    targs = y.numpy().astype(np.float64)
    k_rep = min(args.iters, args.test_iters)
    p_hz, t_hz, p_aux = split_outputs(traj[:, k_rep].numpy().astype(np.float64), targs, cond)
    p1_hz, _, _ = split_outputs(traj[:, 0].numpy().astype(np.float64), targs, cond)
    np.save(os.path.join(out, "preds_test.npy"), p_hz)
    np.save(os.path.join(out, "preds_aux_test.npy"), p_aux)
    np.save(os.path.join(out, "preds_test_stage1.npy"), p1_hz)
    np.save(os.path.join(out, "targs_test.npy"), t_hz)
    np.savez_compressed(os.path.join(out, "refine_test.npz"), theta=traj.numpy(), discrepancy=disc.numpy(),
                        target=targs, aux=aux_test, noise_source=noise_src_test)
    with open(os.path.join(out, "refine_test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[Test] preds_test.npy = iteration {k_rep}", flush=True)
    rae_by_noise_file(p_hz, t_hz, noise_src_test, sim_cfg, out, "Test")


if __name__ == "__main__":
    main()
