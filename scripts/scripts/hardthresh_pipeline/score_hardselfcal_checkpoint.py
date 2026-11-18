#!/usr/bin/env python3
"""
STANDALONE SCORER: loads an already-trained checkpoint (best_model.pt) and runs ONLY the test/external
scoring pass -- no training. Same recovery pattern as score_selfcal_checkpoint.py, built after the
train_PNG100 fold's external-scoring OOM (2026-09-30, jobs 1826846/1826804/1826862 -- fixed via the
TEST_LIMIT_N default patch, but these three runs had already burned their walltime before the fix landed).

Original module's docstring follows unchanged below.
HardSelfCalibratingFusionNet: same self-calibration idea as train_selfcal_fusion.py, but the level
estimate driving the per-row threshold recentering comes from a FIXED uniform hard-threshold branch
instead of SoftDwell -- tests whether the learnable/soft part is actually necessary to get a useful level
estimate, or whether a cheap fixed-ladder histogram alone is enough. No live O(T) computation anywhere
(both branches are grind_hist_gpu histograms; the fixed one is precomputed ONCE like ordinary
hard-threshold training), so this should train close to plain hard-threshold speed, not SoftDwell speed.

Pipeline per training step:
  1. HardSelfCalibratingFusionNet.fixed_stage(fixed_hist) -> f_fixed, baseline_hat_pA, amp_hat_pA
     (fixed_hist precomputed ONCE outside the loop, uniform FIXED_K/FIXED_J ladder)
  2. per-row ladder from (baseline_hat_pA.detach(), amp_hat_pA.detach()), grind_hist_gpu -> selfcal_hist
     (rebuilt fresh every batch -- this is the only per-batch histogram cost, and it's a plain
     grind_hist_gpu call like train_custom_thresh.py, not a live differentiable O(T) computation)
  3. HardSelfCalibratingFusionNet.finish(f_fixed, selfcal_hist, baseline_hat_pA, amp_hat_pA) -> 9 outputs
     (non-detached baseline_hat_pA/amp_hat_pA explicitly fed to the head, same as v2 SelfCalibratingFusionNet)

level_head is trained by an explicit auxiliary LogCosh loss against true (baseline_pA, amplitude_pA),
weighted by LEVEL_AUX_WEIGHT, added to the main 9-output loss (same convention as train_selfcal_fusion.py).

Env: H5_PATH, LIMIT_N, TEST_H5S; FIXED_K(24), FIXED_J(12), FIXED_THR_LO_PA(-1.67), FIXED_THR_HI_PA(21.67);
SELFCAL_K(24), SELFCAL_J(12), MARGIN_PA(1.0), BASE_SCALE_PA(2.0), AMP_MIN_PA(1.0), LEVEL_AUX_WEIGHT(0.3);
BATCH_SIZE(256), LR(1.4e-3), EPOCHS, PATIENCE, OUTPUT_ROOT.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time

import numpy as np
import torch

from scripts.common.h5_ram_loader import parallel_hdf5_load_xy
from scripts.common.splits import build_or_load_splits
from scripts.hardthresh_pipeline.config import HardThreshConfig
from scripts.hardthresh_pipeline.gpu_hist import grind_hist_gpu, make_dwell_lut
from scripts.hardthresh_pipeline.hist_cache import build_hist_stacks, spec_from_cfg
from scripts.hardthresh_pipeline.labels import read_labels, make_targets, denormalize_extras, save_norm, EXTRA_NAMES
from scripts.softdwell_pipeline.metrics import LogCoshLoss, EarlyStopping
from scripts.softdwell_pipeline.model import HardSelfCalibratingFusionNet

N_RATE = 6


def env(name, default, cast=str):
    v = os.environ.get(name)
    return cast(v) if v is not None else default


def build_perrow_hard_hist(x_counts_b, baseline_hat_pA, amp_hat_pA, lut, k, j, margin_pA, counts_per_pA, k_group):
    lo = baseline_hat_pA - margin_pA
    hi = baseline_hat_pA + amp_hat_pA + margin_pA
    t = torch.linspace(0.0, 1.0, k, device=x_counts_b.device).unsqueeze(0)
    thr_pA = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * t
    thr_counts = torch.round(thr_pA * counts_per_pA).clamp(-32768, 32767).to(torch.int16)
    hist, _ = grind_hist_gpu(x_counts_b, thr_counts, lut, j, k_group=k_group)
    return torch.log10(hist + 1.0)


def main() -> None:
    cfg = HardThreshConfig().apply_env_overrides()
    os.makedirs(cfg.output_root, exist_ok=True)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    fixed_k, fixed_j = env("FIXED_K", 24, int), env("FIXED_J", 12, int)
    fixed_thr_lo, fixed_thr_hi = env("FIXED_THR_LO_PA", -1.67, float), env("FIXED_THR_HI_PA", 21.67, float)
    selfcal_k, selfcal_j = env("SELFCAL_K", 24, int), env("SELFCAL_J", 12, int)
    margin_pA = env("MARGIN_PA", 1.0, float)
    base_scale_pA = env("BASE_SCALE_PA", 2.0, float)
    amp_min_pA = env("AMP_MIN_PA", 1.0, float)
    level_aux_weight = env("LEVEL_AUX_WEIGHT", 0.3, float)
    # NOTE: BATCH_SIZE/EVAL_BS default to SMALL values, unlike plain hard-threshold training (which caches
    # its histogram once and can use big batches). Here selfcal_hist is rebuilt fresh via grind_hist_gpu
    # over the FULL T=1,000,000-sample trace every batch/eval step (same cost driver as train_selfcal_fusion.py's
    # v2 script) -- 256/512 (the old defaults, copied from plain hard-threshold before this was noticed) OOM'd
    # a 40k-row pooled LOO run within the first epoch (2026-09-30, jobs 1826795-99, CUDA OOM in grind_hist_gpu's
    # scatter_add_). Override explicitly if VRAM allows more.
    batch_size, lr, epochs, patience = env("BATCH_SIZE", 64, int), env("LR", 1.4e-3, float), env("EPOCHS", 1000, int), env("PATIENCE", 8, int)
    eval_bs = env("EVAL_BS", 32, int)
    k_group = cfg.k_group
    print(f"[cfg] fixed K={fixed_k} J={fixed_j} thr={fixed_thr_lo}..{fixed_thr_hi}pA | selfcal K={selfcal_k} J={selfcal_j} margin={margin_pA}pA | "
          f"base_scale={base_scale_pA}pA amp_min={amp_min_pA}pA level_aux_weight={level_aux_weight} | bs={batch_size} lr={lr}", flush=True)

    # ---------------- fixed branch: precomputed ONCE, uniform ladder (cheap, like ordinary hard-threshold) ---------------- #
    cfg.num_detectors, cfg.num_bins = fixed_k, fixed_j
    cfg.thr_lo_pA, cfg.thr_hi_pA = fixed_thr_lo, fixed_thr_hi
    fixed_spec = spec_from_cfg(cfg)
    t0 = time.perf_counter()
    fixed_hist, _ = build_hist_stacks(cfg, [fixed_spec], device="cuda")[fixed_spec]
    N = int(fixed_hist.shape[0])
    print(f"[hist] fixed {fixed_spec.tag()} ready: {tuple(fixed_hist.shape)} in {time.perf_counter() - t0:.1f}s", flush=True)
    X_fixed = torch.log10(torch.from_numpy(fixed_hist) + 1.0).to(device)

    # ---------------- raw traces in RAM (int16 counts) -- needed to rebuild the self-cal histogram per batch ---------------- #
    t0 = time.perf_counter()
    X_raw_counts, _ = parallel_hdf5_load_xy(
        path=cfg.h5_path, x_key=cfg.traces_key, y_key=cfg.q_key, y_log10=False,
        limit_n=(cfg.limit_n if cfg.limit_n > 0 else None), num_workers=2, chunk_size=256,
    )
    assert X_raw_counts.shape[0] == N, (X_raw_counts.shape, N)
    T = X_raw_counts.shape[-1]
    X_raw_counts = torch.from_numpy(X_raw_counts).short().squeeze(1)   # (N,T) int16, CPU
    print(f"[raw] traces loaded: {tuple(X_raw_counts.shape)} in {time.perf_counter() - t0:.1f}s", flush=True)
    lut_selfcal = torch.from_numpy(make_dwell_lut(T, cfg.fs_hz, cfg.dwell_min_s, cfg.dwell_max_s, selfcal_j)).to(device)

    # ---------------- labels ---------------- #
    rates, extras, info = read_labels(cfg, N)
    train_idx, val_idx, test_idx = build_or_load_splits(cfg, N, 0)
    print(f"[info] N={N} T={T} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}", flush=True)
    y_np, norm = make_targets(rates, extras, train_idx, cfg.target_scale)
    save_norm(os.path.join(cfg.output_root, "norm.json"), norm, info)
    Y = torch.from_numpy(y_np).to(device)
    extras_t = torch.from_numpy(extras.astype(np.float32)).to(device)
    tr, va, te = (torch.from_numpy(a).long().to(device) for a in (train_idx, val_idx, test_idx))

    def get_raw_counts(idx):
        return X_raw_counts[idx.cpu()].to(device, non_blocking=True)   # (B,T) int16

    # ---------------- model ---------------- #
    separate_heads = env("SEPARATE_HEADS", "false").lower() in ("1", "true", "yes")
    backbone = env("BACKBONE", "inception", str)   # must match the checkpoint's own training config
    # must also match the checkpoint's HIDDEN/DEPTH/KERNEL_SIZE if it was a non-default convnext_v2 run (doc 11)
    convnext_hidden = env("HIDDEN", 64, int)
    convnext_depth = env("DEPTH", 4, int)
    convnext_kernel_size = env("KERNEL_SIZE", 7, int)
    core = HardSelfCalibratingFusionNet(fixed_k, fixed_j, selfcal_k, selfcal_j, separate_heads=separate_heads,
                                         base_scale_pA=base_scale_pA, amp_min_pA=amp_min_pA, backbone=backbone,
                                         convnext_hidden=convnext_hidden, convnext_depth=convnext_depth,
                                         convnext_kernel_size=convnext_kernel_size
                                         ).to(device)
    # channels_last requires rank-4 (NCHW) tensors -- SmallCNN3DBackbone's Conv3d weights are rank-5 (same
    # fix as train_hardselfcal_fusion.py, job 1827072/1827108 hit this here too since this script was
    # copied before that fix landed).
    if backbone != "small_cnn_3d":
        core = core.to(memory_format=torch.channels_last)
    print(f"[cfg] separate_heads={separate_heads} backbone={backbone}", flush=True)

    weight_decay = env("WEIGHT_DECAY", 0.0, float)
    opt = torch.optim.AdamW(core.parameters(), lr=lr, weight_decay=weight_decay)

    use_compile = env("USE_COMPILE", "true").lower() in ("1", "true", "yes")
    if use_compile:
        # Both stages are plain conv/linear ops on precomputed/cheap-to-build histograms (no live O(T)
        # softdwell computation anywhere), so this is much lower-risk to compile than the SoftDwell
        # version -- but keep eval eager for the same batch-size-variation reason as everywhere else.
        fixed_stage = torch.compile(core.fixed_stage, mode="reduce-overhead")
        finish = torch.compile(core.finish, mode="reduce-overhead")
        print("[cfg] use_compile=True -> fixed_stage/finish compiled separately; eval uses eager core.fixed_stage/core.finish", flush=True)
    else:
        fixed_stage, finish = core.fixed_stage, core.finish
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.1, patience=4, min_lr=1e-6)
    es = EarlyStopping(patience=patience, min_delta=1e-5, restore_best=False)
    crit = LogCoshLoss(reduction="mean")
    level_crit = LogCoshLoss(reduction="mean")

    def to_native_extras(z):
        return denormalize_extras(z.float().cpu().numpy(), norm)

    def forward_batch(idx_b, fs, fn, raw_counts_fn):
        xc = raw_counts_fn(idx_b)
        f_fixed, base_hat, amp_hat = fs(X_fixed[idx_b])
        with torch.no_grad():
            selfcal_hist = build_perrow_hard_hist(xc, base_hat.detach(), amp_hat.detach(),
                                                   lut_selfcal, selfcal_k, selfcal_j, margin_pA, cfg.counts_per_pA, k_group)
        p = fn(f_fixed, selfcal_hist, base_hat, amp_hat)
        return p, base_hat, amp_hat

    @torch.no_grad()
    def evaluate(idx, Yt=None, raw_counts_fn=None, X_fixed_=None):
        core.eval(); Yt = Yt if Yt is not None else Y
        raw_counts_fn = raw_counts_fn if raw_counts_fn is not None else get_raw_counts
        X_fixed_ = X_fixed_ if X_fixed_ is not None else X_fixed
        outs, loss_sum, nb = [], 0.0, 0
        for s in range(0, idx.numel(), eval_bs):
            b = idx[s:s + eval_bs]
            xc = raw_counts_fn(b)
            f_fixed, base_hat, amp_hat = core.fixed_stage(X_fixed_[b])
            selfcal_hist = build_perrow_hard_hist(xc, base_hat.detach(), amp_hat.detach(),
                                                   lut_selfcal, selfcal_k, selfcal_j, margin_pA, cfg.counts_per_pA, k_group)
            p = core.finish(f_fixed, selfcal_hist, base_hat, amp_hat).float()
            loss_sum += float(crit(p, Yt[b])) * b.numel(); nb += b.numel()
            outs.append(p)
        p = torch.cat(outs); yb = Yt[idx]
        rae = torch.sqrt(torch.abs(p[:, :N_RATE] - yb[:, :N_RATE]) + 1e-12).mean(0).cpu().numpy()
        mae = np.abs(to_native_extras(p[:, N_RATE:]) - to_native_extras(yb[:, N_RATE:])).mean(0)
        return loss_sum / max(nb, 1), rae, mae, p

    best_path = os.path.join(cfg.output_root, "best_model.pt")
    assert os.path.exists(best_path), f"no checkpoint found at {best_path} -- this script only scores an EXISTING best_model.pt"
    print(f"[scorer] loading existing checkpoint (no training): {best_path}", flush=True)
    best_val = float("nan")

    # ---------------- test (internal + external) ---------------- #
    core.load_state_dict(torch.load(best_path, map_location=device))
    _, test_rae, test_mae, _ = evaluate(te)
    print(f"Test RAE per output: {' '.join(f'{r:.4f}' for r in test_rae)} | Overall: {float(test_rae.mean()):.4f}", flush=True)
    print("[extras] TEST MAE pA  " + "  ".join(f"{n}={m_:.4f}" for n, m_ in zip(EXTRA_NAMES, test_mae)), flush=True)
    with open(os.path.join(cfg.output_root, "test_summary.json"), "w") as f:
        json.dump({"test_rae_per_rate": test_rae.tolist(), "test_rae_overall": float(test_rae.mean()),
                   "test_mae_pA": dict(zip(EXTRA_NAMES, map(float, test_mae))), "n_test": int(len(test_idx)),
                   "best_val": best_val, "fixed_K": fixed_k, "fixed_J": fixed_j, "selfcal_K": selfcal_k, "selfcal_J": selfcal_j,
                   "margin_pA": margin_pA}, f, indent=2)

    for path in [q.strip() for q in re.split(r"[,;:]", cfg.test_h5s) if q.strip()]:
        c2 = copy.copy(cfg); c2.h5_path = path; c2.limit_n = env("TEST_LIMIT_N", 10000, int)  # 0 previously OOM'd loading all 100k rows of the train_PNG100 external file (2026-09-30, job 1826871)
        tag = os.path.splitext(os.path.basename(path))[0][-40:]
        fixed_hist2, _ = build_hist_stacks(c2, [fixed_spec], device="cuda")[fixed_spec]
        n2 = int(fixed_hist2.shape[0])
        X_fixed2 = torch.log10(torch.from_numpy(fixed_hist2) + 1.0).to(device)
        raw2, _ = parallel_hdf5_load_xy(path=path, x_key=cfg.traces_key, y_key=cfg.q_key, y_log10=False,
                                          limit_n=(n2 if n2 else None), num_workers=2, chunk_size=256)
        raw2 = torch.from_numpy(raw2).short().squeeze(1)   # (n2,T)
        r2, e2, _ = read_labels(c2, n2)
        y_ext = np.concatenate([np.log10(np.clip(r2, 1e-9, None)),
                                 (e2 - np.asarray(norm["extra_mean"])) / np.asarray(norm["extra_std"]) * norm["target_scale"]], axis=1).astype(np.float32)
        Y2 = torch.from_numpy(y_ext).to(device)
        idx2 = torch.arange(n2, device=device)

        def get_raw_counts2(idx_b, _raw2=raw2):
            return _raw2[idx_b.cpu()].to(device, non_blocking=True)

        _, rae2, mae2, preds2 = evaluate(idx2, Yt=Y2, raw_counts_fn=get_raw_counts2, X_fixed_=X_fixed2)
        print(f"[external] {tag}  RAE per rate: {' '.join(f'{v:.4f}' for v in rae2)} | Overall: {float(rae2.mean()):.4f}", flush=True)
        print(f"[external] MAE pA  " + "  ".join(f"{n}={v:.4f}" for n, v in zip(EXTRA_NAMES, mae2)), flush=True)
        with open(os.path.join(cfg.output_root, f"external_{tag}.json"), "w") as f:
            json.dump({"rae_per_rate": rae2.tolist(), "rae_overall": float(rae2.mean()), "mae_pA": dict(zip(EXTRA_NAMES, map(float, mae2))), "n": n2}, f)
        pred_native = np.concatenate([10 ** preds2[:, :N_RATE].cpu().numpy(), to_native_extras(preds2[:, N_RATE:])], axis=1)
        np.save(os.path.join(cfg.output_root, f"external_preds_{tag}.npy"), pred_native)


if __name__ == "__main__":
    main()
