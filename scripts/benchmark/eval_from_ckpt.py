#!/usr/bin/env python3
import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset

from scripts.benchmark.config import TrainConfig
from scripts.common.hdf5_dataset import HDF5IonChannelDataset
from scripts.common.canonicalize import choose_canonicalizer
from scripts.benchmark.model import SoftDwellThenTFIR
from scripts.benchmark.softdwell import ChannelWiseSoftDwell
from scripts.common.splits import build_or_load_splits
from scripts.benchmark.metrics import rae_log  # ✅ to compute RAE like training

@torch.no_grad()
def main():
    cfg = TrainConfig().apply_env_overrides()

    # --- make eval robust & match training run ---
    cfg.output_root = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/tau_unbounded"  # ← match training
    cfg.batch_size = 1        # ← safe on 1M samples; bump if you have headroom
    cfg.num_workers = 0
    cfg.splits_dir = os.path.join(cfg.output_root, "splits")
    os.makedirs(cfg.splits_dir, exist_ok=True)

    ckpt_path = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/tau_unbounded/best_model.pt"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] loading checkpoint: {ckpt_path}")

    # ---- dataset (same split as training) ----
    canon = choose_canonicalizer(cfg.topology, cfg.symmetry)
    full = HDF5IonChannelDataset(
        cfg.h5_path, x_key=cfg.x_key, y_key=cfg.y_key,
        x_order=cfg.x_order, y_scale=cfg.y_scale, y_log10=cfg.y_log10,
        dtype=torch.float32, canonicalizer=canon,
        start=None, stop=(cfg.limit_n if cfg.limit_n > 0 else None),
    )

    # ---- build model ----
    softdwell = ChannelWiseSoftDwell(
        num_detectors=cfg.num_detectors,
        dwell_min=cfg.dwell_min, dwell_max=cfg.dwell_max,
        num_bins=cfg.num_bins, length_ratio=cfg.length_ratio,
        average_segments=True,
        norm_min=cfg.norm_min, norm_max=cfg.norm_max,
        theta_init_min=cfg.theta_init_min, theta_init_max=cfg.theta_init_max,
        tau_init=cfg.tau_init,
        dtype=torch.float32,
    )
    model = HOHDThenTFIR(
        softdwell, num_outputs=cfg.num_outputs
    ).to(device)

    # ---- build net lazily with one dummy forward ----
    xb0, _ = full[0]                 # (C,T)
    fake = xb0.unsqueeze(0).to(device).float()  # (1,C,T)
    with torch.no_grad():
        _ = model(fake)              # now self.net is created inside

    # ---- now load weights (both softdwell + net.* exist) ----
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("\n[load] --- MISSING KEYS ---")
    for k in missing:
        print("  MISSING:", k)
    print("\n[load] --- UNEXPECTED KEYS ---")
    for k in unexpected:
        print("  UNEXPECTED:", k)
    print(f"\n[load] Summary: missing={len(missing)}, unexpected={len(unexpected)}\n")

    N = len(full)
    _, _, test_idx = build_or_load_splits(cfg, N, rank=0)  # will now read .../freeze_softdwell/splits/...
    test_ds = Subset(full, test_idx.tolist())
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    # ---- inference ----
    model.eval()
    preds_log_chunks, targs_log_chunks = [], []  # same domain as training

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        with torch.inference_mode():
            y_pred = model(xb)
        preds_log_chunks.append(y_pred.detach().cpu())
        targs_log_chunks.append(yb.detach().cpu())

    preds_log = torch.cat(preds_log_chunks, dim=0)   # tensors in training domain (log10 if cfg.y_log10)
    targs_log = torch.cat(targs_log_chunks, dim=0)

    
    # ---- quick diagnostic: log-domain and linear-domain ranges ----
    print("[debug] preds_log range:", float(preds_log.min()), "→", float(preds_log.max()))
    print("[debug] targs_log range:", float(targs_log.min()), "→", float(targs_log.max()))

    if cfg.y_log10:
        preds_lin = 10 ** preds_log
        targs_lin = 10 ** targs_log
        print("[debug] preds_lin range (Hz):", float(preds_lin.min()), "→", float(preds_lin.max()))
        print("[debug] targs_lin range (Hz):", float(targs_lin.min()), "→", float(targs_lin.max()))

    # ---- RAE in training domain ----
    rae_per_out = rae_log(preds_log, targs_log).mean(dim=0).numpy()
    rae_overall = float(rae_per_out.mean())

    # ---- save npy in linear domain (matches your usual npy files) ----
    preds_np = preds_log.numpy()
    targs_np = targs_log.numpy()
    if cfg.y_log10:
        preds_np = 10 ** preds_np
        targs_np = 10 ** targs_np

    pred_path = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/tau_unbounded/predictions_nn.npy"
    targ_path = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/tau_unbounded/targets_nn.npy"
    np.save(pred_path, preds_np)
    np.save(targ_path, targs_np)

    print(f"[done] saved:\n  {pred_path}\n  {targ_path}")
    print(f"Shapes: preds={preds_np.shape}, targets={targs_np.shape}")
    print("RAE per output (training domain):", " ".join(f"{r:.4f}" for r in rae_per_out))
    print(f"RAE overall (mean): {rae_overall:.6f}")

if __name__ == "__main__":
    main()
