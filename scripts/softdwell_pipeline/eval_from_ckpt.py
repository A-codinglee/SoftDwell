#!/usr/bin/env python3
import os
import csv
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from scripts.softdwell_pipeline.config import TrainConfig
from scripts.common.hdf5_dataset import HDF5IonChannelDataset
from scripts.common.canonicalize import choose_canonicalizer
from scripts.softdwell_pipeline.model import SoftDwellThenTFIR
from scripts.softdwell_pipeline.softdwell import ChannelWiseSoftDwell
from scripts.softdwell_pipeline.metrics import rae_log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument(
        "--h5",
        type=str,
        default="/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_5_100000_kij_fixed_16k_8k_4k_2k_1k_500_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_04_16.h5",
        help="Path to inference HDF5 file",
    )
    parser.add_argument("--outdir", type=str, default=None, help="Directory to save outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj

def q_to_chain_rates_torch(yb: torch.Tensor) -> torch.Tensor:
    """
    Convert Q-matrix targets to the 6 adjacent chain rates used by the model.

    Accepts:
      [B, 4, 4] or [B, 1, 4, 4]

    Returns:
      [B, 6]
    """
    if yb.ndim == 4 and yb.shape[1] == 1:
        yb = yb[:, 0]  # [B, 4, 4]

    rate_positions = [
        (0, 1), (1, 0),
        (1, 2), (2, 1),
        (2, 3), (3, 2),
    ]
    return torch.stack([yb[:, i, j] for (i, j) in rate_positions], dim=1)

@torch.no_grad()
def main():
    args = parse_args()
    cfg = TrainConfig().apply_env_overrides()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.isfile(args.h5):
        raise FileNotFoundError(f"HDF5 file not found: {args.h5}")

    cfg.output_root = os.path.dirname(args.ckpt)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.h5_path = args.h5

    save_dir = args.outdir if args.outdir is not None else cfg.output_root
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] checkpoint : {args.ckpt}")
    print(f"[eval] h5 file    : {args.h5}")
    print(f"[eval] save dir   : {save_dir}")
    print(f"[eval] device     : {device}")

    canon = choose_canonicalizer(cfg.topology, cfg.symmetry)
    full = HDF5IonChannelDataset(
        cfg.h5_path,
        x_key=cfg.x_key,
        y_key=cfg.y_key,
        y_log10=cfg.y_log10,
        dtype=torch.float32,
        canonicalizer=canon,
        start=None,
        stop=(cfg.limit_n if cfg.limit_n > 0 else None),
    )

    loader = DataLoader(
        full,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Match the trained run
    num_detectors = 4
    num_bins = 24

    softdwell = ChannelWiseSoftDwell(
        num_detectors=num_detectors,
        dwell_min=cfg.dwell_min,
        dwell_max=cfg.dwell_max,
        num_bins=num_bins,
        average_segments=False,
        norm_min=cfg.norm_min,
        norm_max=cfg.norm_max,
        theta_init_min=cfg.theta_init_min,
        theta_init_max=cfg.theta_init_max,
        tau_init=cfg.tau_init,
        dtype=torch.float32,
    ).to(device)

    # Infer actual SoftDwell output shape first
    xb0, _ = full[0]
    fake = xb0.unsqueeze(0).to(device).float()   # [1, C, T]
    logH0 = softdwell(fake)

    print("[debug] fake shape         :", tuple(fake.shape))
    print("[debug] softdwell out shape:", tuple(logH0.shape))

    K = logH0.shape[1]
    hist_hw = tuple(logH0.shape[-2:])

    print("[debug] inferred K       :", K)
    print("[debug] inferred hist_hw :", hist_hw)

    model = SoftDwellThenTFIR(
        softdwell_layer=softdwell,
        K=K,
        hist_hw=hist_hw,
        num_outputs=cfg.num_outputs,
        use_bn=False,
    ).to(device)

    ckpt_obj = torch.load(args.ckpt, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)

    print("[debug] checkpoint head weight shape:",
          tuple(state_dict["backbone.head.weight"].shape))
    print("[debug] model head weight shape     :",
          tuple(model.backbone.head.weight.shape))

    model.load_state_dict(state_dict, strict=True)
    print("[load] checkpoint loaded successfully\n")

    model.eval()

    preds_log_chunks = []
    targs_log_chunks = []

    first_batch = True

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True)

        # Convert Q targets to 6-rate targets to match model output
        if yb.ndim in (3, 4):
            yb = q_to_chain_rates_torch(yb)

        y_pred = model(xb)

        if first_batch:
            print("[debug] first pred shape :", tuple(y_pred.shape))
            print("[debug] first targ shape :", tuple(yb.shape))
            first_batch = False

        preds_log_chunks.append(y_pred.cpu())
        targs_log_chunks.append(yb.cpu())

    preds_log = torch.cat(preds_log_chunks, dim=0)
    targs_log = torch.cat(targs_log_chunks, dim=0)

    print("[debug] preds_log range:", float(preds_log.min()), "->", float(preds_log.max()))
    print("[debug] targs_log range:", float(targs_log.min()), "->", float(targs_log.max()))

    rae_matrix = rae_log(preds_log, targs_log).numpy()
    rae_per_out = rae_matrix.mean(axis=0)
    rae_overall = float(rae_per_out.mean())

    csv_path = os.path.join(save_dir, "rae_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rae_mean"])
        for value in rae_matrix.mean(axis=1):
            writer.writerow([float(value)])

    preds_np = preds_log.numpy()
    targs_np = targs_log.numpy()

    if cfg.y_log10:
        preds_np = 10 ** preds_np
        targs_np = 10 ** targs_np

    pred_path = os.path.join(save_dir, "predictions_nn.npy")
    targ_path = os.path.join(save_dir, "targets_nn.npy")
    np.save(pred_path, preds_np)
    np.save(targ_path, targs_np)

    print(f"[done] saved: {pred_path}")
    print(f"[done] saved: {targ_path}")
    print(f"[done] saved: {csv_path}")
    print("Shapes: preds =", preds_np.shape, ", targets =", targs_np.shape)
    print("RAE per output:", " ".join(f"{r:.4f}" for r in rae_per_out))
    print(f"RAE overall: {rae_overall:.6f}")


if __name__ == "__main__":
    main()