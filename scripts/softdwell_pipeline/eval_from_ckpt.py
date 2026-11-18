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

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to best_model.pt",
    )

    parser.add_argument(
        "--h5",
        type=str,
        required=True,
        help="Path to inference HDF5 file. For COC, this must be a COC dataset.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: checkpoint directory.",
    )

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Optional override for cfg.topology, e.g. COC or COCO.",
    )

    parser.add_argument(
        "--symmetry",
        type=str,
        default=None,
        help="Optional override for cfg.symmetry.",
    )

    return parser.parse_args()


def extract_state_dict(ckpt_obj):
    """
    Accept either:
      - raw state_dict
      - checkpoint dict containing 'model_state_dict'
    """
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def infer_num_outputs_from_state_dict(state_dict, fallback=None):
    """
    Infer output dimension from the final head weight.

    For example:
      COC  checkpoint -> backbone.head.weight shape [4, hidden]
      COCO checkpoint -> backbone.head.weight shape [6, hidden]
    """
    key = "backbone.head.weight"

    if key in state_dict:
        return state_dict[key].shape[0]

    if fallback is not None:
        print(
            f"[warn] Could not find '{key}' in checkpoint. "
            f"Falling back to cfg.num_outputs={fallback}."
        )
        return fallback

    raise KeyError(
        f"Could not infer num_outputs because '{key}' was not found in checkpoint."
    )


def q_to_chain_rates_torch(yb: torch.Tensor) -> torch.Tensor:
    """
    Convert chain Q-matrix targets to adjacent bidirectional rates.

    Accepts:
      [B, S, S]
      [B, 1, S, S]
      [B, D]          already rate vector

    Returns:
      [B, 2 * (S - 1)]

    Examples:
      COC:
        S = 3
        rates = (0,1), (1,0), (1,2), (2,1)
        output shape [B, 4]

      COCO:
        S = 4
        rates = (0,1), (1,0), (1,2), (2,1), (2,3), (3,2)
        output shape [B, 6]
    """
    if yb.ndim == 4 and yb.shape[1] == 1:
        yb = yb[:, 0]

    # Already rate-vector target
    if yb.ndim == 2:
        return yb

    if yb.ndim != 3:
        raise ValueError(f"Unsupported target shape: {tuple(yb.shape)}")

    if yb.shape[-1] != yb.shape[-2]:
        raise ValueError(
            f"Expected square Q matrix, got target shape {tuple(yb.shape)}"
        )

    num_states = yb.shape[-1]

    rate_positions = []
    for i in range(num_states - 1):
        rate_positions.append((i, i + 1))
        rate_positions.append((i + 1, i))

    rates = torch.stack([yb[:, i, j] for (i, j) in rate_positions], dim=1)

    return rates

def canonicalize_rate_batch_torch(
    yb_rates: torch.Tensor,
    canonicalizer,
) -> torch.Tensor:
    """
    Apply the existing 1D numpy canonicalizer to a batch of rate vectors.

    Input:
      [B, D]

    Output:
      [B, D]
    """
    if canonicalizer is None:
        return yb_rates

    device = yb_rates.device
    dtype = yb_rates.dtype

    y_np = yb_rates.detach().cpu().numpy()
    y_can = np.stack([canonicalizer(row) for row in y_np], axis=0)

    return torch.as_tensor(y_can, device=device, dtype=dtype)


def prepare_targets_torch(
    yb: torch.Tensor,
    canonicalizer,
    y_log10: bool,
) -> torch.Tensor:
    """
    Prepare targets consistently for evaluation.

    Handles:
      Q matrix targets: [B, S, S] or [B, 1, S, S]
      rate-vector targets: [B, D]

    Steps:
      1. Convert Q matrix to adjacent chain rates
      2. Canonicalize rate vector if needed
      3. Apply log10 if the model was trained in log10 space
    """
    yb_rates = q_to_chain_rates_torch(yb)

    yb_rates = canonicalize_rate_batch_torch(
        yb_rates,
        canonicalizer=canonicalizer,
    )

    if y_log10:
        if torch.any(yb_rates <= 0):
            raise ValueError(
                "Cannot apply log10 because target rates contain non-positive values. "
                f"Target range: {float(yb_rates.min())} -> {float(yb_rates.max())}"
            )

        yb_rates = torch.log10(yb_rates)

    return yb_rates


@torch.no_grad()
def main():
    args = parse_args()

    cfg = TrainConfig().apply_env_overrides()

    if args.topology is not None:
        cfg.topology = args.topology

    if args.symmetry is not None:
        cfg.symmetry = args.symmetry

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
    print(f"[eval] topology   : {cfg.topology}")
    print(f"[eval] symmetry   : {cfg.symmetry}")
    print(f"[eval] y_log10    : {cfg.y_log10}")

    ckpt_obj = torch.load(args.ckpt, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)

    num_outputs = infer_num_outputs_from_state_dict(
        state_dict,
        fallback=getattr(cfg, "num_outputs", None),
    )

    print(f"[debug] inferred checkpoint num_outputs: {num_outputs}")

    # This canonicalizer expects a 1D rate vector.
    # Therefore, do NOT pass it directly to HDF5IonChannelDataset
    # when the HDF5 target is a Q matrix.
    vector_canon = choose_canonicalizer(cfg.topology, cfg.symmetry)

    full = HDF5IonChannelDataset(
        cfg.h5_path,
        x_key=cfg.x_key,
        y_key=cfg.y_key,

        # Important:
        # Load raw targets first.
        # We will convert Q -> rates and apply log10 manually after that.
        y_log10=False,

        dtype=torch.float32,
        canonicalizer=None,

        # Full-file evaluation, no train/val/test split
        start=None,
        stop=None,
    )

    print("[debug] number of evaluated samples:", len(full))
    loader = DataLoader(
        full,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # Must match the trained SoftDwell run.
    # For your current SoftDwell run you used 4 detectors and 24 bins.
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

    # Infer SoftDwell output shape.
    xb0, yb0 = full[0]
    fake = xb0.unsqueeze(0).to(device).float()
    logH0 = softdwell(fake)

    print("[debug] fake input shape     :", tuple(fake.shape))
    print("[debug] softdwell out shape  :", tuple(logH0.shape))

    K = logH0.shape[1]
    hist_hw = tuple(logH0.shape[-2:])

    print("[debug] inferred K           :", K)
    print("[debug] inferred hist_hw     :", hist_hw)

    yb0_check = yb0.unsqueeze(0).to(device)

    yb0_rates = prepare_targets_torch(
        yb0_check,
        canonicalizer=vector_canon,
        y_log10=cfg.y_log10,
    )

    print("[debug] first raw target shape :", tuple(yb0_check.shape))
    print("[debug] first rate target shape:", tuple(yb0_rates.shape))

    if yb0_rates.shape[1] != num_outputs:
        raise ValueError(
            f"Checkpoint/data output mismatch before model loading:\n"
            f"  checkpoint num_outputs = {num_outputs}\n"
            f"  target rate dimension  = {yb0_rates.shape[1]}\n"
            f"This usually means you are using a COCO checkpoint with COC data, "
            f"or a COC checkpoint with COCO data."
        )

    model = SoftDwellThenTFIR(
        softdwell_layer=softdwell,
        K=K,
        hist_hw=hist_hw,
        num_outputs=num_outputs,
        use_bn=False,
    ).to(device)

    if "backbone.head.weight" in state_dict:
        print(
            "[debug] checkpoint head weight shape:",
            tuple(state_dict["backbone.head.weight"].shape),
        )

    print(
        "[debug] model head weight shape     :",
        tuple(model.backbone.head.weight.shape),
    )

    model.load_state_dict(state_dict, strict=True)
    print("[load] checkpoint loaded successfully\n")

    model.eval()

    preds_log_chunks = []
    targs_log_chunks = []

    first_batch = True

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True)

        yb = prepare_targets_torch(
            yb,
            canonicalizer=vector_canon,
            y_log10=cfg.y_log10,
        )

        y_pred = model(xb)

        if y_pred.shape != yb.shape:
            raise ValueError(
                f"Prediction/target shape mismatch:\n"
                f"  pred   = {tuple(y_pred.shape)}\n"
                f"  target = {tuple(yb.shape)}\n"
                f"This usually means checkpoint topology and HDF5 topology do not match."
            )

        if first_batch:
            print("[debug] first pred shape:", tuple(y_pred.shape))
            print("[debug] first targ shape:", tuple(yb.shape))
            first_batch = False

        preds_log_chunks.append(y_pred.cpu())
        targs_log_chunks.append(yb.cpu())

    preds_log = torch.cat(preds_log_chunks, dim=0)
    targs_log = torch.cat(targs_log_chunks, dim=0)

    print(
        "[debug] preds_log range:",
        float(preds_log.min()),
        "->",
        float(preds_log.max()),
    )
    print(
        "[debug] targs_log range:",
        float(targs_log.min()),
        "->",
        float(targs_log.max()),
    )

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