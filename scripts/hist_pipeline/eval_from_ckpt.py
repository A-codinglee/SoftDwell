#!/usr/bin/env python3
import os
import csv
import h5py
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from pathlib import Path
from typing import List, Tuple

from scripts.hist_pipeline.config_hist import HistTrainConfig
from scripts.softdwell_pipeline.model import TFExactInceptionResNet
from scripts.softdwell_pipeline.metrics import rae_log


def canonicalize_by_permutations_batch(
    y: np.ndarray,
    perms: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y)
    N, D = y.shape

    perms_arr = [np.asarray(p, dtype=np.int64) for p in perms]
    y_best = y[:, perms_arr[0]].copy()
    which = np.zeros(N, dtype=np.int64)

    for pi in range(1, len(perms_arr)):
        yp = y[:, perms_arr[pi]]
        undecided = np.ones(N, dtype=bool)
        better = np.zeros(N, dtype=bool)

        for j in range(D):
            lt = (yp[:, j] < y_best[:, j]) & undecided
            gt = (yp[:, j] > y_best[:, j]) & undecided
            better[lt] = True
            undecided[lt | gt] = False
            if not np.any(undecided):
                break

        if np.any(better):
            y_best[better] = yp[better]
            which[better] = pi

    return y_best, which


def apply_canonicalization(X: np.ndarray, y: np.ndarray, topology="COCOC"):
    """
    Canonicalize only when the topology is symmetric under reversal.
    COC and COCOC are symmetric; COCO is not.
    """
    if topology is None:
        return X, y

    topology = str(topology).upper()

    symmetric_topologies = {"COC", "COCOC"}

    if topology not in symmetric_topologies:
        print(f"[Canon] Topology={topology} is not symmetric. Skipping canonicalization.", flush=True)
        return X, y

    print(f"[Canon] Applying {topology} symmetry rules...", flush=True)

    D = y.shape[1]
    perms = [
        list(range(D)),               # identity
        list(range(D - 1, -1, -1)),   # reversal
    ]

    y_final, which = canonicalize_by_permutations_batch(y, perms)

    num_swapped = int((which != 0).sum())
    N = y.shape[0]
    print(f"[Canon] Canonicalized {num_swapped}/{N} ({num_swapped/N*100:.1f}%)", flush=True)

    return X, y_final


def build_chain_rate_positions(num_states: int):
    pos = []
    for i in range(num_states - 1):
        pos.append((i, i + 1))
        pos.append((i + 1, i))
    return pos


def load_hist_data(h5_path: str, log10_targets: bool = True, topology: str = "COCOC"):
    with h5py.File(h5_path, "r") as f:
        grids = f["histograms/grids"][:]
        Q = f["labels/Q"][:]

    X_np = grids.astype(np.float32)
    if X_np.ndim == 3:
        X_np = X_np[:, None, :, :]
    elif X_np.ndim == 4 and X_np.shape[-1] == 1 and X_np.shape[1] != 1:
        X_np = np.transpose(X_np, (0, 3, 1, 2))
    elif X_np.ndim != 4:
        raise ValueError(f"Unsupported histogram shape: {X_np.shape}")

    X_np = np.log10(X_np + 1.0).astype(np.float32)

    S = Q.shape[1]
    rate_positions = build_chain_rate_positions(S)
    y_np = np.stack([Q[:, i, j] for (i, j) in rate_positions], axis=1).astype(np.float32)

    X_np, y_np = apply_canonicalization(X_np, y_np, topology=topology)

    if log10_targets:
        y_np = np.log10(np.maximum(y_np, 1e-9)).astype(np.float32)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)
    return X, y


def _fmt_frac(x: float) -> str:
    return f"{x:.3f}".replace(".", "p")


def get_split_seed(cfg):
    return int(getattr(cfg, "split_seed", cfg.seed))


def get_splits_dir(cfg):
    if hasattr(cfg, "splits_dir"):
        p = Path(cfg.splits_dir)
    else:
        p = Path(cfg.h5_path).resolve().parent / "_saved_splits"
    p.mkdir(parents=True, exist_ok=True)
    return p


def split_file_path(cfg, N: int) -> Path:
    stem = Path(cfg.h5_path).stem
    topology = str(getattr(cfg, "topology", "NA"))
    split_seed = get_split_seed(cfg)
    fname = (
        f"{stem}"
        f"__topo_{topology}"
        f"__N_{N}"
        f"__splitseed_{split_seed}"
        f"__test_{_fmt_frac(cfg.test_fraction)}"
        f"__val_{_fmt_frac(cfg.val_fraction)}.npz"
    )
    return get_splits_dir(cfg) / fname


def load_test_split(cfg, N: int):
    path = split_file_path(cfg, N)
    if not path.exists():
        raise FileNotFoundError(f"Saved split not found: {path}")
    data = np.load(path)
    return data["test_idx"]


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj:
        return ckpt_obj["model_state_dict"]
    return ckpt_obj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--h5", type=str, \
                   default="/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_5_100000_kij_fixed_bestmodelof_HOHD_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_hohd_single_10us_100ms_J11_unclamped_2026_04_15.h5")
    p.add_argument("--outdir", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--split-mode", choices=["full", "test"], default="full")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = HistTrainConfig()
    if hasattr(cfg, "apply_env_overrides"):
        cfg = cfg.apply_env_overrides()

    if args.h5 is not None:
        cfg.h5_path = args.h5

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    if not os.path.isfile(cfg.h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {cfg.h5_path}")

    save_dir = args.outdir or os.path.dirname(args.ckpt)
    os.makedirs(save_dir, exist_ok=True)

    topology = getattr(cfg, "topology", "COCOC")
    y_log10 = bool(getattr(cfg, "log10_targets", True))
    use_bn = bool(getattr(cfg, "use_bn", False))
    separate_heads = bool(getattr(cfg, "separate_heads", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] checkpoint : {args.ckpt}")
    print(f"[eval] h5 file    : {cfg.h5_path}")
    print(f"[eval] save dir   : {save_dir}")
    print(f"[eval] device     : {device}")
    print(f"[eval] split mode : {args.split_mode}")

    X, y = load_hist_data(cfg.h5_path, log10_targets=y_log10, topology=topology)
    dataset = TensorDataset(X, y)

    if args.split_mode == "test":
        test_idx = load_test_split(cfg, len(dataset))
        dataset = Subset(dataset, test_idx.tolist())
        print(f"[eval] Using saved test split: {len(test_idx)} samples")
    else:
        print(f"[eval] Using full dataset: {len(dataset)} samples")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    in_channels = int(X.shape[1])
    H, W = int(X.shape[2]), int(X.shape[3])
    num_outputs = int(y.shape[1])

    model = TFExactInceptionResNet(
        in_channels=in_channels,
        num_outputs=num_outputs,
        use_bn=use_bn,
        separate_heads=separate_heads,
        input_hw=(H, W),
    ).to(device)

    ckpt_obj = torch.load(args.ckpt, map_location=device)
    state_dict = extract_state_dict(ckpt_obj)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("[load] checkpoint loaded successfully")

    preds_log_chunks = []
    targs_log_chunks = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb)
        preds_log_chunks.append(pred.cpu())
        targs_log_chunks.append(yb.cpu())

    preds_log = torch.cat(preds_log_chunks, dim=0)
    targs_log = torch.cat(targs_log_chunks, dim=0)

    rae_matrix = rae_log(preds_log, targs_log).numpy()
    rae_per_out = rae_matrix.mean(axis=0)
    rae_overall = float(rae_per_out.mean())

    preds_log_np = preds_log.numpy()
    targs_log_np = targs_log.numpy()

    np.save(os.path.join(save_dir, "predictions_log.npy"), preds_log_np)
    np.save(os.path.join(save_dir, "targets_log.npy"), targs_log_np)

    if y_log10:
        preds_np = 10.0 ** np.clip(preds_log_np.astype(np.float64), -30.0, 30.0)
        targs_np = 10.0 ** np.clip(targs_log_np.astype(np.float64), -30.0, 30.0)
    else:
        preds_np = preds_log_np.astype(np.float64)
        targs_np = targs_log_np.astype(np.float64)

    np.save(os.path.join(save_dir, "predictions_nn.npy"), preds_np)
    np.save(os.path.join(save_dir, "targets_nn.npy"), targs_np)

    csv_path = os.path.join(save_dir, "rae_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rae_mean"])
        for value in rae_matrix.mean(axis=1):
            writer.writerow([float(value)])

    print("Shapes: preds =", preds_np.shape, ", targets =", targs_np.shape)
    print("RAE per output:", " ".join(f"{r:.4f}" for r in rae_per_out))
    print(f"RAE overall: {rae_overall:.6f}")
    print(f"[done] saved: {save_dir}")


if __name__ == "__main__":
    main()