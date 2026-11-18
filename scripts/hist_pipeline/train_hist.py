#!/usr/bin/env python3
import os
import time
import h5py
import argparse
import numpy as np
import csv
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path

# --- Imports from your project ---
from scripts.hist_pipeline.config_hist import HistTrainConfig
from scripts.softdwell_pipeline.model import TFExactInceptionResNet
from scripts.softdwell_pipeline.metrics import LogCoshLoss, rae_log, EarlyStopping

# -------------------------------------------------------------------------
# HELPER: Canonicalization Logic
# -------------------------------------------------------------------------
def canonicalize_by_permutations_batch(
    y: np.ndarray,
    perms: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each row y[i], return lexicographically smallest among y[i][p] for p in perms.
    Returns:
      y_best: (N, D)
      which:  (N,) permutation index chosen
    """
    y = np.asarray(y)
    assert y.ndim == 2
    N, D = y.shape
    assert len(perms) >= 1

    perms_arr = [np.asarray(p, dtype=np.int64) for p in perms]
    for p in perms_arr:
        assert p.shape == (D,), f"perm must have length {D}"

    # initialize with first perm
    y_best = y[:, perms_arr[0]].copy()
    which = np.zeros(N, dtype=np.int64)

    for pi in range(1, len(perms_arr)):
        yp = y[:, perms_arr[pi]]  # (N, D)

        undecided = np.ones(N, dtype=bool)
        better = np.zeros(N, dtype=bool)

        # lex compare yp < y_best (row-wise)
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
# -------------------------------------------------------------------------
# 1. DATA LOADING FUNCTION (Runs once in Parent)
# -------------------------------------------------------------------------
def build_chain_rate_positions(num_states: int):
    # Adjacent chain rates: (0<->1), (1<->2), ...
    pos = []
    for i in range(num_states - 1):
        pos.append((i, i + 1))
        pos.append((i + 1, i))
    return pos

def load_data_to_ram(h5_path, log10_targets=True, topology="COCOC"):
    print(f"[RAM] Opening {h5_path}...", flush=True)
    with h5py.File(h5_path, "r") as f:
        grids = f["histograms/grids"][:]
        Q = f["labels/Q"][:]              # (N, S, S)

    N = grids.shape[0]
    S = Q.shape[1]
    assert Q.shape[1] == Q.shape[2], f"Q must be square, got {Q.shape}"
    print(f"[RAM] Raw loaded. grids={grids.shape}, Q={Q.shape} (S={S})", flush=True)

    # Ensure inputs are (N, 1, H, W)
    X_np = grids.astype(np.float32)
    if X_np.ndim == 4 and X_np.shape[-1] == 1 and X_np.shape[1] != 1:
        # likely (N, H, W, 1) -> (N, 1, H, W)
        X_np = np.transpose(X_np, (0, 3, 1, 2))

    # Histogram scaling: log10(x+1)
    X_np = np.log10(X_np + 1.0).astype(np.float32)

    # Targets: chain adjacent rates, size = 2*(S-1)
    rate_positions = build_chain_rate_positions(S)
    y_vals = [Q[:, i, j] for (i, j) in rate_positions]
    y_np = np.stack(y_vals, axis=1).astype(np.float32)

    # Canonicalize only if topology supports it (COCOC), otherwise no-op
    X_canon, y_canon = apply_canonicalization(X_np, y_np, topology=topology)

    if log10_targets:
        eps = 1e-9
        y_canon = np.log10(np.maximum(y_canon, eps)).astype(np.float32)

    X = torch.from_numpy(X_canon)
    y = torch.from_numpy(y_canon)

    # Share memory so mp.spawn workers access same RAM
    X.share_memory_()
    y.share_memory_()

    print(f"[RAM] Data ready. X: {X.shape}, y: {y.shape} (Shared)", flush=True)
    return X, y

def _fmt_frac(x: float) -> str:
    return f"{x:.3f}".replace(".", "p")


def get_split_seed(cfg):
    # If cfg.split_seed exists, use it; otherwise fall back to cfg.seed
    return int(getattr(cfg, "split_seed", cfg.seed))


def get_splits_dir(cfg):
    """
    IMPORTANT:
    This directory must be stable across runs.
    Do NOT put it inside a seed-dependent output directory.
    """
    if hasattr(cfg, "splits_dir"):
        p = Path(cfg.splits_dir)
    else:
        # default: stable folder next to the dataset file
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


def build_or_load_splits(cfg, N: int) -> str:
    """
    Save/load deterministic train/val/test indices for this dataset.
    Returns path to the .npz file.
    """
    path = split_file_path(cfg, N)

    if path.exists():
        data = np.load(path)
        train_idx = data["train_idx"]
        val_idx = data["val_idx"]
        test_idx = data["test_idx"]

        total = len(train_idx) + len(val_idx) + len(test_idx)
        if total == N:
            print(f"[Split] Loaded existing split: {path}", flush=True)
            print(
                f"[Split] Sizes -> train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
                flush=True
            )
            return str(path)

        print(f"[Split][warn] Existing split has wrong size for N={N}. Rebuilding...", flush=True)

    split_seed = get_split_seed(cfg)
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(N).astype(np.int64)

    n_test = int(N * cfg.test_fraction)
    n_val = int((N - n_test) * cfg.val_fraction)
    n_train = N - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    np.savez(
        path,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        N=np.int64(N),
        n_train=np.int64(n_train),
        n_val=np.int64(n_val),
        n_test=np.int64(n_test),
        split_seed=np.int64(split_seed),
        test_fraction=np.float32(cfg.test_fraction),
        val_fraction=np.float32(cfg.val_fraction),
    )

    print(f"[Split] Saved new split: {path}", flush=True)
    print(
        f"[Split] Sizes -> train={n_train}, val={n_val}, test={n_test} | split_seed={split_seed}",
        flush=True
    )
    return str(path)

def init_metrics_csv(csv_path: str, num_outputs: int):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["epoch", "epoch_time_s", "train_loss", "val_loss", "lr"]
        header += [f"val_rae_out_{i+1}" for i in range(num_outputs)]
        writer.writerow(header)


def append_metrics_csv(
    csv_path: str,
    epoch: int,
    epoch_time_s: float,
    train_loss: float,
    val_loss: float,
    lr: float,
    val_rae_per: list,
):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        row = [epoch, epoch_time_s, train_loss, val_loss, lr] + list(val_rae_per)
        writer.writerow(row)
# -------------------------------------------------------------------------
# 2. WORKER FUNCTION (Runs on each GPU)
# -------------------------------------------------------------------------
def train_worker(rank, world_size, cfg, X_shared, y_shared, split_path):
    """
    The main training loop for a single GPU.
    """
    # -- A. Environment Setup --
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Optimization flags
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_master = (rank == 0)
    if is_master:
        print(f"[Worker 0] Initialized DDP world_size={world_size}")
        os.makedirs(cfg.output_root, exist_ok=True)

    # -- B. Dataset from Shared Tensors --
    full_dataset = TensorDataset(X_shared, y_shared)

    # Fixed split loaded from disk
    split_data = np.load(split_path)
    train_idx = split_data["train_idx"]
    val_idx = split_data["val_idx"]
    test_idx = split_data["test_idx"]

    train_ds = Subset(full_dataset, train_idx)
    val_ds = Subset(full_dataset, val_idx)
    test_ds = Subset(full_dataset, test_idx)

    if is_master:
        print(
            f"[Worker 0] Using split file: {split_path} | "
            f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            flush=True
        )

    # -- C. Loaders --
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    loader_kwargs = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=True)

    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)

    # Infer shapes
    in_channels = int(X_shared.shape[1])
    H, W = int(X_shared.shape[2]), int(X_shared.shape[3])
    num_outputs = int(y_shared.shape[1])

    # CSV path for per-epoch metrics
    metrics_csv = os.path.join(cfg.output_root, "train_val_metrics.csv")
    if is_master:
        init_metrics_csv(metrics_csv, num_outputs)

    # -- D. Model --
    model = TFExactInceptionResNet(
        in_channels=in_channels,
        num_outputs=num_outputs,
        use_bn=False,
        separate_heads=False,
        input_hw=(H, W),
    ).to(device)

    model = DDP(model, device_ids=[rank])

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    crit = LogCoshLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.1,
        patience=4,
        min_lr=1e-6,
    )

    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta, restore_best=True)

    best_val = float("inf")
    best_path = os.path.join(cfg.output_root, "best_model.pt")

    # -- E. Training Loop --
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_sampler.set_epoch(epoch)

        # ----------------- Train -----------------
        model.train()
        tr_loss_accum = torch.zeros(1, device=device)
        tr_count = torch.zeros(1, device=device)

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            tr_loss_accum += loss.detach() * xb.size(0)
            tr_count += xb.size(0)

        dist.all_reduce(tr_loss_accum)
        dist.all_reduce(tr_count)
        train_loss = (tr_loss_accum / tr_count).item()

        # ----------------- Val -----------------
        model.eval()

        val_loss_sum = torch.zeros(1, device=device)
        val_rae_sum = torch.zeros(num_outputs, device=device)
        val_count = torch.zeros(1, device=device, dtype=torch.long)

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                bsz = yb.size(0)

                pred = model(xb)
                loss = crit(pred, yb)

                if (not torch.isfinite(pred).all()) or (not torch.isfinite(loss)):
                    if is_master:
                        mx = float(torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0).abs().max())
                        print(f"[val][warn] non-finite pred/loss; max|pred|={mx}. Skipping batch.", flush=True)
                    continue

                val_loss_sum += loss.detach() * bsz
                val_rae_sum += rae_log(pred, yb).sum(dim=0)
                val_count += bsz

        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_rae_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_count, op=dist.ReduceOp.SUM)

        den = val_count.clamp_min(1).float()
        val_loss = (val_loss_sum / den).item()
        val_rae_per = (val_rae_sum / den).detach().cpu().tolist()

        scheduler.step(val_loss)

        dt = time.time() - t0
        current_lr = opt.param_groups[0]["lr"]

        if is_master:
            print(
                f"[Epoch {epoch:03d}] {dt:.2f}s | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Val RAE per out: {' '.join(f'{r:.4f}' for r in val_rae_per)}",
                flush=True
            )

            append_metrics_csv(
                csv_path=metrics_csv,
                epoch=epoch,
                epoch_time_s=dt,
                train_loss=train_loss,
                val_loss=val_loss,
                lr=current_lr,
                val_rae_per=val_rae_per,
            )

        # ----------------- Early stopping + checkpoint -----------------
        stop_local = False
        if is_master:
            stop_local = es.step(val_loss, model.module, epoch)

            if val_loss < best_val - cfg.min_delta:
                best_val = val_loss
                torch.save(model.module.state_dict(), best_path)
                print(f"  -> Saved best model (val={best_val:.6f})", flush=True)

        stop_flag = torch.tensor([1 if stop_local else 0], device=device, dtype=torch.uint8)
        dist.broadcast(stop_flag, src=0)

        if stop_flag.item() == 1:
            if is_master:
                print(f"[Early Stopping] No improvement for {cfg.patience} epochs. Stopping.", flush=True)
            break

    # -- F. Testing (Master Only) --
    if is_master:
        print(f"\n[Test] Loading best model...")

        model = TFExactInceptionResNet(
            in_channels=in_channels,
            num_outputs=num_outputs,
            use_bn=False,
            separate_heads=False,
            input_hw=(H, W),
        ).to(device)

        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        test_loader = DataLoader(
            test_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=0, pin_memory=True
        )

        preds_log_parts, targs_log_parts = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred_log = model(xb)

                preds_log_parts.append(pred_log.cpu())
                targs_log_parts.append(yb.cpu())

        preds_log = torch.cat(preds_log_parts).numpy()
        targs_log = torch.cat(targs_log_parts).numpy()

        y_is_log10 = True

        if y_is_log10:
            preds_log = np.clip(preds_log.astype(np.float64), -30.0, 30.0)
            targs_log = np.clip(targs_log.astype(np.float64), -30.0, 30.0)
            preds_hz = 10.0 ** preds_log
            targs_hz = 10.0 ** targs_log
        else:
            preds_hz = preds_log.astype(np.float64)
            targs_hz = targs_log.astype(np.float64)

        np.save(os.path.join(cfg.output_root, "preds_test.npy"), preds_hz)
        np.save(os.path.join(cfg.output_root, "targs_test.npy"), targs_hz)

        rmse_hz = np.sqrt(np.mean((preds_hz - targs_hz) ** 2))
        mae_hz = np.mean(np.abs(preds_hz - targs_hz))
        rel_err = np.mean(np.abs(preds_hz - targs_hz) / np.maximum(targs_hz, 1e-12))
        print(f"[Test] RMSE (Hz): {rmse_hz:.6e} | MAE (Hz): {mae_hz:.6e} | Mean Rel Err: {rel_err:.6f}")

    dist.destroy_process_group()

# -------------------------------------------------------------------------
# 3. PARENT MAIN
# -------------------------------------------------------------------------
def main():
    cfg = HistTrainConfig()
    # Optional overrides (e.g. to train on a freshly simulated file)
    cfg.h5_path = os.environ.get("HIST_H5_PATH", cfg.h5_path)
    cfg.topology = os.environ.get("HIST_TOPOLOGY", cfg.topology)
    cfg.epochs = int(os.environ.get("HIST_EPOCHS", cfg.epochs))
    if "HIST_OUTPUT_ROOT" in os.environ:
        cfg.output_root = os.environ["HIST_OUTPUT_ROOT"]
        cfg.splits_dir = os.path.join(cfg.output_root, "splits")
    cfg.output_root = os.path.join(cfg.output_root, f"seed{cfg.seed}")

    # 1. Load data into parent RAM
    X_shared, y_shared = load_data_to_ram(
        cfg.h5_path,
        log10_targets=True,
        topology=cfg.topology
    )

    # 2. Build/load a fixed split ONCE for this dataset
    N = int(X_shared.shape[0])
    split_path = build_or_load_splits(cfg, N)

    # 3. Check GPUs
    world_size = torch.cuda.device_count()
    if world_size < 1:
        print("No GPUs found!")
        return
    print(f"[Parent] Spawning {world_size} workers...", flush=True)

    # 4. Spawn
    mp.spawn(
        train_worker,
        nprocs=world_size,
        args=(world_size, cfg, X_shared, y_shared, split_path),
        join=True
    )

if __name__ == "__main__":
    main()