#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import argparse
import csv
import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from scripts.softdwell_pipeline.config import TrainConfig
from scripts.softdwell_pipeline.metrics import LogCoshLoss, rae_log, EarlyStopping
from scripts.softdwell_pipeline.softdwell import ChannelWiseSoftDwell
from scripts.softdwell_pipeline.model import SoftDwellThenTFIR
from scripts.softdwell_pipeline.checkpoint import save_checkpoint, load_checkpoint, is_rank0
from scripts.common.splits import build_or_load_splits
from scripts.common.canonicalize import choose_canonicalizer
from scripts.softdwell_pipeline.softdwell_logging import log_sd_params_epoch
from scripts.common.ddp_utils import (
    ddp_setup,
    cleanup_ddp,
    is_main_process,
    rprint,
    ddp_allreduce_sum,
    gather_concat_varlen,
)
from scripts.common.h5_ram_loader import parallel_hdf5_load_xy
from scripts.common.inmemory_dataset import InMemoryDataset

# ---- cuDNN / matmul tuning ----
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# -------------------- WORKER: runs on each GPU -------------------- #

def train_worker(rank: int, world_size: int, cfg: TrainConfig, X: torch.Tensor, y: torch.Tensor):
    """
    Single worker process for DDP. Rank identifies the GPU (0..world_size-1).
    X, y are shared-memory tensors created by the parent.
    """
    # ---- 1) Setup env vars so ddp_setup() can use env:// ----
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize DDP and device via your existing helper
    is_ddp, rank, world_size, local_rank, device = ddp_setup()
    assert is_ddp, "train_worker should run in DDP mode"

    os.makedirs(cfg.output_root, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    # ---------------- Dataset & splits (from shared RAM) ---------------- #
    # X, y are already canonicalized + log10 (if requested) in the parent.
    full = InMemoryDataset(X, y)

    # info about canonicalizer only for logging
    canon = choose_canonicalizer(cfg.topology, cfg.symmetry)
    if is_rank0():
        if canon is None:
            print(
                f"[canon] disabled (topology={cfg.topology} symmetry={cfg.symmetry})",
                flush=True,
            )
        else:
            print(
                f"[canon] enabled (topology={cfg.topology} symmetry={cfg.symmetry})",
                flush=True,
            )
        print(
            f"[cfg] y_log10={cfg.y_log10} "
            f"topology={cfg.topology} symmetry={cfg.symmetry}",
            flush=True,
        )

    N = len(full)
    # Splits must be consistent across ranks; build_or_load_splits handles that
    train_idx, val_idx, test_idx = build_or_load_splits(cfg, N, rank)

    train_ds = Subset(full, train_idx.tolist())
    val_ds   = Subset(full, val_idx.tolist())
    test_ds  = Subset(full, test_idx.tolist())

    # ---------------- DataLoaders ---------------- #
    print(f"[rank {rank}] device {device.type}", flush=True)

    common_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,  # can be small (0–2) since data is in RAM
        pin_memory=False,
        persistent_workers=False,
    )
    base_seed = cfg.split_seed

    if is_ddp:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=base_seed,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=base_seed,
        )
        test_sampler = DistributedSampler(
            test_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
            seed=base_seed,
        )
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(
        train_ds, shuffle=(not is_ddp), sampler=train_sampler, drop_last=True, **common_loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **common_loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, sampler=test_sampler, drop_last=False, **common_loader_kwargs
    )

    # sanity check batch
    _xb, _yb = next(iter(train_loader))

    def safe_minmax(t: torch.Tensor):
        m = torch.isfinite(t)
        if not m.any():
            return float("nan"), float("nan")
        t = t[m]
        return float(torch.amin(t).cpu()), float(torch.amax(t).cpu())

    xb_fin = torch.isfinite(_xb).all().item()
    yb_fin = torch.isfinite(_yb).all().item()
    yb_min, yb_max = safe_minmax(_yb)

    rprint(rank, f"[data] xb finite={xb_fin} yb finite={yb_fin} yb min={yb_min} yb max={yb_max}")

    if is_main_process(rank):
        print(
            f"[info] train_size={len(train_ds)} | world_size={world_size} | "
            f"perGPU_batch={cfg.batch_size} | steps_per_epoch={len(train_loader)}",
            flush=True,
        )

    # fixed 1-sample monitor batch (CPU -> GPU later)
    xb_mon_cpu = _xb[0:1].contiguous()

    # ---------------- Model ---------------- #
    softdwell = ChannelWiseSoftDwell(
        num_detectors=cfg.num_detectors,
        dwell_min=cfg.dwell_min,
        dwell_max=cfg.dwell_max,
        num_bins=cfg.num_bins,
        average_segments=False,
        norm_min=cfg.norm_min,
        norm_max=cfg.norm_max,
        eps=1e-6,
        theta_init_min=cfg.theta_init_min,
        theta_init_max=cfg.theta_init_max,
        tau_init=cfg.tau_init,
        dtype=torch.float32,
        timing=False,
        logN_gamma=cfg.logN_gamma,
    ).to(device)



    if is_rank0():
        print(
            f"[cfg] h5_path={cfg.h5_path}\n"
            f"[cfg] output_root={cfg.output_root} "
            f"ckpt_dir={cfg.ckpt_dir} resume={cfg.resume}",
            flush=True,
        )

    # ---- Infer (K, Hh, Wh) once, before building the wrapper/backbone ----
    with torch.no_grad():
        fake_ts = _xb[0:1].to(device, non_blocking=True).float()   # [1, C, T]
        H_fake = softdwell(fake_ts)                                # [1, K, Hh, Wh]
        K = H_fake.shape[1]
        hist_hw = (H_fake.shape[2], H_fake.shape[3])

    model = SoftDwellThenTFIR(
        softdwell,
        K=K,
        hist_hw=hist_hw,
        num_outputs=cfg.num_outputs,
        use_bn=False,
        separate_heads=False,
    ).to(device).to(memory_format=torch.channels_last)
    

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    core = model.module if is_ddp else model

    csv_out = os.path.join(cfg.output_root, "softdwell_params.csv")
    if is_main_process(rank):
        need_header = (not os.path.exists(csv_out)) or (os.path.getsize(csv_out) == 0)
        if need_header:
            log_sd_params_epoch(core, -1, csv_path=csv_out, stats_only=False)

    crit = LogCoshLoss()

    # ---- split params: softdwell (theta/tau) vs others ----
    sd_params, bb_params = [], []
    for n, p in core.named_parameters():
        if not p.requires_grad:
            continue
        (sd_params if "softdwell" in n else bb_params).append(p)

    groups = [
        {"params": bb_params, "lr": cfg.lr, "weight_decay": 0.0},
        {
            "params": sd_params,
            "lr": getattr(cfg, "lr_sd", cfg.lr),
            "weight_decay": 0.0,
        },
    ]
    opt = torch.optim.Adam(groups)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.1,
        patience=4,
        min_lr=1e-6,
    )

    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # ---- RESUME (shared ckpt dir, rank0 picks) ----
    resume_path = cfg.resume
    if resume_path == "auto":
        os.makedirs(cfg.ckpt_dir, exist_ok=True)
        pts = [
            os.path.join(cfg.ckpt_dir, f)
            for f in os.listdir(cfg.ckpt_dir)
            if f.endswith(".pt")
        ]
        resume_path = max(pts, key=os.path.getmtime) if pts else ""
    if resume_path and os.path.isfile(resume_path):
        if is_rank0():
            print(f"[rank0] Resuming from {os.path.abspath(resume_path)}", flush=True)

        start_epoch, global_step, extra = load_checkpoint(
            resume_path, model, opt, scheduler
        )
        if "earlystop" in extra:
            es.load_state_dict(extra["earlystop"])
        start_epoch += 1  # next epoch
    else:
        start_epoch, global_step = 0, 0

    es.patience = cfg.patience
    es.min_delta = cfg.min_delta

    best_val = es.best

    # ----------------- Train ----------------- #
    best_path = os.path.join(cfg.output_root, cfg.best_model_filename)

    if is_rank0():
        print(f"[best path] {os.path.abspath(best_path)}", flush=True)

    for epoch in range(start_epoch, cfg.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_epoch0 = time.perf_counter()

        if is_ddp and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        model.train()
        tr_loss_sum = torch.zeros(1, device=device)
        n_tr_samples = torch.zeros(1, device=device, dtype=torch.long)

        PROFILE = False  # keep profiling for now

        epoch_load = 0.0
        epoch_cpu_gpu = 0.0
        epoch_fwd = 0.0
        epoch_bwd = 0.0
        epoch_step = 0.0
        num_steps = 0

        if PROFILE:
            train_iter = iter(train_loader)
            num_steps = len(train_loader)

            for batch_idx in range(num_steps):
                # ---- DATALOADER (CPU/RAM only) ----
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                xb_cpu, yb_cpu = next(train_iter)
                data_time = time.perf_counter() - t0
                epoch_load += data_time

                # ---- CPU -> GPU ----
                t0 = time.perf_counter()
                xb = xb_cpu.to(device, non_blocking=True).float()
                yb = yb_cpu.to(device, non_blocking=True).float()
                cpu_gpu_time = time.perf_counter() - t0
                epoch_cpu_gpu += cpu_gpu_time

                bs = yb.size(0)
                opt.zero_grad(set_to_none=True)

                # ---- FWD ----
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                pred = model(xb)
                loss = crit(pred, yb)
                torch.cuda.synchronize(device)
                fwd_time = time.perf_counter() - t0
                epoch_fwd += fwd_time

                tr_loss_sum += loss.detach() * bs
                n_tr_samples += bs

                # ---- BWD ----
                t0 = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize(device)
                bwd_time = time.perf_counter() - t0
                epoch_bwd += bwd_time

                # ---- STEP ----
                t0 = time.perf_counter()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                torch.cuda.synchronize(device)
                step_time = time.perf_counter() - t0
                epoch_step += step_time

                global_step += 1

                if batch_idx % 5 == 0 and is_main_process(rank):
                    print(
                        f"[epoch {epoch+1} batch {batch_idx}] "
                        f"load={data_time:.4f}s | "
                        f"cpu->gpu={cpu_gpu_time:.4f}s | "
                        f"fwd={fwd_time:.4f}s | "
                        f"bwd={bwd_time:.4f}s | "
                        f"step={step_time:.4f}s",
                        flush=True,
                    )

            if is_main_process(rank):
                steps = max(1, num_steps)
                print(
                    f"[timing][epoch {epoch+1}] "
                    f"avg_load={epoch_load/steps:.6f}s | "
                    f"avg_cpu2gpu={epoch_cpu_gpu/steps:.6f}s | "
                    f"avg_fwd={epoch_fwd/steps:.6f}s | "
                    f"avg_bwd={epoch_bwd/steps:.6f}s | "
                    f"avg_step={epoch_step/steps:.6f}s",
                    flush=True,
                )

        else:
            for batch_idx, (xb_cpu, yb_cpu) in enumerate(train_loader):
                num_steps += 1
                xb = xb_cpu.to(device, non_blocking=True).float()
                yb = yb_cpu.to(device, non_blocking=True).float()
                bs = yb.size(0)

                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = crit(pred, yb)
                tr_loss_sum += loss.detach() * bs
                n_tr_samples += bs

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                global_step += 1

        ddp_allreduce_sum(tr_loss_sum)
        ddp_allreduce_sum(n_tr_samples)
        avg_train_loss_global = (tr_loss_sum / n_tr_samples.clamp_min(1)).item()

        # ----------------- Validate ----------------- #
        model.eval()
        val_loss_sum = torch.zeros(1, device=device)
        rae_sum = torch.zeros(cfg.num_outputs, device=device)
        n_val_samples = torch.zeros(1, device=device, dtype=torch.long)

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).float()
                yb = yb.to(device, non_blocking=True).float()
                bsz = yb.size(0)

                pred = model(xb)
                loss = crit(pred, yb)

                if not torch.isfinite(pred).all():
                    if is_main_process(rank):
                        print(
                            "[val][warn] non-finite predictions; max|pred|=",
                            float(pred.abs().max()),
                        )
                    continue

                if not torch.isfinite(loss):
                    if is_main_process(rank):
                        mx = float((pred - yb).abs().max())
                        print(
                            f"[val][warn] non-finite loss; max|residual|={mx}. Skipping batch."
                        )
                    continue

                val_loss_sum += loss.detach() * bsz
                rae_sum += rae_log(pred, yb).sum(dim=0)
                n_val_samples += bsz

        ddp_allreduce_sum(val_loss_sum)
        ddp_allreduce_sum(rae_sum)
        ddp_allreduce_sum(n_val_samples)

        den = n_val_samples.clamp_min(1).float()
        val_loss_global = (val_loss_sum / den).item()
        val_rae_global = (rae_sum / den).detach().cpu().tolist()

        if scheduler is not None:
            scheduler.step(val_loss_global)
            opt.param_groups[1]["lr"] = cfg.lr_sd
            if is_main_process(rank):
                cur_lrs = [g["lr"] for g in opt.param_groups]
                print(f"[lr] after epoch {epoch+1}: {cur_lrs}", flush=True)         

        if is_main_process(rank):
            log_sd_params_epoch(core, epoch, csv_path=csv_out, stats_only=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t_epoch0

        rprint(
            rank,
            f"Epoch {epoch+1} ({epoch_sec:.2f}s): "
            f"Train {avg_train_loss_global:.6f} | "
            f"Val(glob) {val_loss_global:.6f} | "
            f"Val RAE per out: {' '.join(f'{r:.4f}' for r in val_rae_global)}",
        )

        if is_main_process(rank):
            metrics_path = os.path.join(cfg.output_root, "metrics.csv")
            need_header = not os.path.exists(metrics_path) or os.path.getsize(metrics_path) == 0
            with open(metrics_path, "a") as f:
                if need_header:
                    f.write("epoch,train_loss,val_loss\n")
                f.write(f"{epoch+1},{avg_train_loss_global},{val_loss_global}\n")

        stop_now = es.step(val_loss_global)

        if is_main_process(rank): 
            if(val_loss_global < best_val - cfg.min_delta):
                print(
                    f"[best val updated] {best_val:.6f} -> {val_loss_global:.6f} "
                    f"at epoch {epoch+1}",
                    flush=True,
                )
                best_val = val_loss_global
                torch.save((model.module if is_ddp else model).state_dict(), best_path)

            print(
                f"[ES] Status Epoch {epoch+1} | "
                f"Curr: {val_loss_global:.6f} | "
                f"Best: {es.best:.6f} | "
                f"Count: {es.count}/{es.patience}",
                flush=True
            )
            
            ckpt_path_epoch = os.path.join(cfg.ckpt_dir, f"epoch-{epoch+1:04d}.pt")
            save_checkpoint(
                ckpt_path_epoch,
                model,
                opt,
                scheduler,
                epoch,
                global_step,
                cfg,
                extra={"earlystop": es.state_dict()},
            )

        if stop_now:
            rprint(rank, f"Early stopping at epoch {epoch+1}, best val {best_val:.6f}")
            break

    # ----------------- Test (best) ----------------- #
    state = torch.load(best_path, map_location=device)
    missing, unexpected = (model.module if is_ddp else model).load_state_dict(
        state, strict=False
    )
    rprint(rank, "[load] missing:", missing)
    rprint(rank, "[load] unexpected:", unexpected)

    model.eval()
    pred_chunks, targ_chunks = [], []
    rae_sum_local = torch.zeros(cfg.num_outputs, device=device)
    n_tot_local = 0

    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            pred = model(xb)
            pred_chunks.append(pred.detach())
            targ_chunks.append(yb.detach())
            rae_sum_local += rae_log(pred, yb).sum(dim=0)
            n_tot_local += yb.size(0)

    rae_sum_global = ddp_allreduce_sum(rae_sum_local.clone())
    n_tot_global = ddp_allreduce_sum(
        torch.tensor([n_tot_local], device=device, dtype=torch.long)
    )
    test_rae_per = (rae_sum_global / n_tot_global.clamp_min(1)).detach().cpu().numpy()
    test_rae_overall = float(test_rae_per.mean())
    rprint(
        rank,
        "Test RAE per output:",
        " ".join(f"{r:.4f}" for r in test_rae_per),
        "| Overall:",
        f"{test_rae_overall:.4f}",
    )

    pred_local = (
        torch.cat(pred_chunks, dim=0)
        if pred_chunks
        else torch.empty(0, cfg.num_outputs, device=device)
    )
    targ_local = (
        torch.cat(targ_chunks, dim=0)
        if targ_chunks
        else torch.empty(0, cfg.num_outputs, device=device)
    )

    pred_all = gather_concat_varlen(pred_local, dim=0)
    targ_all = gather_concat_varlen(targ_local, dim=0)

    if is_main_process(rank):
        preds_out = pred_all.detach().cpu().numpy()
        targs_out = targ_all.detach().cpu().numpy()
        if cfg.y_log10:
            preds_out = 10**preds_out
            targs_out = 10**targs_out
        print("Predictions range:", preds_out.min(), preds_out.max())
        print("Targets range:", targs_out.min(), targs_out.max())
        np.save(os.path.join(cfg.output_root, cfg.predictions_filename), preds_out)
        np.save(os.path.join(cfg.output_root, cfg.targets_filename), targs_out)
        rprint(
            rank,
            f"[save] predictions -> {cfg.predictions_filename}, "
            f"targets -> {cfg.targets_filename}",
        )

    # clean up dist
    cleanup_ddp(is_ddp=True)


# -------------------- PARENT: load once, spawn workers -------------------- #
def main():
    import gc  # <--- add this at the top of the file or inside main

    cfg = TrainConfig().apply_env_overrides()

    parser = argparse.ArgumentParser(description="SoftDwell Trainig")
    parser.add_argument("--res", type=int, default=11, help="Resolution setting")
    parser.add_argument("--k", type=int, default=4, help="Number of Detectors")
    parser.add_argument("--bs", type=int, default=32, help="Batch Size")
    parser.add_argument("--exp-name", type=str, required=True, help="Experiment name for output directories")

    args, unknown = parser.parse_known_args()

    cfg.num_bins = args.res
    cfg.num_detectors = args.k
    cfg.batch_size = args.bs


    if is_rank0():
        print(f"[config] Overrides: res={args.res} k={args.k} bs={args.bs}")
        print(f"[config] output_root={cfg.output_root}")

    canon = choose_canonicalizer(cfg.topology, cfg.symmetry)
 
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    print(f"[parent] world_size={world_size}", flush=True)

    # Required for env://
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    print(f"[cfg] h5_path={cfg.h5_path}", flush=True)
    print("[RAM] Loading full dataset into memory (parent only)...", flush=True)
    t0 = time.perf_counter()

    X_np, y_np = parallel_hdf5_load_xy(
        path=cfg.h5_path,
        x_key=cfg.x_key,
        y_key=cfg.y_key,
        y_log10=cfg.y_log10,
        canonicalizer=canon,
        y_from_Q=cfg.y_from_Q,
        limit_n=(cfg.limit_n if cfg.limit_n > 0 else None),
        num_workers=4,
        chunk_size=1024,
        t_start=cfg.t_start,
        t_len=cfg.t_len
    )

    t_load = time.perf_counter() - t0
    print(f"[RAM] Loaded dataset in {t_load:.2f}s", flush=True)
    print(f"[RAM] X_np={X_np.shape}, y_np={y_np.shape}", flush=True)

    # ---- Convert to shared-memory tensors for mp.spawn workers ----
    # First create normal tensors that share storage with the NumPy arrays
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    # Move their storage into shared memory so all spawned workers see the same data
    X.share_memory_()
    y.share_memory_()

    # ---- Now it's safe to drop the NumPy arrays ----
    # The tensor storage is now owned by PyTorch; X_np/y_np objects can go.
    del X_np
    del y_np
    gc.collect()
    print("[RAM] Dropped NumPy arrays after sharing tensors.", flush=True)

    # Now spawn DDP workers; each gets a handle to the same shared storage
    mp.spawn(
        train_worker,
        nprocs=world_size,
        args=(world_size, cfg, X, y),
    )


if __name__ == "__main__":
    main()
