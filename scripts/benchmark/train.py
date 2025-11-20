#!/usr/bin/env python3
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from scripts.benchmark.config import TrainConfig
from scripts.benchmark.metrics import LogCoshLoss, rae_log, EarlyStopping
from scripts.common.hdf5_dataset import HDF5IonChannelDataset
from scripts.benchmark.softdwell import ChannelWiseSoftDwell
from scripts.benchmark.model import SoftDwellThenTFIR
from scripts.benchmark.checkpoint import save_checkpoint, load_checkpoint, is_rank0
from scripts.common.splits import build_or_load_splits
from scripts.common.canonicalize import choose_canonicalizer
from scripts.benchmark.softdwell_logging import log_sd_params_epoch
from scripts.common.ddp_utils import (
    ddp_setup,
    cleanup_ddp,
    is_main_process,
    rprint,
    ddp_allreduce_sum,
    gather_concat_varlen,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def main():
    cfg = TrainConfig().apply_env_overrides()

    if is_rank0():
        print(
            f"[cfg] output_root={cfg.output_root} "
            f"ckpt_dir={cfg.ckpt_dir} resume={cfg.resume}",
            flush=True,
        )

    os.makedirs(cfg.output_root, exist_ok=True)
    os.makedirs(cfg.ckpt_dir, exist_ok=True)

    is_ddp, rank, world_size, local_rank, device = ddp_setup()

    # Dataset & splits
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
    train_idx, val_idx, test_idx = build_or_load_splits(cfg, N, rank)

    train_ds = Subset(full, train_idx.tolist())
    val_ds = Subset(full, val_idx.tolist())
    test_ds = Subset(full, test_idx.tolist())

    # Loaders
    common_loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
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
        train_ds, shuffle=(not is_ddp), sampler=train_sampler, **common_loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, shuffle=False, sampler=val_sampler, drop_last=False, **common_loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, shuffle=False, sampler=test_sampler, drop_last=False, **common_loader_kwargs
    )

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

    # Model
    softdwell = ChannelWiseSoftDwell(
        num_detectors=cfg.num_detectors,
        dwell_min=cfg.dwell_min,
        dwell_max=cfg.dwell_max,
        num_bins=cfg.num_bins,
        length_ratio=cfg.length_ratio,
        average_segments=True,
        norm_min=cfg.norm_min,
        norm_max=cfg.norm_max,
        eps=1e-6,
        theta_init_min=cfg.theta_init_min,
        theta_init_max=cfg.theta_init_max,
        tau_init=cfg.tau_init,
        dtype=torch.float32,
        timing=False,
    ).to(device)

    model = SoftDwellThenTFIR(
        softdwell,
        num_outputs=cfg.num_outputs,
        use_bn=False,
    ).to(device).to(memory_format=torch.channels_last)

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    core = model.module if is_ddp else model
    # use one sample from the batch we already fetched
    fake = _xb[0:1].to(device).float()      # shape [1, C, T]
    was_training = core.training
    core.eval()
    with torch.no_grad():
        _ = core(fake)                      # runs softdwell + builds backbone
    core.train(was_training)

    csv_out = os.path.join(cfg.output_root, "softdwell_params.csv")
    if is_main_process(rank):
        need_header = (not os.path.exists(csv_out)) or (os.path.getsize(csv_out) == 0)
        if need_header:
            log_sd_params_epoch(core, -1, csv_path=csv_out, stats_only=False)

    crit = LogCoshLoss()
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)

    if (not is_ddp) or rank == 0:
        n_params = sum(p.numel() for p in params)
        print(f"[opt] single group | trainable params: {n_params}", flush=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.1,
        patience=4,
        min_lr=1e-6,
    )

    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)

    # ---- RESUME ----
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

    # ----------------- Train -----------------
    best_val = float("inf")
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

        for xb_cpu, yb_cpu in train_loader:
            xb = xb_cpu.to(device, non_blocking=True).float()
            yb = yb_cpu.to(device, non_blocking=True).float()
            bs = yb.size(0)

            opt.zero_grad(set_to_none=True)

            # ---------- FWD ----------
            pred = model(xb)
            loss = crit(pred, yb)

            tr_loss_sum += loss.detach() * bs
            n_tr_samples += bs

            # ---------- BACKWARD ----------
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            global_step += 1

        ddp_allreduce_sum(tr_loss_sum)
        ddp_allreduce_sum(n_tr_samples)
        avg_train_loss_global = (tr_loss_sum / n_tr_samples.clamp_min(1)).item()

        # ----------------- Validate -----------------
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
                loss = crit(pred, yb)  # mean over batch

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

        # Reduce across ranks (SUMs)
        ddp_allreduce_sum(val_loss_sum)
        ddp_allreduce_sum(rae_sum)
        ddp_allreduce_sum(n_val_samples)

        den = n_val_samples.clamp_min(1).float()
        val_loss_global = (val_loss_sum / den).item()
        val_rae_global = (rae_sum / den).detach().cpu().tolist()

        # LR scheduling
        if scheduler is not None:
            scheduler.step(val_loss_global)
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

        # --- save checkpoints every epoch (rank 0 only) ---
        if is_main_process(rank):
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

        # Save best model (rank 0)
        if is_main_process(rank) and (val_loss_global < best_val - cfg.min_delta):
            best_val = val_loss_global
            torch.save((model.module if is_ddp else model).state_dict(), best_path)

        if es.step(val_loss_global):
            rprint(rank, f"Early stopping at epoch {epoch+1}, best val {best_val:.6f}")
            break

    # ----------------- Test (best) -----------------
    # Load best on all ranks
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
            pred = model(xb)  # [B, num_outputs]
            pred_chunks.append(pred.detach())
            targ_chunks.append(yb.detach())
            rae_sum_local += rae_log(pred, yb).sum(dim=0)
            n_tot_local += yb.size(0)

    # Reduce metrics across ranks
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

    # Gather full predictions/targets to rank 0
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

    # Only rank 0 saves .npy
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


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_ddp(is_ddp=True if "RANK" in os.environ else False)
