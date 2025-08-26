#!/usr/bin/env python3
import numpy as np
import torch
import os
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DistributedDataParallel as DDP

from config import TrainConfig
from ddp_utils import (
    ddp_init_from_torchrun, is_main_process, ddp_cleanup, ddp_allreduce_mean
)
from data import PackedMemmapDataset
from model import TFExactInceptionResNet
from metrics import LogCoshLoss, rae_log, EarlyStopping

# Speed knobs
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def make_dataloaders(cfg: TrainConfig, device):
    ds = PackedMemmapDataset(
        cfg.hists_path, cfg.labels_path,
        log10_labels=True, normalize_hist=True, label_scale=1.0
    )
    indices = np.arange(len(ds))
    train_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx  = train_test_split(train_idx, test_size=0.1, random_state=42)

    train_subset = Subset(ds, train_idx)
    val_subset   = Subset(ds, val_idx)
    test_subset  = Subset(ds, test_idx)

    is_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    if is_ddp:
        train_sampler = DistributedSampler(train_subset, shuffle=True, drop_last=True)
        val_sampler   = DistributedSampler(val_subset,   shuffle=False, drop_last=False)
        test_sampler  = DistributedSampler(test_subset,  shuffle=False, drop_last=False)
    else:
        train_sampler = val_sampler = test_sampler = None

    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
        multiprocessing_context="spawn",
    )
    if cfg.num_workers > 0:
        common["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(train_subset, sampler=train_sampler, shuffle=(train_sampler is None), **common)
    val_loader   = DataLoader(val_subset,   sampler=val_sampler,   shuffle=False, **common)
    test_loader  = DataLoader(test_subset,  sampler=test_sampler,  shuffle=False, **common)
    return train_loader, val_loader, test_loader, train_sampler


def main_worker(cfg: TrainConfig):
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    # DDP init: picks up torchrun env automatically; otherwise stays single-process
    is_ddp, local_rank, world_size, global_rank = ddp_init_from_torchrun()
    device = torch.device(f"cuda:{local_rank}" if (use_cuda and is_ddp) else ("cuda" if use_cuda else "cpu"))

    train_loader, val_loader, test_loader, train_sampler = make_dataloaders(cfg, device)

    model = TFExactInceptionResNet(
        in_channels=1, num_outputs=cfg.num_outputs,
        use_bn=cfg.use_bn, separate_heads=cfg.separate_heads
    ).to(device)
    model = model.to(memory_format=torch.channels_last)

    try:
        model = torch.compile(model, backend="inductor")
    except Exception as e:
        if is_main_process():
            print("[warn] torch.compile disabled:", e)

    if is_ddp:
        if device.type == "cuda":
            model = DDP(model, device_ids=[device.index], output_device=device.index, find_unused_parameters=False, gradient_as_bucket_view=False)
        else:
            # CPU DDP: no device_ids/output_device
            model = DDP(model, find_unused_parameters=False, gradient_as_bucket_view=False)

    crit   = LogCoshLoss()
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.epochs,
        pct_start=0.1, div_factor=10.0, final_div_factor=1e3
    )

    es = EarlyStopping(patience=cfg.patience, min_delta=cfg.min_delta)
    best_val = float("inf")

    # ---- Train (per-batch OneCycle step, same as your original) ----
    for epoch in range(cfg.epochs):
        if torch.distributed.is_initialized() and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        tr_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                pred = model(xb)
                loss = crit(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            tr_loss_sum += float(loss)

        tr_loss = tr_loss_sum / max(1, len(train_loader))
        tr_loss_t = torch.tensor([tr_loss], device=device)
        tr_loss_t = ddp_allreduce_mean(tr_loss_t)
        tr_loss = tr_loss_t.item()

        # ---- Validate ----
        model.eval()
        val_loss_accum = 0.0
        val_rae_sum = torch.zeros(cfg.num_outputs, device=device)
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                yb = yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type == "cuda")):
                    pred = model(xb)
                    loss = crit(pred, yb)
                val_loss_accum += float(loss)
                val_rae_sum += rae_log(pred, yb).mean(dim=0)

        val_loss = val_loss_accum / max(1, len(val_loader))
        val_loss_t = torch.tensor([val_loss], device=device)
        val_loss_t = ddp_allreduce_mean(val_loss_t)
        val_loss = val_loss_t.item()

        val_rae_per = val_rae_sum / max(1, len(val_loader))
        val_rae_per = ddp_allreduce_mean(val_rae_per)
        val_rae_np = val_rae_per.detach().cpu().numpy()

        if is_main_process():
            print(
                f"Epoch {epoch+1}: Train {tr_loss:.6f} | Val {val_loss:.6f} | "
                f"Val RAE per output: {' '.join(f'{r:.4f}' for r in val_rae_np)}"
            )

        # Save best
        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            if is_main_process():
                to_save = model.module if isinstance(model, DDP) else model
                torch.save(to_save.state_dict(), "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/best_model.pt")

        # Early stopping (synced)
        stop_local = es.step(val_loss)
        stop_flag = torch.tensor([int(stop_local)], device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stop_flag, op=torch.distributed.ReduceOp.MAX)
        if stop_flag.item() > 0:
            if is_main_process():
                print(f"Early stopping at epoch {epoch+1}, best val {best_val:.6f}")
            break

    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    # ---- Test ----
    if isinstance(model, DDP):
        model.module.load_state_dict(torch.load("/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/best_model.pt", map_location=device))
    else:
        model.load_state_dict(torch.load("/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/best_model.pt", map_location=device))

    model.eval()
    preds_parts, targs_parts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            preds_parts.append(pred.cpu().numpy())
            targs_parts.append(yb.cpu().numpy())

    preds_log_all = np.concatenate(preds_parts, axis=0) if preds_parts else np.empty((0, cfg.num_outputs))
    targs_log_all = np.concatenate(targs_parts, axis=0) if targs_parts else np.empty((0, cfg.num_outputs))

    # If DDP, gather to rank 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        payload = {'preds': preds_log_all, 'targs': targs_log_all}
        gathered = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gathered, payload)
        if is_main_process():
            preds_log_all = np.concatenate([g['preds'] for g in gathered], axis=0)
            targs_log_all = np.concatenate([g['targs'] for g in gathered], axis=0)

    if is_main_process():
        preds_hz = 10 ** preds_log_all
        targs_hz = 10 ** targs_log_all
        print("Predictions range:", preds_hz.min(), preds_hz.max())
        print("Targets range:", targs_hz.min(), targs_hz.max())
        np.save("/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/predictions_nn.npy", preds_hz)
        np.save("/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/targets_nn.npy", targs_hz)

    ddp_cleanup()


def main():
    cfg = TrainConfig()  # <- single source of truth (config.py)

    world_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_env > 1:
        return main_worker(cfg=cfg)
    return main_worker(cfg=cfg)


if __name__ == "__main__":
    main()
