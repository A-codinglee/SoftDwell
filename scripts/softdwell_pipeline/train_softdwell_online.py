#!/usr/bin/env python3
# scripts/softdwell_pipeline/train_softdwell_online.py
"""
SoftDwell model (ChannelWiseSoftDwell -> backbone) trained on freshly simulated traces instead of an HDF5
file, like scripts/hist_pipeline/train_hist_online.py. The backbone (--backbone) is ConvNeXt V2 as in
train_selfcal_online.py by default, or train_ram.py's Inception-ResNet (SoftDwellThenTFIR) with
--backbone inception.

  - CPU simulator processes (scripts.simulation) keep simulating raw int16 traces into a shared queue;
    every sample is used exactly once (train_hist_online.py --no-reuse). The SoftDwell layer is far
    slower per sample than the simulator, so the trainer practically never waits for data.
  - The GPU trainer(s) (DDP, one process per GPU) turn each trace into soft dwell histograms with the
    learnable thresholds / temperatures and train them together with the Inception-ResNet backbone.
  - Validation and test sets are simulated once at start-up and stay fixed.

Everything about the simulated data comes from the JSON given with --sim-config, e.g. configs/sim/COC.json.
The SoftDwell input normalisation (norm_min / norm_max of train_ram.py's TrainConfig) defaults to the
config's analysis.grind range (raw counts), 18000..24000 for COC.json, as in the offline runs.

Taken from train_ram.py: model, Adam with separate learning rates for the backbone and the SoftDwell
parameters (the SoftDwell one is held fixed), gradient clip, batch size. Taken from
train_hist_online.py: the epoch definition (--steps-per-epoch), learning-rate schedule and early
stopping patience, since online epochs are much shorter than a pass over the 81k-sample file.

To match another run's training setup (e.g. a train_selfcal_online.py run): --clip-grad-norm,
--weight-decay, --target-scale and --batch-size x GPUs = that run's batch size.

Long runs: with --resume, a checkpoint (checkpoint.pt) is written after every epoch, and a run started
again in the same folder carries on from it (model, optimizer, schedule, early stopping, and the
training-sample counter, so no sample is used twice). When --max-hours ends a job, it then exits without
testing and the next job continues; the test runs once early stopping (or --epochs) ends the training.
Chain jobs with sbatch --dependency=afterany:<previous job>; a job started after the run finished exits.

Writes the same files as train_hist_online.py (preds_test.npy, targs_test.npy, aux_test.npy, ...), so
plot_online_results.py and plot_noise_cv.py work, plus softdwell_params.csv as train_ram.py does.
best_model.pt holds softdwell.* and backbone.*; with --backbone inception it is a SoftDwellThenTFIR state
dict, as written by train_ram.py.

Usage:
  python3 -m scripts.softdwell_pipeline.train_softdwell_online --sim-config configs/sim/COC.json --k 4 --res 24
  # the Inception-ResNet of the offline runs instead:
  python3 -m scripts.softdwell_pipeline.train_softdwell_online --sim-config configs/sim/COC.json --backbone inception
"""
import argparse
import contextlib
import functools
import json
import os
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.common.ddp_utils import gather_concat_varlen
from scripts.hist_pipeline.train_hist import append_metrics_csv, init_metrics_csv
from scripts.hist_pipeline.train_hist_online import (LR_FACTOR, LR_PATIENCE, MIN_LR, STREAM_TEST, STREAM_VAL,
                                                     _allreduce_sum, _pool_init, condition_norm, evaluate,
                                                     generator_loop, rae_by_noise_file, simulate_fixed_set,
                                                     split_outputs, take_batch, write_noise_spectra)
from scripts.scripts.softdwell_pipeline.model import ConvNeXtV2Backbone
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.metrics import EarlyStopping, LogCoshLoss
from scripts.softdwell_pipeline.model import TFExactInceptionResNet
from scripts.softdwell_pipeline.softdwell import ChannelWiseSoftDwell
from scripts.softdwell_pipeline.softdwell_logging import log_sd_params_epoch

# Settings of train_ram.py that are not command-line options there either.
MODEL_KWARGS = dict(use_bn=False, separate_heads=False)
CHECKPOINT = "checkpoint.pt"
DONE_FILE = "DONE"          # written once the test has run; a resumed job then has nothing to do


class FromTrace(torch.nn.Module):
    """int16 traces (B, T) as the simulators hand them over -> SoftDwellThen, which expects float (B, 1, T)."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x.float().unsqueeze(1))


class SoftDwellThen(torch.nn.Module):
    """
    SoftDwell layer -> backbone. The same attribute names as SoftDwellThenTFIR (softdwell, backbone), so with
    --backbone inception the state dict is the one train_ram.py writes, and softdwell_logging finds theta / tau.
    """

    def __init__(self, softdwell, backbone):
        super().__init__()
        self.softdwell, self.backbone = softdwell, backbone

    def forward(self, x):
        return self.backbone(self.softdwell(x))


def make_backbone(args, in_channels, num_outputs, J):
    """--backbone: the Inception-ResNet of train_ram.py, or ConvNeXt V2 as train_selfcal_online.py builds it."""
    if args.backbone == "inception":
        return TFExactInceptionResNet(in_channels=in_channels, num_outputs=num_outputs, input_hw=(J, J),
                                      **MODEL_KWARGS)
    Backbone = functools.partial(ConvNeXtV2Backbone, hidden=args.hidden, depth=args.depth,
                                 kernel_size=args.kernel_size, use_grn=(args.backbone != "convnext_v2_nogrn"))
    return Backbone(in_channels=in_channels, num_outputs=num_outputs, input_hw=(J, J), **MODEL_KWARGS)


def build_model(args, num_outputs, device):
    """SoftDwell layer as train_ram.py builds it; J (histogram size) is the number of unique dwell lengths."""
    softdwell = ChannelWiseSoftDwell(
        num_detectors=args.k,
        dwell_min=args.dwell_min,
        dwell_max=args.dwell_max,
        num_bins=args.res,
        average_segments=False,
        norm_min=args.norm_range[0],
        norm_max=args.norm_range[1],
        eps=1e-6,
        theta_init_min=args.theta_init[0],
        theta_init_max=args.theta_init[1],
        tau_init=args.tau_init,
        dtype=torch.float32,
        timing=False,
        logN_gamma=args.logN_gamma,
    )
    J = softdwell.inner.J
    net = SoftDwellThen(softdwell, make_backbone(args, args.k, num_outputs, J))
    return FromTrace(net.to(device).to(memory_format=torch.channels_last))


def train_worker(rank, world_size, args, shared):
    (X_buf, y_buf, data_lock, n_written, n_read, next_index, X_val, y_val, X_test, y_test, cond, sim_cfg,
     noise_src_test, ckpt) = shared

    use_ddp = torch.cuda.is_available()
    if use_ddp:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ.setdefault("MASTER_PORT", "29500")      # main() picks a free one
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    else:
        device = torch.device("cpu")

    is_master = (rank == 0)
    num_outputs = int(y_buf.shape[1])

    metrics_csv = os.path.join(args.output_root, "train_val_metrics.csv")
    sd_csv = os.path.join(args.output_root, "softdwell_params.csv")
    ckpt_path = os.path.join(args.output_root, CHECKPOINT)
    torch.manual_seed(args.seed)
    model = build_model(args, num_outputs, device)
    core = model                                        # FromTrace; core.net is what gets saved
    if ckpt is not None:
        core.net.load_state_dict(ckpt["model"])
    if is_master:
        print(f"[Worker 0] device={device} world_size={world_size} steps_per_epoch={args.steps_per_epoch} "
              f"batch_size={args.batch_size} | backbone={args.backbone} "
              f"({sum(p.numel() for p in core.net.backbone.parameters()):,} parameters) | "
              f"SoftDwell K={args.k} J={core.net.softdwell.inner.J} "
              f"lengths={core.net.softdwell.inner.L_list}", flush=True)
        if ckpt is None:
            init_metrics_csv(metrics_csv, num_outputs)
            if os.path.isfile(sd_csv):
                os.remove(sd_csv)                       # left over from an earlier run in this folder
            log_sd_params_epoch(core.net, -1, csv_path=sd_csv, stats_only=False)
        else:
            print(f"[Worker 0] resumed after epoch {ckpt['epoch']} (best val {ckpt['best_val']:.6f}, "
                  f"{ckpt['next_index']:,} training samples drawn so far)", flush=True)
    if use_ddp:
        model = DDP(model, device_ids=[rank])           # weights broadcast from rank 0

    # backbone vs SoftDwell (theta / tau) parameters, as in train_ram.py
    sd_params, bb_params = [], []
    for n, p in core.named_parameters():
        (sd_params if "softdwell" in n else bb_params).append(p)
    # AdamW: the same as train_ram.py's Adam with the default --weight-decay 0
    opt = torch.optim.AdamW([{"params": bb_params, "lr": args.lr, "weight_decay": args.weight_decay},
                             {"params": sd_params, "lr": args.lr_sd, "weight_decay": 0.0}])
    crit = LogCoshLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    es = EarlyStopping(patience=args.patience, min_delta=args.min_delta, restore_best=False)

    best_val, first_epoch = float("inf"), 1
    if ckpt is not None:
        opt.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        es.best, es.best_epoch, es.count = ckpt["early_stopping"]
        best_val, first_epoch = ckpt["best_val"], ckpt["epoch"] + 1
    finished = ckpt is not None and ckpt.get("finished", False)     # training ended, only the test is left
    best_path = os.path.join(args.output_root, "best_model.pt")
    val_idx = torch.arange(rank, X_val.shape[0], world_size)        # each rank its own slice
    samples_per_epoch = args.steps_per_epoch * args.batch_size * world_size
    micro = args.micro_batch if args.micro_batch > 0 else args.batch_size

    out_of_time = False
    for epoch in range(first_epoch, args.epochs + 1):
        if finished:
            break
        t0 = time.time()
        written0 = n_written.value
        waited = 0.0

        # ----------------- Train -----------------
        model.train()
        tr_loss_accum = torch.zeros(1, device=device)
        tr_count = torch.zeros(1, device=device)

        for _ in range(args.steps_per_epoch):
            xb, yb, w = take_batch(X_buf, y_buf, data_lock, n_written, n_read, args.batch_size)
            waited += w
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            # --micro-batch: the batch in chunks, gradients summed; the same gradient as the whole batch at once
            # (no batch norm: ConvNeXt's LayerNorm / GRN and the SoftDwell layer work per sample)
            opt.zero_grad(set_to_none=True)
            chunks = list(zip(xb.split(micro), yb.split(micro)))
            loss = 0.0
            for c, (xc, yc) in enumerate(chunks):
                sync = c == len(chunks) - 1 or not use_ddp           # DDP all-reduce only after the last chunk
                with (contextlib.nullcontext() if sync else model.no_sync()):
                    part = crit(model(xc), yc) * (xc.size(0) / xb.size(0))
                    part.backward()
                loss = loss + part.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad_norm)
            opt.step()

            tr_loss_accum += loss.detach() * xb.size(0)
            tr_count += xb.size(0)

        _allreduce_sum(tr_loss_accum)
        _allreduce_sum(tr_count)
        train_loss = (tr_loss_accum / tr_count).item()
        t_train = time.time() - t0

        # ----------------- Val (fixed set) -----------------
        model.eval()
        val_loss_sum, val_rae_sum, val_count, _ = evaluate(
            model, X_val, y_val, val_idx, micro, device, crit, num_outputs)
        _allreduce_sum(val_loss_sum)
        _allreduce_sum(val_rae_sum)
        _allreduce_sum(val_count)

        den = val_count.clamp_min(1).float()
        val_loss = (val_loss_sum / den).item()
        val_rae_per = (val_rae_sum / den).detach().cpu().tolist()

        scheduler.step(val_loss)
        opt.param_groups[1]["lr"] = args.lr_sd             # SoftDwell learning rate stays fixed (train_ram.py)

        dt = time.time() - t0
        current_lr = opt.param_groups[0]["lr"]

        if is_master:
            fresh = n_written.value - written0
            print(
                f"[Epoch {epoch:03d}] {dt:.2f}s (train {t_train:.1f}s, val {dt - t_train:.1f}s) | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Val RAE per out: {' '.join(f'{r:.4f}' for r in val_rae_per)}\n"
                f"            no reuse: {samples_per_epoch} new samples, each used once | "
                f"simulated at {fresh / max(dt, 1e-9):.1f}/s | this trainer waited {waited:.1f}s for data "
                f"({100 * waited / max(dt, 1e-9):.0f}% of the epoch) | total used {n_read.value} | "
                f"lr {current_lr:g}"
                + (f" | GPU memory peak {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
                   if device.type == "cuda" else ""),
                flush=True,
            )
            append_metrics_csv(csv_path=metrics_csv, epoch=epoch, epoch_time_s=dt, train_loss=train_loss,
                               val_loss=val_loss, lr=current_lr, val_rae_per=val_rae_per)
            log_sd_params_epoch(core.net, epoch - 1, csv_path=sd_csv, stats_only=False)
            log_sd_params_epoch(core.net, epoch - 1, stats_only=True)

        # ----------------- Early stopping + checkpoint -----------------
        stop_local, out_of_time, early = False, False, False
        if is_master:
            early = stop_local = es.step(val_loss, None, epoch)
            if val_loss < best_val - args.min_delta:
                best_val = val_loss
                torch.save(core.net.state_dict(), best_path)
                print(f"  -> Saved best model (val={best_val:.6f})", flush=True)
            # --max-hours: stop while one more epoch still fits, so the test below runs before the job ends
            out_of_time = args.max_hours > 0 and time.time() - args.t_start + dt > args.max_hours * 3600
            if out_of_time and not stop_local:
                print(f"[Time limit] another epoch would pass --max-hours {args.max_hours:g}. Stopping.", flush=True)
            done = early or epoch == args.epochs          # training over: only the test is left
            out_of_time = out_of_time and not done
            stop_local = done or out_of_time
            if args.resume:
                # after every epoch: everything needed to carry on. next_index: no training sample index
                # below it is drawn again (the ones still waiting in the queue are skipped)
                tmp = ckpt_path + ".tmp"
                torch.save({"model": core.net.state_dict(), "optimizer": opt.state_dict(),
                            "scheduler": scheduler.state_dict(), "early_stopping": (es.best, es.best_epoch, es.count),
                            "best_val": best_val, "epoch": epoch, "next_index": int(next_index.value),
                            "finished": done}, tmp)
                os.replace(tmp, ckpt_path)

        if use_ddp:
            flags = torch.tensor([int(stop_local), int(out_of_time)], device=device, dtype=torch.uint8)
            dist.broadcast(flags, src=0)
            stop_local, out_of_time = bool(flags[0].item()), bool(flags[1].item())

        if stop_local:
            if is_master and early:
                print(f"[Early Stopping] No improvement for {args.patience} epochs. Stopping.", flush=True)
            break

    if out_of_time and args.resume:
        if is_master:
            print(f"[Time limit] checkpoint written to {ckpt_path}; start the same command again to continue. "
                  "No test yet.", flush=True)
        if use_ddp:
            dist.destroy_process_group()
        return

    # -- Testing (fixed test set, split over all ranks like validation) --
    if use_ddp:
        dist.barrier()                                  # best_model.pt is written
    if is_master:
        print("\n[Test] Loading best model...", flush=True)
    core.net.load_state_dict(torch.load(best_path, map_location=device))
    core.eval()
    n_test = X_test.shape[0]
    test_idx = torch.arange(rank, n_test, world_size)
    _, rae_sum, count, preds = evaluate(core, X_test, y_test, test_idx, micro, device, crit, num_outputs)
    _allreduce_sum(rae_sum)
    _allreduce_sum(count)
    preds = torch.cat(preds) if preds else torch.empty((0, num_outputs))
    preds = gather_concat_varlen(preds.to(device), dim=0).cpu()          # rank 0's rows, then rank 1's, ...

    if is_master:
        order = torch.cat([torch.arange(r, n_test, world_size) for r in range(world_size)])
        preds_all = np.empty((n_test, num_outputs), dtype=np.float64)
        preds_all[order.numpy()] = preds.numpy()
        targs_all = y_test.numpy().astype(np.float64)
        preds_hz, targs_hz, preds_aux = split_outputs(preds_all, targs_all, cond)
        n_rates = preds_hz.shape[1]

        np.save(os.path.join(args.output_root, "preds_test.npy"), preds_hz)
        np.save(os.path.join(args.output_root, "targs_test.npy"), targs_hz)

        # RAE is only meaningful for the rates; "Overall" stays the mean over the rates
        test_rae_per = (rae_sum / count.clamp_min(1).float()).cpu().tolist()
        rel_err = np.mean(np.abs(preds_hz - targs_hz) / np.maximum(targs_hz, 1e-12))
        print(f"[Test] RAE per out: {' '.join(f'{r:.4f}' for r in test_rae_per)} | "
              f"Overall: {float(np.mean(test_rae_per[:n_rates])):.4f} | Mean Rel Err: {rel_err:.6f}", flush=True)
        rae_by_noise_file(preds_hz, targs_hz, noise_src_test, sim_cfg, args.output_root, "Test")

        if cond:
            np.save(os.path.join(args.output_root, "preds_aux_test.npy"), preds_aux)
            mae = (np.abs(preds_all[:, n_rates:] - targs_all[:, n_rates:]).mean(axis=0)
                   * np.asarray(cond["std"]) / cond["counts_per_unit"])
            print(f"[Test] Mean abs error ({cond['unit']}): "
                  + " | ".join(f"{nm} {e:.4g}" for nm, e in zip(cond["names"], mae)), flush=True)
        with open(os.path.join(args.output_root, DONE_FILE), "w") as f:
            f.write("test done\n")

    if use_ddp:
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True, help="simulation JSON, e.g. configs/sim/COC.json")
    p.add_argument("--output-root", default=None,
                   help="default: outputs/softdwell_online/<sim-config name>/<backbone>/<K>Det_<res>Res")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--predict-conditions", action="store_true",
                   help="also predict baseline, amplitude and noise sigma of each sample (those that vary in "
                        "the sim-config), as extra outputs behind the rates")
    p.add_argument("--holdout-noise", default=None, metavar="NAME",
                   help="of the sim-config's noise files, test on the one whose name contains NAME and "
                        "train on all others")
    p.add_argument("--train-noise", default=None, metavar="NAME",
                   help="of the sim-config's noise files, train on the one whose name contains NAME only "
                        "and test on all others")

    # SoftDwell layer (names as in train_ram.py: --k detectors, --res dwell-length bins)
    p.add_argument("--k", type=int, default=4, help="number of soft detectors (thresholds)")
    p.add_argument("--res", type=int, default=24, help="dwell-length bins J (histograms are J x J)")
    p.add_argument("--dwell-min", type=int, default=1, help="shortest dwell length, samples")
    p.add_argument("--dwell-max", type=int, default=10000, help="longest dwell length, samples")
    p.add_argument("--norm-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                   help="raw counts mapped to 0..1 before the detectors (default: the sim-config's "
                        "analysis.grind range)")
    p.add_argument("--theta-init", type=float, nargs=2, default=[0.1, 0.9], metavar=("MIN", "MAX"))
    p.add_argument("--tau-init", type=float, default=0.1)
    p.add_argument("--logN-gamma", type=float, default=0.0)

    # backbone behind the SoftDwell layer (choices and ConvNeXt knobs as in train_selfcal_online.py)
    p.add_argument("--backbone", default="convnext_v2", choices=["convnext_v2", "convnext_v2_nogrn", "inception"],
                   help="convnext_v2: the ConvNeXt V2 of the self-calibrating network; inception: train_ram.py's "
                        "TFExactInceptionResNet")
    p.add_argument("--hidden", type=int, default=64, help="ConvNeXt width")
    p.add_argument("--depth", type=int, default=4, help="ConvNeXt blocks")
    p.add_argument("--kernel-size", type=int, default=7, help="ConvNeXt depthwise kernel")

    # data: the buffer is only a queue (every sample is used once); one trace of 10 s at 100 kHz is 2 MB
    p.add_argument("--buffer-size", type=int, default=512, help="traces waiting for the trainers")
    p.add_argument("--gen-workers", type=int, default=0,
                   help="simulator processes (0 = all CPUs not used by trainers)")
    p.add_argument("--train-gpus", type=int, default=0, help="GPUs to train on (0 = all in the job)")
    p.add_argument("--n-val", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=10000)

    # training
    p.add_argument("--batch-size", type=int, default=8, help="per GPU")
    p.add_argument("--micro-batch", type=int, default=0,
                   help="split each GPU's batch into chunks of this many traces and add up their gradients "
                        "(same result, less GPU memory; 8 fits a 10 s trace on an 80 GB A100). 0 = whole batch")
    p.add_argument("--steps-per-epoch", type=int, default=0,
                   help="0 = 8192 samples per epoch: 8192 / (batch_size * n_gpus)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-3, help="backbone")
    p.add_argument("--lr-sd", type=float, default=1e-3, help="SoftDwell theta / tau (held fixed)")
    p.add_argument("--max-hours", type=float, default=0.0,
                   help="stop training once another epoch would pass this many hours since start-up, so the "
                        "test still runs within the job's time limit (0 = no limit); with --resume the job "
                        "exits without testing and the next one continues")
    p.add_argument("--resume", action="store_true",
                   help="checkpoint after every epoch and carry on from the folder's checkpoint if there is one")
    p.add_argument("--clip-grad-norm", type=float, default=1.0, help="1.0 as train_ram.py; selfcal runs use 5.0")
    p.add_argument("--weight-decay", type=float, default=0.0, help="AdamW, backbone only")
    p.add_argument("--target-scale", type=float, default=1.0,
                   help="--predict-conditions: standardized baseline / amplitude / sigma targets are multiplied by "
                        "this (train_selfcal_online.py --target-scale; its runs use 1.155)")
    return p.parse_args()


def main():
    args = parse_args()
    args.t_start = time.time()
    if args.output_root is None:
        name = os.path.splitext(os.path.basename(args.sim_config))[0]
        args.output_root = os.path.join("outputs", "softdwell_online", name, args.backbone, f"{args.k}Det_{args.res}Res")
    args.output_root = os.path.join(args.output_root, f"seed{args.seed}")
    os.makedirs(args.output_root, exist_ok=True)
    ckpt = None
    if args.resume:
        if os.path.isfile(os.path.join(args.output_root, DONE_FILE)):
            print(f"[Parent] {args.output_root} is finished (tested). Nothing to do.", flush=True)
            return
        ckpt_path = os.path.join(args.output_root, CHECKPOINT)
        if os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu")
            print(f"[Parent] resuming from {ckpt_path} (epoch {ckpt['epoch']})", flush=True)

    # The copy written here (absolute paths, this run's seed) is what the simulator reads.
    config_path = prepare_sim_config(args.sim_config, os.path.join(args.output_root, "sim_config.json"), args.seed,
                                     holdout_noise=args.holdout_noise, train_noise=args.train_noise)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len = probe.trace_len
    grind = probe.config.get("analysis", {}).get("grind", {})       # already in raw counts
    probe.close()
    if args.norm_range is None:
        if "min" not in grind or "max" not in grind:
            raise SystemExit(f"{args.sim_config} has no analysis.grind range: give --norm-range MIN MAX")
        args.norm_range = [float(grind["min"]), float(grind["max"])]
    if args.dwell_max > trace_len:
        raise SystemExit(f"--dwell-max {args.dwell_max} is longer than the trace ({trace_len} samples)")

    K = sim_cfg.num_states
    off_chain = [(r, c) for r in range(K) for c in range(K) if abs(r - c) > 1 and sim_cfg.q_hi[r, c] > 0]
    if off_chain:
        raise SystemExit(f"only linear chains are supported as training targets; non-adjacent rates at {off_chain}")
    n_rates = 2 * (K - 1)

    cpp = sim_cfg.counts_per_pA
    unit, sc = ("pA", cpp) if cpp > 0 else ("counts", 1.0)
    print(f"[Parent] sim-config={args.sim_config} topology={sim_cfg.topology} trace_len={trace_len} | "
          f"SoftDwell norm range {args.norm_range[0]:g}..{args.norm_range[1]:g} counts", flush=True)

    # extra targets; conditions.json tells eval_hist_online / plot_online_results how to read the outputs
    cond = None
    cond_path = os.path.join(args.output_root, "conditions.json")
    if args.predict_conditions:
        cond = condition_norm(sim_cfg)
        if not cond["index"]:
            raise SystemExit(f"--predict-conditions: baseline, amplitude and sigma are all fixed in {args.sim_config}")
        cond["std"] = [s / args.target_scale for s in cond["std"]]
        cond["unit"], cond["counts_per_unit"] = unit, sc
        with open(cond_path, "w") as f:
            json.dump(cond, f, indent=2)
        print(f"[Parent] extra targets: {', '.join(cond['names'])}", flush=True)
    elif os.path.isfile(cond_path):
        os.remove(cond_path)                    # left over from an earlier run in this folder
    y_dim = n_rates + (len(cond["index"]) if cond else 0)

    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.output_root)
    names = lambda paths: ", ".join(os.path.basename(p) for p in paths)
    print(f"[Parent] noise: {sim_cfg.noise_mode}\n"
          f"[Parent]   train / val: {names(sim_cfg.noise_paths)}\n"
          f"[Parent]   test:        {names(sim_cfg.noise_test_paths) or 'same files as training'}", flush=True)

    gpus_in_job = torch.cuda.device_count()
    n_gpus = min(gpus_in_job, args.train_gpus) if args.train_gpus > 0 else gpus_in_job
    world_size = max(1, n_gpus)
    n_cpus = len(os.sched_getaffinity(0))
    if args.gen_workers <= 0:
        args.gen_workers = max(1, n_cpus - world_size - 1)
    if args.buffer_size < args.batch_size * world_size:
        raise SystemExit("--buffer-size must be at least batch size x number of GPUs")
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = max(1, 8192 // (args.batch_size * world_size))
    print(f"[Parent] gpus={n_gpus} (of {gpus_in_job} in the job) cpus={n_cpus} gen_workers={args.gen_workers} "
          f"buffer={args.buffer_size} traces of {trace_len} points y={y_dim}", flush=True)

    train_config = {
        "model": {"name": SoftDwellThen.__name__, "backbone": args.backbone,
                  **({} if args.backbone == "inception" else
                     {"hidden": args.hidden, "depth": args.depth, "kernel_size": args.kernel_size}),
                  "num_detectors": args.k, "num_bins": args.res,
                  "dwell_min": args.dwell_min, "dwell_max": args.dwell_max, "norm_range": args.norm_range,
                  "theta_init": args.theta_init, "tau_init": args.tau_init, "logN_gamma": args.logN_gamma,
                  "average_segments": False, "num_outputs": y_dim, "num_rates": n_rates,
                  "extra_outputs": cond["names"] if cond else [], **MODEL_KWARGS},
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
                      "lr_softdwell": args.lr_sd, "lr_softdwell_fixed": True},
        "loss": LogCoshLoss.__name__, "target_scale": args.target_scale,
        "lr_scheduler": {"name": "ReduceLROnPlateau", "monitor": "val_loss", "factor": LR_FACTOR,
                         "patience": LR_PATIENCE, "min_lr": MIN_LR},
        "early_stopping": {"monitor": "val_loss", "patience": args.patience, "min_delta": args.min_delta},
        "grad_clip_norm": args.clip_grad_norm,
        "batch_size": args.batch_size,          # per GPU
        "batch_size_total": args.batch_size * world_size,
        "micro_batch": args.micro_batch or args.batch_size,
        "resume": args.resume,
        "train_gpus": world_size if n_gpus else 0,
        "steps_per_epoch": args.steps_per_epoch,
        "samples_per_epoch": args.steps_per_epoch * args.batch_size * world_size,
        "max_epochs": args.epochs,
        "seed": args.seed,
        "data": {"sim_config": args.sim_config, "no_reuse": True, "buffer_size": args.buffer_size,
                 "gen_workers": args.gen_workers, "n_val": args.n_val, "n_test": args.n_test,
                 "trace_len": trace_len,
                 "noise_mode": sim_cfg.noise_mode, "noise_files_train_val": sim_cfg.noise_paths,
                 "noise_files_test": sim_cfg.noise_test_paths or sim_cfg.noise_paths},
    }
    with open(os.path.join(args.output_root, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    ctx = mp.get_context("spawn")

    # 1. Fixed val / test sets, as raw traces (all CPUs, before the background simulators start)
    t0 = time.time()
    with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                  initargs=(config_path, sim_cfg, args.seed, spectrum_path, cond, True)) as pool:
        X_val, y_val, _, _ = simulate_fixed_set(pool, STREAM_VAL, args.n_val, (trace_len,), y_dim, torch.int16)
        X_test, y_test, aux_test, noise_src_test = simulate_fixed_set(pool, STREAM_TEST, args.n_test, (trace_len,),
                                                                      y_dim, torch.int16)
    np.save(os.path.join(args.output_root, "aux_test.npy"), aux_test)
    np.save(os.path.join(args.output_root, "noise_source_test.npy"), noise_src_test)
    n_fixed = args.n_val + args.n_test
    print(f"[Parent] Simulated val={args.n_val} test={args.n_test} in {time.time() - t0:.1f}s "
          f"({n_fixed / max(time.time() - t0, 1e-9):.1f} samples/s)", flush=True)

    # 2. Queue of fresh training traces + background simulators
    X_buf = torch.zeros((args.buffer_size, trace_len), dtype=torch.int16).share_memory_()
    y_buf = torch.zeros((args.buffer_size, y_dim), dtype=torch.float32).share_memory_()
    data_lock = ctx.Lock()
    next_index = ctx.Value("q", ckpt["next_index"] if ckpt else 0)    # first training sample to draw
    n_written = ctx.Value("q", 0, lock=False)   # only modified under data_lock
    n_read = ctx.Value("q", 0, lock=False)      # samples handed to trainers; only modified under data_lock
    stop = ctx.Value("b", 0, lock=False)

    gens = [
        ctx.Process(
            target=generator_loop,
            args=(config_path, sim_cfg, args.seed, spectrum_path,
                  X_buf, y_buf, data_lock, next_index, n_written, stop, n_read, cond, True),
            daemon=True,
        )
        for _ in range(args.gen_workers)
    ]
    for g in gens:
        g.start()

    try:
        shared = (X_buf, y_buf, data_lock, n_written, n_read, next_index, X_val, y_val, X_test, y_test, cond,
                  sim_cfg, noise_src_test, ckpt)
        if n_gpus >= 1:
            # a port nobody on this node uses: two of our jobs can share a node
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                os.environ["MASTER_PORT"] = str(sock.getsockname()[1])
            print(f"[Parent] Spawning {world_size} trainer(s)...", flush=True)
            mp.spawn(train_worker, nprocs=world_size, args=(world_size, args, shared), join=True)
        else:
            print("[Parent] No GPU found, training in this process on CPU.", flush=True)
            train_worker(0, 1, args, shared)
    finally:
        stop.value = 1
        for g in gens:
            g.join(timeout=30)
            if g.is_alive():
                g.terminate()


if __name__ == "__main__":
    main()
