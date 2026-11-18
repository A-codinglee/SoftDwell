#!/usr/bin/env python3
# scripts/hist_pipeline/train_selfcal_online.py
"""
HardSelfCalibratingFusionNet (scripts/scripts/hardthresh_pipeline/train_hardselfcal_fusion.py) trained
on freshly simulated samples, like train_hist_online.py: same simulator, same per-sample draws, same
noise options (--holdout-noise / --train-noise), every sample used once.

The network looks at each sample twice:
  1. a fixed threshold ladder  -> histograms -> first backbone -> estimate of baseline and amplitude
  2. a ladder placed around that estimate, per sample -> histograms -> second backbone
  3. both feature vectors + the level estimate -> 9 outputs (6 log10 rates, baseline, amplitude, sigma)
Step 2 needs the raw trace at training time, so the simulator processes hand over int16 traces here
(not histograms) and both sets of histograms are computed on the GPU with grind_hist_gpu.

Kept from train_hist_online.py: batch size, initial learning rate, learning-rate schedule
(ReduceLROnPlateau with the factor and patience set there) and early-stopping patience. Taken from the
hard-threshold script: optimizer (AdamW; without weight decay it is the same as Adam), gradient clip, histogram shapes, the auxiliary loss on the level estimate and the
scale of the three extra targets.

Writes the same files as train_hist_online.py, so plot_online_results.py and plot_noise_cv.py work.

Usage:
  python3 -m scripts.hist_pipeline.train_selfcal_online --sim-config configs/sim/COCO_noise5.json \
      --holdout-noise 2026_02_20 --output-root <dir>
"""
import argparse
import json
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.common.gitinfo import git_info
from scripts.hist_pipeline.train_hist import append_metrics_csv, init_metrics_csv
from scripts.hist_pipeline.train_hist_online import (CONDITIONS, LR_FACTOR, LR_PATIENCE, MIN_LR, SIM_DETAIL,
                                                     SIM_DETAIL_NAMES, STREAM_TEST, STREAM_TRAIN, STREAM_VAL,
                                                     _pool_init, condition_norm, generator_loop, rae_by_noise_file,
                                                     simulate_fixed_set, split_outputs, write_noise_spectra)
from scripts.hist_pipeline.gpu_hist_fast import grind_hist_fast
from scripts.scripts.hardthresh_pipeline.gpu_hist import grind_hist_gpu, make_dwell_lut, make_thresholds
from scripts.scripts.softdwell_pipeline.model import HardSelfCalibratingFusionNet
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.metrics import EarlyStopping, LogCoshLoss, rae_log

DWELL_MIN_S, DWELL_MAX_S = 1e-5, 0.1            # dwell-time range of the histograms, as in configs/sim


class Hists:
    """
    The two GPU histogram stacks of a batch of raw traces. Building them is most of the GPU work of a
    step, so the batch is split over all GPUs of the job (the model itself lives on the first one).
    """

    def __init__(self, args, trace_len, fs, counts_per_pA, device, devices=None):
        """devices: the GPUs that build the histograms (default: all of the job, or --hist-gpus of them)."""
        self.a, self.c, self.dev = args, counts_per_pA, device
        n = torch.cuda.device_count() if device.type == "cuda" else 0
        if args.hist_gpus > 0:
            n = min(n, args.hist_gpus)
        self.devices = devices or [torch.device(f"cuda:{i}") for i in range(n)] or [device]
        thr = torch.from_numpy(make_thresholds(args.fixed_thr_pA[0] * counts_per_pA,
                                               args.fixed_thr_pA[1] * counts_per_pA, args.fixed_k))
        lut = lambda j: torch.from_numpy(make_dwell_lut(trace_len, fs, DWELL_MIN_S, DWELL_MAX_S, j))
        self.thr_fixed = [thr.to(d) for d in self.devices]
        self.lut_fixed = [lut(args.fixed_j).to(d) for d in self.devices]
        self.lut_selfcal = [lut(args.selfcal_j).to(d) for d in self.devices]
        self.pool = ThreadPoolExecutor(len(self.devices))
        if args.hist_impl == "reference":
            self.grind = lambda x, thr, lut, j: grind_hist_gpu(x, thr, lut, j, k_group=args.k_group)[0]
        elif args.hist_impl == "cuda":    # identical histograms in one pass (scripts/hist_pipeline/bench_gpu_hist.py)
            from scripts.hist_pipeline import gpu_hist_cuda
            gpu_hist_cuda.load()          # compile now, not in the first step
            self.grind = gpu_hist_cuda.grind_hist_cuda
        else:       # identical histograms, about half the GPU time (scripts/hist_pipeline/bench_gpu_hist.py)
            self.grind = lambda x, thr, lut, j: grind_hist_fast(x, thr, lut, j, k_group=args.k_group,
                                                                accumulate="scatter")

    def _map(self, fn, *per_device):
        """fn(device index, ...) on every device at once; results concatenated on the model's device."""
        futures = [self.pool.submit(fn, i, *vals) for i, vals in enumerate(zip(*per_device))]
        return torch.cat([f.result().to(self.dev, non_blocking=True) for f in futures])

    def load_async(self, x, extra=()):
        """
        Start copying x (page-locked, on the CPU) to the devices, one part each, on a separate copy stream per
        device, so the copy can run while the devices compute on the previous batch. extra: further
        page-locked CPU tensors for the model's device. Returns a pending batch for ready() / copied().
        """
        if not hasattr(self, "copy_streams"):
            self.copy_streams = [torch.cuda.Stream(d) for d in self.devices]
        parts, events, ex = [], [], []
        for i, (xp, d, st) in enumerate(zip(torch.tensor_split(x, len(self.devices)), self.devices, self.copy_streams)):
            with torch.cuda.stream(st):
                if i == 0:
                    ex = [e.to(self.dev, non_blocking=True) for e in extra]
                parts.append(xp.to(d, non_blocking=True))
                ev = torch.cuda.Event()
                ev.record(st)
            events.append(ev)
        return parts, ex, events

    def ready(self, pending):
        """Make the devices' compute streams wait for a pending batch's copies; returns (parts, extra)."""
        parts, ex, events = pending
        for g, d, ev in zip(parts, self.devices, events):
            cs = torch.cuda.current_stream(d)
            cs.wait_event(ev)
            g.record_stream(cs)                 # memory made on the copy stream, used on the compute stream
        cs0 = torch.cuda.current_stream(self.dev)
        for e in ex:
            e.record_stream(cs0)
        return parts, ex

    @staticmethod
    def copied(pending):
        """Wait (on the CPU) until a pending batch's copies are done, so its buffer slots can be freed."""
        for ev in pending[2]:
            ev.synchronize()

    def load(self, x):
        """x (B, T) int16 on the CPU -> one part per device."""
        parts = torch.tensor_split(x, len(self.devices))
        return list(self.pool.map(lambda pd: pd[0].to(pd[1], non_blocking=True), zip(parts, self.devices)))

    @torch.no_grad()
    def fixed(self, parts):
        """-> (B, K, J, J) log10(count + 1) at the fixed ladder."""
        a = self.a

        def work(i, x):
            out = [self.grind(x[s:s + a.hist_rows], self.thr_fixed[i], self.lut_fixed[i], a.fixed_j)
                   for s in range(0, x.shape[0], a.hist_rows)]
            return torch.log10(torch.cat(out) + 1.0) if out else x.new_zeros((0, a.fixed_k, a.fixed_j, a.fixed_j),
                                                                              dtype=torch.float32)
        return self._map(work, parts)

    @torch.no_grad()
    def selfcal(self, parts, baseline_pA, amp_pA):
        """Per row: K thresholds from baseline - margin to baseline + amplitude + margin."""
        a = self.a
        lo = baseline_pA - a.margin_pA
        hi = baseline_pA + amp_pA + a.margin_pA
        t = torch.linspace(0.0, 1.0, a.selfcal_k, device=self.dev).unsqueeze(0)
        thr = torch.round((lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * t) * self.c).clamp(-32768, 32767).to(torch.int16)
        thr_parts = torch.split(thr, [p.shape[0] for p in parts])

        def work(i, x, th):
            th = th.to(x.device)
            out = [self.grind(x[s:s + a.hist_rows], th[s:s + a.hist_rows], self.lut_selfcal[i], a.selfcal_j)
                   for s in range(0, x.shape[0], a.hist_rows)]
            return torch.log10(torch.cat(out) + 1.0) if out else x.new_zeros((0, a.selfcal_k, a.selfcal_j, a.selfcal_j),
                                                                              dtype=torch.float32)
        return self._map(work, parts, thr_parts)


def pin_host(t):
    """
    Page-lock an existing CPU tensor in place (works for shared memory too), so copies to the GPUs go
    straight by DMA instead of through a CPU staging copy. Returns True if it worked.
    """
    if not torch.cuda.is_available() or t.numel() == 0:
        return False
    try:
        torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        return bool(t.is_pinned())
    except Exception:
        return False


def gpu_noise_producer(noise_buf, mag, devices, n_written, stop_flag, chunk=16):
    """
    BENCHMARK ONLY (--bench-noise gpu): synthesize noise on the given GPUs, as SpectralNoise does, and
    copy it into the shared ring noise_buf, staying about one ring ahead of the simulators. It charges the
    real costs of moving noise synthesis to the GPUs (GPU time, copies back to CPU memory); which noise a
    sample gets is not tracked, so a run like this is for timing only.
    """
    R, N = noise_buf.shape
    mags = [torch.from_numpy(np.ascontiguousarray(mag[0])).to(d) for d in devices]
    produced = 0
    while not stop_flag[0]:
        if produced - n_written.value > R - chunk:
            time.sleep(0.002)
            continue
        i = (produced // chunk) % len(devices)
        d = devices[i]
        with torch.cuda.device(d):
            phi = torch.rand(chunk, mags[i].numel(), device=d) * (2 * np.pi)
            Y = torch.polar(mags[i].expand(chunk, -1), phi)
            x = torch.fft.irfft(Y, n=2 * (mags[i].numel() - 1))[:, :N]
            x = x - x.mean(dim=1, keepdim=True)
            x = (x * (5000.0 / x.std(dim=1, keepdim=True))).clamp(-32767, 32767).round().to(torch.int16)
            s0 = produced % R
            noise_buf[s0:s0 + chunk].copy_(x)
        produced += chunk


def forward(core, hists, x, fixed_hist=None, parts=None):
    """
    x (B, T) int16 on the CPU, or parts = hists.load(x) already on the GPUs.
    Returns (9 outputs, baseline estimate pA, amplitude estimate pA).
    """
    if parts is None:
        parts = hists.load(x)
    if fixed_hist is None:
        fixed_hist = hists.fixed(parts)
    f_fixed, base_hat, amp_hat = core.fixed_stage(fixed_hist)
    selfcal_hist = hists.selfcal(parts, base_hat.detach(), amp_hat.detach())
    return core.finish(f_fixed, selfcal_hist, base_hat, amp_hat), base_hat, amp_hat


def build_core(args, y_dim, device):
    return HardSelfCalibratingFusionNet(args.fixed_k, args.fixed_j, args.selfcal_k, args.selfcal_j, num_outputs=y_dim,
                                        base_scale_pA=args.base_scale_pA, amp_min_pA=args.amp_min_pA,
                                        backbone=args.backbone, convnext_hidden=args.hidden,
                                        convnext_depth=args.depth, convnext_kernel_size=args.kernel_size).to(device)


def timed(tm, name, fn, devices):
    """--profile-epoch: run fn with the GPUs synchronized before and after it; append its wall time (ms) to tm[name]."""
    for d in devices:
        torch.cuda.synchronize(d)
    t = time.perf_counter()
    with torch.profiler.record_function(name):       # the stage's span in profile_trace.json
        out = fn()
        for d in devices:
            torch.cuda.synchronize(d)
    tm.setdefault(name, []).append(1000 * (time.perf_counter() - t))
    return out


class TwoLook(torch.nn.Module):
    """Both looks in one forward(), so DistributedDataParallel can wrap them (it averages the gradients in backward)."""

    def __init__(self, core, hists):
        super().__init__()
        self.core, self.hists = core, hists
        self.tm = None                          # a dict while profiling: forward() then times its stages into it

    def forward(self, parts, fixed_hist):
        if self.tm is None:
            return forward(self.core, self.hists, None, fixed_hist, parts)
        tm, devs = self.tm, self.hists.devices
        f_fixed, base_hat, amp_hat = timed(tm, "1st backbone + level estimate",
                                           lambda: self.core.fixed_stage(fixed_hist), devs)
        selfcal = timed(tm, "self-calibrated histograms (GPUs)",
                        lambda: self.hists.selfcal(parts, base_hat.detach(), amp_hat.detach()), devs)
        pred = timed(tm, "2nd backbone + head", lambda: self.core.finish(f_fixed, selfcal, base_hat, amp_hat), devs)
        return pred, base_hat, amp_hat


STAGES_DDP = ["wait for simulated traces", "copy batch to the GPUs", "1st backbone + level estimate",
              "self-calibrated histograms (GPUs)", "2nd backbone + head", "losses",
              "backward (+ gradient all-reduce)", "clip + optimizer step", "free the buffer slots", "STEP TOTAL"]


def ddp_helper(rank, world, args, trace_len, fs, c, y_dim, n_rates, cond, X_buf, y_buf, H_buf, ready, steps_allowed,
               lr_now, stop):
    """
    --train-gpus N: trainer rank 1..N-1 on GPU <rank>. Step k uses buffer slots k * B .. (k + 1) * B (as the main
    trainer, rank 0, which takes slice 0 and frees the slots); this rank takes slice <rank> of them, builds
    its histograms on its own GPU, and DDP averages the gradients with the other ranks. It runs the steps
    rank 0 allows (steps_allowed, raised at the start of each epoch) at the learning rate rank 0 sets.
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group("nccl", rank=rank, world_size=world)
    if args.pin_memory:
        for t in (X_buf, H_buf):
            pin_host(t)
    hists = Hists(args, trace_len, fs, c, device, devices=[device])
    torch.manual_seed(args.seed)
    core = build_core(args, y_dim, device)
    net = DDP(TwoLook(core, hists), device_ids=[rank])       # weights broadcast from rank 0
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit, level_crit = LogCoshLoss(), LogCoshLoss()
    lvl_mean = torch.tensor(cond["mean"][:2], device=device, dtype=torch.float32)
    lvl_std = torch.tensor(cond["std"][:2], device=device, dtype=torch.float32)
    B, b = args.batch_size, args.batch_size // world
    S, step, tm = args.steps_per_epoch, 0, {}
    devs = [device]
    while True:
        while step >= steps_allowed.value and not stop.value:
            time.sleep(0.002)
        if step >= steps_allowed.value:
            break
        for g in opt.param_groups:
            g["lr"] = lr_now.value
        profiling = step // S + 1 == args.profile_epoch
        # while profiling, run() times a stage with this GPU synchronized around it; otherwise it just runs it
        run = (lambda name, fn: timed(tm, name, fn, devs)) if profiling else (lambda name, fn: fn())
        net.module.tm = tm if profiling else None
        t_step = time.perf_counter()
        a = (step * B) % args.buffer_size + rank * b

        def wait():
            while not bool(ready[a:a + b].all()):
                if stop.value:
                    return False
                time.sleep(0.002)
            return True
        if not run("wait for simulated traces", wait):
            return
        parts, yb, hb = run("copy batch to the GPUs", lambda: (
            hists.load(X_buf[a:a + b]), y_buf[a:a + b].to(device), H_buf[a:a + b].to(device)))
        torch.cuda.synchronize(device)          # copied before rank 0 can free the slots (after the all-reduce)
        opt.zero_grad(set_to_none=True)
        pred, base_hat, amp_hat = net(parts, hb)
        level_true = (yb[:, n_rates:n_rates + 2] * lvl_std + lvl_mean) / c
        loss = run("losses", lambda: crit(pred, yb) + args.level_aux_weight * level_crit(
            torch.stack([base_hat, amp_hat], dim=1), level_true))
        run("backward (+ gradient all-reduce)", loss.backward)
        run("clip + optimizer step", lambda: (
            torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=args.clip_grad_norm), opt.step()))
        if profiling:
            tm.setdefault("STEP TOTAL", []).append(1000 * (time.perf_counter() - t_step))
        step += 1
        if step % S == 0:       # rank 0 validates on this GPU too: leave it the memory this rank's cache holds
            torch.cuda.empty_cache()
        if profiling and step % S == 0:         # the profiled epoch is over: hand the times to rank 0
            path = os.path.join(args.output_root, f"profile_rank{rank}.json")
            with open(path + ".tmp", "w") as f:
                json.dump(tm, f)
            os.replace(path + ".tmp", path)
    dist.destroy_process_group()


@torch.no_grad()
def evaluate(core, hists, X, fixed, y, batch_size, device, crit, streams=False, tm=None):
    """
    X (N, T) int16 on the CPU, fixed (N, K, J, J) on the device. Returns (loss, RAE per output, preds).
    streams (X page-locked): copy the next batch while the current one is computed.
    tm (--profile-epoch, without streams): a dict; each stage of each batch is timed into it (GPUs synchronized).
    """
    core.eval()
    loss_sum, rae_sum, preds = 0.0, torch.zeros(y.shape[1], device=device), []
    starts = list(range(0, X.shape[0], batch_size))
    pend = hists.load_async(X[:batch_size], [y[:batch_size]]) if streams else None
    for n, s in enumerate(starts):
        if streams:
            cur = pend
            if n + 1 < len(starts):
                t = starts[n + 1]
                pend = hists.load_async(X[t:t + batch_size], [y[t:t + batch_size]])
            parts, (yb,) = hists.ready(cur)
            p, _, _ = forward(core, hists, None, fixed[s:s + batch_size], parts=parts)
        elif tm is not None:
            devs, fb = hists.devices, fixed[s:s + batch_size]
            parts, yb = timed(tm, "copy batch to the GPUs", lambda: (
                hists.load(X[s:s + batch_size]), y[s:s + batch_size].to(device)), devs)
            f_fixed, base_hat, amp_hat = timed(tm, "1st backbone + level estimate", lambda: core.fixed_stage(fb), devs)
            selfcal = timed(tm, "self-calibrated histograms (GPUs)",
                            lambda: hists.selfcal(parts, base_hat, amp_hat), devs)
            p = timed(tm, "2nd backbone + head", lambda: core.finish(f_fixed, selfcal, base_hat, amp_hat), devs)
        else:
            yb = y[s:s + batch_size].to(device, non_blocking=True)
            p, _, _ = forward(core, hists, X[s:s + batch_size], fixed[s:s + batch_size])
        t_m = time.perf_counter()
        p = p.float()
        loss_sum += float(crit(p, yb)) * yb.shape[0]
        rae_sum += rae_log(p, yb).sum(dim=0)
        preds.append(p.cpu())
        if tm is not None:
            tm.setdefault("loss + RAE + predictions to the CPU", []).append(1000 * (time.perf_counter() - t_m))
    n = X.shape[0]
    return loss_sum / n, (rae_sum / n).cpu().tolist(), torch.cat(preds)


def take_traces(X_buf, y_buf, data_lock, n_written, n_read, batch_size, timeout_s=900.0):
    """
    The next batch_size unused traces, in arrival order; waits for the simulators if needed.
    Like train_hist_online.take_batch, but the 2 MB traces are copied outside the lock: the simulators
    never write into a slot that has not been read, and a slot counts as read only once n_read moves.
    Returns (x, y, seconds waited for data). Used only when a batch can wrap around the buffer end;
    otherwise see peek_traces.
    """
    size = X_buf.shape[0]
    t0 = time.time()
    while n_written.value - n_read.value < batch_size:
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no new samples for {timeout_s:.0f}s; did the simulator processes die?")
        time.sleep(0.002)
    waited = time.time() - t0
    a = n_read.value % size
    b = a + batch_size
    if b <= size:
        x, y = X_buf[a:b].clone(), y_buf[a:b].clone()
    else:
        x, y = torch.cat([X_buf[a:], X_buf[:b - size]]), torch.cat([y_buf[a:], y_buf[:b - size]])
    with data_lock:
        n_read.value += batch_size
    return x, y, waited


def thread_sched():
    """{thread id: (name, seconds on a CPU, seconds runnable but waiting for one)} of this process's threads."""
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/schedstat") as f:
                run_ns, wait_ns, _ = f.read().split()
            with open(f"/proc/self/task/{tid}/comm") as f:
                name = f.read().strip()
        except OSError:                         # the thread ended meanwhile
            continue
        out[tid] = (name, int(run_ns) * 1e-9, int(wait_ns) * 1e-9)
    return out


def peek_traces(X_buf, y_buf, ready, n_read, batch_size, timeout_s=900.0):
    """
    Like take_traces, but returns views into the shared buffer (no 256 MB copy on the CPU), for buffers
    that are a multiple of the batch size, so a batch never wraps around the end. The slots stay
    reserved until release_traces() moves n_read, which the caller does once the traces are on the GPUs.
    Returns (x view, y view, seconds waited for data).
    """
    size = X_buf.shape[0]
    t0 = time.time()
    a = n_read.value % size
    while not bool(ready[a:a + batch_size].all()):          # every slot of the batch filled
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no new samples for {timeout_s:.0f}s; did the simulator processes die?")
        time.sleep(0.002)
    return X_buf[a:a + batch_size], y_buf[a:a + batch_size], time.time() - t0


def release_traces(data_lock, n_read, batch_size, ready):
    a = n_read.value % ready.shape[0]
    ready[a:a + batch_size] = 0                 # before n_read moves, so a refilled slot can't be cleared
    with data_lock:
        n_read.value += batch_size


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--holdout-noise", default=None, metavar="NAME")
    p.add_argument("--train-noise", default=None, metavar="NAME")

    # data: the buffer is only a queue (every sample is used once); one trace is 2 MB
    p.add_argument("--buffer-size", type=int, default=2048, help="traces waiting for the trainer")
    p.add_argument("--gen-workers", type=int, default=0, help="simulator processes (0 = all CPUs but two)")
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=10000)

    # training: batch size, epoch length and patience as in train_hist_online.py
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=156)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad-norm", type=float, default=5.0)
    p.add_argument("--level-aux-weight", type=float, default=0.3)
    p.add_argument("--target-scale", type=float, default=1.155, help="scale of the 3 standardized extra targets")

    # network
    p.add_argument("--backbone", default="convnext_v2",
                   choices=["inception", "small_cnn", "small_cnn_3d", "convnext_v2", "convnext_v2_nogrn"])
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--kernel-size", type=int, default=7)
    p.add_argument("--base-scale-pA", type=float, default=2.0)
    p.add_argument("--amp-min-pA", type=float, default=1.0)

    # histograms
    p.add_argument("--fixed-k", type=int, default=24)
    p.add_argument("--fixed-j", type=int, default=12)
    p.add_argument("--fixed-thr-pA", type=float, nargs=2, default=[-1.67, 21.67], metavar=("LO", "HI"))
    p.add_argument("--selfcal-k", type=int, default=24)
    p.add_argument("--selfcal-j", type=int, default=12)
    p.add_argument("--margin-pA", type=float, default=1.0)
    p.add_argument("--hist-rows", type=int, default=32, help="traces per GPU histogram call (bounds GPU memory)")
    p.add_argument("--k-group", type=int, default=24, help="thresholds per GPU histogram group")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false",
                   help="don't page-lock the shared buffer and the val / test traces (copies to the GPUs then go "
                        "through a CPU staging copy)")
    p.add_argument("--cuda-streams", action="store_true",
                   help="copy the next batch to the GPUs on separate CUDA streams while the current one is "
                        "computed (needs page-locked memory)")
    p.add_argument("--noise-on-gpu", action="store_true",
                   help="make the training samples' noise on the GPUs (all but the first, which trains) instead "
                        "of in the simulator processes; validation / test noise stays on the CPU")
    p.add_argument("--bench-noise", default="none", choices=["none", "cached", "gpu"],
                   help="BENCHMARK ONLY, results are not valid: 'cached' reuses one noise per recording (CPU "
                        "noise cost zero), 'gpu' makes noise on GPUs 2.. and hands it to the simulators")
    p.add_argument("--hist-impl", default="fast", choices=["fast", "cuda", "reference"],
                   help="GPU histogram code: the lean scripts/hist_pipeline/gpu_hist_fast.py, the one-pass CUDA "
                        "kernel scripts/hist_pipeline/gpu_hist_cuda.py (both identical results) or the copied "
                        "reference grind_hist_gpu")
    p.add_argument("--hist-gpus", type=int, default=0, help="GPUs that build histograms (0 = all in the job)")
    p.add_argument("--fixed-hist-on-gpu", action="store_true",
                   help="build the fixed-ladder histograms on the GPUs (default: the simulator processes build them "
                        "on the CPU, with the simulator's own histogram code, which gives identical histograms)")
    p.add_argument("--synth-on-gpu", action="store_true",
                   help="synthesize the training traces on the GPUs (scripts/hist_pipeline/gpu_synth.py): the "
                        "simulators only run Gillespie, the GPUs filter, add the GPU noise and round; implies "
                        "--noise-on-gpu, needs --fixed-hist-on-gpu and one trainer; validation / test stay on the CPU")
    p.add_argument("--profile-epoch", type=int, default=0, metavar="N",
                   help="PROFILING: in epoch N (path without --cuda-streams), time every stage of each training "
                        "step with the GPUs synchronized around it, and run torch.profiler on its last steps; "
                        "writes profile_epoch.txt and profile_trace.json next to the results")
    p.add_argument("--train-gpus", type=int, default=1,
                   help="GPUs that train the network (data parallel, one process each; the batch is split between "
                        "them and each builds its share's histograms). Validation still uses all GPUs")
    p.add_argument("--deterministic", action="store_true",
                   help="for one-to-one comparisons: the trainer gets the samples in index order (sample i always "
                        "in buffer slot i %% buffer size) and the GPUs use deterministic algorithms, so two runs with "
                        "the same options train identically; slower, so time with the normal mode (needs one "
                        "trainer)")
    args = p.parse_args()
    if args.synth_on_gpu:
        args.noise_on_gpu = True
    return args


def main():
    args = parse_args()
    args.output_root = os.path.join(args.output_root, f"seed{args.seed}")
    os.makedirs(args.output_root, exist_ok=True)
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # needed by cuBLAS, before its first use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not args.deterministic      # benchmark mode picks algorithms by timing
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)      # warns about any op without a deterministic version

    config_path = prepare_sim_config(args.sim_config, os.path.join(args.output_root, "sim_config.json"), args.seed,
                                     holdout_noise=args.holdout_noise, train_noise=args.train_noise)
    fixed_on_cpu = not args.fixed_hist_on_gpu
    if fixed_on_cpu:
        # the simulator's histograms (sd_simulate_hist) become the network's fixed ladder
        with open(config_path) as f:
            jcfg = json.load(f)
        jcfg["analysis"] = {"hist_backend": "grind",
                            "hist2d": {"tmin_s": DWELL_MIN_S, "tmax_s": DWELL_MAX_S, "bins": args.fixed_j},
                            "grind": {"min_pA": args.fixed_thr_pA[0], "max_pA": args.fixed_thr_pA[1],
                                      "count": args.fixed_k}}
        with open(config_path, "w") as f:
            json.dump(jcfg, f, indent=2)
        if args.buffer_size % args.batch_size:
            raise SystemExit("with the fixed histograms on the CPU, --buffer-size must be a multiple of --batch-size")
    probe = TraceSimulator(config_path)
    if fixed_on_cpu and (probe.hist_detectors, probe.hist_bins) != (args.fixed_k, args.fixed_j):
        raise SystemExit(f"simulator histograms {probe.hist_detectors}x{probe.hist_bins} != fixed ladder "
                         f"{args.fixed_k}x{args.fixed_j}")
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c = probe.trace_len, sim_cfg.counts_per_pA
    fs = float(probe.config["sampling"]["rate_hz"])
    kernel = probe.kernel() if args.synth_on_gpu else None
    probe.close()
    n_rates = 2 * (sim_cfg.num_states - 1)
    if c <= 0:
        raise SystemExit("the sim-config needs an 'amplifier' block (levels in pA)")

    # targets: 6 log10 rates + standardized [baseline, amplitude, sigma] * target_scale
    cond = condition_norm(sim_cfg)
    if cond["index"] != [0, 1, 2]:
        raise SystemExit("baseline, amplitude and sigma must all vary in the sim-config for this network")
    cond["std"] = [s / args.target_scale for s in cond["std"]]
    cond["unit"], cond["counts_per_unit"] = "pA", c
    with open(os.path.join(args.output_root, "conditions.json"), "w") as f:
        json.dump(cond, f, indent=2)
    y_dim = n_rates + 3

    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.output_root)
    names = lambda paths: ", ".join(os.path.basename(p) for p in paths)
    print(f"[Parent] sim-config={args.sim_config} topology={sim_cfg.topology} device={device}\n"
          f"[Parent]   train / val: {names(sim_cfg.noise_paths)}\n"
          f"[Parent]   test:        {names(sim_cfg.noise_test_paths) or 'same files as training'}", flush=True)

    n_cpus = len(os.sched_getaffinity(0))
    if args.buffer_size < args.batch_size:
        raise SystemExit("--buffer-size must be at least the batch size")

    world = max(1, min(args.train_gpus, torch.cuda.device_count()))
    if args.deterministic and world > 1:
        raise SystemExit("--deterministic needs one trainer (--train-gpus 1)")
    if world > 1:
        bad = [msg for cond_, msg in [
            (not fixed_on_cpu, "the fixed-ladder histograms built on the CPU"),
            (args.cuda_streams, "no --cuda-streams"),
            (args.batch_size % world, f"a batch size divisible by {world}"),
            (args.buffer_size < 2 * args.batch_size, "--buffer-size of at least 2 batches")] if cond_]
        if bad:
            raise SystemExit(f"--train-gpus {world} needs: {'; '.join(bad)}")
    if args.synth_on_gpu:
        bad = [msg for cond_, msg in [
            (fixed_on_cpu, "--fixed-hist-on-gpu"), (world > 1, "one trainer"), (args.cuda_streams, "no --cuda-streams"),
            (args.bench_noise != "none", "no --bench-noise"), (device.type != "cuda", "a GPU"),
            (sim_cfg.noise_mode != "spectral_resample", "spectral_resample noise")] if cond_]
        if bad:
            raise SystemExit(f"--synth-on-gpu needs: {'; '.join(bad)}")
    if args.gen_workers <= 0:
        args.gen_workers = max(1, n_cpus - 2 - (world - 1))          # the other trainers need cores too
    hists = Hists(args, trace_len, fs, c, device)
    # with several trainers, each builds its own share's training histograms on its own GPU
    train_hists = Hists(args, trace_len, fs, c, device, devices=[device]) if world > 1 else hists
    core = build_core(args, y_dim, device)
    n_params = sum(p.numel() for p in core.parameters())

    train_config = {
        "model": {"name": "HardSelfCalibratingFusionNet", "backbone": args.backbone, "hidden": args.hidden,
                  "depth": args.depth, "kernel_size": args.kernel_size, "parameters": n_params,
                  "num_outputs": y_dim, "num_rates": n_rates, "extra_outputs": cond["names"],
                  "base_scale_pA": args.base_scale_pA, "amp_min_pA": args.amp_min_pA},
        "histogram_devices": len(hists.devices), "histogram_code": args.hist_impl,
        "training_noise_made_on": "gpu" if args.noise_on_gpu else "cpu (simulators)",
        "training_traces_synthesized_on": "gpu" if args.synth_on_gpu else "cpu (simulators)",
        "cuda_streams": args.cuda_streams, "train_gpus": world, "deterministic": args.deterministic,
        "fixed_histograms_built_on": "cpu (simulator)" if fixed_on_cpu else "gpu",
        "histograms": {"fixed": {"thresholds": args.fixed_k, "bins": args.fixed_j, "range_pA": args.fixed_thr_pA},
                       "selfcal": {"thresholds": args.selfcal_k, "bins": args.selfcal_j, "margin_pA": args.margin_pA},
                       "dwell_range_s": [DWELL_MIN_S, DWELL_MAX_S]},
        "optimizer": {"name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay},
        "loss": "LogCoshLoss", "level_aux_weight": args.level_aux_weight, "target_scale": args.target_scale,
        "lr_scheduler": {"name": "ReduceLROnPlateau", "monitor": "val_loss", "factor": LR_FACTOR,
                         "patience": LR_PATIENCE, "min_lr": MIN_LR},
        "early_stopping": {"monitor": "val_loss", "patience": args.patience, "min_delta": args.min_delta},
        "grad_clip_norm": args.clip_grad_norm,
        "batch_size": args.batch_size, "steps_per_epoch": args.steps_per_epoch,
        "samples_per_epoch": args.steps_per_epoch * args.batch_size, "max_epochs": args.epochs, "seed": args.seed,
        "data": {"sim_config": args.sim_config, "no_reuse": True, "buffer_size": args.buffer_size,
                 "gen_workers": args.gen_workers, "n_val": args.n_val, "n_test": args.n_test,
                 "noise_mode": sim_cfg.noise_mode, "noise_files_train_val": sim_cfg.noise_paths,
                 "noise_files_test": sim_cfg.noise_test_paths or sim_cfg.noise_paths},
    }
    train_config["git"] = git_info()
    with open(os.path.join(args.output_root, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)
    print(f"[Parent] backbone={args.backbone} params={n_params:,} cpus={n_cpus} gen_workers={args.gen_workers} "
          f"buffer={args.buffer_size} traces of {trace_len} points | histograms on {len(hists.devices)} device(s)",
          flush=True)

    ctx = mp.get_context("spawn")

    # 1. Fixed val / test sets, as raw traces
    t0 = time.time()
    mode = "both" if fixed_on_cpu else True
    h_shape = (args.fixed_k, args.fixed_j, args.fixed_j) if fixed_on_cpu else None
    with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                  initargs=(config_path, sim_cfg, args.seed, spectrum_path, cond, mode)) as pool:
        val = simulate_fixed_set(pool, STREAM_VAL, args.n_val, (trace_len,), y_dim, torch.int16, h_shape)
        test = simulate_fixed_set(pool, STREAM_TEST, args.n_test, (trace_len,), y_dim, torch.int16, h_shape)
    X_val, y_val = val[0], val[1]
    if args.pin_memory:
        t_pin = time.time()
        pinned = [pin_host(t) for t in (X_val, test[0])]
        print(f"[Parent] page-locked val / test traces: {pinned} in {time.time() - t_pin:.1f}s", flush=True)
    X_test, y_test, aux_test, noise_src_test = test[:4]
    np.save(os.path.join(args.output_root, "aux_test.npy"), aux_test)
    np.save(os.path.join(args.output_root, "noise_source_test.npy"), noise_src_test)
    dt = time.time() - t0
    print(f"[Parent] Simulated val={args.n_val} test={args.n_test} in {dt:.1f}s "
          f"({(args.n_val + args.n_test) / max(dt, 1e-9):.1f} samples/s)", flush=True)

    # their fixed-ladder histograms never change: build them once
    t0 = time.time()
    fixed_of = lambda X: torch.cat([hists.fixed(hists.load(X[s:s + args.batch_size]))
                                    for s in range(0, X.shape[0], args.batch_size)])
    if fixed_on_cpu:
        fixed_val, fixed_test = val[4].to(device), test[4].to(device)
    else:
        fixed_val, fixed_test = fixed_of(X_val), fixed_of(X_test)
        print(f"[Parent] fixed-ladder histograms of val / test built in {time.time() - t0:.1f}s", flush=True)

    # 2. Queue of fresh training traces + background simulators
    if args.buffer_size % args.batch_size:
        raise SystemExit("--buffer-size must be a multiple of --batch-size")
    zero_copy = True
    X_buf = torch.zeros((args.buffer_size, trace_len), dtype=torch.int16).share_memory_()
    y_buf = torch.zeros((args.buffer_size, y_dim), dtype=torch.float32).share_memory_()
    H_buf = (torch.zeros((args.buffer_size, *h_shape), dtype=torch.float32).share_memory_() if fixed_on_cpu
             else None)
    if args.pin_memory:
        print(f"[Parent] page-locked training buffer: {[pin_host(t) for t in (X_buf, H_buf) if t is not None]}",
              flush=True)
    data_lock = ctx.Lock()
    next_index = ctx.Value("q", 0)
    n_written = ctx.Value("q", 0, lock=False)
    n_read = ctx.Value("q", 0, lock=False)
    stop = ctx.Value("b", 0, lock=False)
    n_reserved = ctx.Value("q", 0, lock=False)  # only modified under data_lock
    ready = torch.zeros(args.buffer_size, dtype=torch.uint8).share_memory_()
    # --synth-on-gpu: X_buf holds the uint16 averaged levels, M_buf per slot [sample index, lo, hi, noise scale]
    M_buf = torch.zeros((args.buffer_size, 4), dtype=torch.float64).share_memory_() if args.synth_on_gpu else None
    noise_buf = None
    if args.bench_noise == "gpu":
        noise_buf = torch.zeros((1024, trace_len), dtype=torch.int16).share_memory_()
        if args.pin_memory:
            pin_host(noise_buf)
    if args.bench_noise != "none":
        print(f"[Parent] BENCHMARK: --bench-noise {args.bench_noise}; results of this run are for timing only",
              flush=True)
    idle = ctx.Value("d", 0.0)                  # simulator seconds spent waiting for room in the buffer
    sim_times = ctx.Array("d", SIM_DETAIL)      # simulator seconds: [GPU noise wait, simulate, room wait, copy,
                                                #   and within simulate: Gillespie, trace synthesis, histograms; SIM_DETAIL_NAMES]
    producer = None
    if args.noise_on_gpu:
        from scripts.hist_pipeline.gpu_noise import GPUNoiseProducer
        # all of the job's GPUs but the first, also with --hist-gpus 1 (then GPU 0 trains, the others make noise)
        all_gpus = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        noise_devs = all_gpus[1:] or hists.devices
        # --synth-on-gpu: the noise stays on the (first) noise GPU; a larger ring, since it is freed only once the
        # trainer has used a sample, so it must cover every sample between the buffer's oldest and the newest
        producer = GPUNoiseProducer(np.load(spectrum_path), trace_len, noise_devs,
                                    list(sim_cfg.noise_sources(test=False)), args.seed, STREAM_TRAIN,
                                    ring=4096 if args.synth_on_gpu else 1024,
                                    pin=pin_host if args.pin_memory else None,
                                    ring_device=noise_devs[0] if args.synth_on_gpu else None)
        producer.start()
        print(f"[Parent] training noise made on {[str(d) for d in producer.devices]}", flush=True)
    gsynth = None
    if args.synth_on_gpu:
        from scripts.hist_pipeline.gpu_synth import GPUSynth
        gsynth = GPUSynth(kernel, producer, hists.devices)
        print(f"[Parent] training traces synthesized on {[str(d) for d in hists.devices]} "
              f"({len(kernel)}-sample step response), noise ring on {producer.ring.device}", flush=True)
    gens = [ctx.Process(target=generator_loop,
                        args=(config_path, sim_cfg, args.seed, spectrum_path, X_buf, y_buf, data_lock, next_index,
                              n_written, stop, n_read, cond, "levels" if gsynth else mode, H_buf, n_reserved, ready,
                              idle, None if args.bench_noise == "none" else args.bench_noise, noise_buf,
                              producer.handles() if producer and not gsynth else None, sim_times, args.deterministic,
                              M_buf), daemon=True)
            for _ in range(args.gen_workers)]
    producer_stop = [False]
    if args.bench_noise == "gpu":
        import threading
        noise_devices = hists.devices[1:] or hists.devices
        threading.Thread(target=gpu_noise_producer, daemon=True,
                         args=(noise_buf, np.load(spectrum_path), noise_devices, n_written, producer_stop)).start()
        print(f"[Parent] BENCHMARK: noise made on {[str(d) for d in noise_devices]}", flush=True)
    for g in gens:
        g.start()

    streams = args.cuda_streams and device.type == "cuda" and args.pin_memory and bool(X_buf.is_pinned())
    if args.cuda_streams and not streams:
        print("[Parent] --cuda-streams needs CUDA and page-locked memory; copying without streams", flush=True)
    pend, fetch_wait = None, [0.0]
    print(f"[Parent] CUDA streams for copies: {streams}", flush=True)

    def fetch(block):
        """Start copying the next unreleased batch if all its slots are filled (block: wait for them)."""
        t0 = time.time()
        a = n_read.value % args.buffer_size
        while not bool(ready[a:a + args.batch_size].all()):
            if not block:
                return None
            time.sleep(0.002)
        fetch_wait[0] += time.time() - t0
        extra = [y_buf[a:a + args.batch_size]] + ([H_buf[a:a + args.batch_size]] if fixed_on_cpu else [])
        return hists.load_async(X_buf[a:a + args.batch_size], extra)

    def profiled_epoch(n_prof=10):
        """
        --profile-epoch: the training steps of one epoch as in the path without streams, with every GPU
        synchronized before and after each stage, so a stage's wall time is its own (what normally overlaps,
        e.g. the CPU queuing the next kernels while the GPU works, no longer does: the epoch runs a little
        slower). The last n_prof steps also run under torch.profiler.
        Returns (main loss sum, level loss sum, seconds waited for data, {stage: [ms per step]}, profiler).
        """
        tm = {}

        def run(name, fn):
            return timed(tm, name, fn, train_hists.devices)

        B = args.batch_size
        if net is not None:     # several trainers: as in the training loop, this one takes slice 0 of each batch
            b = B // world
            prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA])
            tr, lv, waited = 0.0, 0.0, 0.0
            net.module.tm = tm
            for s in range(args.steps_per_epoch):
                if s == args.steps_per_epoch - n_prof:
                    prof.start()
                t_step = time.perf_counter()
                xb, yb, w = run("wait for simulated traces", lambda: peek_traces(X_buf, y_buf, ready, n_read, B))
                waited += w
                a = n_read.value % args.buffer_size
                parts, yb, hb = run("copy batch to the GPUs", lambda: (
                    train_hists.load(xb[:b]), yb[:b].to(device), H_buf[a:a + b].to(device)))
                opt.zero_grad(set_to_none=True)
                pred, base_hat, amp_hat = net(parts, hb)
                main_loss, level_loss = run("losses", lambda: (crit(pred, yb), level_crit(
                    torch.stack([base_hat, amp_hat], dim=1), (yb[:, n_rates:n_rates + 2] * lvl_std + lvl_mean) / c)))
                run("backward (+ gradient all-reduce)",
                    lambda: (main_loss + args.level_aux_weight * level_loss).backward())
                run("clip + optimizer step", lambda: (
                    torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=args.clip_grad_norm), opt.step()))
                tr += float(main_loss.detach())
                lv += float(level_loss.detach())
                run("free the buffer slots", lambda: release_traces(data_lock, n_read, B, ready))
                tm.setdefault("STEP TOTAL", []).append(1000 * (time.perf_counter() - t_step))
            prof.stop()
            net.module.tm = None
            return tr, lv, waited, {k: tm[k] for k in STAGES_DDP}, prof

        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                  torch.profiler.ProfilerActivity.CUDA])
        tr, lv, waited = 0.0, 0.0, 0.0
        for s in range(args.steps_per_epoch):
            if s == args.steps_per_epoch - n_prof:
                prof.start()
            t_step = time.perf_counter()
            xb, yb, w = run("wait for simulated traces", lambda: peek_traces(X_buf, y_buf, ready, n_read, B))
            waited += w
            a = n_read.value % args.buffer_size
            parts, yb, hb = run("copy batch to the GPUs", lambda: (
                hists.load(xb), yb.to(device), H_buf[a:a + B].to(device) if fixed_on_cpu else None))
            if gsynth:
                parts = run("noise + trace synthesis (GPUs)", lambda: gsynth(parts, M_buf[a:a + B]))
            if not fixed_on_cpu:
                hb = run("fixed-ladder histograms (GPUs)", lambda: hists.fixed(parts))
            run("free the buffer slots", lambda: release_traces(data_lock, n_read, B, ready))
            opt.zero_grad(set_to_none=True)
            f_fixed, base_hat, amp_hat = run("1st backbone + level estimate", lambda: core.fixed_stage(hb))
            selfcal = run("self-calibrated histograms (GPUs)",
                          lambda: hists.selfcal(parts, base_hat.detach(), amp_hat.detach()))
            main_loss, level_loss = run("2nd backbone + head + losses", lambda: (
                crit(core.finish(f_fixed, selfcal, base_hat, amp_hat), yb),
                level_crit(torch.stack([base_hat, amp_hat], dim=1), (yb[:, n_rates:n_rates + 2] * lvl_std + lvl_mean) / c)))
            run("backward", lambda: (main_loss + args.level_aux_weight * level_loss).backward())
            run("clip + optimizer step", lambda: (
                torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=args.clip_grad_norm), opt.step()))
            tr += float(main_loss.detach())
            lv += float(level_loss.detach())
            tm.setdefault("STEP TOTAL", []).append(1000 * (time.perf_counter() - t_step))
        prof.stop()
        return tr, lv, waited, tm, prof

    def report_profile(tm, prof, t_val, normal_epoch_s, val_tm, sim_dt, epoch_s, noise_dt=None, th_dt=None):
        steps = args.steps_per_epoch
        total = np.mean(tm["STEP TOTAL"])
        lines = [f"=== profile of epoch {args.profile_epoch}: {steps} training steps of batch {args.batch_size}, "
                 f"{f'{world} trainers (rank 0 shown), ' if world > 1 else ''}"
                 f"histograms on {len(train_hists.devices)} GPU(s), fixed-ladder histograms built on "
                 f"{'the CPU' if fixed_on_cpu else 'the GPUs'}, noise made on "
                 f"{'the GPUs' if producer else 'the CPU'}, training traces synthesized on "
                 f"{'the GPUs' if gsynth else 'the CPU'}",
                 f"{'stage':38s}{'ms/step':>9s}{'share':>8s}{'s/epoch':>9s}{'median':>9s}{'max':>9s}"]
        for k, v in tm.items():
            v = np.array(v)
            lines.append(f"{k:38s}{v.mean():9.1f}{100 * v.mean() / total:7.1f}%{v.mean() * steps / 1000:9.2f}"
                         f"{np.median(v):9.1f}{v.max():9.1f}")
        lines.append(f"{'validation (whole)':38s}{'':9s}{'':8s}{t_val:9.2f}")
        if val_tm:
            n_vb = len(val_tm["loss + RAE + predictions to the CPU"])
            lines.append(f"\n--- validation of this epoch: {n_vb} batches of {args.batch_size} "
                         f"(fixed-ladder histograms made once at the start), GPUs synchronized around each stage")
            lines.append(f"{'stage':38s}{'ms/batch':>9s}{'share':>8s}{'s/epoch':>9s}")
            v_tot = sum(np.sum(v) for v in val_tm.values()) / 1000
            for k, v in val_tm.items():
                lines.append(f"{k:38s}{np.mean(v):9.1f}{100 * np.sum(v) / 1000 / t_val:7.1f}%{np.sum(v) / 1000:9.2f}")
            lines.append(f"{'(between batches, not in a stage)':38s}{'':9s}{100 * (t_val - v_tot) / t_val:7.1f}%"
                         f"{t_val - v_tot:9.2f}")
        if sim_dt is not None:
            per = sim_dt / (args.gen_workers * epoch_s)
            names = ["waiting for noise from the GPU", "simulating trace + 1st-look histograms",
                     "waiting for a free buffer slot", "copying into the buffer"]
            lines.append(f"\n--- the {args.gen_workers} simulators during this epoch ({epoch_s:.1f}s, training and "
                         f"validation): share of each process's time, and seconds per trace")
            n_tr = args.steps_per_epoch * args.batch_size
            for j, (nm, sh, t) in enumerate(zip(names, per[:4], sim_dt[:4])):
                lines.append(f"{nm:44s}{100 * sh:6.1f}%{1000 * t / n_tr:9.1f} ms/trace")
                if j == 1 and len(sim_dt) > 4 and sim_dt[4:7].sum() > 0:     # the simulator's own split
                    parts = ["Gillespie (channel states)",
                             "averaged levels for the GPU (uint16)" if gsynth else "trace synthesis (filter + noise + int16)",
                             "1st-look histograms (Grind)"]
                    for nm2, sh2, t2 in zip(parts, per[4:7], sim_dt[4:7]):
                        lines.append(f"  {nm2:42s}{100 * sh2:6.1f}%{1000 * t2 / n_tr:9.1f} ms/trace")
                    rest = sim_dt[1] - sim_dt[4:7].sum()
                    lines.append(f"  {'parameter draws, Python, copies':42s}{100 * rest / (args.gen_workers * epoch_s):6.1f}%"
                                 f"{1000 * rest / n_tr:9.1f} ms/trace")
            # a timer is counted once its trace is done (flushed every 8): mostly slot waits still going on at the
            # end of the epoch (the trainer stops reading during prof.stop() and validation)
            lines.append(f"{'not counted yet (mostly slot waits going on)':44s}{100 * (1 - per[:4].sum()):6.1f}%")
            if len(sim_dt) >= SIM_DETAIL:
                d = dict(zip(SIM_DETAIL_NAMES, sim_dt[7:]))
                loop, win = d["whole loop (wall)"], args.gen_workers * epoch_s
                head, after = d["loop head (stop flag, next-index lock)"], d["after the copy (timer flush, n_written lock)"]
                cpu, rq = d["on a CPU (kernel)"], d["runnable, waiting for a CPU (kernel)"]
                lines.append(f"  where 'other' goes: the processes' own loop time (counted, like the timers above, when "
                             f"a trace is done, flushed every 8) is {100 * loop / win:.1f}% of {args.gen_workers} x "
                             f"the epoch; of that loop time:")
                lines.append(f"    {'the four timers above':40s}{100 * sim_dt[:4].sum() / loop:6.1f}%")
                for nm, t in [("loop head (stop flag, next-index lock)", head),
                              ("after the copy (timer flush, n_written)", after)]:
                    lines.append(f"    {nm:40s}{100 * t / loop:6.1f}%{1000 * t / n_tr:9.2f} ms/trace")
                lines.append(f"    {'not covered by any timer':40s}{100 * (loop - sim_dt[:4].sum() - head - after) / loop:6.1f}%")
                lines.append(f"  in the slot wait: lock waits {1000 * d['lock waits in the slot wait'] / n_tr:.2f} ms/trace, "
                             f"{d['sleep polls in the slot wait (count)'] / n_tr:.2f} sleep polls (5 ms) per trace")
                lines.append(f"  kernel scheduler, share of the loop time: on a CPU {100 * cpu / loop:.1f}%, runnable but "
                             f"waiting for a CPU {100 * rq / loop:.1f}%, asleep or blocked {100 * (loop - cpu - rq) / loop:.1f}%; "
                             f"per trace: {1000 * cpu / n_tr:.1f} ms on a CPU, {1000 * rq / n_tr:.1f} ms waiting for one, "
                             f"{d['involuntary context switches (count)'] / n_tr:.1f} involuntary and "
                             f"{d['voluntary context switches (count)'] / n_tr:.1f} voluntary context switches")
        if th_dt:
            lines.append(f"\n--- the trainer process's threads during this epoch ({epoch_s:.1f}s), by thread name "
                         f"(kernel scheduler)")
            lines.append(f"{'threads':28s}{'count':>6s}{'on a CPU':>10s}{'waiting for a CPU':>19s}")
            for nm, (n, run, wait) in sorted(th_dt.items(), key=lambda kv: -kv[1][1])[:12]:
                lines.append(f"{nm:28s}{n:6d}{run:9.2f}s{wait:18.2f}s")
        if noise_dt:
            lines.append(f"\n--- the GPU noise threads during this epoch ({epoch_s:.1f}s): busy = making a chunk of "
                         f"noise and copying it {'into the GPU ring' if gsynth else 'to the CPU'}, waiting = for free "
                         f"slots ({'trainer' if gsynth else 'simulators'} behind)")
            lines.append(f"{'thread':10s}{'chunks':>8s}{'traces/s':>10s}{'ms/chunk':>10s}{'busy':>8s}{'waiting':>9s}")
            for dev_, n_c, busy, wait in noise_dt:
                lines.append(f"{dev_:10s}{n_c:8.0f}{n_c * producer.chunk / epoch_s:10.0f}{1000 * busy / max(n_c, 1):10.1f}"
                             f"{100 * busy / epoch_s:7.1f}%{100 * wait / epoch_s:8.1f}%")
        if world > 1:           # the same stages on every trainer (each times its own GPU), mean ms per step
            per = [tm]
            for r in range(1, world):
                path, t_end = os.path.join(args.output_root, f"profile_rank{r}.json"), time.time() + 120
                while not os.path.exists(path) and time.time() < t_end:
                    time.sleep(0.5)
                per.append(json.load(open(path)) if os.path.exists(path) else {})
            lines.append(f"\n--- per trainer (GPU), ms/step: rank 0 also waits for and frees the slots")
            lines.append(f"{'stage':38s}" + "".join(f"{f'GPU {r}':>9s}" for r in range(world)))
            for k in STAGES_DDP:
                lines.append(f"{k:38s}" + "".join(f"{np.mean(p[k]):9.1f}" if p.get(k) else f"{'-':>9s}" for p in per))
        lines.append(f"training part of this profiled epoch: {total * steps / 1000:.1f}s; of the epoch before "
                     f"(normal, not synchronized): {normal_epoch_s:.1f}s" if normal_epoch_s else "")
        ka = prof.key_averages()
        lines += [f"\n--- GPU operations of the last profiled steps (all GPUs), by GPU time",
                  ka.table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=50),
                  f"\n--- the same steps, by CPU time spent in the operation itself",
                  ka.table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=50)]
        txt = "\n".join(lines)
        print(txt, flush=True)
        with open(os.path.join(args.output_root, "profile_epoch.txt"), "w") as f:
            f.write(txt + "\n")
        prof.export_chrome_trace(os.path.join(args.output_root, "profile_trace.json"))

    if args.profile_epoch and (streams or not zero_copy):
        raise SystemExit("--profile-epoch profiles the path without --cuda-streams")
    t_train_last = None

    trainers, net = [], None
    if world > 1:
        steps_allowed, lr_shared = ctx.Value("q", 0, lock=False), ctx.Value("d", args.lr, lock=False)
        with socket.socket() as sock:                   # a port nobody on this node uses
            sock.bind(("127.0.0.1", 0))
            os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "127.0.0.1", str(sock.getsockname()[1])
        trainers = [ctx.Process(target=ddp_helper, daemon=True,
                                args=(r, world, args, trace_len, fs, c, y_dim, n_rates, cond, X_buf, y_buf, H_buf,
                                      ready, steps_allowed, lr_shared, stop))
                    for r in range(1, world)]
        for t in trainers:
            t.start()
        dist.init_process_group("nccl", rank=0, world_size=world)
        net = DDP(TwoLook(core, train_hists), device_ids=[device.index or 0])
        print(f"[Parent] training on {world} GPUs (data parallel): {args.batch_size // world} samples of each batch "
              f"per GPU", flush=True)

    try:
        opt = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE,
                                                               min_lr=MIN_LR)
        es = EarlyStopping(patience=args.patience, min_delta=args.min_delta, restore_best=False)
        crit, level_crit = LogCoshLoss(), LogCoshLoss()
        lvl_mean = torch.tensor(cond["mean"][:2], device=device, dtype=torch.float32)
        lvl_std = torch.tensor(cond["std"][:2], device=device, dtype=torch.float32)

        metrics_csv = os.path.join(args.output_root, "train_val_metrics.csv")
        init_metrics_csv(metrics_csv, y_dim)
        best_path = os.path.join(args.output_root, "best_model.pt")
        best_val = float("inf")
        samples_per_epoch = args.steps_per_epoch * args.batch_size

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            written0, waited, idle0 = n_written.value, 0.0, idle.value
            core.train()
            tr_sum, lvl_sum = 0.0, 0.0
            if net is not None:                         # let the other trainers run this epoch's steps
                lr_shared.value = opt.param_groups[0]["lr"]
                steps_allowed.value += args.steps_per_epoch
            profiling = epoch == args.profile_epoch
            if profiling:
                sim0, th0 = np.array(sim_times[:]), thread_sched()
                noise0 = producer.thread_stats() if producer else None
                tr_sum, lvl_sum, waited, prof_times, prof = profiled_epoch()
            for _ in range(0 if profiling else args.steps_per_epoch):
                if net is not None:     # several trainers: this one takes slice 0 and frees the slots after the step
                    b = args.batch_size // world
                    xb, yb, w = peek_traces(X_buf, y_buf, ready, n_read, args.batch_size)
                    a = n_read.value % args.buffer_size
                    parts, yb, hb = train_hists.load(xb[:b]), yb[:b].to(device), H_buf[a:a + b].to(device)
                    torch.cuda.synchronize(device)
                elif streams:
                    cur = pend if pend is not None else fetch(block=True)
                    waited += fetch_wait[0]; fetch_wait[0] = 0.0
                    hists.copied(cur)                                   # on the GPUs: free its slots
                    release_traces(data_lock, n_read, args.batch_size, ready)
                    pend = fetch(block=False)                           # next batch copies during this step
                    parts, ex = hists.ready(cur)
                    yb, hb = ex[0], (ex[1] if fixed_on_cpu else None)
                    w = 0.0
                elif zero_copy:
                    xb, yb, w = peek_traces(X_buf, y_buf, ready, n_read, args.batch_size)
                    parts = hists.load(xb)
                    yb = yb.to(device)
                    if gsynth:
                        a = n_read.value % args.buffer_size
                        parts = gsynth(parts, M_buf[a:a + args.batch_size])
                    if fixed_on_cpu:
                        a = (n_read.value % args.buffer_size)
                        hb = H_buf[a:a + args.batch_size].to(device)
                    for d in parts:                 # the copies must be done before the slots are freed
                        if d.is_cuda:
                            torch.cuda.synchronize(d.device)
                    release_traces(data_lock, n_read, args.batch_size, ready)
                else:
                    xb, yb, w = take_traces(X_buf, y_buf, data_lock, n_written, n_read, args.batch_size)
                    parts = hists.load(xb)
                waited += w
                yb = yb.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                if net is not None:
                    pred, base_hat, amp_hat = net(parts, hb)
                else:
                    pred, base_hat, amp_hat = forward(core, hists, None, fixed_hist=hb if fixed_on_cpu else None,
                                                      parts=parts)
                main_loss = crit(pred, yb)
                # true baseline / amplitude in pA, recovered from the standardized targets
                level_true = (yb[:, n_rates:n_rates + 2] * lvl_std + lvl_mean) / c
                level_loss = level_crit(torch.stack([base_hat, amp_hat], dim=1), level_true)
                (main_loss + args.level_aux_weight * level_loss).backward()
                torch.nn.utils.clip_grad_norm_(core.parameters(), max_norm=args.clip_grad_norm)
                opt.step()
                tr_sum += float(main_loss.detach())
                lvl_sum += float(level_loss.detach())
                if net is not None:     # every trainer has its slice on its GPU once the gradients were averaged
                    release_traces(data_lock, n_read, args.batch_size, ready)
            train_loss = tr_sum / args.steps_per_epoch
            t_train = time.time() - t0

            val_tm = {} if profiling and not streams else None
            val_loss, val_rae, _ = evaluate(core, hists, X_val, fixed_val, y_val, args.batch_size, device, crit,
                                            tm=val_tm,
                                            streams=streams and bool(X_val.is_pinned()))
            scheduler.step(val_loss)
            dt = time.time() - t0
            if profiling:
                noise_dt = ([(a[0], *(np.array(a[1:]) - np.array(b[1:]))) for a, b in zip(producer.thread_stats(), noise0)]
                            if producer else None)
                th_dt = {}                              # thread name: [threads, s on a CPU, s waiting for one]
                for tid, (nm, run, wait) in thread_sched().items():
                    r0, w0 = th0[tid][1:] if tid in th0 else (0.0, 0.0)
                    e = th_dt.setdefault("main thread" if tid == str(os.getpid()) else nm, [0, 0.0, 0.0])
                    e[0] += 1
                    e[1] += run - r0
                    e[2] += wait - w0
                report_profile(prof_times, prof, dt - t_train, t_train_last, val_tm, np.array(sim_times[:]) - sim0, dt,
                               noise_dt, th_dt)
            t_train_last = t_train
            lr_now = opt.param_groups[0]["lr"]
            fresh = n_written.value - written0
            print(f"[Epoch {epoch:03d}] {dt:.2f}s (train {t_train:.1f}s, val {dt - t_train:.1f}s) | Train: {train_loss:.6f} | "
                  f"LevelAux: {lvl_sum / args.steps_per_epoch:.6f} | Val: {val_loss:.6f} | "
                  f"Val RAE per out: {' '.join(f'{r:.4f}' for r in val_rae)}\n"
                  f"            no reuse: {samples_per_epoch} new samples, each used once | simulated at "
                  f"{fresh / max(dt, 1e-9):.1f}/s | trainer waited {waited:.1f}s for data "
                  f"({100 * waited / max(dt, 1e-9):.0f}% of the epoch) | simulators idle "
                  f"{100 * (idle.value - idle0) / max(dt * args.gen_workers, 1e-9):.0f}% (buffer full) | "
                  f"total used {n_read.value} | lr {lr_now:g}",
                  flush=True)
            append_metrics_csv(csv_path=metrics_csv, epoch=epoch, epoch_time_s=dt, train_loss=train_loss,
                               val_loss=val_loss, lr=lr_now, val_rae_per=val_rae)

            stop_now = es.step(val_loss, None, epoch)
            if val_loss < best_val - args.min_delta:
                best_val = val_loss
                torch.save(core.state_dict(), best_path)
                print(f"  -> Saved best model (val={best_val:.6f})", flush=True)
            if stop_now:
                print(f"[Early Stopping] No improvement for {args.patience} epochs. Stopping.", flush=True)
                break
    finally:
        stop.value = 1
        producer_stop[0] = True
        if producer is not None:
            producer.close()
        for g in gens + trainers:
            g.join(timeout=30)
            if g.is_alive():
                g.terminate()
        if net is not None:
            dist.destroy_process_group()

    # 3. Test with the best model
    print("\n[Test] Loading best model...", flush=True)
    core.load_state_dict(torch.load(best_path, map_location=device))
    _, test_rae, preds = evaluate(core, hists, X_test, fixed_test, y_test, args.batch_size, device, LogCoshLoss(),
                                  streams=streams and bool(X_test.is_pinned()))
    preds_all = preds.numpy().astype(np.float64)
    targs_all = y_test.numpy().astype(np.float64)
    preds_hz, targs_hz, preds_aux = split_outputs(preds_all, targs_all, cond)
    np.save(os.path.join(args.output_root, "preds_test.npy"), preds_hz)
    np.save(os.path.join(args.output_root, "targs_test.npy"), targs_hz)
    np.save(os.path.join(args.output_root, "preds_aux_test.npy"), preds_aux)
    print(f"[Test] RAE per out: {' '.join(f'{r:.4f}' for r in test_rae)} | "
          f"Overall: {float(np.mean(test_rae[:n_rates])):.4f}", flush=True)
    rae_by_noise_file(preds_hz, targs_hz, noise_src_test, sim_cfg, args.output_root, "Test")
    mae = np.abs(preds_aux - aux_test).mean(axis=0) / c
    print(f"[Test] Mean abs error (pA): " + " | ".join(f"{nm} {e:.4g}" for nm, e in zip(CONDITIONS, mae)), flush=True)


if __name__ == "__main__":
    main()
