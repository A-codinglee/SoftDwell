#!/usr/bin/env python3
# scripts/hist_pipeline/train_hist_online.py
"""
Histogram pipeline trained on a refreshing buffer instead of a fixed file.

  - CPU generator processes keep simulating new samples (C++ simulator via
    scripts.simulation) and overwrite the oldest slot of a shared in-RAM buffer.
  - GPU trainer(s) draw random batches from whatever is currently in the buffer,
    so the GPU never waits for the simulator once the buffer has its initial fill.
  - With --no-reuse the buffer is a queue instead: every sample is used exactly once, so each
    epoch sees only new samples and the trainer waits whenever the simulator is slower.
  - Validation and test sets are simulated once at start-up and stay fixed.

Everything about the simulated data (topology, rate / level / noise ranges, histogram
settings) comes from the JSON given with --sim-config, e.g. configs/sim/COC.json.

There is no natural epoch any more; one "epoch" is --steps-per-epoch optimizer steps.

With --predict-conditions the network also outputs the recording conditions of each sample
(baseline, amplitude, noise sigma) behind the rates; see condition_norm().
"""
import argparse
import json
import os
import resource
import socket
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from scripts.common.canonicalize import choose_canonicalizer, extract_chain_rates_from_Q
from scripts.hist_pipeline.train_hist import append_metrics_csv, init_metrics_csv
from scripts.hist_pipeline.gpu_noise import release_noise, take_noise
from scripts.simulation.noise import NOISE_I16_STD, SpectralNoise, source_spectra
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.metrics import EarlyStopping, LogCoshLoss, rae_log
from scripts.softdwell_pipeline.model import TFExactInceptionResNet

# Independent random streams, so train / val / test never share a sample.
STREAM_TRAIN, STREAM_VAL, STREAM_TEST = 0, 1, 2

CONDITIONS = ("baseline", "amplitude", "sigma")     # columns of aux

# Training settings that are not command-line options. main() writes them, together with the options,
# to train_config.json in the run folder.
MODEL_KWARGS = dict(use_bn=False, separate_heads=False)
LR_FACTOR, LR_PATIENCE, MIN_LR = 0.7, 10, 1e-6      # ReduceLROnPlateau on the validation loss
GRAD_CLIP_NORM = 1.0


def condition_norm(sim_cfg: SampleSpec, n: int = 4096) -> dict:
    """
    --predict-conditions: which columns of aux = [closed level, level span, noise sigma] vary in this
    config (a fixed one is not a target), and their mean / std in raw counts. The extra targets are
    (aux - mean) / std, so they are on a similar scale as the log10 rates.
    Estimated from n draws with a fixed seed.
    """
    rng = np.random.default_rng(0)
    a = np.stack([sim_cfg.sample_levels_sigma(rng)[2] for _ in range(n)]).astype(np.float64)
    mean, std = a.mean(axis=0), a.std(axis=0)
    index = [j for j in range(a.shape[1]) if std[j] > 1e-9 * max(1.0, abs(mean[j]))]
    return {"names": [CONDITIONS[j] for j in index], "index": index,
            "mean": mean[index].tolist(), "std": std[index].tolist()}


def split_outputs(preds, targs, cond):
    """
    Network outputs / targets (n, y_dim) -> (predicted rates Hz, true rates Hz, predicted conditions).
    Conditions are (n, 3) raw counts in the column order of aux, NaN where not predicted; None without cond.
    """
    n_rates = preds.shape[1] - (len(cond["index"]) if cond else 0)
    preds_hz = 10.0 ** np.clip(preds[:, :n_rates], -30.0, 30.0)
    targs_hz = 10.0 ** np.clip(targs[:, :n_rates], -30.0, 30.0)
    preds_aux = None
    if cond:
        preds_aux = np.full((len(preds), len(CONDITIONS)), np.nan, dtype=np.float32)
        preds_aux[:, cond["index"]] = preds[:, n_rates:] * np.asarray(cond["std"]) + np.asarray(cond["mean"])
    return preds_hz, targs_hz, preds_aux


# -------------------------------------------------------------------------
# 1. SIMULATION HELPERS (run in CPU processes)
# -------------------------------------------------------------------------
def simulate_xy(sim: TraceSimulator, sim_cfg: SampleSpec, canon, seed: int, stream: int, i: int, synth=None,
                cond=None, trace=False, noise_given=None):
    """
    One training pair, with the same transforms train_hist.load_data_to_ram applies:
      x = log10(grids + 1)                       (D, B, B) float32
      y = log10(canonicalized adjacent rates)    (2*(S-1),) float32
          with cond (--predict-conditions): followed by the normalized conditions, see condition_norm()
    Also returns aux = [closed level, level span, noise sigma] of this sample (raw counts), followed
    by the row of sim_cfg.all_noise_paths its noise came from.
    Depends only on (seed, stream, i).
    trace=True: x is the raw int16 trace (trace_len,) of the same sample instead of its histograms.
    trace="both": x = (trace, log10(histograms + 1)), from one simulation.
    noise_given = (int16 noise, unit, recording row): noise made elsewhere (scripts/hist_pipeline/gpu_noise.py);
    then no recording is drawn here and synth is not used.
    trace="levels": only Gillespie, the trace is synthesized on the GPU (scripts/hist_pipeline/gpu_synth.py), with
    GPU noise: x = (u as int16 (trace_len,), [lo, hi, noise scale]), see TraceSimulator.simulate_levels; the
    noise scale is sigma / NOISE_I16_STD, the unit of the GPU noise. Draws the same random numbers as with
    noise_given, so the sample is the one the simulator would synthesize with that noise.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream, i]))
    Q = sim_cfg.sample_Q(rng)
    sim_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    noise_index = int(rng.integers(0, 2**32, dtype=np.uint64))
    levels, sigma, aux = sim_cfg.sample_levels_sigma(rng)
    noise, k = None, 0
    if noise_given is not None:
        noise, k = (noise_given[0], noise_given[1]), noise_given[2]
    elif synth is not None and trace != "levels":                    # spectral_resample
        src = sim_cfg.noise_sources(test=(stream == STREAM_TEST))
        # no draw when there is only one file, so single-file runs keep their random sequence
        k = src[int(rng.integers(len(src)))] if len(src) > 1 else src[0]
        noise = synth.synthesize(rng, k)
    if trace == "levels":
        u, lo, hi = sim.simulate_levels(Q, levels, seed=sim_seed)
        x = (u.view(np.int16), np.array([lo, hi, sigma / NOISE_I16_STD]))
    elif trace == "both":
        tr, grids, _ = sim.simulate_trace_and_hist(Q, levels, seed=sim_seed, noise_index=noise_index, sigma=sigma,
                                                   noise=noise)
        x = (tr, np.log10(grids + 1.0).astype(np.float32))
    elif trace:
        x = sim.simulate_trace(Q, levels, seed=sim_seed, noise_index=noise_index, sigma=sigma, noise=noise)
    else:
        grids, _ = sim.simulate_hist(Q, levels, seed=sim_seed, noise_index=noise_index, sigma=sigma, noise=noise)
        x = np.log10(grids + 1.0).astype(np.float32)
    y = extract_chain_rates_from_Q(Q).astype(np.float32)
    if canon is not None:
        y = canon(y)
    y = np.log10(np.maximum(y, 1e-9)).astype(np.float32)
    if cond is not None:
        z = (aux[cond["index"]] - np.asarray(cond["mean"])) / np.asarray(cond["std"])
        y = np.concatenate([y, z.astype(np.float32)])
    return x, y, np.append(aux, np.float32(k))


_POOL = {}


def write_noise_spectra(sim_cfg: SampleSpec, trace_len: int, out_dir: str, reuse: bool = False) -> str:
    """
    spectral_resample: measure the spectrum of every noise file once (rows as in
    sim_cfg.all_noise_paths) and save it; every process then synthesizes from that file.
    Returns its path, or "" for recorded noise. reuse: keep a matching file that is already there.
    """
    if sim_cfg.noise_mode != "spectral_resample":
        return ""
    path = os.path.join(out_dir, "noise_spectrum.npy")
    if reuse and os.path.isfile(path):
        if len(np.atleast_2d(np.load(path, mmap_mode="r"))) == len(sim_cfg.all_noise_paths):
            return path
    np.save(path, source_spectra(sim_cfg.all_noise_paths, trace_len, sim_cfg.noise_avg_seg,
                                 sim_cfg.noise_avg_overlap, sim_cfg.noise_generate_factor))
    return path


class _CachedNoise:
    """BENCHMARK ONLY (--bench-noise cached): one noise per recording, reused; costs no CPU time."""

    def __init__(self, synth):
        self.synth, self.cache = synth, {}

    def synthesize(self, rng, source=0):
        if source not in self.cache:
            self.cache[source] = self.synth.synthesize(rng, source)[0].copy()
        return self.cache[source], NOISE_I16_STD


class _BufferNoise:
    """BENCHMARK ONLY (--bench-noise gpu): noise read from a shared ring filled by a GPU producer."""

    def __init__(self, noise_buf, trace_len):
        self.buf = noise_buf.numpy()
        self.q = np.empty(trace_len, dtype=np.int16)
        self.i = os.getpid() * 7919

    def synthesize(self, rng, source=0):
        self.i += 1
        np.copyto(self.q, self.buf[self.i % len(self.buf)])      # the 2 MB read a real hand-over would cost
        return self.q, NOISE_I16_STD


def make_synth(spectrum_path, trace_len):
    """Per-process spectral noise synthesizer, or None for recorded noise."""
    return SpectralNoise(np.load(spectrum_path), trace_len) if spectrum_path else None


def _pool_init(config_path, sim_cfg, seed, spectrum_path, cond=None, trace=False):
    torch.set_num_threads(1)
    _POOL["sim"] = TraceSimulator(config_path)
    _POOL["synth"] = make_synth(spectrum_path, _POOL["sim"].trace_len)
    _POOL["cfg"] = sim_cfg
    _POOL["seed"] = seed
    _POOL["cond"] = cond
    _POOL["trace"] = trace
    _POOL["canon"] = choose_canonicalizer(sim_cfg.topology, "auto")


def _pool_job(job):
    stream, i = job
    return i, simulate_xy(_POOL["sim"], _POOL["cfg"], _POOL["canon"], _POOL["seed"], stream, i, _POOL["synth"],
                          _POOL["cond"], _POOL["trace"])


def simulate_fixed_set(pool, stream: int, n: int, x_shape, y_dim: int, x_dtype=torch.float32, h_shape=None):
    """Returns (X, y, aux, noise source); with h_shape (pool started with trace="both") also the histograms H."""
    H = torch.empty((n, *h_shape), dtype=torch.float32) if h_shape else None
    X = torch.empty((n, *x_shape), dtype=x_dtype)
    y = torch.empty((n, y_dim), dtype=torch.float32)
    aux = np.empty((n, len(CONDITIONS) + 1), dtype=np.float32)
    for i, (xi, yi, ai) in pool.imap_unordered(_pool_job, [(stream, i) for i in range(n)], chunksize=4):
        if H is not None:
            xi, hi = xi
            H[i] = torch.from_numpy(hi)
        X[i] = torch.from_numpy(xi)
        y[i] = torch.from_numpy(yi)
        aux[i] = ai
    out = (X.share_memory_(), y.share_memory_(), aux[:, :-1], aux[:, -1].astype(np.int64))
    return out + (H.share_memory_(),) if H is not None else out


def rae_by_noise_file(preds_hz, targs_hz, noise_src, sim_cfg: SampleSpec, out_dir: str, tag: str):
    """Test RAE (mean over the rates) per noise file; printed and written to test_by_noise_file.json."""
    rae = np.sqrt(np.abs(np.log10(np.maximum(preds_hz, 1e-300)) - np.log10(targs_hz)) + 1e-12).mean(axis=1)
    rows = [{"noise_file": os.path.basename(sim_cfg.all_noise_paths[k]), "n": int((noise_src == k).sum()),
             "rae": float(rae[noise_src == k].mean())} for k in np.unique(noise_src)]
    with open(os.path.join(out_dir, "test_by_noise_file.json"), "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(f"[{tag}] RAE on {r['noise_file']} (n={r['n']}): {r['rae']:.4f}", flush=True)


# generator_loop's optional profiling entries times[7:SIM_DETAIL]
SIM_DETAIL_NAMES = ["loop head (stop flag, next-index lock)", "after the copy (timer flush, n_written lock)",
                    "lock waits in the slot wait", "sleep polls in the slot wait (count)", "whole loop (wall)",
                    "on a CPU (kernel)", "runnable, waiting for a CPU (kernel)",
                    "involuntary context switches (count)", "voluntary context switches (count)"]
SIM_DETAIL = 7 + len(SIM_DETAIL_NAMES)


def _sched_counters():
    """This thread's [seconds on a CPU, seconds runnable but waiting for one, involuntary and voluntary
    context switches] since it started (Linux /proc and getrusage)."""
    with open("/proc/thread-self/schedstat") as f:
        run_ns, wait_ns, _ = f.read().split()
    ru = resource.getrusage(resource.RUSAGE_THREAD)
    return [int(run_ns) * 1e-9, int(wait_ns) * 1e-9, ru.ru_nivcsw, ru.ru_nvcsw]


def generator_loop(config_path, sim_cfg, seed, spectrum_path, X_buf, y_buf, data_lock, next_index, n_written, stop,
                   n_read=None, cond=None, trace=False, H_buf=None, n_reserved=None, ready=None, idle=None,
                   bench_noise=None, noise_buf=None, gpu_noise=None, times=None, ordered=False, M_buf=None):
    """
    Background producer: simulate forever, overwrite the oldest buffer slot each time.
    With n_read (--no-reuse) the buffer is a queue instead: never overwrite a sample the
    trainer has not taken yet, wait for room.
    With n_reserved and ready (queue only): a slot is reserved under the lock but filled outside it, then
    flagged in ready, so the 62 processes don't take turns copying 2 MB traces while holding one lock.
    gpu_noise: handles of a gpu_noise.GPUNoiseProducer; the noise of sample i is taken from there.
    idle (shared double): seconds this process spent waiting for room, summed over all processes.
    times (shared doubles, for profiling): seconds spent [waiting for GPU noise, simulating (trace and, with
    H_buf, its histograms), waiting for room, copying into the buffer], summed over all processes; with 7
    entries also the simulator's own split of the simulating part [Gillespie, trace synthesis, histograms];
    with SIM_DETAIL entries also the SIM_DETAIL_NAMES (the parts outside the four timers, and the kernel's
    scheduler counters of the process).
    ordered (queue with ready only): sample i always goes into slot i % size, so the trainer, which reads the
    slots in order, gets the samples in index order whichever process finishes first (--deterministic).
    M_buf (queue with ready only, trace="levels"): the trace is synthesized on the GPU; X_buf gets the averaged
    levels (uint16 in int16), M_buf the slot's [sample index, lo, hi, noise scale].
    """
    torch.set_num_threads(1)
    sim = TraceSimulator(config_path)
    synth = make_synth(spectrum_path, sim.trace_len)
    if bench_noise == "cached":
        synth = _CachedNoise(synth)
    elif bench_noise == "gpu":
        synth = _BufferNoise(noise_buf, sim.trace_len)
    canon = choose_canonicalizer(sim_cfg.topology, "auto")
    Xn, yn = X_buf.numpy(), y_buf.numpy()      # views onto the shared memory
    Hn = H_buf.numpy() if H_buf is not None else None
    Mn = M_buf.numpy() if M_buf is not None else None
    size = Xn.shape[0]

    rd = ready.numpy() if ready is not None else None
    n_tt = len(times) if times is not None else 4
    tt = [0.0] * n_tt

    def add(k, t0):
        """times[k] += seconds since t0 (local sum, flushed to the shared array every 8 samples)."""
        now = time.time()
        tt[k] += now - t0
        return now
    detail = n_tt >= SIM_DETAIL
    s_last = _sched_counters() if detail else None
    n_done = 0
    t_it = time.time()
    while not stop.value:
        with next_index.get_lock():
            i = next_index.value
            next_index.value += 1
        t_s = time.time()
        if detail:
            tt[7] += t_s - t_it
        if gpu_noise is not None:
            got = take_noise(gpu_noise, i, stop)
            if got is None:
                return
            t_s = add(0, t_s)
            x, y, _ = simulate_xy(sim, sim_cfg, canon, seed, STREAM_TRAIN, i, synth, cond, trace, got[:3])
            release_noise(gpu_noise, got[3])                 # the trace is built, the slot can be refilled
        else:
            x, y, _ = simulate_xy(sim, sim_cfg, canon, seed, STREAM_TRAIN, i, synth, cond, trace)
        t_s = add(1, t_s)
        if n_tt > 4:
            lt = sim.last_timings()
            if lt is not None:
                for k in range(3):
                    tt[4 + k] += lt[k]
        if rd is not None:
            t0 = time.time()
            while True:                              # reserve a slot the trainer has already read
                t_l = time.time()
                with data_lock:
                    if detail:
                        tt[9] += time.time() - t_l
                    if ordered:
                        if i - n_read.value < size:      # slot i % size has been read
                            slot = i % size
                            break
                    elif n_reserved.value - n_read.value < size:
                        slot = n_reserved.value % size
                        n_reserved.value += 1
                        break
                if stop.value:
                    return
                if detail:
                    tt[10] += 1
                time.sleep(0.005)
            if idle is not None:
                with idle.get_lock():
                    idle.value += time.time() - t0
            t_s = add(2, t_s)
            if H_buf is not None:
                Xn[slot], Hn[slot] = x
            elif Mn is not None:
                Xn[slot] = x[0]
                Mn[slot, 0] = i
                Mn[slot, 1:] = x[1]
            else:
                Xn[slot] = x
            yn[slot] = y
            rd[slot] = 1                             # the trainer reads a slot only once this is set
            t_e = add(3, t_s)
            n_done += 1
            if times is not None and n_done % 8 == 0:
                if detail:
                    s_now = _sched_counters()
                    for k in range(4):
                        tt[12 + k] += s_now[k] - s_last[k]
                    s_last = s_now
                with times.get_lock():
                    for k in range(n_tt):
                        times[k] += tt[k]
                tt = [0.0] * n_tt
            with data_lock:
                n_written.value += 1
            if detail:
                now = time.time()
                tt[8] += now - t_e
                tt[11] += now - t_it
                t_it = now
            continue
        # Slot is chosen at write time so the filled region is always [0, min(n_written, size)).
        # The lock keeps a reader from seeing the new x with the old y.
        while True:
            with data_lock:
                if n_read is None or n_written.value - n_read.value < size:
                    slot = n_written.value % size
                    if H_buf is not None:                # trace="both": x = (trace, histograms)
                        Xn[slot] = x[0]
                        Hn[slot] = x[1]
                    else:
                        Xn[slot] = x
                    yn[slot] = y
                    n_written.value += 1
                    break
            if stop.value:
                return
            time.sleep(0.01)


def sample_batch(X_buf, y_buf, data_lock, n_written, batch_size, gen):
    with data_lock:
        n = min(n_written.value, X_buf.shape[0])
        idx = torch.randint(0, n, (batch_size,), generator=gen)
        return X_buf[idx], y_buf[idx]          # advanced indexing copies


def take_batch(X_buf, y_buf, data_lock, n_written, n_read, batch_size, timeout_s=900.0):
    """
    --no-reuse: the next batch_size samples nobody has used yet, in arrival order; waits for the
    generators if there are not enough. Returns (x, y, seconds waited).
    """
    size = X_buf.shape[0]
    t0 = time.time()
    while True:
        with data_lock:
            if n_written.value - n_read.value >= batch_size:
                idx = torch.arange(n_read.value, n_read.value + batch_size) % size
                n_read.value += batch_size
                return X_buf[idx], y_buf[idx], time.time() - t0
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no new samples for {timeout_s:.0f}s; did the generator processes die?")
        time.sleep(0.002)


# -------------------------------------------------------------------------
# 2. TRAINER (one per GPU, or a single CPU process if there is no GPU)
# -------------------------------------------------------------------------
def _allreduce_sum(t):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def evaluate(model, X, y, idx, batch_size, device, crit, num_outputs):
    """Loss / RAE sums over the given indices. Returns (loss_sum, rae_sum, count, preds)."""
    loss_sum = torch.zeros(1, device=device)
    rae_sum = torch.zeros(num_outputs, device=device)
    count = torch.zeros(1, device=device, dtype=torch.long)
    preds = []
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b = idx[s:s + batch_size]
            xb = X[b].to(device, non_blocking=True)
            yb = y[b].to(device, non_blocking=True)
            pred = model(xb)
            loss = crit(pred, yb)
            preds.append(pred.cpu())
            if (not torch.isfinite(pred).all()) or (not torch.isfinite(loss)):
                continue
            loss_sum += loss.detach() * yb.size(0)
            rae_sum += rae_log(pred, yb).sum(dim=0)
            count += yb.size(0)
    return loss_sum, rae_sum, count, preds


def train_worker(rank, world_size, args, shared):
    (X_buf, y_buf, data_lock, n_written, n_read, X_val, y_val, X_test, y_test, cond, sim_cfg, noise_src_test) = shared

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
    in_channels = int(X_buf.shape[1])
    H, W = int(X_buf.shape[2]), int(X_buf.shape[3])
    num_outputs = int(y_buf.shape[1])

    metrics_csv = os.path.join(args.output_root, "train_val_metrics.csv")
    if is_master:
        print(f"[Worker 0] device={device} world_size={world_size} "
              f"steps_per_epoch={args.steps_per_epoch} batch_size={args.batch_size}", flush=True)
        init_metrics_csv(metrics_csv, num_outputs)

    def build_model():
        return TFExactInceptionResNet(
            in_channels=in_channels,
            num_outputs=num_outputs,
            input_hw=(H, W),
            **MODEL_KWARGS,
        ).to(device)

    model = build_model()
    if use_ddp:
        model = DDP(model, device_ids=[rank])
    core = model.module if use_ddp else model

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = LogCoshLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LR,
    )
    es = EarlyStopping(patience=args.patience, min_delta=args.min_delta, restore_best=False)

    best_val = float("inf")
    best_path = os.path.join(args.output_root, "best_model.pt")

    # each rank draws different batches, and evaluates its own slice of the fixed val set
    batch_gen = torch.Generator().manual_seed(args.seed * 1000 + rank)
    val_idx = torch.arange(rank, X_val.shape[0], world_size)

    samples_per_epoch = args.steps_per_epoch * args.batch_size * world_size

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        written0 = n_written.value
        waited = 0.0

        # ----------------- Train -----------------
        model.train()
        tr_loss_accum = torch.zeros(1, device=device)
        tr_count = torch.zeros(1, device=device)

        for _ in range(args.steps_per_epoch):
            if args.no_reuse:
                xb, yb, w = take_batch(X_buf, y_buf, data_lock, n_written, n_read, args.batch_size)
                waited += w
            else:
                xb, yb = sample_batch(X_buf, y_buf, data_lock, n_written, args.batch_size, batch_gen)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            opt.step()

            tr_loss_accum += loss.detach() * xb.size(0)
            tr_count += xb.size(0)

        _allreduce_sum(tr_loss_accum)
        _allreduce_sum(tr_count)
        train_loss = (tr_loss_accum / tr_count).item()

        # ----------------- Val (fixed set) -----------------
        model.eval()
        val_loss_sum, val_rae_sum, val_count, _ = evaluate(
            model, X_val, y_val, val_idx, args.batch_size, device, crit, num_outputs)
        _allreduce_sum(val_loss_sum)
        _allreduce_sum(val_rae_sum)
        _allreduce_sum(val_count)

        den = val_count.clamp_min(1).float()
        val_loss = (val_loss_sum / den).item()
        val_rae_per = (val_rae_sum / den).detach().cpu().tolist()

        scheduler.step(val_loss)

        dt = time.time() - t0
        current_lr = opt.param_groups[0]["lr"]

        if is_master:
            fresh = n_written.value - written0
            reuse = samples_per_epoch / max(1, fresh)
            if args.no_reuse:
                data_line = (f"            no reuse: {samples_per_epoch} new samples, each used once | "
                             f"simulated at {fresh / max(dt, 1e-9):.1f}/s | this trainer waited {waited:.1f}s "
                             f"for data ({100 * waited / max(dt, 1e-9):.0f}% of the epoch) | "
                             f"total used {n_read.value}")
            else:
                data_line = (f"            buffer: {fresh} fresh samples ({fresh / max(dt, 1e-9):.1f}/s), "
                             f"{samples_per_epoch} consumed ({samples_per_epoch / max(dt, 1e-9):.1f}/s), "
                             f"~{reuse:.1f} uses per fresh sample | total simulated {n_written.value}")
            print(
                f"[Epoch {epoch:03d}] {dt:.2f}s | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Val RAE per out: {' '.join(f'{r:.4f}' for r in val_rae_per)}\n" + data_line,
                flush=True,
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
            stop_local = es.step(val_loss, None, epoch)
            if val_loss < best_val - args.min_delta:
                best_val = val_loss
                torch.save(core.state_dict(), best_path)
                print(f"  -> Saved best model (val={best_val:.6f})", flush=True)

        if use_ddp:
            stop_flag = torch.tensor([1 if stop_local else 0], device=device, dtype=torch.uint8)
            dist.broadcast(stop_flag, src=0)
            stop_local = bool(stop_flag.item())

        if stop_local:
            if is_master:
                print(f"[Early Stopping] No improvement for {args.patience} epochs. Stopping.", flush=True)
            break

    # -- Testing (master only, fixed test set) --
    if is_master:
        print("\n[Test] Loading best model...", flush=True)
        model = build_model()
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        test_idx = torch.arange(X_test.shape[0])
        _, rae_sum, count, preds = evaluate(
            model, X_test, y_test, test_idx, args.batch_size, device, crit, num_outputs)

        preds_all = torch.cat(preds).numpy().astype(np.float64)
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
            # same rows / columns as aux_test.npy (raw counts), NaN where a condition is not predicted
            np.save(os.path.join(args.output_root, "preds_aux_test.npy"), preds_aux)
            mae = (np.abs(preds_all[:, n_rates:] - targs_all[:, n_rates:]).mean(axis=0)
                   * np.asarray(cond["std"]) / cond["counts_per_unit"])
            print(f"[Test] Mean abs error ({cond['unit']}): "
                  + " | ".join(f"{nm} {e:.4g}" for nm, e in zip(cond["names"], mae)), flush=True)

    if use_ddp:
        dist.destroy_process_group()


# -------------------------------------------------------------------------
# 3. PARENT MAIN5e-
# -------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim-config", required=True, help="simulation JSON, e.g. configs/sim/COC.json")
    p.add_argument("--output-root", default=None, help="default: outputs/hist_train_online/<sim-config name>")
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

    # buffer
    p.add_argument("--buffer-size", type=int, default=20000, help="samples kept in RAM")
    p.add_argument("--no-reuse", action="store_true",
                   help="use every simulated sample exactly once: each epoch trains on new samples only, "
                        "and the trainer waits for the simulator when it runs out")
    p.add_argument("--min-fill", type=int, default=0, help="samples needed before training starts (0 = buffer/4)")
    p.add_argument("--gen-workers", type=int, default=0, help="simulator processes (0 = all CPUs not used by trainers)")
    p.add_argument("--train-gpus", type=int, default=0,
                   help="GPUs to train on (0 = all in the job). Use 1 when extra GPUs were requested only for "
                        "their CPU cores: batch size and steps per epoch then stay as in a 1-GPU run")
    p.add_argument("--n-val", type=int, default=10000)
    p.add_argument("--n-test", type=int, default=10000)

    # training
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=0, help="0 = buffer_size / (batch_size * n_gpus)")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-3)
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_root is None:
        name = os.path.splitext(os.path.basename(args.sim_config))[0]
        args.output_root = os.path.join("outputs", "hist_train_online", name)
    args.output_root = os.path.join(args.output_root, f"seed{args.seed}")
    os.makedirs(args.output_root, exist_ok=True)

    # The copy written here (absolute paths, this run's seed) is what the simulator reads.
    config_path = prepare_sim_config(args.sim_config, os.path.join(args.output_root, "sim_config.json"), args.seed,
                                     holdout_noise=args.holdout_noise, train_noise=args.train_noise)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    x_shape = (probe.hist_detectors, probe.hist_bins, probe.hist_bins)
    trace_len = probe.trace_len
    probe.close()
    if min(x_shape) <= 0:
        raise SystemExit(f"{args.sim_config}: analysis.grind / analysis.hist2d is missing or invalid")

    K = sim_cfg.num_states
    off_chain = [(r, c) for r in range(K) for c in range(K) if abs(r - c) > 1 and sim_cfg.q_hi[r, c] > 0]
    if off_chain:
        raise SystemExit(f"only linear chains are supported as training targets; non-adjacent rates at {off_chain}")
    n_rates = 2 * (K - 1)

    cpp = sim_cfg.counts_per_pA
    unit, sc = ("pA", cpp) if cpp > 0 else ("counts", 1.0)
    fmt = lambda rg: f"[{rg[0] / sc:g}, {rg[1] / sc:g}]"
    levels_txt = (f"baseline {fmt(sim_cfg.baseline)} amplitude {fmt(sim_cfg.amplitude)} open_sign {sim_cfg.open_sign:+g}"
                  if sim_cfg.amplitude is not None
                  else " ".join(f"{lab} {fmt(rg)}" for lab, rg in sorted(sim_cfg.level_groups.items())))
    k_used = sim_cfg.q_hi > 0
    print(f"[Parent] sim-config={args.sim_config} topology={sim_cfg.topology}\n"
          f"[Parent] rates [{sim_cfg.q_lo[k_used].min():g}, {sim_cfg.q_hi[k_used].max():g}] Hz | "
          f"levels ({unit}): {levels_txt} | sigma ({unit}): {fmt(sim_cfg.sigma)}", flush=True)

    # extra targets; conditions.json tells eval_hist_online / plot_online_results how to read the outputs
    cond = None
    cond_path = os.path.join(args.output_root, "conditions.json")
    if args.predict_conditions:
        cond = condition_norm(sim_cfg)
        if not cond["index"]:
            raise SystemExit(f"--predict-conditions: baseline, amplitude and sigma are all fixed in {args.sim_config}")
        cond["unit"], cond["counts_per_unit"] = unit, sc
        with open(cond_path, "w") as f:
            json.dump(cond, f, indent=2)
        print(f"[Parent] extra targets: {', '.join(cond['names'])}", flush=True)
    elif os.path.isfile(cond_path):
        os.remove(cond_path)                    # left over from an earlier run in this folder
    y_dim = n_rates + (len(cond["index"]) if cond else 0)

    t0 = time.time()
    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.output_root)
    names = lambda paths: ", ".join(os.path.basename(p) for p in paths)
    if spectrum_path:
        print(f"[Parent] noise: spectral_resample (spectra measured in {time.time() - t0:.1f}s)\n"
              f"[Parent]   train / val: {names(sim_cfg.noise_paths)}\n"
              f"[Parent]   test:        {names(sim_cfg.noise_test_paths) or 'same files as training'}", flush=True)
    else:
        print(f"[Parent] noise: recorded_noise from {sim_cfg.noise_path}", flush=True)

    gpus_in_job = torch.cuda.device_count()
    n_gpus = min(gpus_in_job, args.train_gpus) if args.train_gpus > 0 else gpus_in_job
    world_size = max(1, n_gpus)
    n_cpus = len(os.sched_getaffinity(0))
    if args.gen_workers <= 0:
        args.gen_workers = max(1, n_cpus - world_size - 1)
    if args.min_fill <= 0:
        args.min_fill = max(args.batch_size, args.buffer_size // 4)
    args.min_fill = min(args.min_fill, args.buffer_size)
    if args.no_reuse:
        if args.buffer_size < args.batch_size * world_size:
            raise SystemExit("--no-reuse needs --buffer-size >= batch size x number of GPUs")
        args.min_fill = args.batch_size * world_size    # the buffer is only a queue; start right away
    if args.steps_per_epoch <= 0:
        args.steps_per_epoch = max(1, args.buffer_size // (args.batch_size * world_size))

    print(f"[Parent] mode: {'no reuse (new samples every epoch)' if args.no_reuse else 'refreshing buffer'}", flush=True)
    print(f"[Parent] gpus={n_gpus} (of {gpus_in_job} in the job) cpus={n_cpus} gen_workers={args.gen_workers} "
          f"buffer={args.buffer_size} min_fill={args.min_fill} x={x_shape} y={y_dim}", flush=True)

    # everything the training part of this run uses, after the defaults above were resolved
    train_config = {
        "model": {"name": TFExactInceptionResNet.__name__, "in_channels": x_shape[0],
                  "input_hw": list(x_shape[1:]), "num_outputs": y_dim, "num_rates": n_rates,
                  "extra_outputs": cond["names"] if cond else [], **MODEL_KWARGS},
        "optimizer": {"name": "Adam", "lr": args.lr},
        "loss": LogCoshLoss.__name__,
        "lr_scheduler": {"name": "ReduceLROnPlateau", "monitor": "val_loss", "factor": LR_FACTOR,
                         "patience": LR_PATIENCE, "min_lr": MIN_LR},
        "early_stopping": {"monitor": "val_loss", "patience": args.patience, "min_delta": args.min_delta},
        "grad_clip_norm": GRAD_CLIP_NORM,
        "batch_size": args.batch_size,          # per GPU
        "train_gpus": world_size if n_gpus else 0,
        "steps_per_epoch": args.steps_per_epoch,
        "samples_per_epoch": args.steps_per_epoch * args.batch_size * world_size,
        "max_epochs": args.epochs,
        "seed": args.seed,
        "data": {"sim_config": args.sim_config, "no_reuse": args.no_reuse, "buffer_size": args.buffer_size,
                 "min_fill": args.min_fill, "gen_workers": args.gen_workers,
                 "n_val": args.n_val, "n_test": args.n_test,
                 "noise_mode": sim_cfg.noise_mode, "noise_files_train_val": sim_cfg.noise_paths,
                 "noise_files_test": sim_cfg.noise_test_paths or sim_cfg.noise_paths},
    }
    with open(os.path.join(args.output_root, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    ctx = mp.get_context("spawn")

    # 1. Fixed val / test sets (all CPUs, before the background generators start)
    t0 = time.time()
    with ctx.Pool(max(1, n_cpus - 1), initializer=_pool_init,
                  initargs=(config_path, sim_cfg, args.seed, spectrum_path, cond)) as pool:
        X_val, y_val, _, _ = simulate_fixed_set(pool, STREAM_VAL, args.n_val, x_shape, y_dim)
        X_test, y_test, aux_test, noise_src_test = simulate_fixed_set(pool, STREAM_TEST, args.n_test, x_shape, y_dim)
    # per-sample [closed level, level span, noise sigma] of the test set (raw counts), row-aligned with preds_test.npy
    np.save(os.path.join(args.output_root, "aux_test.npy"), aux_test)
    # per-sample row of the noise file list (training files, then test files) the test noise came from
    np.save(os.path.join(args.output_root, "noise_source_test.npy"), noise_src_test)
    n_fixed = args.n_val + args.n_test
    print(f"[Parent] Simulated val={args.n_val} test={args.n_test} in {time.time() - t0:.1f}s "
          f"({n_fixed / max(time.time() - t0, 1e-9):.1f} samples/s)", flush=True)

    # 2. Shared buffer + background generators
    X_buf = torch.zeros((args.buffer_size, *x_shape), dtype=torch.float32).share_memory_()
    y_buf = torch.zeros((args.buffer_size, y_dim), dtype=torch.float32).share_memory_()
    data_lock = ctx.Lock()
    next_index = ctx.Value("q", 0)
    n_written = ctx.Value("q", 0, lock=False)   # only modified under data_lock
    n_read = ctx.Value("q", 0, lock=False)      # --no-reuse: samples handed to trainers; only modified under data_lock
    stop = ctx.Value("b", 0, lock=False)

    gens = [
        ctx.Process(
            target=generator_loop,
            args=(config_path, sim_cfg, args.seed, spectrum_path,
                  X_buf, y_buf, data_lock, next_index, n_written, stop, n_read if args.no_reuse else None,
                  cond),
            daemon=True,
        )
        for _ in range(args.gen_workers)
    ]
    for g in gens:
        g.start()

    try:
        # 3. Wait for the initial fill
        t0 = time.time()
        last = t0
        while n_written.value < args.min_fill:
            if not any(g.is_alive() for g in gens):
                raise RuntimeError("all generator processes died during the initial fill")
            time.sleep(0.5)
            if time.time() - last >= 10.0:
                last = time.time()
                print(f"[Parent] filling buffer: {n_written.value}/{args.min_fill} "
                      f"({n_written.value / (last - t0):.1f} samples/s)", flush=True)
        print(f"[Parent] Initial fill of {args.min_fill} done in {time.time() - t0:.1f}s. Training starts.",
              flush=True)

        # 4. Train
        shared = (X_buf, y_buf, data_lock, n_written, n_read, X_val, y_val, X_test, y_test, cond,
                  sim_cfg, noise_src_test)
        if n_gpus >= 1:
            # a port nobody on this node uses: two of our jobs can share a node (one GPU each)
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
