# scripts/hist_pipeline/gpu_synth.py
"""
Trace synthesis of the training samples on the GPUs (train_selfcal_online.py --synth-on-gpu).

The simulator processes only run Gillespie and hand over u, the noise-free current averaged over each sample
interval, as uint16 (TraceSimulator.simulate_levels; 2 MB per trace, as the trace itself). The GPU filters u
with the step response, adds the sample's GPU noise (gpu_noise.GPUNoiseProducer, kept on the GPU) and rounds
to int16: the same arithmetic as the simulator's synthesizeTraceInt16CoreF, rearranged into one FIR filter
(see synth_cuda.cu), so traces agree with the CPU's up to a count at a few rounding ties (float sums in a
different order, u quantized to 1/65535 of the level span between events).

The library is compiled with nvcc on first use into $GRIND_CUDA_DIR (default $TMPDIR or /tmp), as
gpu_hist_cuda.py does, and called through ctypes on the caller's current CUDA stream.
"""
import ctypes
import os
import subprocess
import threading
import time

import numpy as np
import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synth_cuda.cu")
_lib, _lock = None, threading.Lock()


def load():
    global _lib
    with _lock:
        if _lib is None:
            out_dir = os.environ.get("GRIND_CUDA_DIR") or os.environ.get("TMPDIR") or "/tmp"
            so = os.path.join(out_dir, f"libsynth_cuda_{int(os.path.getmtime(SRC))}.so")
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}"
                subprocess.run(["nvcc", "-O3", "-arch=sm_80", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, SRC],
                               check=True)
                os.replace(tmp, so)              # atomic: other processes see the whole file or none
            lib = ctypes.CDLL(so)
            lib.synth_traces_cuda.restype = ctypes.c_int
            lib.synth_traces_cuda.argtypes = [ctypes.c_int] + [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 \
                + [ctypes.c_void_p]
            _lib = lib
    return _lib


def taps(kernel):
    """FIR taps g[m] = kernel[m+1] - kernel[m] (float32, as Filter.cpp's float kernel)."""
    k = np.asarray(kernel, dtype=np.float32)
    return k[1:] - k[:-1]


@torch.no_grad()
def synth_traces(uq, noise, meta, g):
    """uq (B, T) int16 holding uint16 u; noise (B, T) int16; meta (B, 3) float32 [lo, hi, noise scale]; g (G,)
    float32; all on one device. Returns the traces (B, T) int16."""
    B, T = uq.shape
    dev = uq.device
    uq, noise = uq.contiguous(), noise.contiguous()
    meta = meta.to(dev, torch.float32).contiguous()
    g = g.to(dev, torch.float32).contiguous()
    out = torch.empty((B, T), dtype=torch.int16, device=dev)
    if B:
        rc = load().synth_traces_cuda(dev.index, torch.cuda.current_stream(dev).cuda_stream, uq.data_ptr(),
                                      noise.data_ptr(), meta.data_ptr(), g.data_ptr(), int(g.numel()), B, T,
                                      out.data_ptr())
        if rc:
            raise RuntimeError(f"synth_traces_cuda failed (code {rc})")
    return out


@torch.no_grad()
def synth_traces_reference(uq, noise, meta, g):
    """The same in float64 with torch ops (for checks): sum_m g[m] u[n-m] + noise * scale, rounded half away
    from zero as std::round."""
    B, T = uq.shape
    meta = meta.to(uq.device, torch.float64)
    u = meta[:, :1] + (uq.to(torch.int32) & 0xFFFF).to(torch.float64) * ((meta[:, 1:2] - meta[:, :1]) / 65535.0)
    G = int(g.numel())
    up = torch.cat([u[:, :1].expand(B, G - 1), u], dim=1)
    y = torch.zeros_like(u)
    for m, gm in enumerate(g.to(torch.float64).tolist()):
        y += gm * up[:, G - 1 - m:G - 1 - m + T]
    y += noise.to(torch.float64) * meta[:, 2:3]
    return (torch.sign(y) * torch.floor(y.abs() + 0.5)).clamp(-32768, 32767).to(torch.int16)


class GPUSynth:
    """
    Builds a batch's traces on the histogram GPUs from its uint16 parts (already on the GPUs) and the batch's
    noise from the producer's GPU ring, then frees the noise slots.
    """

    def __init__(self, kernel, producer, devices, timeout_s=900.0):
        self.producer, self.timeout_s = producer, timeout_s
        self.g = {d: torch.from_numpy(taps(kernel)).to(d) for d in devices}
        self.noise_wait_s = 0.0                 # seconds spent waiting for noise (summed)
        load()                                  # compile now, not in the first step

    def __call__(self, parts, meta):
        """parts: per device (b_i, T) int16 u; meta (B, 4) float64 on the CPU [sample index, lo, hi, scale].
        Returns per device (b_i, T) int16 traces."""
        p = self.producer
        idx = meta[:, 0].to(torch.int64)
        slots = idx % p.R
        t0 = time.time()
        while not bool((p.seq[slots] == idx).all()):
            if time.time() - t0 > self.timeout_s:
                raise RuntimeError(f"no GPU noise for samples {idx[p.seq[slots] != idx][:4].tolist()} after "
                                   f"{self.timeout_s:.0f}s")
            time.sleep(0.0005)
        self.noise_wait_s += time.time() - t0
        rows = p.ring.index_select(0, slots.to(p.ring.device))
        out, s = [], 0
        for x in parts:
            b, d = x.shape[0], x.device
            nz = rows[s:s + b].to(d, non_blocking=True)
            out.append(synth_traces(x, nz, meta[s:s + b, 1:].to(torch.float32), self.g[d]))
            s += b
        p.seq[slots] = -1                       # the noise rows are copied (in stream order), the slots can be refilled
        return out
