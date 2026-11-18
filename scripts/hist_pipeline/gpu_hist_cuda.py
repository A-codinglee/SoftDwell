# scripts/hist_pipeline/gpu_hist_cuda.py
"""
grind_hist_fast in one pass: the CUDA kernels of grind_cuda.cu scan each trace once (one warp per chunk of a
trace, one lane per threshold) and count the dwell pairs straight into the histograms, instead of building the
(rows x thresholds x T) boolean arrays, nonzero() and scatter of grind_hist_fast. Identical histograms (checked
by scripts/hist_pipeline/bench_gpu_hist.py).

The library is compiled with nvcc on first use into $GRIND_CUDA_DIR (default $TMPDIR or /tmp) and called
through ctypes on the caller's current CUDA stream.
"""
import ctypes
import os
import subprocess
import threading

import torch

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grind_cuda.cu")
_lib, _lock = None, threading.Lock()
_cache = {}            # (device, id of lut) -> int32 copy of the LUT


def load():
    global _lib
    with _lock:
        if _lib is None:
            out_dir = os.environ.get("GRIND_CUDA_DIR") or os.environ.get("TMPDIR") or "/tmp"
            so = os.path.join(out_dir, f"libgrind_cuda_{int(os.path.getmtime(SRC))}.so")
            if not os.path.exists(so):
                tmp = f"{so}.{os.getpid()}"
                subprocess.run(["nvcc", "-O3", "-arch=sm_80", "-shared", "-Xcompiler", "-fPIC", "-o", tmp, SRC],
                               check=True)
                os.replace(tmp, so)              # atomic: other processes see the whole file or none
            lib = ctypes.CDLL(so)
            lib.grind_hist_cuda.restype = ctypes.c_int
            lib.grind_hist_cuda.argtypes = [ctypes.c_int, ctypes.c_void_p] + [ctypes.c_void_p] * 2 + [ctypes.c_int] \
                + [ctypes.c_void_p] + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2
            _lib = lib
    return _lib


def n_chunks(B):
    """Chunks per trace: enough warps to fill the GPU (about 4096), a multiple of 8 warps per block."""
    return max(8, -(-4096 // max(B, 1)) // 8 * 8)


@torch.no_grad()
def grind_hist_cuda(x, thr, lut, J, k_group=None):
    """Same arguments and result as grind_hist_fast: x (B, T) int16 on the device; thr (K,) shared or (B, K) per
    row; lut (T+1,). Returns hist (B, K, J, J) float32. k_group is ignored (all thresholds in one pass)."""
    B, T = x.shape
    per_row = thr.dim() == 2
    K = int(thr.shape[-1])
    dev = x.device
    x = x.contiguous()
    thr32 = thr.to(torch.int32).contiguous()
    key = (dev, lut.data_ptr())
    if key not in _cache:
        _cache[key] = lut.to(torch.int32).contiguous()
    lut32 = _cache[key]
    C = n_chunks(B)
    hist = torch.zeros((B, K, J, J), device=dev, dtype=torch.int32)
    summ = torch.empty((B, K, C, 5), device=dev, dtype=torch.int32)
    if B and T >= 2:
        rc = load().grind_hist_cuda(dev.index, torch.cuda.current_stream(dev).cuda_stream, x.data_ptr(),
                                    thr32.data_ptr(), K if per_row else 0, lut32.data_ptr(), B, T, K, J, C,
                                    hist.data_ptr(), summ.data_ptr())
        if rc:
            raise RuntimeError(f"grind_hist_cuda failed (code {rc}; needs K <= 32 and T a multiple of 8)")
    return hist.to(torch.float32)
