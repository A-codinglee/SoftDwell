# ===== Incremental ΔW (exact) dwell frontend =====
# Same method as your earlier SoftDwellLayer:
# - log-spaced dwell lengths via length_ratio
# - exact prefix-sum ΔW construction
# - logsumexp pairing, returned in linear domain
#
# API kept compatible with your current calls:
#   SoftDwell2DLayer(num_detectors, dwell_min, dwell_max, num_bins, **kw)
# NOTE: num_bins is IGNORED in this exact-method implementation (kept for signature compatibility).

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F  
# from checkpoint import is_rank0  # for debug prints

__all__ = ["SoftDwell2DLayer", "ChannelWiseSoftDwell"]

# ---------- utilities (from your incremental-ΔW version) ----------

def normalize_no_clamp(x: torch.Tensor, vmin: float, vmax: float, eps: float = 1e-12) -> torch.Tensor:
    denom = torch.clamp(
        torch.as_tensor(vmax, dtype=x.dtype, device=x.device) -
        torch.as_tensor(vmin, dtype=x.dtype, device=x.device),
        min=eps
    )
    return (x - vmin) / denom

def log_spaced_lengths(lmin: int, lmax: int, ratio: float):
    """Return sorted unique integer lengths: lmin, ceil(lmin*r), ceil(...), <= lmax."""
    lmin = max(1, int(lmin))
    lmax = max(lmin, int(lmax))
    r = max(1.0000001, float(ratio))
    L = []
    v = lmin
    seen = set()
    while v <= lmax:
        zz = int(v)
        if zz < 1:
            zz = 1
        if zz not in seen:
            L.append(zz); seen.add(zz)
        v = math.ceil(v * r)
        if len(L) > 1 and L[-1] == L[-2]:  # safety if r≈1
            v = L[-1] + 1
    if L[-1] != lmax and lmax > L[-1]:
        L.append(lmax)
    return L  # python list of ints

# ---------- main (exact) layer under your current class name ----------

class SoftDwell2DLayer(nn.Module):
    """
    EXACT incremental-ΔW soft dwell histograms (your original method), aliased under
    the old name 'SoftDwell2DLayer' for compatibility with your training code.

    Args:
      num_detectors: K
      dwell_min, dwell_max: dwell length range
      num_bins: IGNORED (kept for API compatibility)
      length_ratio: log spacing ratio for dwell lengths (like your original)
      average_segments, apply_signal_gauss, gauss_kernel, gauss_sigma, norm_min, norm_max, etc.
    """
    def __init__(
        self,
        num_detectors: int,
        dwell_min: int,
        dwell_max: int,
        num_bins: int,               # <-- IGNORED (kept for signature compatibility)
        *,
        length_ratio: float = 1.6,   # <-- NEW: same as your original
        average_segments: bool = True,
        norm_min: float = 0.0,
        norm_max: float = 4000.0,
        eps: float = 1e-6,
        theta_init_min: float = 0.2,
        theta_init_max: float = 0.8,
        tau_init: float = 0.2,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        timing: bool = False,
    ):
        super().__init__()
        # store core
        self.K = int(num_detectors)
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.average_segments = bool(average_segments)
        self.eps = float(eps)
        self.dtype  = dtype
        self.timing = bool(timing)

        # stability bounds as buffers (no device args)
        tau_min_val, tau_max_val = 1e-3, 1e3
        theta_min_val, theta_max_val = 1e-3, 1.0 - 1e-3
        self.register_buffer("tau_min",   torch.as_tensor(tau_min_val,   dtype=self.dtype))
        self.register_buffer("tau_max",   torch.as_tensor(tau_max_val,   dtype=self.dtype))
        self.register_buffer("theta_min", torch.as_tensor(theta_min_val, dtype=self.dtype))
        self.register_buffer("theta_max", torch.as_tensor(theta_max_val, dtype=self.dtype))

        # dwell lengths using your exact log-spaced scheme
        L_py = log_spaced_lengths(dwell_min, dwell_max, length_ratio)
        for z in range(1, len(L_py)):
            if L_py[z] <= L_py[z - 1]:
                L_py[z] = L_py[z - 1] + 1
        self.L_list = L_py
        self.J = len(self.L_list)

        # Δℓ list for averaging (NO device here)
        dL, prev = [], 0
        for L in self.L_list:
            dL.append(L - prev); prev = L
        self.register_buffer("delta_L", torch.tensor(dL, dtype=self.dtype))          # (J,)
        self.register_buffer("lengths", torch.tensor(self.L_list, dtype=torch.long)) # (J,)

        # ----- parameters (NO device here) -----
        theta_init = torch.linspace(theta_init_min, theta_init_max, steps=self.K, dtype=self.dtype)
        self.raw_theta = nn.Parameter(theta_init)  # (K,)

        tau_init = torch.as_tensor(tau_init, dtype=self.dtype)
        self.raw_tau = nn.Parameter(tau_init.expand(self.K).clone())  # (K,)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T) raw signal
        returns H: (B, K, J, J) linear-domain histograms (one orientation: 1→0)
        Vectorized over batch(B) and detectors(K); only loops over JxJ remain.
        """
        # ---- device/dtype: keep dwell math in float32 for stability ----
        dev = self.raw_theta.device          # infer the live device from a real Parameter
        if x.device != dev:
            x = x.to(dev)
        if x.dtype != torch.float32:
            x = x.float()

        B, T = x.shape

        # ---- normalize + optional smoothing ----
        x_norm = normalize_no_clamp(x, self.norm_min, self.norm_max)  # (B,T)                     

        # ---- detectors via broadcasting: (B, K, T) ----
        theta = self.raw_theta.view(1, self.K, 1)  # (1,K,1)
        tau   = self.raw_tau.view(1, self.K, 1)    # (1,K,1)
        
        y = torch.sigmoid((x_norm[:, None, :] - theta) / tau)  # (B,K,T)

        y = y.clamp(self.eps, 1.0 - self.eps)

        # ---- prefix sums over time (exclusive) for all (B,K) ----
        logy   = torch.log(y)                                          # (B,K,T)
        log1my = torch.log1p(-y)                                       # (B,K,T)
        S1 = F.pad(torch.cumsum(logy,   dim=-1), (1, 0))               # (B,K,T+1)
        S0 = F.pad(torch.cumsum(log1my, dim=-1), (1, 0))               # (B,K,T+1)

        # ---- Δ windows for each dwell length j (vectorized over B,K) ----

        L_list = self.lengths.tolist()                                 # list[int], J
        J = len(L_list)
        dW1_list, dW0_list = [], []
        prev = 0
        for Lj in L_list:
            Nj = T - Lj + 1
            if Nj > 0:
                # (B,K,Nj) = S[..., Lj:] - S[..., prev:prev+Nj]
                dW1 = S1[..., Lj:] - S1[..., prev:prev + Nj]
                dW0 = S0[..., Lj:] - S0[..., prev:prev + Nj]
            else:
                dW1 = x.new_empty((B, self.K, 0))
                dW0 = x.new_empty((B, self.K, 0))
            dW1_list.append(dW1)
            dW0_list.append(dW0)
            prev = Lj

        # ---- pair (i, j): batch-wise slices, no loops over B or K ----
        logH = x.new_full((B, self.K, J, J), float("-inf"))
        delta_L = self.delta_L  # (J,) float tensor
        for j, Lj in enumerate(L_list):
            A = dW1_list[j]                      # (B,K,Nj)
            Nj = A.size(-1)
            if Nj == 0:
                continue
            for i, Li in enumerate(L_list):
                N = Nj - Li                      # valid starts: T - Lj - Li + 1
                if N <= 0:
                    continue
                a = A[..., :N]                   # (B,K,N)
                b = dW0_list[i][..., Lj:Lj+N]    # (B,K,N) shifted by Lj
                if self.average_segments:
                    a = a / delta_L[j]
                    b = b / delta_L[i]
                logH[..., i, j] = torch.logsumexp(a + b, dim=-1)  # -> (B,K)

        H = torch.exp(logH)                       # (B,K,J,J) linear domain
        return H

# ---------- channel-wise wrapper (same name your code expects) ----------

class ChannelWiseSoftDwell(nn.Module):
    """x: [B, C, T] -> H: [B, C*K, J, J]  (exact incremental-ΔW method)"""
    def __init__(self, num_detectors: int, dwell_min: int, dwell_max: int, num_bins: int, **kw):
        super().__init__()
        self.K = int(num_detectors)
        # Forward to SoftDwell2DLayer with same signature; num_bins is ignored inside exact method.
        self.inner = SoftDwell2DLayer(num_detectors, dwell_min, dwell_max, num_bins, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"ChannelWiseSoftDwell expects (B,C,T); got {tuple(x.shape)}"
        B, C, T = x.shape
        # run exact dwell per channel (float32 inside)
        H = self.inner(x.reshape(B * C, T))             # (B*C, K, J, J)
        H = H.reshape(B, C, self.K, H.size(-2), H.size(-1))
        return H.flatten(1, 2)                          # (B, C*K, J, J)
