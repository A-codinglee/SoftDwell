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
        num_bins: int,               
        *,
        # length_ratio: float = 1.6
        average_segments: bool = True,
        norm_min: float = 0.0,
        norm_max: float = 4000.0,
        eps: float = 1e-6,
        theta_init_min: float = 0.1,
        theta_init_max: float = 0.9,
        tau_init: float = 0.1,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        timing: bool = False,
        logN_gamma: float = 0.0
    ):
        super().__init__()
        # store core
        self.K = int(num_detectors)
        self.norm_min = float(norm_min)
        self.norm_max = float(norm_max)
        self.norm_denom = max(self.norm_max - self.norm_min, 1e-12)
        self.average_segments = bool(average_segments)
        self.eps = float(eps)
        self.dtype  = dtype
        self.timing = bool(timing)
        self.logN_gamma = float(logN_gamma)

        # stability bounds as buffers (no device args)
        tau_min_val, tau_max_val = 1e-3, 1e3
        theta_min_val, theta_max_val = 1e-3, 1.0 - 1e-3
        self.register_buffer("tau_min",   torch.as_tensor(tau_min_val,   dtype=self.dtype))
        self.register_buffer("tau_max",   torch.as_tensor(tau_max_val,   dtype=self.dtype))
        self.register_buffer("theta_min", torch.as_tensor(theta_min_val, dtype=self.dtype))
        self.register_buffer("theta_max", torch.as_tensor(theta_max_val, dtype=self.dtype))

        log_min = math.log10(dwell_min)
        log_max = math.log10(dwell_max)

        log_edges = torch.linspace(log_min, log_max, num_bins + 1, dtype=torch.float32)
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:])
        L_float = 10 ** log_centers

        lengths = torch.round(L_float).clamp(min=dwell_min, max=dwell_max).to(torch.long)
        lengths = torch.unique_consecutive(lengths)

        self.register_buffer("lengths", lengths)
        self.J = int(lengths.numel())
        # CPU list for fast looping without GPU sync
        self.lengths_cpu = [int(v) for v in lengths.cpu().tolist()]
        self.L_list = self.lengths_cpu

        if self.J != num_bins:
            print(f"[SoftDwell] requested J={num_bins} but got unique J={self.J} after rounding/unique.", flush=True)

        # Δℓ list for averaging (NO device here)
        dL, prev = [], 0
        for L in self.L_list:
            dL.append(L - prev); prev = L
        self.register_buffer("delta_L", torch.tensor(dL, dtype=self.dtype))          # (J,)

        # ----- parameters (NO device here) -----
        theta_init = torch.linspace(theta_init_min, theta_init_max, steps=self.K, dtype=self.dtype)
        self.raw_theta = nn.Parameter(theta_init)  # (K,)

        tau_init = torch.as_tensor(tau_init, dtype=self.dtype)
        self.raw_tau = nn.Parameter(tau_init.expand(self.K).clone())  # (K,)


    def forward(self, x: torch.Tensor, return_debug: bool = False) -> torch.Tensor:
        """
        x: (B,T) raw signal
        returns H: (B, K, J, J) linear-domain histograms (one orientation: 1→0)
        Vectorized over batch(B) and detectors(K); only loops over JxJ remain.
        """
        debug = {} if return_debug else None

        # ---- device/dtype: keep dwell math in float32 for stability ----
        dev = self.raw_theta.device          # infer the live device from a real Parameter
        if x.device != dev:
            x = x.to(dev)
        if x.dtype != torch.float32:
            x = x.float()

        B, T = x.shape

        # ---- normalize + optional smoothing ----
        x_norm = (x - self.norm_min) / self.norm_denom  # (B,T)

        if return_debug:
            debug["x_norm_minmax"] = (x_norm.min().item(), x_norm.max().item())                     

        # ---- detectors via broadcasting: (B, K, T) ----
        theta = self.raw_theta.view(1, self.K, 1)  # (1,K,1)
        tau   = self.raw_tau.view(1, self.K, 1)    # (1,K,1)
        
        u = (x_norm[:, None, :] - theta) / tau
        
        if return_debug:
            debug["u_minmax"] = (u.min().item(), u.max().item())
        # ---- prefix sums over time (exclusive) for all (B,K) ----
        logy   = F.logsigmoid(u)                                          # (B,K,T)
        log1my = F.logsigmoid(-u)                                       # (B,K,T)

        if return_debug:
            debug["logy_minmax"]   = (logy.min().item(), logy.max().item())
            debug["log1my_minmax"] = (log1my.min().item(), log1my.max().item())
            
        S1 = F.pad(torch.cumsum(logy,   dim=-1), (1, 0))               # (B,K,T+1)
        S0 = F.pad(torch.cumsum(log1my, dim=-1), (1, 0))               # (B,K,T+1)

        # ---- Δ windows for each dwell length j (vectorized over B,K) ----
        L_list = self.L_list
        J = self.J

        dW1_list, dW0_list = [], []
        for Lj in L_list:
            Nj = T - Lj + 1
            if Nj > 0:
                # dW*(..., s) = S*(s+Lj) - S*(s)  for s=0..Nj-1
                dW1 = S1[..., Lj:] - S1[..., :Nj]   # (B,K,Nj)
                dW0 = S0[..., Lj:] - S0[..., :Nj]   # (B,K,Nj)
            else:
                dW1 = x.new_empty((B, self.K, 0))
                dW0 = x.new_empty((B, self.K, 0))

            dW1_list.append(dW1)
            dW0_list.append(dW0)

        # ---- pair (i, j): batch-wise slices, no loops over B or K ----
        logH = x.new_full((B, self.K, J, J), float("-inf"))
        if return_debug:
            logH_raw = logH.new_full(logH.shape, float("nan"))  # (B,K,J,J)
            logN_map = logH.new_full(logH.shape, float("nan"))  # (B,K,J,J)
        
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
                    a = a / Lj
                    b = b / Li
                scores = a + b  # (B,K,N)
                logH_ij = torch.logsumexp(scores, dim=-1)  # (B,K)

                if return_debug:
                    logH_raw[..., i, j] = logH_ij
                    logN_map[..., i, j] = math.log(N)

                logH[..., i, j] = logH_ij - self.logN_gamma * math.log(N)

        # ---- debug for logP ----
        if return_debug:
            fin = torch.isfinite(logH)
            debug["logH_finite_frac"] = fin.float().mean().item()
            debug["logH_minmax"] = (
                (logH[fin].min().item(), logH[fin].max().item()) if fin.any() else (float("nan"), float("nan"))
            )
            if torch.isposinf(logH).any() or torch.isnan(logH).any():
                raise RuntimeError("logH has +inf or NaN")
            
            debug["logH_raw"] = logH_raw
            debug["logN_map"] = logN_map

            # optional but useful: map of (Li + Lj) for “length bias” correlation checks
            L = logH.new_tensor(L_list, dtype=torch.float32)   # (J,)
            debug["len_map"] = (L[:, None] + L[None, :])       # (J, J)

            debug["logN_gamma"] = self.logN_gamma

            return logH, debug

        return logH

        # return logH

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
