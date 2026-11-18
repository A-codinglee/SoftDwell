"""
Exact GPU port of the reference "grind" hard-threshold dwell histogram
(3DFit_unclamped_developing/src/Grind.cpp, compute_grind_hist_stack).

Semantics, transcribed from the C++ (do not "improve" without re-validating
against grind_hist_reference below, which mirrors the C++ state machine):

  * K absolute thresholds: linspace(thr_min, thr_max, K), each floored to an
    integer and clamped to int16 (K==1 -> the midpoint). Label of a sample for
    threshold k is  y > thr[k]  (1 = above, 0 = below).
  * Per threshold, a "dwell" is a maximal run of constant label. Only
    COMPLETED dwells count: the first run (starting at sample 0) counts as a
    normal dwell once its end is seen, the last (unterminated) run is never
    completed and is therefore never used.
  * A pair is emitted for every two consecutive completed dwells (both
    directions, above->below and below->above). The pair is filed at
    grid[Y_bin, X_bin] where X = the below-threshold (label 0) dwell and
    Y = the above-threshold (label 1) dwell.
  * Dwell length L samples -> t = L / fs; bin = floor((log10 t - log10 tmin) /
    (log10 tmax - log10 tmin) * B). A dwell with t < tmin or t >= tmax has no
    bin (-1) and the WHOLE pair is dropped (not clamped).
  * N_events[k] = number of emitted pairs.

Output layout: hist (B, K, J, J), axis 2 = Y (above-dwell bin), axis 3 = X
(below-dwell bin), i.e. hist[b, k, y, x] -- the same [by * B + bx] order the C++
writes into each grid.

Implementation: no per-segment padding and no segment cap (unlike the
SoftDwell hybrid_hard RLE, which had to truncate at max_segs): transitions are
found with nonzero() on the label-change mask, so memory scales with the
number of transitions actually present, and heavy noise chatter is handled
exactly. Thresholds are processed in groups of `k_group` to bound peak memory.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Threshold ladder + dwell-length -> bin LUT (both built exactly like Grind.cpp)
# ---------------------------------------------------------------------------
def make_thresholds(thr_min: float, thr_max: float, count: int) -> np.ndarray:
    """int16 ladder, identical to build_grind_params()."""
    assert count >= 1 and (count == 1 or thr_max > thr_min)

    def floor_clamped_i16(v: float) -> int:
        f = math.floor(v)
        return int(max(np.iinfo(np.int16).min, min(np.iinfo(np.int16).max, f)))

    if count == 1:
        return np.array([floor_clamped_i16(0.5 * (thr_min + thr_max))], dtype=np.int16)
    delta = (thr_max - thr_min) / float(count - 1)
    return np.array([floor_clamped_i16(thr_min + delta * k) for k in range(count)], dtype=np.int16)


def make_dwell_lut(max_len: int, fs_hz: float, tmin_s: float, tmax_s: float, J: int) -> np.ndarray:
    """LUT[L] = bin of a dwell of L samples (or -1 = dropped), for L in [0, max_len].

    Same float64 formula as Grind.cpp's bin_idx(): t = L/fs, drop if t < tmin or
    t >= tmax, idx = floor((log10 t - log10 tmin) * (1/(log10 tmax - log10 tmin)) * B).
    """
    assert tmax_s > tmin_s > 0 and J > 0
    a = math.log10(tmin_s)
    b = math.log10(tmax_s)
    inv_span = 1.0 / (b - a)
    L = np.arange(max_len + 1, dtype=np.float64)
    t = L / fs_hz
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (np.log10(np.where(t > 0, t, 1.0)) - a) * inv_span
        idx = np.floor(x * float(J)).astype(np.int64)
    ok = (t > 0) & (t >= tmin_s) & (t < tmax_s) & (idx >= 0) & (idx < J)
    return np.where(ok, idx, -1).astype(np.int16)


# ---------------------------------------------------------------------------
# Reference (slow, CPU): literal transcription of the C++ per-threshold state
# machine. Used only to validate the GPU implementation.
# ---------------------------------------------------------------------------
def grind_hist_reference(y: np.ndarray, thr: np.ndarray, lut: np.ndarray, J: int) -> Tuple[np.ndarray, np.ndarray]:
    K = len(thr)
    hist = np.zeros((K, J, J), dtype=np.float32)
    nev = np.zeros(K, dtype=np.int64)
    N = len(y)
    if N < 2:
        return hist, nev
    for k in range(K):
        lab = (y.astype(np.int64) > int(thr[k]))
        cur_start = 0
        cur_lab = int(lab[0])
        have_prev = False
        prev_len = 0
        prev_lab = 0
        for i in range(1, N):
            if int(lab[i]) == cur_lab:
                continue
            cur_len = i - cur_start
            if have_prev and prev_len != 0 and cur_len != 0:
                bx = lut[prev_len] if prev_lab == 0 else lut[cur_len]
                by = lut[cur_len] if prev_lab == 0 else lut[prev_len]
                if bx >= 0 and by >= 0:
                    hist[k, by, bx] += 1.0
                    nev[k] += 1
            prev_lab = cur_lab
            prev_len = cur_len
            have_prev = True
            cur_lab = int(lab[i])
            cur_start = i
    return hist, nev


# ---------------------------------------------------------------------------
# GPU implementation
# ---------------------------------------------------------------------------
@torch.no_grad()
def grind_hist_gpu(
    x: torch.Tensor,          # (B, T) int16 (or any integer dtype), on the compute device
    thr: torch.Tensor,        # (K,) integer thresholds shared by every row, OR (B,K) one ladder per row
    lut: torch.Tensor,        # (T+1,) int16 dwell-length -> bin LUT (-1 = dropped), same device
    J: int,
    k_group: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (hist (B,K,J,J) float32, n_events (B,K) int64).

    thr.dim()==1: the classic global ladder (same K thresholds for every row -- e.g. the uniform
    hard-threshold reference, or the fixed learned-theta custom-threshold experiment).
    thr.dim()==2 (B,K): a PER-ROW ladder -- e.g. self-calibrating thresholds built from a per-row
    baseline/amplitude estimate (SelfCalibratingFusionNet). Everything downstream (event detection,
    dwell binning, the (row = b*kg+kk) indexing into the output) is unchanged either way; only how
    thr_g is broadcast against x differs.
    """
    assert x.dim() == 2
    B, T = x.shape
    per_row = thr.dim() == 2
    if per_row:
        assert thr.shape[0] == B, f"thr (B,K) must have B={B} rows, got {thr.shape}"
        K = int(thr.shape[1])
    else:
        K = int(thr.numel())
    dev = x.device
    assert lut.numel() >= T + 1, f"LUT too short: {lut.numel()} < {T + 1}"

    hist = torch.zeros(B * K * J * J, device=dev, dtype=torch.float32)
    nev = torch.zeros(B * K, device=dev, dtype=torch.int64)
    if T < 2:
        return hist.view(B, K, J, J), nev.view(B, K)

    for k0 in range(0, K, k_group):
        if per_row:
            thr_g = thr[:, k0:k0 + k_group]                        # (B, kg)
            kg = int(thr_g.shape[1])
            thr_b = thr_g.unsqueeze(2)                              # (B, kg, 1) -- broadcasts per row
        else:
            thr_g = thr[k0:k0 + k_group]                            # (kg,)
            kg = int(thr_g.numel())
            thr_b = thr_g.view(1, kg, 1)                            # (1, kg, 1) -- broadcasts globally
        R = B * kg

        s = (x.unsqueeze(1) > thr_b).reshape(R, T)     # (R,T) bool, row = b*kg + kk
        diff = s[:, 1:] != s[:, :-1]                                   # (R,T-1)
        nz = diff.nonzero()                                            # (M,2), sorted by row then position
        del diff
        M = int(nz.shape[0])
        if M < 2:
            continue

        row = nz[:, 0]
        idx = nz[:, 1] + 1                       # sample index where a new run starts (== C++ `i`)
        del nz

        same_prev = torch.zeros(M, dtype=torch.bool, device=dev)
        same_prev[1:] = row[1:] == row[:-1]
        prev_idx = torch.where(same_prev, torch.roll(idx, 1), torch.zeros_like(idx))
        length = idx - prev_idx                  # length of the dwell that ends at this transition
        del prev_idx, idx

        counts = torch.bincount(row, minlength=R)
        offs = torch.cumsum(counts, 0) - counts
        j_in_row = torch.arange(M, device=dev) - offs[row]     # dwell index within its row
        s0 = s[:, 0]                                            # label of the first dwell
        del s, counts, offs

        pair = same_prev[1:]                     # entry e+1 continues entry e's row -> (dwell e, dwell e+1) is a pair
        a = length[:-1][pair]
        b = length[1:][pair]
        row_p = row[:-1][pair]
        j_p = j_in_row[:-1][pair]
        del length, same_prev, row, j_in_row, pair

        lab_a = s0[row_p] ^ ((j_p & 1) == 1)     # label of the first dwell of the pair (True = above)
        X = torch.where(lab_a, b, a)             # below-threshold dwell
        Y = torch.where(lab_a, a, b)             # above-threshold dwell
        bx = lut[X].long()
        by = lut[Y].long()
        ok = (bx >= 0) & (by >= 0)
        bb = torch.div(row_p, kg, rounding_mode="floor")
        kk = k0 + (row_p - bb * kg)
        flat_cell = ((bb * K + kk) * J + by) * J + bx
        ones = torch.ones(int(ok.sum()), device=dev, dtype=torch.float32)
        hist.scatter_add_(0, flat_cell[ok], ones)
        nev.scatter_add_(0, (bb * K + kk)[ok], torch.ones_like(flat_cell[ok]))
        del X, Y, bx, by, ok, bb, kk, flat_cell, a, b, row_p, j_p, lab_a, s0, ones

    return hist.view(B, K, J, J), nev.view(B, K)
