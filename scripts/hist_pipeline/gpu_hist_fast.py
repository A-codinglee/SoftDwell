# scripts/hist_pipeline/gpu_hist_fast.py
"""
A leaner version of grind_hist_gpu (scripts/scripts/hardthresh_pipeline/gpu_hist.py) that gives the
identical histograms (checked by check_against_reference below) with less GPU work:

  * no per-threshold event counts (training never uses them; they cost a second scatter)
  * transitions found with one flat nonzero() instead of a (row, position) pair per transition
  * the label of a dwell read directly from the thresholded trace instead of via a per-row
    bincount / cumsum / parity bookkeeping
  * counts accumulated with bincount (accumulate="bincount") or scatter_add (accumulate="scatter")

Semantics are those of the reference (Grind.cpp): per threshold, every pair of consecutive completed
dwells is filed at [above-dwell bin, below-dwell bin]; the first dwell counts once it ends, the last
(unterminated) one never does; a pair with a dwell outside the LUT range is dropped.
"""
import torch


@torch.no_grad()
def grind_hist_fast(x, thr, lut, J, k_group=8, accumulate="bincount"):
    """x (B, T) int16 on the device; thr (K,) shared or (B, K) per row; lut (T+1,). Returns hist (B, K, J, J)."""
    B, T = x.shape
    per_row = thr.dim() == 2
    K = int(thr.shape[1]) if per_row else int(thr.numel())
    dev = x.device
    n_cells = B * K * J * J
    hist = torch.zeros(n_cells, device=dev, dtype=torch.float32)
    if T < 2:
        return hist.view(B, K, J, J)
    lut = lut.long()
    for k0 in range(0, K, k_group):
        thr_g = thr[:, k0:k0 + k_group] if per_row else thr[k0:k0 + k_group]
        kg = int(thr_g.shape[-1])
        thr_b = thr_g.unsqueeze(2) if per_row else thr_g.view(1, kg, 1)
        R = B * kg
        s = (x.unsqueeze(1) > thr_b).reshape(R, T)                    # row = b*kg + kk
        f = (s[:, 1:] != s[:, :-1]).reshape(-1).nonzero().squeeze(1)  # flat transition positions, sorted
        if f.numel() < 2:
            continue
        row = torch.div(f, T - 1, rounding_mode="floor")
        idx = f - row * (T - 1) + 1                                    # sample where the new run starts
        del f
        same = row[1:] == row[:-1]                                     # transition e and e+1 in one row
        prev = torch.zeros_like(idx)
        prev[1:] = torch.where(same, idx[:-1], torch.zeros_like(idx[:-1]))
        length = idx - prev                                            # dwell ending at each transition
        # label of the dwell ending at transition e = the trace's label just before it
        lab = s.reshape(-1)[row * T + idx - 1]
        del s, prev
        a, b = length[:-1][same], length[1:][same]
        lab_a = lab[:-1][same]
        row_p = row[:-1][same]
        del length, lab, row, idx, same
        X = torch.where(lab_a, b, a)                                   # below-threshold dwell
        Y = torch.where(lab_a, a, b)                                   # above-threshold dwell
        bx, by = lut[X], lut[Y]
        ok = (bx >= 0) & (by >= 0)
        bb = torch.div(row_p, kg, rounding_mode="floor")
        kk = k0 + (row_p - bb * kg)
        cell = (((bb * K + kk) * J + by) * J + bx)[ok]
        if accumulate == "bincount":
            hist += torch.bincount(cell, minlength=n_cells).to(torch.float32)
        else:
            hist.scatter_add_(0, cell, torch.ones(cell.numel(), device=dev, dtype=torch.float32))
    return hist.view(B, K, J, J)


def check_against_reference(x, thr, lut, J, k_group=8):
    """True if grind_hist_fast equals the reference grind_hist_gpu, both accumulation modes."""
    from scripts.scripts.hardthresh_pipeline.gpu_hist import grind_hist_gpu
    ref, _ = grind_hist_gpu(x, thr, lut, J, k_group=k_group)
    return all(torch.equal(ref, grind_hist_fast(x, thr, lut, J, k_group, acc)) for acc in ("bincount", "scatter"))
