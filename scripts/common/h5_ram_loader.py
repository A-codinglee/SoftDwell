# scripts/common/h5_ram_loader.py
import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from typing import Optional, Callable, Tuple

import torch
import torch.distributed as dist
from scripts.common.canonicalize import extract_chain_rates_from_Q


# ===============================================================
# Worker: receives ONE tuple argument (required for imap_unordered)
# ===============================================================
def _load_xy_chunk(args) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Worker: load a slice [start:end) of X and Y from HDF5 and return it.
    """
    (
        path,
        x_key,
        y_key,
        start,
        end,
        y_log10,
        canonicalizer,
        y_from_Q,
        t_start,
        t_len,
    ) = args

    with h5py.File(path, "r") as f:
        x_ds = f[x_key]
        y_ds = f[y_key]

        # ---- slice time during read ----
        if t_len is None:
            x_raw = x_ds[start:end]
        else:
            t0 = int(t_start)
            t1 = int(t_start + t_len)
            if x_ds.ndim == 2:        # (N, T)
                x_raw = x_ds[start:end, t0:t1]
            elif x_ds.ndim == 3:      # (N, C, T)
                x_raw = x_ds[start:end, :, t0:t1]
            else:
                raise ValueError(f"Unsupported x_ds ndim={x_ds.ndim}, shape={x_ds.shape}")

        x = np.asarray(x_raw, dtype=np.float32)
        y_raw = np.asarray(y_ds[start:end])

    # Ensure X is (M, C, T)
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim != 3:
        raise ValueError(f"X slice must be 3D (M,C,T), got {x.shape}")

    # Y processing
    if y_from_Q:
        y = extract_chain_rates_from_Q(np.asarray(y_raw))
        y = y.astype(np.float32)
    else:
        y = np.asarray(y_raw, dtype=np.float32)
        
    if canonicalizer is not None:
        y = np.stack([canonicalizer(row) for row in y], axis=0)

    if y_log10:
        y = np.log10(np.clip(y, 1e-30, None))

    return start, end, x, y


# ===============================================================
# Main parallel HDF5 loader (memory-safe)
# ===============================================================
def parallel_hdf5_load_xy(
    path: str,
    x_key: str,
    y_key: str,
    y_log10: bool,
    canonicalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    limit_n: Optional[int] = None,
    num_workers: int = 4,
    chunk_size: int = 1024,
    *,
    y_from_Q: bool = False,
    t_start: int = 0,
    t_len: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load (X, Y) from HDF5 into RAM using multiprocessing.

    - Preallocate final arrays once.
    - Workers load chunks into temporary arrays.
    - Main process copies chunks into the preallocated arrays.
    - Shows a tqdm progress bar over *samples*.
    """
    # 1) Determine shapes + preallocate
    with h5py.File(path, "r") as f:
        if x_key not in f or y_key not in f:
            raise KeyError(f"{x_key} or {y_key} not found. Keys: {list(f.keys())}")

        x_ds = f[x_key]
        y_ds = f[y_key]

        n = min(len(x_ds), len(y_ds))
        if limit_n is not None and limit_n > 0:
            n = min(n, limit_n)

        if n <= 0:
            raise ValueError("No samples to load.")

        sample_x = np.asarray(x_ds[0])

        # Case 1: (T,)  -> treat as (1, T)
        if sample_x.ndim == 1:
            C, T_full = 1, sample_x.shape[0]
        elif sample_x.ndim == 2:
            C, T_full = sample_x.shape[0], sample_x.shape[-1]
        elif sample_x.ndim == 3:
            C, T_full = sample_x.shape[0], sample_x.shape[-1]
        else:
            raise ValueError(f"Unsupported X sample shape: {sample_x.shape}")

        T = T_full if t_len is None else int(t_len)
        X_np = np.empty((n, C, T), dtype=np.float32)
        
        sample_y = np.asarray(y_ds[0])
        if y_from_Q:
            if sample_y.ndim != 2 or sample_y.shape[0] != sample_y.shape[1]:
                raise ValueError(f"y_from_Q=True but Y sample is not square (S,S): got {sample_y.shape}")
            S = sample_y.shape[0]
            D = 2 * (S - 1)
            y_shape = (D,)
        else:
            y_shape = y_ds.shape[1:]

        y_np = np.empty((n, *y_shape), dtype=np.float32)

    # 2) Build tasks
    tasks = [
        (path, x_key, y_key, start, min(start + chunk_size, n), y_log10, canonicalizer, y_from_Q, t_start, t_len)
        for start in range(0, n, chunk_size)
    ]

    n_workers = min(num_workers, len(tasks))
    ctx = mp.get_context("spawn")

    # 3) Parallel load with per-sample progress
    with ctx.Pool(n_workers) as pool:
        with tqdm(
            total=n,
            desc="Loading HDF5 into RAM",
            unit="samples",
        ) as pbar:
            for start, end, x_chunk, y_chunk in pool.imap_unordered(_load_xy_chunk, tasks):
                X_np[start:end] = x_chunk
                y_np[start:end] = y_chunk
                pbar.update(end - start)

    return X_np, y_np
