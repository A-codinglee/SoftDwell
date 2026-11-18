# scripts/common/h5_ram_loader.py
import h5py
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # Keep native dtype here (int16 in this dataset) rather than upcasting
        # to float32 -- halves both this transient chunk and the persistent
        # X_np array it gets copied into. train_ram.py already casts to
        # float32 per-batch right before GPU use, so nothing downstream needs
        # the wider dtype any earlier than that.
        x = np.asarray(x_raw)
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
        # Match the source HDF5 dtype (int16 for this dataset) rather than
        # float32 -- see _load_xy_chunk for why; halves host RAM use for the
        # full preloaded array.
        X_np = np.empty((n, C, T), dtype=x_ds.dtype)
        
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

    # 3) Parallel load with per-sample progress.
    #
    # Threads, not processes: each worker's chunk is cast to float32 before
    # being handed back (~4MB/sample), and a process Pool would have to
    # pickle+pipe that whole array across a process boundary to return it --
    # cheap on a node with CPU/bandwidth to spare, but the dominant cost on a
    # thin CPU allocation (e.g. 4 cores/GPU) where that serialization work
    # competes directly with the read/cast work for the same few cores.
    # Threads share memory with the parent, so fut.result() is a plain
    # reference, no serialization. Each task opens its own h5py.File handle
    # (see _load_xy_chunk), so there's no shared-handle thread-safety issue.
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_load_xy_chunk, t) for t in tasks]
        with tqdm(
            total=n,
            desc="Loading HDF5 into RAM",
            unit="samples",
        ) as pbar:
            for fut in as_completed(futures):
                start, end, x_chunk, y_chunk = fut.result()
                X_np[start:end] = x_chunk
                y_np[start:end] = y_chunk
                pbar.update(end - start)

    return X_np, y_np
