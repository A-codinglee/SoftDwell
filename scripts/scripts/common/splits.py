# splits.py
import os
import numpy as np
import torch
from pathlib import Path
import torch.distributed as dist

def split_file_path(cfg, N: int) -> str:
    stem = Path(cfg.h5_path).stem
    fname = f"{stem}__{cfg.split_name}_seed{cfg.split_seed}_N{N}.npz"
    return os.path.join(cfg.splits_dir, fname)

def build_or_load_splits(cfg, N: int, rank: int):
    path = split_file_path(cfg, N)
    if os.path.isfile(path):
        if rank == 0:
            print(f"[splits] loading fixed split from {path}", flush=True)
        data = np.load(path, allow_pickle=False)
        return data["train_idx"], data["val_idx"], data["test_idx"]

    if rank == 0:
        os.makedirs(cfg.splits_dir, exist_ok=True)
        n_test = max(1, int(cfg.test_split * N))
        n_val  = max(1, int(cfg.val_split  * (N - n_test)))
        n_train = N - n_val - n_test

        g = torch.Generator().manual_seed(cfg.split_seed)
        perm = torch.randperm(N, generator=g).tolist()

        train_idx = np.array(perm[:n_train], dtype=np.int64)
        val_idx   = np.array(perm[n_train:n_train+n_val], dtype=np.int64)
        test_idx  = np.array(perm[n_train+n_val:], dtype=np.int64)

        assert len(train_idx) + len(val_idx) + len(test_idx) == N
        assert not (set(train_idx) & set(val_idx) or
                    set(train_idx) & set(test_idx) or
                    set(val_idx)  & set(test_idx))

        np.savez_compressed(path, train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
        print(f"[splits] wrote fixed split to {path} "
              f"(train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)})", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    data = np.load(path, allow_pickle=False)
    return data["train_idx"], data["val_idx"], data["test_idx"]
