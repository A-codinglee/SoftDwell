import os, re, hashlib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def _cache_dir():
    base = os.environ.get("XDG_CACHE_HOME", os.path.join(Path.home(), ".cache"))
    d = os.path.join(base, "mfpb_idx")
    os.makedirs(d, exist_ok=True)
    return d

def _idx_path_for(npy_path):
    st = os.stat(npy_path)
    sig = f"{npy_path}|{st.st_size}|{int(st.st_mtime)}".encode()
    h = hashlib.sha1(sig).hexdigest()[:16]
    return os.path.join(_cache_dir(), f"{os.path.basename(npy_path)}.{h}.idx.npy")

def build_or_load_index(npy_path):
    idx_path = _idx_path_for(npy_path)
    if os.path.exists(idx_path):
        return np.load(idx_path, allow_pickle=False)

    offsets = []
    with open(npy_path, "rb") as f:
        while True:
            try:
                off = f.tell()
                _ = np.load(f, allow_pickle=True)
                offsets.append(off)
            except (ValueError, EOFError):
                break

    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.size == 0:
        raise RuntimeError(f"No records found in {npy_path}")

    try:
        np.save(idx_path, offsets)
    except PermissionError:
        return offsets
    return offsets

class PackedMemmapDataset(Dataset):
    def __init__(self, hist_path, label_path, log10_labels=True, normalize_hist=True, label_scale=1.0):
        self.hist_path, self.label_path = hist_path, label_path
        self.log10, self.norm, self.scale = log10_labels, normalize_hist, float(label_scale)
        self._h = None
        self._y = None

    def _ensure(self):
        def _is_npy(path: str) -> bool:
            with open(path, "rb") as f:
                return f.read(6) == b"\x93NUMPY"

        # --- Histograms ---
        if self._h is None:
            if _is_npy(self.hist_path):
                H = np.load(self.hist_path, mmap_mode="r")
                assert H.ndim == 3 and H.shape[1:] == (60, 60), f"unexpected hist shape {H.shape}"
                assert H.dtype == np.float32, f"hist dtype {H.dtype} != float32"
                self._h = H
            else:
                h = np.memmap(self.hist_path, dtype=np.float32, mode="r")
                assert h.size % (60 * 60) == 0, "raw hist file size not divisible by 60*60"
                N = h.size // (60 * 60)
                self._h = h.reshape(N, 60, 60)

        # --- Labels ---
        if self._y is None:
            if _is_npy(self.label_path):
                Y = np.load(self.label_path, mmap_mode="r")
                assert Y.ndim == 2 and Y.shape[1] == 8, f"unexpected label shape {Y.shape}"
                assert Y.dtype == np.float32, f"label dtype {Y.dtype} != float32"
                self._y = Y
            else:
                y = np.memmap(self.label_path, dtype=np.float32, mode="r")
                assert y.size % 8 == 0, "raw label file size not divisible by 8"
                self._y = y.reshape(-1, 8)

        # --- Consistency check ---
        assert self._y.shape[0] == self._h.shape[0], \
            f"hists N={self._h.shape[0]} vs labels N={self._y.shape[0]} mismatch"


    def __len__(self):
        self._ensure()
        return self._h.shape[0]

    def __getitem__(self, i):
        self._ensure()
        hist = self._h[i]
        if self.norm:
            s = hist.sum(dtype=np.float32)
            if s > 0:
                hist = hist / s
        y = self._y[i] / self.scale
        if self.log10:
            y = np.log10(np.clip(y, 1e-12, None))
        x = torch.from_numpy(hist).unsqueeze(0)
        y = torch.from_numpy(y.astype(np.float32, copy=False))
        return x, y