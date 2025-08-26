# hdf5_dataset.py
import numpy as np, torch, h5py
from torch.utils.data import Dataset

class HDF5IonChannelDataset(Dataset):
    """
    Returns:
      x: (C, T) float32
      y: float32 (same shape as stored)
    """
    def __init__(self, h5_path, x_key="timeseries", y_key="labels",
                 x_order="auto", y_scale=1.0, y_log10=False, dtype=torch.float32,
                 start=None, stop=None):
        self.h5_path, self.x_key, self.y_key = h5_path, x_key, y_key
        self.x_order, self.y_scale, self.y_log10, self.dtype = x_order, y_scale, y_log10, dtype
        self._h5 = None; self._x = None; self._y = None
        with h5py.File(self.h5_path, "r") as h5:
            if self.x_key not in h5 or self.y_key not in h5:
                raise KeyError(f"Datasets not found. Available: {list(h5.keys())}")
            n = min(len(h5[self.x_key]), len(h5[self.y_key]))
        self.i0 = 0 if start is None else max(0, start)
        self.i1 = n if stop is None else min(n, stop)
        if self.i0 >= self.i1:
            raise ValueError(f"Invalid slice [{self.i0}:{self.i1}] of {n}")

    def _need(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._x = self._h5[self.x_key]
            self._y = self._h5[self.y_key]

    @staticmethod
    def _infer_order_2d(x2d: np.ndarray) -> str:
        return "CT" if x2d.shape[1] >= x2d.shape[0] else "TC"

    def __len__(self): return self.i1 - self.i0

    def __getitem__(self, idx):
        self._need()
        i = self.i0 + idx
        x = np.asarray(self._x[i])
        if x.ndim == 1:
            x = x[None, :]
        elif x.ndim == 2:
            order = self.x_order if self.x_order != "auto" else self._infer_order_2d(x)
            if order == "TC": x = x.T
        else:
            if x.shape[-1] < x.shape[-2]:
                raise ValueError(f"Ambiguous feature shape {x.shape}; set x_order explicitly.")
            x = x.reshape(int(np.prod(x.shape[:-1])), x.shape[-1])
        x = torch.from_numpy(x).to(self.dtype)

        y = np.asarray(self._y[i]).astype(np.float32, copy=False)
        if self.y_log10: y = np.log10(np.clip(y, 1e-30, None))
        if self.y_scale != 1.0: y = y / self.y_scale
        y = torch.from_numpy(y).to(self.dtype)
        return x, y

    def close(self):
        try:
            if self._h5 is not None: self._h5.close()
        except Exception: pass
