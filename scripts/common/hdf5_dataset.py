# hdf5_dataset.py
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple


class HDF5IonChannelDataset(Dataset):
    """
    Ion-channel dataset backed by an HDF5 file.

    Assumptions (project-specific):
      - X is stored as a 2D array per sample with shape (C, T) in float32.
      - Y is stored as a 1D or 2D array per sample (e.g. (8,) rates).

    Each __getitem__(idx) returns:
      x: torch.Tensor of shape (C, T)
      y: torch.Tensor with same shape as stored per sample (after canonicalization / transforms)
    """

    def __init__(
        self,
        h5_path: str,
        x_key: str = "timeseries",
        y_key: str = "labels",
        y_log10: bool = False,
        dtype: torch.dtype = torch.float32,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        canonicalizer: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        self.h5_path = h5_path
        self.x_key = x_key
        self.y_key = y_key
        self.y_log10 = y_log10
        self.dtype = dtype
        self.canonicalizer = canonicalizer

        # Lazy-open handles (per worker / per process)
        self._h5 = None
        self._x = None
        self._y = None

        # Probe file once to get dataset length and check keys.
        with h5py.File(self.h5_path, "r") as h5:
            if self.x_key not in h5 or self.y_key not in h5:
                raise KeyError(
                    f"Datasets not found in {self.h5_path!r}. "
                    f"Available keys: {list(h5.keys())}"
                )
            nx = len(h5[self.x_key])
            ny = len(h5[self.y_key])
            n = min(nx, ny)
            if nx != ny:
                print(
                    f"[HDF5IonChannelDataset] Warning: len(x)={nx} != len(y)={ny}. "
                    f"Using n={n} (minimum of both)."
                )

        # Slice [i0, i1) over the n samples.
        self.i0 = 0 if start is None else max(0, start)
        self.i1 = n if stop is None else min(n, stop)
        if self.i0 >= self.i1:
            raise ValueError(f"Invalid slice [{self.i0}:{self.i1}] of total n={n}")

    # ---- internal helpers ----

    def _ensure_open(self) -> None:
        """Open the HDF5 file lazily (per worker / per process)."""
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
            self._x = self._h5[self.x_key]
            self._y = self._h5[self.y_key]

    def _load_x(self, i: int) -> np.ndarray:
        """
        Load one X sample as a numpy array and normalize to shape (C, T), float32.

        In this cleaned-up version, we assume the stored shape is already (C, T).
        """
        x = np.asarray(self._x[i])
        if x.ndim == 1:
            x = x[None, :]  # (T,) -> (1, T)
        elif x.ndim != 2:
            raise ValueError(
                f"Expected X[i] to be 2D (C, T), got shape {x.shape} for index {i}"
            )
        return x.astype(np.float32, copy=False)

    def _load_y(self, i: int) -> np.ndarray:
        """
        Load one Y sample as a numpy array and apply:
          - canonicalization (if provided)
          - log10 transform (if enabled)
        """
        y = np.asarray(self._y[i]).astype(np.float32, copy=False)

        if self.canonicalizer is not None:
            y = self.canonicalizer(y)

        if self.y_log10:
            # Guard against zeros or negative values before log10
            y = np.log10(np.clip(y, 1e-30, None))

        return y

    # ---- Dataset interface ----

    def __len__(self) -> int:
        return self.i1 - self.i0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_open()
        i = self.i0 + idx

        x_np = self._load_x(i)  # (C, T), float32
        y_np = self._load_y(i)  # float32

        x = torch.from_numpy(x_np).to(self.dtype)
        y = torch.from_numpy(y_np).to(self.dtype)
        return x, y

    def close(self) -> None:
        """Optional manual cleanup."""
        try:
            if self._h5 is not None:
                self._h5.close()
        except Exception:
            pass
