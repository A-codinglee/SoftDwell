# scripts/hist_pipeline/dataset_hist.py
import h5py
import torch
from torch.utils.data import Dataset

# 5-state chain:
# 0 <-> 1 <-> 2 <-> 3 <-> 4
RATE_POSITIONS = [
    (0, 1),  # k01
    (1, 0),  # k10
    (1, 2),  # k12
    (2, 1),  # k21
    (2, 3),  # k23
    (3, 2),  # k32
    (3, 4),  # k34
    (4, 3),  # k43
]

class HistogramRateDataset(Dataset):
    def __init__(self, h5_path: str, log10_targets: bool = True):
        self.h5_path = h5_path
        self.log10_targets = log10_targets
        with h5py.File(h5_path, "r") as f:
            self.n_samples = f["histograms/grids"].shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_path, "r") as f:
            grids = f["histograms/grids"][idx]      #
            Q     = f["labels/Q"][idx]             # (5, 5)

        x = torch.from_numpy(grids).float()   

        # targets: pick 8 off-diagonals in fixed order
        y_vals = [Q[i, j] for (i, j) in RATE_POSITIONS]
        y = torch.tensor(y_vals, dtype=torch.float32)  # (8,)

        if self.log10_targets:
            # Q entries are in [10, 1e6], so log10 brings them to [1, 6]
            eps = 1e-12
            y = torch.log10(torch.clamp(y, min=eps))

        return x, y
