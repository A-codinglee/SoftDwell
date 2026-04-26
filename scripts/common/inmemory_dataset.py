# scripts/common/inmemory_dataset.py
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Tuple


class InMemoryDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        """
        X: (N, C, T) tensor in RAM
        y: (N, ...) tensor in RAM
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        return x, self.y[idx]
