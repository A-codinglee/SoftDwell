import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rae_log(pred_log: torch.Tensor, y_log: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(pred_log - y_log) + 1e-12)

class LogCoshLoss(nn.Module):
    def __init__(self, reduction="mean"): super().__init__(); self.reduction = reduction
    def forward(self, input, target):
        x = input - target
        logcosh = x + F.softplus(-2.0 * x) - math.log(2.0)
        return logcosh.mean() if self.reduction=="mean" else logcosh.sum() if self.reduction=="sum" else logcosh

class EarlyStopping:
    """
    Stop if val_loss fails to improve by > min_delta for `patience` checks.
    Call with the (globally averaged) validation loss.
    """
    def __init__(self, patience: int = 12, min_delta: float = 0.0, restore_best: bool = True):
        self.patience = patience
        self.min_delta = float(min_delta)
        self.best = float("inf")
        self.best_epoch = -1
        self.count = 0
        self.restore_best = restore_best
        self._best_state = None

    def step(self, val_loss: float, model: torch.nn.Module | None = None, epoch: int | None = None) -> bool:
        if not math.isfinite(val_loss):
            # treat non-finite as no improvement
            self.count += 1
            return self.count >= self.patience

        improved = val_loss < (self.best - self.min_delta)
        if improved:
            self.best = val_loss
            self.count = 0
            if epoch is not None:
                self.best_epoch = epoch
            if self.restore_best and model is not None:
                # lightweight copy of state dict (CPU)
                self._best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.count += 1

        return self.count >= self.patience

    def restore(self, model: torch.nn.Module):
        if self.restore_best and self._best_state is not None:
            model.load_state_dict(self._best_state)