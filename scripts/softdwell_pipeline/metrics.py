import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def rae_log(pred_log: torch.Tensor, y_log: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.abs(pred_log - y_log) + 1e-12)

class LogCoshLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r = input - target
        v = torch.abs(r)
        c = torch.log(torch.tensor(2.0, device=v.device, dtype=v.dtype))
        loss = v + F.softplus(-2.0 * v) - c
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

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

    def state_dict(self):
        # Only save lightweight training progress; do NOT save _best_state (model weights).
        return {
            "best": self.best,
            "best_epoch": self.best_epoch,
            "count": self.count,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "restore_best": self.restore_best,
        }

    def load_state_dict(self, d: dict):
        # Restore dynamic counters and config so patience continues after resume.
        # _best_state remains None after resume (that’s fine since you also save best_model.pt separately).
        self.best = float(d.get("best", self.best))
        self.best_epoch = int(d.get("best_epoch", self.best_epoch))
        self.count = int(d.get("count", self.count))
        self.patience = int(d.get("patience", self.patience))
        self.min_delta = float(d.get("min_delta", self.min_delta))
        self.restore_best = bool(d.get("restore_best", self.restore_best))