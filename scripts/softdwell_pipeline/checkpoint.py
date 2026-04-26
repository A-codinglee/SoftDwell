# checkpoint.py
import os
import time
import torch

def is_rank0():
    return (not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0)

def ddp_unwrap(m):
    return m.module if hasattr(m, "module") else m

def atomic_save(obj, tmp, final):
    torch.save(obj, tmp)
    os.replace(tmp, final)

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, cfg, extra=None):
    """
    Save training state. 'extra' is a free-form dict (e.g., {'earlystop': es.state_dict()}).
    No RNG states are saved.
    """
    if not is_rank0():
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "epoch": int(epoch),
        "global_step": int(step),

        "model": ddp_unwrap(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,

        # free-form extras (EarlyStopping, EMA, whatever you like)
        "extra": extra or {},

        "saved_time": time.time(),
        # optionally keep cfg snapshot if you want:
        # "cfg": getattr(cfg, "__dict__", None),
    }
    atomic_save(obj, path + ".tmp", path)

def load_checkpoint(path, model, optimizer, scheduler):
    """
    Load training state. Returns (epoch, global_step, extra_dict).
    """
    ckpt = torch.load(path, map_location="cpu")

    # core states
    ddp_unwrap(model).load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    epoch = int(ckpt.get("epoch", 0))
    step  = int(ckpt.get("global_step", 0))
    extra = ckpt.get("extra", {}) or {}
    return epoch, step, extra
