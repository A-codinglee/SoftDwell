import os, tempfile
import torch
import torch.distributed as dist
from datetime import timedelta

def ddp_init_from_torchrun(backend=None):
    """Return (is_ddp, local_rank, world_size, global_rank). Works for torchrun or single-process."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        world_size = int(os.environ["WORLD_SIZE"])
        global_rank = int(os.environ["RANK"])
        # Be robust if LOCAL_RANK is missing
        ndev = max(1, torch.cuda.device_count())
        local_rank = int(os.environ.get("LOCAL_RANK", global_rank % ndev))

        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        dist.init_process_group(backend=backend, timeout=timedelta(minutes=60))
        return True, local_rank, world_size, global_rank

    # single-process path
    return False, 0, 1, 0

def is_main_process():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

def ddp_cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def ddp_allreduce_mean(tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= dist.get_world_size()
    return tensor

def choose_store_dir():
    for cand in (os.environ.get("SLURM_TMPDIR"), "/dev/shm", "/tmp"):
        if cand and os.path.isdir(cand) and os.access(cand, os.W_OK):
            import os
            d = os.path.join(cand, f"ddp_store_{os.getpid()}")
            os.makedirs(d, exist_ok=True)
            return d
    return tempfile.mkdtemp(prefix="ddp_store_")