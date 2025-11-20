# scripts/common/ddp_utils.py
import os
import torch
import torch.distributed as dist


def ddp_setup():
    """
    Initialize DDP if launched with torchrun.

    Returns:
        is_ddp (bool): whether DDP is used
        rank (int): global rank
        world_size (int): world size
        local_rank (int): local rank (GPU index on the node)
        device (torch.device): device for this process
    """
    if "RANK" not in os.environ:
        # single-process (no DDP)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return False, 0, 1, 0, device

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device(f"cuda:{local_rank}")
    return True, rank, world_size, local_rank, device


def is_main_process(rank: int) -> bool:
    return rank == 0


def cleanup_ddp(is_ddp: bool):
    """Safely tear down the distributed process group."""
    if is_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rprint(rank, *args, **kwargs):
    """Print only on rank 0."""
    if is_main_process(rank):
        print(*args, **kwargs, flush=True)


def ddp_allreduce_sum(t: torch.Tensor):
    """All-reduce a tensor by SUM in-place (no-op if not in DDP)."""
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def gather_concat_varlen(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    All-gather a tensor along 'dim' when each rank may have a different size
    on that dim. Returns the concatenated tensor on every rank.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x

    world_size = dist.get_world_size()
    device = x.device

    # lengths per rank on 'dim'
    local_len = torch.tensor([x.size(dim)], device=device, dtype=torch.long)
    lens = [torch.zeros_like(local_len) for _ in range(world_size)]
    dist.all_gather(lens, local_len)
    lens = torch.stack(lens).squeeze(1)  # [world_size]
    max_len = int(lens.max().item())

    # pad to max_len on 'dim'
    if x.size(dim) < max_len:
        new_shape = list(x.shape)
        new_shape[dim] = max_len
        x_pad = x.new_zeros(new_shape)
        sl = [slice(None)] * x.dim()
        sl[dim] = slice(0, x.size(dim))
        x_pad[tuple(sl)] = x
    else:
        x_pad = x

    # gather padded tensors
    gather_list = [torch.empty_like(x_pad) for _ in range(world_size)]
    dist.all_gather(gather_list, x_pad)

    # trim back to original sizes and concat
    pieces = []
    for r, g in enumerate(gather_list):
        if lens[r] > 0:
            sl = [slice(None)] * g.dim()
            sl[dim] = slice(0, int(lens[r].item()))
            pieces.append(g[tuple(sl)])
    return torch.cat(pieces, dim=dim)
