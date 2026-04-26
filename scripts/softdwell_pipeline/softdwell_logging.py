# softdwell_logging.py
import os, io, torch
from scripts.softdwell_pipeline.checkpoint import is_rank0 

def _unwrap(model):
    # Works for DDP or plain module
    return getattr(model, "module", model)

def get_sd_inner(model):
    """Return the SoftDwell2DLayer no matter how the model is wrapped."""
    m = _unwrap(model)
    # Typical: SoftDwellThenTFIR.softdwell (ChannelWiseSoftDwell).inner (SoftDwell2DLayer)
    if hasattr(m, "softdwell") and hasattr(m.softdwell, "inner"):
        return m.softdwell.inner
    if hasattr(m, "inner"):  # ChannelWiseSoftDwell passed directly
        return m.inner
    return m  # assume the inner itself was passed

@torch.no_grad()
def _read_params_cpu(inner):
    theta = inner.raw_theta.detach().to(device='cpu', dtype=torch.float32)
    tau = inner.raw_tau.detach().to(device='cpu', dtype=torch.float32)
    return theta, tau


def log_sd_params_epoch(model, epoch: int, csv_path: str | None = None, stats_only: bool = True):
    """Rank-0 only: print θ/τ stats or append per-detector rows to CSV."""
    if not is_rank0():
        return
    inner = get_sd_inner(model)
    theta, tau = _read_params_cpu(inner)

    if stats_only:
        print(
            f"[softdwell] e{epoch+1} "
            f"theta mean={float(theta.mean()):.5f} min={float(theta.min()):.5f} max={float(theta.max()):.5f} | "
            f"tau mean={float(tau.mean()):.5f} min={float(tau.min()):.5f} max={float(tau.max()):.5f}",
            flush=True,
        )
        return

    if not csv_path:
        raise ValueError("csv_path must be provided when stats_only=False")

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    header_needed = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    buf = io.StringIO()
    if header_needed:
        buf.write("epoch,k,theta,tau\n")
    for k in range(theta.numel()):
        buf.write(
            f"{epoch+1},{k},"
            f"{float(theta[k]):.8g},{float(tau[k]):.8g}\n"
        )
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(buf.getvalue())
