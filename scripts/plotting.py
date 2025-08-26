#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# Fixed filenames produced by train.py
PRED_FILE = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/predictions_nn.npy"
TARG_FILE = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/targets_nn.npy"

preds_hz = np.load(PRED_FILE)
targs_hz = np.load(TARG_FILE)

for i in range(preds_hz.shape[1]):
    x = np.clip(targs_hz[:, i], 1e-12, None)
    y = np.clip(preds_hz[:, i], 1e-12, None)
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=5, alpha=0.5, rasterized=True)
    lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
    plt.plot([lo, hi], [lo, hi], linestyle='--', linewidth=1)
    plt.xscale("log"); plt.yscale("log")
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel(f"Ground Truth (Hz) output {i}")
    plt.ylabel(f"Prediction (Hz) output {i}")
    plt.title(f"Scatter (log–log) for Output {i}")
    plt.grid(alpha=0.3, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/graphs/scatter_loglog_output_{i}.png", dpi=150)
    plt.close()
