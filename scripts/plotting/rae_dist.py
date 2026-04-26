#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def _ensure_parent_dir(path: Path) -> None:
    parent = path.parent
    if str(parent) != "":
        parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute mean RAE (log10 domain) and export a single-column CSV.")
    parser.add_argument("--preds", default="predictions_nn.npy", help="Path to predictions .npy (shape: N x 8)")
    parser.add_argument("--targs", default="targets_nn.npy", help="Path to targets .npy (shape: N x 8)")
    parser.add_argument("--out", default="rae_mean_only.csv", help="Output path for mean-only CSV")
    args = parser.parse_args()

    preds_path = Path(args.preds)
    targs_path = Path(args.targs)
    out_path = Path(args.out)

    if not preds_path.exists():
        raise FileNotFoundError(f"preds file not found: {preds_path}")
    if not targs_path.exists():
        raise FileNotFoundError(f"targs file not found: {targs_path}")

    preds_hz = np.load(preds_path)
    targs_hz = np.load(targs_path)

    if preds_hz.shape != targs_hz.shape:
        raise ValueError(f"Shape mismatch: preds {preds_hz.shape} vs targs {targs_hz.shape}")
    if preds_hz.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N, 8). Got preds ndim={preds_hz.ndim}")
    if preds_hz.shape[1] != targs_hz.shape[1]:
        raise ValueError(f"Expected second dim = 8. Got {preds_hz.shape[1]}")

    preds_log = np.log10(preds_hz)
    targs_log = np.log10(targs_hz)

    print("[debug] preds_log range:", float(preds_log.min()), "→", float(preds_log.max()))
    print("[debug] targs_log range:", float(targs_log.min()), "→", float(targs_log.max()))

    rae_matrix = np.sqrt(np.abs(preds_log - targs_log) + 1e-12)
    rae_mean = rae_matrix.mean(axis=1)

    _ensure_parent_dir(out_path)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rae_mean"])
        for v in rae_mean:
            writer.writerow([float(v)])

    print(f"[done] saved MEAN-ONLY CSV → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
