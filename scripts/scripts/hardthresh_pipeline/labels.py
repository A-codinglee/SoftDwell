"""
Targets for the raw-trace datasets (generator: 3DFit_unclamped_for_sims,
level_mode "baseline_plus"):

  /labels/Q          (N, S, S)  rates, s^-1
  /labels/levels     (N, S)     per-state level in raw counts: closed label = baseline,
                                open label = baseline + amplitude   (C-O-C-O -> [b, b+a, b, b+a])
  /labels/noise_sigma (N,)      noise sigma of the row (see sigma_units)

Network targets (D = 2*(S-1) + 3 = 9 for the 4-state chain):
  [ log10 of the 2*(S-1) chain rates | baseline_pA | amplitude_pA | sigma_pA ]
The three extras are standardized on the TRAIN split (mean/std saved to
norm.json) and scaled by cfg.target_scale, so they sit on a scale comparable
with the log10 rates.
"""
from __future__ import annotations

import json
from typing import Tuple

import h5py
import numpy as np

from scripts.common.canonicalize import extract_chain_rates_from_Q, choose_canonicalizer

EXTRA_NAMES = ("baseline_pA", "amplitude_pA", "sigma_pA")


def read_labels(cfg, n_rows: int) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Returns (rates_hz (N,D_r) float64, extras_pA (N,3) float64, info dict)."""
    with h5py.File(cfg.h5_path, "r") as f:
        r0 = cfg.row_start
        Q = f[cfg.q_key][r0:r0 + n_rows]
        levels = f[cfg.levels_key][r0:r0 + n_rows].astype(np.float64)
        sigma = f[cfg.sigma_key][r0:r0 + n_rows].astype(np.float64)

    rates = extract_chain_rates_from_Q(Q).astype(np.float64)
    canon = choose_canonicalizer(cfg.topology, cfg.symmetry)
    if canon is not None:
        rates = np.stack([canon(r) for r in rates], axis=0)

    S = levels.shape[1]
    assert S >= 2, f"need >= 2 states, got {levels.shape}"
    baseline = levels[:, 0]
    amplitude = levels[:, 1] - levels[:, 0]

    # baseline_plus: states of the same label share a level (C-O-C-O -> b, b+a, b, b+a).
    if S >= 4:
        tol = 1e-3 * max(1.0, float(np.abs(levels).max()))
        assert np.allclose(levels[:, 2], levels[:, 0], atol=tol) and np.allclose(levels[:, 3], levels[:, 1], atol=tol), (
            "levels are not [b, b+a, b, b+a] -- this pipeline assumes the C-O-C-O baseline_plus layout; "
            "check the dataset / adapt read_labels()"
        )

    units = cfg.sigma_units
    if units == "auto":
        units = "counts" if float(np.median(sigma)) > 5.0 else "pA"
    sigma_pA = sigma / cfg.counts_per_pA if units == "counts" else sigma

    extras = np.stack([baseline / cfg.counts_per_pA, amplitude / cfg.counts_per_pA, sigma_pA], axis=1)
    info = {
        "sigma_units_used": units,
        "n_rows": int(rates.shape[0]),
        "range_pA": {n: [float(extras[:, i].min()), float(extras[:, i].max())] for i, n in enumerate(EXTRA_NAMES)},
        "rate_range_hz": [float(rates.min()), float(rates.max())],
    }
    return rates, extras, info


def make_targets(rates_hz: np.ndarray, extras_pA: np.ndarray, train_idx: np.ndarray, target_scale: float):
    """Build the (N, D) float32 network targets + the normalization dict."""
    y_rates = np.log10(np.clip(rates_hz, 1e-9, None))
    mean = extras_pA[train_idx].mean(axis=0)
    std = extras_pA[train_idx].std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    z = (extras_pA - mean) / std * target_scale
    y = np.concatenate([y_rates, z], axis=1).astype(np.float32)
    norm = {"extra_mean": mean.tolist(), "extra_std": std.tolist(), "target_scale": float(target_scale),
            "n_rates": int(y_rates.shape[1]), "extra_names": list(EXTRA_NAMES)}
    return y, norm


def denormalize_extras(z: np.ndarray, norm: dict) -> np.ndarray:
    """(…, 3) network-space extras -> pA."""
    mean = np.asarray(norm["extra_mean"])
    std = np.asarray(norm["extra_std"])
    return z / norm["target_scale"] * std + mean


def save_norm(path: str, norm: dict, info: dict) -> None:
    with open(path, "w") as f:
        json.dump({"norm": norm, "label_info": info}, f, indent=2)
