"""
Config for the hard-threshold pipeline on raw-trace H5 datasets.

Every field can be overridden by an environment variable named after the field
(upper-cased), e.g. LIMIT_N=10000 K=24 J=12. `num_detectors` -> env K and
`num_bins` -> env J are the two aliases (a bare "num_detectors" env would be
awkward to type). Booleans accept 1/true/yes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields, field


_ALIASES = {"num_detectors": "K", "num_bins": "J"}


@dataclass
class HardThreshConfig:
    # -------- data --------
    h5_path: str = ""                      # REQUIRED (env H5_PATH): raw-trace H5 with /traces + /labels/*
    traces_key: str = "traces"
    q_key: str = "labels/Q"
    levels_key: str = "labels/levels"
    sigma_key: str = "labels/noise_sigma"
    limit_n: int = 0                       # 0 = all rows (from row_start); >0 = that many rows
    row_start: int = 0                     # first row used; e.g. 10000 to score rows a limit_n=10000 run never saw
    t_len: int = 0                         # 0 = full trace; >0 = truncate to first t_len samples (tests only)
    fs_hz: float = 100_000.0
    counts_per_pA: float = 327.68
    sigma_units: str = "auto"              # "auto" | "counts" | "pA": unit of /labels/noise_sigma
    topology: str = "COCO"
    symmetry: str = "off"

    # -------- histogram (hard-threshold / "grind") --------
    num_detectors: int = 32                # K
    num_bins: int = 10                     # J (log-spaced dwell bins, J x J grid per detector)
    # Fixed, absolute threshold ladder in pA, linspace(lo, hi, K) like Grind.cpp.
    # Levels are per-row random (closed: baseline in [-0.5, 0.5] pA, open:
    # baseline + amplitude in [0, 2.0] pA), so the ladder must be global, not
    # per-row; the network estimates baseline/amplitude from which thresholds
    # see crossings. lo/hi are the main open design choice -- default spans all
    # possible level positions with a 0.5 pA margin each side.
    thr_lo_pA: float = -1.0
    thr_hi_pA: float = 2.5
    dwell_min_s: float = 1e-5
    dwell_max_s: float = 0.1
    log10_hist: bool = True                # log10(x+1) on histogram values (hist_pipeline convention)

    # -------- histogram build (GPU, streaming) --------
    read_rows: int = 32                    # rows read from H5 per I/O request
    hist_bs: int = 4                       # rows per GPU histogram batch
    k_group: int = 8                       # thresholds per GPU group (bounds peak memory)
    prefetch: int = 2                      # read-ahead requests
    cache: bool = True                     # cache built histograms on disk, keyed by data+spec
    cache_dir: str = "/home/woody/mfpb/mfpb003h/hardthresh_hist_cache"

    # -------- targets --------
    # 6 chain rates (log10) + baseline + amplitude + sigma. The 3 extras are
    # standardized on the train split and multiplied by target_scale (= std of a
    # log-uniform rate over 4 decades, ~1.155) so their loss magnitude is
    # comparable with the rates'.
    extra_loss_weight: float = 1.0
    target_scale: float = 1.155

    # -------- training (thesis protocol, same as hist_pipeline) --------
    batch_size: int = 128
    lr: float = 1e-3
    epochs: int = 200
    patience: int = 8
    min_delta: float = 0.0
    clip_grad_norm: float = 1.0
    seed: int = 0                          # 0 = random run seed (weights + shuffling); split seed is separate

    # -------- splits (same mechanism as softdwell_pipeline: persisted .npz) --------
    val_split: float = 0.10
    test_split: float = 0.10
    split_seed: int = 42
    # Reserved test block: rows [reserve_start, reserve_start+reserve_n) are the TEST set and are excluded from train and
    # validation. train/val are drawn (seeded) from all other loaded rows. reserve_n = 0 -> ordinary 80/10/10 file split.
    reserve_start: int = -1
    reserve_n: int = 0
    split_name: str = "default"
    splits_dir: str = "/home/woody/mfpb/mfpb003h/hardthresh_splits"

    # -------- context embedding (train_context.py): random partial labels as extra input --------
    context_p_none: float = 0.3            # share of rows with nothing revealed
    context_embed: int = 64                # width of the context embedding
    context_mode: str = "concat"           # "concat" (embedding joins the features before the head) | "film" (also scale-and-shift on the K input channels and the stem)
    context_copy_weight: float = 0.0       # weight of an extra loss on the REVEALED entries (0 = revealed slots are not trained)

    # -------- external test files (e.g. the 4 x 10k noise-source test sets) --------
    test_h5s: str = ""                    # H5 paths separated by ':' (',' breaks sbatch --export) evaluated after training
    test_limit_n: int = 0                  # rows per external test file (0 = all)

    # -------- outputs --------
    output_root: str = "/home/woody/mfpb/mfpb003h/softdwell_outputs/hardthresh/run"

    def apply_env_overrides(self) -> "HardThreshConfig":
        for f in fields(self):
            env_name = _ALIASES.get(f.name, f.name.upper())
            if env_name not in os.environ:
                continue
            raw = os.environ[env_name]
            cur = getattr(self, f.name)
            if isinstance(cur, bool):
                val = raw.lower() in ("1", "true", "yes")
            elif isinstance(cur, int):
                val = int(raw)
            elif isinstance(cur, float):
                val = float(raw)
            else:
                val = raw
            setattr(self, f.name, val)
        return self

    # ---- derived helpers ----
    @property
    def thr_lo_counts(self) -> float:
        return self.thr_lo_pA * self.counts_per_pA

    @property
    def thr_hi_counts(self) -> float:
        return self.thr_hi_pA * self.counts_per_pA
