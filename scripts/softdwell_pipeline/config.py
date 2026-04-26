from dataclasses import dataclass
import os

@dataclass
class TrainConfig:
    # -------- outputs / paths --------
    output_root: str = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs"
    run_name: str = ""
    make_run_subdir: bool = True
    checkpoints_dirname: str = "checkpoints"
    best_model_filename: str = "best_model.pt"
    predictions_filename: str = "predictions_nn.npy"
    targets_filename: str = "targets_nn.npy"
    metrics_csv: str = "metrics.csv"

    # Will be derived in __post_init__
    ckpt_dir: str = None
    splits_dir: str = None

    # -------- data (HDF5) --------
    h5_path: str = "/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_1p6_3p2_100000_kij_100_1000000_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_04_14.h5"

    x_key: str = "traces"
    # x_key: str = "timeseries"
    y_key: str = "labels/Q"
    # y_key: str = "labels"
    y_log10: bool = True
    val_split: float = 0.10
    test_split: float = 0.10
    limit_n: int =  0
    y_from_Q: bool = True
    t_start: int = 0
    t_len: int = None

    # Fixed split controls (for reproducible membership)
    split_seed: int = 42
    split_name: str = "default"    

    topology: str = "COCO"
    symmetry: str = "off"   # "on", "off", or "auto

    # -------- HOHD (incremental-ΔW exact) --------
    use_soft_dwell: bool = True
    num_detectors: int = 1
    sampling_frequency: float = 100_000
    dwell_min: int = 1
    dwell_max: int = 10_000
    num_bins: int = 4
    theta_init_min: float = 0.1
    theta_init_max: float = 0.9
    tau_init: float = 0.1
    norm_min: float = 18000.0
    norm_max: float = 24000.0
    clamp_norm: bool = True
    logN_gamma: float = 0.0

    # -------- model/training --------
    num_outputs: int = 6
    use_bn: bool = False
    amp: bool = True
    lr: float = 1e-3 # from 5e-4
    lr_sd: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 256
    epochs: int = 1000
    patience: int = 8
    min_delta: float = 0.0
    num_workers: int = 0
    prefetch_factor: int = 1
    clip_grad_norm: float = 1.0
    compile_model: bool = False

    # -------- checkpointing / resume --------
    resume: str = "auto"             # "", path, or "auto"

    def __post_init__(self):
        # Derive directories that depend on output_root
        if self.ckpt_dir is None:
            self.ckpt_dir = os.path.join(self.output_root, self.checkpoints_dirname)
        if self.splits_dir is None:
            self.splits_dir = os.path.join(self.output_root, "splits")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

    def apply_env_overrides(self):
        # paths / dirs
        self.output_root = os.environ.get("OUTPUT_ROOT", self.output_root)
        self.checkpoints_dirname = os.environ.get("CHECKPOINTS_DIRNAME", self.checkpoints_dirname)
        # recompute derived dirs if OUTPUT_ROOT changed
        self.ckpt_dir = os.environ.get("CKPT_DIR", os.path.join(self.output_root, self.checkpoints_dirname))
        self.splits_dir = os.environ.get("SPLITS_DIR", os.path.join(self.output_root, "splits"))
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.splits_dir, exist_ok=True)

        # resume / scheduling
        self.resume = os.environ.get("RESUME", self.resume)

        # split controls (optional overrides)
        self.split_name = os.environ.get("SPLIT_NAME", self.split_name)
        self.split_seed = int(os.environ.get("SPLIT_SEED", self.split_seed))
        return self