from dataclasses import dataclass


@dataclass
class TrainConfig:
    hists_path: str = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/data/hists.npy"
    labels_path: str = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/data/labels_fixed.npy"
    batch_size: int = 128
    epochs: int = 200
    patience: int = 12
    min_delta: float = 0.0
    num_outputs: int = 8
    use_bn: bool = False
    separate_heads: bool = False
    amp: bool = True
    lr: float = 3e-3
    weight_decay: float = 1e-2
    num_workers: int = 0
    prefetch_factor: int = 1