from dataclasses import dataclass

@dataclass
class HistTrainConfig:
    # Data
    h5_path: str = "/home/woody/mfpb/mfpb003h/time_series_datasets/COC_SNR_1_100000_kij_10_1000000_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_grind_K24_range_10us_100ms_J24_unclamped_2026_07_01.h5"
    output_root: str = "outputs/hist_train/COC/SNR_1/K24J24"

    topology: str = "COC"

    # Dataloader
    batch_size: int = 128
    num_workers: int = 1

    # Training
    epochs: int = 200
    patience: int = 8
    min_delta: float = 0.0
    lr: float = 1e-3

    # Splits
    val_fraction: float = 0.1
    test_fraction: float = 0.1

    # Reproducibility
    seed: int = 1
    split_seed: int = 123
    splits_dir: str = "outputs/hist_train/COC/SNR_1/K24J24/splits"
