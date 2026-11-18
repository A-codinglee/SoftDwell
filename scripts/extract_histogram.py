#!/usr/bin/env python3
import h5py
import numpy as np

filename = "/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_5_100000_kij_fixed_fordemo_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_04_16.h5"

with h5py.File(filename, "r") as f:
    first_hist = f["histograms/grids"][0]        # (4, 20, 20)

for i in range(first_hist.shape[0]):
    np.savetxt(f"first_histogram_comp{i}.csv", first_hist[i], delimiter=" ")

