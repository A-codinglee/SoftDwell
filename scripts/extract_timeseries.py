#!/usr/bin/env python3
import h5py
import numpy as np

filename = "/home/woody/mfpb/mfpb003h/time_series_datasets/COC_SNR_1_1000_kij_fixed_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_06_30.h5"

with h5py.File(filename, "r") as f:
    first_ts = f["traces"][0]   # first trace only
    np.savetxt("first_trace_demo.csv", first_ts, delimiter=",")

print("Saved first_trace.csv")
print("Shape:", first_ts.shape)