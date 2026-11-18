#!/usr/bin/env python3
import h5py
import numpy as np

filename = "/home/woody/mfpb/mfpb003h/time_series_datasets/COCO_SNR_5_100000_kij_fixed_bestmodelof_softdwell_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_04_15.h5"

with h5py.File(filename, "r") as f:
    Q = f["labels/Q"][:]   # shape: (N, 4, 4)

# off-diagonal mask
offdiag = ~np.eye(Q.shape[1], dtype=bool)

# find entries that are nonzero in at least one sample
mask = np.any(Q != 0, axis=0) & offdiag

positions = np.argwhere(mask)

if len(positions) != 6:
    raise ValueError(
        f"Expected 6 relevant Q entries, but found {len(positions)}.\n"
        f"Positions found: {positions.tolist()}"
    )

# extract those 6 entries per sample
Q6 = Q[:, mask]   # shape: (N, 6)

# headers like Q_0_1, Q_1_0, ...
header = ",".join([f"Q_{i}_{j}" for i, j in positions])

np.savetxt("labels_Q_6.csv", Q6, delimiter=",", header=header, comments="")

print("Saved labels_Q_6.csv")
print("Shape:", Q6.shape)
print("Positions used:", positions.tolist())