#!/usr/bin/env python3
import numpy as np

f1 = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/glorot/4Det_11Res/rae_mean.csv"
f2 = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/glorot/4Det_11Res_2/rae_mean.csv"
# f3 = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train/COCO/SNR_5_100000_kij_100_1000000_samples_1000000_lvl_20000_22000/D32_glorot/rae_mean_3rd.csv"

out = "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/glorot/rae_mean_avg.csv"

a = np.loadtxt(f1, delimiter=",", skiprows=1)  # (n,)
b = np.loadtxt(f2, delimiter=",", skiprows=1)
# c = np.loadtxt(f3, delimiter=",", skiprows=1)

# n = min(a.size, b.size, c.size)  # only average rows present in all three
n = min(a.size, b.size)  # only average rows present in all three

# avg = (a[:n] + b[:n] + c[:n]) / 3.0
avg = (a[:n] + b[:n]) / 2.0

# write with header "rae_mean"
np.savetxt(out, avg, delimiter=",", header="rae_mean", comments="", fmt="%.8f")
print(f"[ok] wrote {out} with {n} rows")
