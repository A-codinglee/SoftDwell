#!/usr/bin/env python3
import numpy as np
import csv

# --- settings ---
N_SAMPLES = 10000      
NUM_OUTPUTS = 6      
LOW, HIGH = 2.0, 6.0  

# --- generate random pred_log
preds_log = np.random.uniform(LOW, HIGH, size=(N_SAMPLES, NUM_OUTPUTS))
targs_log = np.random.uniform(LOW, HIGH, size=(N_SAMPLES, NUM_OUTPUTS))

# --- RAE in log space (same as your rae_log) ---
# rae = sqrt(|pred_log - targ_log| + 1e-12)
rae_matrix = np.sqrt(np.abs(preds_log - targs_log) + 1e-12)  # shape (N, 8)

# mean RAE per sample → shape (N,)
rae_mean = rae_matrix.mean(axis=1)

# --- save ONLY rae_mean to CSV (one column, no id) ---
csv_path = "rae_rnd_coco_2_6_3.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["rae_mean"])  # header
    for v in rae_mean:
        writer.writerow([float(v)])

print(f"[done] saved random RAE distribution CSV → {csv_path}")
print("rae_mean shape:", rae_mean.shape)
