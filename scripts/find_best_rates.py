import numpy as np

pred = np.load("/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_3/predictions_nn.npy")   # shape (N, 6), linear domain
targ = np.load("/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_3/targets_nn.npy")       # shape (N, 6), linear domain

# convert back to log domain safely
eps = 1e-30
pred_log = np.log10(np.maximum(pred, eps))
targ_log = np.log10(np.maximum(targ, eps))

# same formula as your rae_log:
# sqrt(abs(pred_log - targ_log) + 1e-12)
rae_matrix = np.sqrt(np.abs(pred_log - targ_log) + 1e-12)   # shape (N, 6)

# one score per sample = mean over the 6 outputs
sample_rae = rae_matrix.mean(axis=1)

best_idx = np.argmin(sample_rae)

print("best sample index:", best_idx)
print("best sample mean RAE:", sample_rae[best_idx])

rate_names = ["k01", "k10", "k12", "k21", "k23", "k32"]

print("\nPredicted vs target rates for best sample:")
for name, p, t, r in zip(rate_names, pred[best_idx], targ[best_idx], rae_matrix[best_idx]):
    print(f"{name}: pred={p:.6g}, target={t:.6g}, rae={r:.6f}")
print(" ".join(f"{name}" for name in rate_names))
print(" ".join(f"{t:.6g}" for t in targ[best_idx]))