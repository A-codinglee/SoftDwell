import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator

pred = np.load(
    "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res/test/predictions_nn.npy"
)

target = np.load(
    "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res/test/targets_nn.npy"
)

print("prediction shape:", pred.shape)
print("target shape:", target.shape)

# Shape is (1000, 4), assumed order:
# [k12, k21, k23, k32]
rate_names = ["k12", "k21", "k23", "k32"]

pred_rates = pred[:1000]
target_rates = target[:1000]

ground_truth = target_rates[0]

print("\nGround truth from targets_nn.npy:")
for name, value in zip(rate_names, ground_truth):
    print(f"{name}: {value:.6g}")

print("\nTarget variation check:")
for i, name in enumerate(rate_names):
    print(
        f"{name}: min={target_rates[:, i].min():.6g}, "
        f"max={target_rates[:, i].max():.6g}"
    )

x = np.arange(len(rate_names))

plt.figure(figsize=(7, 4.5))

# Prediction dots
rng = np.random.default_rng(0)

for j in range(len(rate_names)):
    jitter = rng.normal(loc=0.0, scale=0.06, size=pred_rates.shape[0])

    plt.scatter(
        np.full(pred_rates.shape[0], x[j]) + jitter,
        pred_rates[:, j],
        s=10,
        alpha=0.20,
        label="Predictions" if j == 0 else None,
        color="C0",
        zorder=2
    )

# Normal gray boxplot
bp = plt.boxplot(
    [pred_rates[:, j] for j in range(len(rate_names))],
    positions=x,
    widths=0.42,
    showfliers=False,
    whis=(5, 95),          # whiskers = 5th and 95th percentiles
    patch_artist=True,
    boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.3),
    medianprops=dict(color="black", linestyle="--", linewidth=1.3),
    whiskerprops=dict(color="black", linewidth=1.3),
    capprops=dict(color="black", linewidth=1.3),
    zorder=3
)

bp["boxes"][0].set_label("Prediction distribution")

# Ground truth as short black horizontal lines
plt.hlines(
    y=ground_truth,
    xmin=x - 0.28,
    xmax=x + 0.28,
    linewidth=2,
    color="C1",
    linestyle="-",
    label="Ground truth",
    zorder=4
)

plt.xticks(x, rate_names, fontsize=13)
plt.ylabel("Transition rate (log scale)", fontsize=13)
plt.title("SoftDwell at SNR=1", fontsize=15)

plt.yscale("log")

ax = plt.gca()
ax.yaxis.set_major_locator(LogLocator(base=10.0))

# Grid only at 10^2, 10^3, 10^4, ...
plt.grid(True, axis="y", which="major", linestyle="-", alpha=0.7)
plt.grid(False, axis="y", which="minor")

plt.legend()
plt.tight_layout()
plt.savefig("/home/hpc/mfpb/mfpb102h/master/SoftDwell/test_graphs/transition_rate_predictions_boxplot_logscale.png", dpi=300)
plt.show()