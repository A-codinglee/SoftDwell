import csv
import statistics
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Put your CSV files here
# Each config has 3 runs
# ------------------------------------------------------------------
configs = {
    "J=24": [
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_1/logs/epoch_times.csv",
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_2/logs/epoch_times.csv",
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_3/logs/epoch_times.csv",
    ],
    "J=31": [
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_31Res_1/logs/epoch_times.csv",
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_31Res_2/logs/epoch_times.csv",
        "/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_31Res_3/logs/epoch_times.csv",
    ],
}

ORANGE = "#ff7f0e"
BLUE   = "#1f77b4"
BLACK  = "black"
BG     = "#eaeaea"

def mean_epoch_time(csv_file):
    vals = []
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals.append(float(row["epoch_time_s"]))
    return statistics.mean(vals)

run_means = {}
for cfg, files in configs.items():
    run_means[cfg] = [mean_epoch_time(f) for f in files]

cfg_names = list(run_means.keys())
means = [statistics.mean(run_means[cfg]) for cfg in cfg_names]
stds = [statistics.stdev(run_means[cfg]) for cfg in cfg_names]

# ------------------------------------------------------------------
# Print run means and overall summary
# ------------------------------------------------------------------
for cfg in cfg_names:
    print(
        f"{cfg} run means = {[round(x, 2) for x in run_means[cfg]]} "
        f"overall mean = {statistics.mean(run_means[cfg]):.2f} "
        f"std = {statistics.stdev(run_means[cfg]):.2f}"
    )

if len(cfg_names) == 2:
    cfg_a, cfg_b = cfg_names
    mean_a = statistics.mean(run_means[cfg_a])
    mean_b = statistics.mean(run_means[cfg_b])

    # assumes cfg_a is the cheaper/faster one if its mean is smaller
    if mean_a < mean_b:
        faster_cfg, slower_cfg = cfg_a, cfg_b
        faster_mean, slower_mean = mean_a, mean_b
    else:
        faster_cfg, slower_cfg = cfg_b, cfg_a
        faster_mean, slower_mean = mean_b, mean_a

    abs_diff = slower_mean - faster_mean
    rel_reduction = abs_diff / slower_mean * 100.0
    speedup = slower_mean / faster_mean

    print()
    print(
        f"{faster_cfg} is faster than {slower_cfg} by {abs_diff:.2f} s/epoch "
        f"({rel_reduction:.1f}% less time, speedup {speedup:.2f}x)"
    )

fig, ax = plt.subplots(figsize=(9, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

ax.grid(axis="y", color="gray", alpha=0.4, linewidth=1)

for i, cfg in enumerate(cfg_names):
    yvals = run_means[cfg]
    offsets = [-0.06, 0.0, 0.06]

    # orange run points
    for j, y in enumerate(yvals):
        ax.plot(i + offsets[j], y, "o", color=ORANGE, markersize=10)

    # black std error bar + caps
    ax.errorbar(
        i,
        means[i],
        yerr=stds[i],
        fmt="none",
        ecolor=BLACK,
        elinewidth=3,
        capsize=10,
        capthick=3,
        zorder=2,
    )

    # blue mean bar
    ax.plot(
        [i - 0.12, i + 0.12],
        [means[i], means[i]],
        color=BLUE,
        linewidth=4,
        solid_capstyle="round",
        zorder=3,
    )

ax.set_xticks(range(len(cfg_names)))
ax.set_xticklabels(cfg_names, fontsize=20)
ax.set_ylabel("Epoch time [s]", fontsize=28)
ax.set_xlabel("J", fontsize=28)
ax.set_title("SoftDwell Epoch Time Comparison (K=4)", fontsize=30, pad=16)

ax.tick_params(axis="y", labelsize=18, width=1.5, length=8)
ax.tick_params(axis="x", width=1.5, length=8)

plt.tight_layout()
plt.savefig("softdwell_j_time_comparison.png", dpi=300, facecolor=fig.get_facecolor())
plt.show()