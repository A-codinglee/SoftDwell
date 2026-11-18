# make_thesis_bottleneck_plots.py
# Matplotlib-only code to reproduce the 3 plots we discussed:
#  1) Per-iteration stacked breakdown (Disk vs RAM)
#  2) Epoch time bars (Disk epoch, RAM epoch, RAM preload+epoch)
#  3) Amortization curve: average time/epoch vs number of epochs (RAM includes preload)
#
# Outputs: PNG + PDF for each plot.

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10})

# -----------------------------
# 1) Paste your measured numbers here
# -----------------------------
disk = {
    "label": "Disk, 1 worker (steady-state)",
    "epoch_time_s": 580.23,
    "avg_load_s": 14.907119,
    "avg_cpu2gpu_s": 0.021915,
    "avg_fwd_s": 0.412229,
    "avg_bwd_s": 1.041455,
    "avg_step_s": 0.020550,
}

ram = {
    "label": "RAM, 1 worker (steady-state)",
    "epoch_time_s": 53.63,
    "avg_load_s": 0.146694,
    "avg_cpu2gpu_s": 0.083893,
    "avg_fwd_s": 0.426645,
    "avg_bwd_s": 0.961727,
    "avg_step_s": 0.020820,
}

# One-time cost (seconds) to preload/copy dataset into RAM
ram_preload_s = 200.0

COMPONENTS = [
    ("avg_load_s", "load"),
    ("avg_cpu2gpu_s", "cpu→gpu"),
    ("avg_fwd_s", "forward"),
    ("avg_bwd_s", "backward"),
    ("avg_step_s", "optimizer step"),
]

def iter_total(run):
    return sum(run[k] for k, _ in COMPONENTS)

def annotate_bars(xs, heights, fmt="{:.1f}"):
    m = max(heights) if heights else 0.0
    dy = 0.01 * m
    for x, h in zip(xs, heights):
        plt.text(x, h + dy, fmt.format(h), ha="center", va="bottom")

# -----------------------------
# 2) Plot 1: Per-iteration stacked breakdown (Disk vs RAM)
# -----------------------------
runs = [disk, ram]
x = np.arange(len(runs))
labels = [r["label"] for r in runs]

fig = plt.figure(figsize=(9.2, 4.9))
bottom = np.zeros(len(runs))
for k, lab in COMPONENTS:
    vals = np.array([r[k] for r in runs], dtype=float)
    plt.bar(x, vals, bottom=bottom, label=lab)
    bottom += vals

plt.xticks(x, labels, rotation=0)
plt.ylabel("Average time per iteration (s)")
plt.title("Per-iteration timing breakdown (disk vs RAM)")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("iter_breakdown_disk_vs_ram.png", dpi=300)
plt.savefig("iter_breakdown_disk_vs_ram.pdf")
plt.close(fig)

# -----------------------------
# 3) Plot 2: Epoch time bars (include RAM preload)
# -----------------------------
epoch_labels = [
    "Disk (epoch)",
    "RAM (epoch)",
    "RAM (preload + epoch)",
]
epoch_vals = [
    disk["epoch_time_s"],
    ram["epoch_time_s"],
    ram_preload_s + ram["epoch_time_s"],
]

fig = plt.figure(figsize=(9.2, 4.9))
xx = np.arange(len(epoch_vals))
plt.bar(xx, epoch_vals)
plt.xticks(xx, epoch_labels, rotation=0)
plt.ylabel("Time (s)")
plt.title("End-to-end time comparison (steady-state vs first RAM epoch)")
annotate_bars(xx, epoch_vals, fmt="{:.1f}")
plt.tight_layout()
plt.savefig("epoch_time_with_preload.png", dpi=300)
plt.savefig("epoch_time_with_preload.pdf")
plt.close(fig)

# -----------------------------
# 4) Plot 3: Amortization curve (avg time/epoch vs number of epochs)
# -----------------------------
E = np.arange(1, 101)  # epochs 1..100
disk_avg = (E * disk["epoch_time_s"]) / E
ram_avg = (ram_preload_s + E * ram["epoch_time_s"]) / E

fig = plt.figure(figsize=(9.2, 4.9))
plt.plot(E, disk_avg, label="Disk: avg time/epoch")
plt.plot(E, ram_avg, label="RAM: avg time/epoch (incl. preload)")
plt.xlabel("Number of epochs (E)")
plt.ylabel("Average time per epoch (s)")
plt.title("Amortization of RAM preload cost")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("amortization_avg_epoch_time.png", dpi=300)
plt.savefig("amortization_avg_epoch_time.pdf")
plt.close(fig)

# -----------------------------
# 5) Print a tiny summary (optional)
# -----------------------------
disk_iter = iter_total(disk)
ram_iter = iter_total(ram)
disk_load_share = 100.0 * disk["avg_load_s"] / disk_iter
ram_load_share = 100.0 * ram["avg_load_s"] / ram_iter

print("=== Summary ===")
print(f"Disk iter total: {disk_iter:.6f}s | load share: {disk_load_share:.1f}%")
print(f"RAM  iter total: {ram_iter:.6f}s | load share: {ram_load_share:.1f}%")
print("Wrote:")
print(" - iter_breakdown_disk_vs_ram.(png/pdf)")
print(" - epoch_time_with_preload.(png/pdf)")
print(" - amortization_avg_epoch_time.(png/pdf)")
