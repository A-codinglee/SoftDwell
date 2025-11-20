import re
import numpy as np
import matplotlib.pyplot as plt

# ------------------------
# Load and parse epoch times
# ------------------------
log_path = "/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/freeze_softdwell/logs/slurm-3138161.out"

epoch_nums = []
epoch_times = []
phases = []  # "full" or "head"

# Match:  Epoch 7 (5842.74s):
pattern = r"Epoch\s+(\d+)\s+\(([\d\.]+)s\)"

with open(log_path, "r") as f:
    for line in f:
        m = re.search(pattern, line)
        if m:
            e = int(m.group(1))
            t = float(m.group(2))

            epoch_nums.append(e)
            epoch_times.append(t)

            # full-phase SoftDwell epochs are much slower (~3000–9000s)
            if t > 2000:   # threshold to separate full vs head
                phases.append("full")
            else:
                phases.append("head")

# ------------------------
# 수동으로 1~7 epoch 추가 (로그에 없던 구간)
# ------------------------
manual_epochs = {
    1: 6755.39,
    2: 5827.51,
    3: 5837.72,
    4: 5833.83,
    5: 5825.30,
    6: 5828.83,
    7: 5812.95,
}

for e, t in manual_epochs.items():
    if e not in epoch_nums:      # 이미 있으면 안 겹치게 체크
        epoch_nums.append(e)
        epoch_times.append(t)
        phases.append("full")    # 1~7은 full phase라서 full로 표시

# numpy 배열로 변환 + epoch 기준 정렬
epoch_nums = np.array(epoch_nums)
epoch_times = np.array(epoch_times)
order = np.argsort(epoch_nums)
epoch_nums = epoch_nums[order]
epoch_times = epoch_times[order]
phases = [phases[i] for i in order]

# ------------------------
# Line plot with colored markers
# ------------------------

colors = ["red" if p == "full" else "blue" for p in phases]

plt.figure(figsize=(10,5))
plt.scatter(epoch_nums, epoch_times, c=colors, s=40)
plt.plot(epoch_nums, epoch_times, alpha=0.3)

plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Time per epoch (seconds)", fontsize=12)
plt.title("Epoch Time Comparison: Full Model vs Head-only (Frozen SoftDwell)", fontsize=14)

# legend
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Full (SoftDwell active)')
blue_patch = mpatches.Patch(color='blue', label='Head-only (cache)')
plt.legend(handles=[red_patch, blue_patch])

plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("epoch_time_freeze_vs_head.png", dpi=300, bbox_inches='tight')
plt.show()
