#!/usr/bin/env python3
import re
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <logfile> <outfile.png>")
    sys.exit(1)

logfile = sys.argv[1]
outfile = sys.argv[2]

# Example line:
# [Epoch 001] 19.28s | Train: 0.470613 | Val: 0.350177 | Val RAE per out: ...
pattern = re.compile(
    r"\[Epoch\s+(\d+)\]\s+[\d.]+s\s+\|\s+Train:\s+([\d.]+)\s+\|\s+Val:\s+([\d.]+)"
)

epochs = []
train_losses = []
val_losses = []

with open(logfile, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        m = pattern.search(line)
        if m:
            epochs.append(int(m.group(1)))
            train_losses.append(float(m.group(2)))
            val_losses.append(float(m.group(3)))

if not epochs:
    print("No epoch lines found in log file.")
    sys.exit(1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Train loss")
plt.plot(epochs, val_losses, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outfile, dpi=300)
print(f"Saved plot to {outfile}")