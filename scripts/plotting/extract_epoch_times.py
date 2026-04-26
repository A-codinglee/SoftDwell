#!/usr/bin/env python3
import re
import csv
import sys
import os

logfile = sys.argv[1]
outfile = sys.argv[2]

epoch_re = re.compile(r"^Epoch\s+(\d+)\s+\(([\d.]+)s\):")

rows = []

with open(logfile, "r", errors="replace") as f:
    for line in f:
        m = epoch_re.match(line.strip())
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "epoch_time_s": float(m.group(2)),
            })

file_exists = os.path.exists(outfile)
write_header = (not file_exists) or os.path.getsize(outfile) == 0

with open(outfile, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["epoch", "epoch_time_s"])
    if write_header:
        writer.writeheader()
    writer.writerows(rows)

print(f"appended {len(rows)} rows to {outfile}")