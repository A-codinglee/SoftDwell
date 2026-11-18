#!/usr/bin/env python3
"""
Split a long-form CSV (epoch,k,theta,tau) into per-k CSV files WITHOUT pandas.

Outputs (by default, with a single-column header and NO epoch column, NO row index):
  theta_k0_values_only.csv
  tau_k0_values_only.csv
  theta_k1_values_only.csv
  tau_k1_values_only.csv
  ...

Usage:
  python split_params_by_k_no_pandas.py input.csv --outdir out

Optional:
  --no-header   # write values only (no header row)
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", type=Path)
    ap.add_argument("--outdir", type=Path, default=Path("."))
    ap.add_argument("--no-header", action="store_true")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    # Store as: theta_map[k][epoch] = theta_value; tau_map[k][epoch] = tau_value
    theta_map: Dict[int, Dict[int, str]] = {}
    tau_map: Dict[int, Dict[int, str]] = {}
    epochs: Set[int] = set()
    ks: Set[int] = set()

    with args.input_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"epoch", "k", "theta", "tau"}
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        for row in reader:
            epoch = int(row["epoch"])
            k = int(row["k"])
            theta = row["theta"]
            tau = row["tau"]

            epochs.add(epoch)
            ks.add(k)

            theta_map.setdefault(k, {})[epoch] = theta
            tau_map.setdefault(k, {})[epoch] = tau

    epoch_list: List[int] = sorted(epochs)
    k_list: List[int] = sorted(ks)

    def write_series(filename: Path, colname: str, values_by_epoch: Dict[int, str]) -> None:
        with filename.open("w", newline="") as out_f:
            w = csv.writer(out_f)
            if not args.no_header:
                w.writerow([colname])

            # Write in increasing epoch order; if a (k,epoch) is missing, write blank
            for e in epoch_list:
                w.writerow([values_by_epoch.get(e, "")])

    for k in k_list:
        write_series(
            args.outdir / f"theta_k{k}_values_only.csv",
            f"theta_k{k}",
            theta_map.get(k, {}),
        )
        write_series(
            args.outdir / f"tau_k{k}_values_only.csv",
            f"tau_k{k}",
            tau_map.get(k, {}),
        )

    print(f"Wrote {2 * len(k_list)} files to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()
