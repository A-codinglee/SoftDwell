#!/usr/bin/env python3
# scripts/hist_pipeline/bench_summary.py
"""
Summary of one run of the fixed pipeline benchmark (bench_pipeline.sh), and comparison with a reference.

  summarize:  python3 -m scripts.hist_pipeline.bench_summary <bench dir>
              -> <bench dir>/summary.json, printed; compared with --reference if given
  reference:  python3 -m scripts.hist_pipeline.bench_summary --make-reference <bench dir> <bench dir> ... \
                  --out <reference.json>
              -> mean and spread over repeated runs of the reference setup

What is measured (epochs 1-2 are excluded from the timings: start-up, first compilation):
  speed:   seconds per epoch (training part, validation part), trainer waiting for data, simulators idle,
           GPU utilization per GPU
  quality: validation rate error (mean of the rates' RAE) at fixed epochs, and at fixed training time
           (interpolated at the reference's time for those epochs: fair when steps per epoch differ),
           test rate error of the best model at the end
"""
import argparse
import csv
import glob
import json
import os
import re

import numpy as np

EPOCH_MARKS = (5, 10, 20, 30)


def summarize(bench_dir):
    run = os.path.join(bench_dir, "seed1")
    rows = list(csv.DictReader(open(os.path.join(run, "train_val_metrics.csv"))))
    t = np.array([float(r["epoch_time_s"]) for r in rows])
    rae = np.array([np.mean([float(r[f"val_rae_out_{i}"]) for i in range(1, 7)]) for r in rows])
    cum = np.cumsum(t)
    s = {"dir": bench_dir, "epochs": len(rows), "epoch_s": float(t[2:].mean()), "total_train_s": float(cum[-1]),
         "val_rae": {str(e): float(rae[e - 1]) for e in EPOCH_MARKS if e <= len(rae)},
         "val_rae_best": float(rae.min()), "time_s": [float(x) for x in cum], "rae_curve": [float(x) for x in rae]}
    log = sorted(glob.glob(os.path.join(run, "slurm-*.out")))
    if log:
        txt = open(log[-1]).read()
        parts = re.findall(r"\(train ([\d.]+)s, val ([\d.]+)s\)", txt)[2:]
        if parts:
            s["train_s"] = float(np.mean([float(a) for a, _ in parts]))
            s["val_s"] = float(np.mean([float(b) for _, b in parts]))
        w = [float(x) for x in re.findall(r"waited ([\d.]+)s for data", txt)[2:]]
        idle = [float(x) for x in re.findall(r"simulators idle (\d+)%", txt)[2:]]
        s["waited_s"] = float(np.mean(w)) if w else None
        s["sim_idle_pct"] = float(np.mean(idle)) if idle else None
        m = re.search(r"\[Test\] RAE on [^\n]*?: ([\d.]+)", txt)
        s["test_rae"] = float(m.group(1)) if m else None
        s["options"] = (re.search(r"BENCH OPTIONS: (.*)", txt) or [None, ""])[1]
    util = glob.glob(os.path.join(run, "gpu-util-*.csv"))
    if util:
        per = {}
        for line in open(util[0]):
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 3 and f[2].split()[0].isdigit():
                per.setdefault(f[1], []).append(float(f[2].split()[0]))
        s["gpu_util_pct"] = {g: float(np.mean(v[int(0.3 * len(v)):])) for g, v in sorted(per.items())}
    top = glob.glob(os.path.join(run, "cpu-top-*.txt"))
    if top:
        s.update(cpu_use(top[0]))
    return s


def cpu_use(path):
    """
    Cores in use by the run's processes (from top every 10 s, the first 30% of snapshots skipped as start-up):
    all of them, the main trainer (the process running train_selfcal_online, all its threads) and the rest
    (simulators and, with --train-gpus, the other trainers). top drops the trainer from most snapshots once
    it is busy, so its share comes from its accumulated CPU time (TIME+) between the first and last snapshot
    that list it, not from %CPU.
    """
    def secs(t):
        p = t.split(":")
        return sum(float(x) * 60 ** i for i, x in enumerate(reversed(p)))

    snaps, cur, now, pid, seen = [], None, 0.0, None, []
    for line in open(path):
        if line.startswith("top -"):
            now = secs(line.split()[2])
            cur = {"others": 0.0}
            snaps.append(cur)
            continue
        f = line.split(None, 11)
        if cur is None or len(f) < 12 or not f[0].isdigit():
            continue
        cmd, pct = f[11], float(f[8])
        if pid is None and "train_selfcal_online" in cmd:
            pid = f[0]
        if f[0] == pid:
            seen.append((now, secs(f[10])))
            continue
        if cmd.startswith("top ") or "nvidia-smi" in cmd:
            continue
        cur["others"] += pct / 100
    snaps = snaps[int(0.3 * len(snaps)):]
    if not snaps:
        return {}
    others = float(np.mean([x["others"] for x in snaps]))
    trainer = float("nan")
    if len(seen) >= 2 and seen[-1][0] - seen[0][0] > 300:
        trainer = (seen[-1][1] - seen[0][1]) / (seen[-1][0] - seen[0][0])
    return {"cpu_cores_used": others + trainer, "cpu_cores_trainer": trainer, "cpu_cores_others": others}


def val_rae_at_time(s, t_s):
    """Validation rate error reached after t_s seconds of training (linear between epochs)."""
    t, r = np.array(s["time_s"]), np.array(s["rae_curve"])
    return float(np.interp(t_s, t, r)) if t_s <= t[-1] else None


def compare(s, ref):
    def line(name, v, m, sd=None, fmt="{:.3f}", lower_better=True):
        if v is None or m is None:
            return
        d = v - m
        tag = ""
        if sd:
            tag = "  within reference spread" if abs(d) <= 2 * sd else ("  BETTER" if (d < 0) == lower_better
                                                                         else "  WORSE")
        rel = f" ({100 * d / m:+.1f}%)" if m else ""
        print(f"  {name:38s} {fmt.format(v):>9s}   reference {fmt.format(m):>9s}" +
              (f" ± {fmt.format(sd)}" if sd else "") + rel + tag)

    print("speed")
    line("seconds per epoch", s["epoch_s"], ref["epoch_s"]["mean"], ref["epoch_s"]["sd"], "{:.1f}")
    line("  training part (s)", s.get("train_s"), ref.get("train_s", {}).get("mean"), None, "{:.1f}")
    line("  validation part (s)", s.get("val_s"), ref.get("val_s", {}).get("mean"), None, "{:.1f}")
    line("trainer waiting for data (s/epoch)", s.get("waited_s"), ref.get("waited_s", {}).get("mean"), None, "{:.2f}")
    line("simulators idle (%)", s.get("sim_idle_pct"), ref.get("sim_idle_pct", {}).get("mean"), None, "{:.0f}",
         lower_better=False)
    for k, name in (("cpu_cores_used", "CPU cores in use, all"), ("cpu_cores_trainer", "  main trainer process"),
                    ("cpu_cores_others", "  simulators (+ other trainers)")):
        if s.get(k) is not None:
            print(f"  {name:38s} {s[k]:9.1f}")
    print("quality (rate RAE, lower is better)")
    for e in EPOCH_MARKS:
        r = ref["val_rae"].get(str(e))
        if r and str(e) in s["val_rae"]:
            line(f"validation after {e} epochs", s["val_rae"][str(e)], r["mean"], r["sd"])
    for e in EPOCH_MARKS:
        r = ref["val_rae"].get(str(e))
        tt = ref.get("time_at_epoch_s", {}).get(str(e))
        if r and tt:
            v = val_rae_at_time(s, tt)
            if v is not None:
                line(f"validation after {tt / 60:.1f} min of training", v, r["mean"], r["sd"])
    line("test after the run (best model)", s.get("test_rae"), ref["test_rae"]["mean"], ref["test_rae"]["sd"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dirs", nargs="+")
    p.add_argument("--reference", default=None, help="reference.json to compare with")
    p.add_argument("--make-reference", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    sums = [summarize(d) for d in a.dirs]
    if a.make_reference:
        def ms(vals):
            vals = [v for v in vals if v is not None]
            return {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "n": len(vals)}
        ref = {"runs": [s["dir"] for s in sums], "options": sums[0].get("options", ""),
               "epoch_s": ms([s["epoch_s"] for s in sums]), "train_s": ms([s.get("train_s") for s in sums]),
               "val_s": ms([s.get("val_s") for s in sums]), "waited_s": ms([s.get("waited_s") for s in sums]),
               "sim_idle_pct": ms([s.get("sim_idle_pct") for s in sums]),
               "test_rae": ms([s.get("test_rae") for s in sums]),
               "val_rae": {str(e): ms([s["val_rae"].get(str(e)) for s in sums]) for e in EPOCH_MARKS
                           if all(str(e) in s["val_rae"] for s in sums)},
               "time_at_epoch_s": {str(e): float(np.mean([s["time_s"][e - 1] for s in sums])) for e in EPOCH_MARKS
                                   if all(len(s["time_s"]) >= e for s in sums)}}
        with open(a.out, "w") as f:
            json.dump(ref, f, indent=2)
        print(json.dumps({k: v for k, v in ref.items() if k not in ("runs",)}, indent=1))
        return
    for s in sums:
        with open(os.path.join(s["dir"], "summary.json"), "w") as f:
            json.dump(s, f, indent=2)
        print(f"== {s['dir']}\n   options: {s.get('options', '')}")
        if a.reference and os.path.isfile(a.reference):
            compare(s, json.load(open(a.reference)))
        else:
            print(json.dumps({k: v for k, v in s.items() if k not in ("time_s", "rae_curve")}, indent=1))


if __name__ == "__main__":
    main()
