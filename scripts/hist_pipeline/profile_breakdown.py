#!/usr/bin/env python3
# scripts/hist_pipeline/profile_breakdown.py
"""
GPU time of each stage of the profiled training steps, per GPU and per kind of GPU operation, from the
profile_trace.json that train_selfcal_online.py --profile-epoch writes (the stages are its record_function
spans). Next to each stage's wall time this shows how much of it the GPUs actually computed; the rest is
the CPU queuing work, Python, or waiting.

A GPU operation belongs to the stage during which the CPU launched it. Work launched by the GPU noise
threads (the threads that run the noise FFT) is shown as its own line, as it runs alongside the stages.

Usage:
  python3 -m scripts.hist_pipeline.profile_breakdown <run>/seed1/profile_trace.json
"""
import json
import sys
from collections import defaultdict

KINDS = [  # (kind, substrings of the GPU operation's name), first match wins
    ("find crossings (nonzero)", ("DeviceSelect", "DeviceCompact", "nonzero")),
    ("count into bins (scatter_add)", ("scatter",)),
    ("copy CPU->GPU", ("Memcpy HtoD",)),
    ("copy GPU->CPU", ("Memcpy DtoH",)),
    ("copy GPU->GPU / memset", ("Memcpy DtoD", "Memset")),
    ("noise FFT", ("fft", "FFT")),
    ("gather (indexing)", ("index_elementwise", "gather")),
    ("convolution / matmul", ("conv", "cudnn", "gemm", "cutlass", "sm80_", "sm90_", "ampere_", "wgrad", "dgrad")),
    ("gradient all-reduce (NCCL)", ("nccl",)),
    ("reductions", ("Reduce", "reduce_kernel")),
]


def kind(name):
    for k, keys in KINDS:
        if any(s in name for s in keys):
            return k
    return "elementwise / other"


def main(path):
    ev = json.load(open(path))["traceEvents"]
    spans = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in ev if e.get("cat") == "user_annotation")
    launch = {e["args"]["correlation"]: e for e in ev
              if e.get("cat") in ("cuda_runtime", "cuda_driver") and "correlation" in e.get("args", {})}
    gpu = [e for e in ev if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e["args"].get("device", -1) >= 0]
    noise_tids = {launch[e["args"]["correlation"]]["tid"] for e in gpu
                  if e["args"].get("correlation") in launch and "fft" in e["name"].lower()}
    stages = list(dict.fromkeys(s[2] for s in spans))
    n_steps = sum(1 for s in spans if s[2] == stages[0])
    wall = defaultdict(float)
    for a, b, nm in spans:
        wall[nm] += (b - a) / 1000 / n_steps
    t0, t1 = spans[0][0], spans[-1][1]

    def stage_of(ts):
        for a, b, nm in spans:           # spans are few (stages x steps)
            if a <= ts <= b:
                return nm
        return "(between stages)"

    busy = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))   # stage -> device -> kind -> ms/step
    devices = set()
    for e in gpu:
        if not (t0 <= e["ts"] <= t1):
            continue
        lc = launch.get(e["args"].get("correlation"))
        dev = e["args"].get("device", -1)
        devices.add(dev)
        st = "GPU noise generation (alongside)" if lc and lc["tid"] in noise_tids else \
            stage_of(lc["ts"]) if lc else "(launch not found)"
        busy[st][dev][kind(e["name"])] += e["dur"] / 1000 / n_steps
    devices = sorted(devices)
    span_ms = (t1 - t0) / 1000 / n_steps

    print(f"{n_steps} profiled steps, {span_ms:.1f} ms per step from the first to the last stage; "
          f"GPU time = sum of the GPU operations launched in the stage, ms per step\n")
    head = f"{'stage':40s}{'wall':>8s}" + "".join(f"{f'GPU {d}':>9s}" for d in devices) + "   main GPU operations"
    print(head)
    print("-" * len(head))
    order = stages + [s for s in busy if s not in stages]
    for st in order:
        per = busy.get(st, {})
        tot_kind = defaultdict(float)
        for d in per:
            for k, v in per[d].items():
                tot_kind[k] += v
        top = sorted(tot_kind.items(), key=lambda kv: -kv[1])[:3]
        tot = sum(tot_kind.values())
        print(f"{st[:40]:40s}{wall.get(st, float('nan')):8.1f}"
              + "".join(f"{sum(per.get(d, {}).values()):9.1f}" for d in devices)
              + "   " + ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in top if tot))
    print("-" * len(head))
    for d in devices:
        tot = sum(sum(busy[st].get(d, {}).values()) for st in busy)
        print(f"GPU {d}: busy {tot:.1f} of {span_ms:.1f} ms per step ({100 * tot / span_ms:.0f}%)")
    print("\nby kind of operation, all stages, ms per step:")
    tot_kind = defaultdict(lambda: defaultdict(float))
    for st in busy:
        for d in busy[st]:
            for k, v in busy[st][d].items():
                tot_kind[k][d] += v
    for k, per in sorted(tot_kind.items(), key=lambda kv: -sum(kv[1].values())):
        print(f"  {k:34s}" + "".join(f"  GPU {d} {per.get(d, 0.0):7.1f}" for d in devices))


if __name__ == "__main__":
    main(sys.argv[1])
