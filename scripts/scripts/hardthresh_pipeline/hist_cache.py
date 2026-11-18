"""
Build hard-threshold ("grind") histogram stacks from a raw-trace H5 in RAM,
streaming the (huge) trace array through the GPU once, and cache the result on
disk so a re-run with the same data + (K, J, ladder) skips the whole pass.

Several (K, J, ladder) specs can be built in ONE read pass over the data
(reading 2 MB/row dominates the cost, so a K/J sweep should call
build_hist_stacks() once with all specs, e.g. via build_cache.py).

Output per spec: hist (N, K, J, J) float32 raw pair counts (axis 2 = above-dwell
bin, axis 3 = below-dwell bin, see gpu_hist), n_events (N, K) int64.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

import h5py
import numpy as np
import torch

from scripts.hardthresh_pipeline.gpu_hist import make_thresholds, make_dwell_lut, grind_hist_gpu

CACHE_VERSION = "grind-v1"   # bump whenever histogram semantics change


@dataclass(frozen=True)
class HistSpec:
    K: int
    J: int
    thr_lo_pA: float
    thr_hi_pA: float

    def tag(self) -> str:
        return f"K{self.K}_J{self.J}"


def spec_from_cfg(cfg) -> HistSpec:
    return HistSpec(cfg.num_detectors, cfg.num_bins, cfg.thr_lo_pA, cfg.thr_hi_pA)


def _n_rows_and_len(cfg) -> Tuple[int, int]:
    with h5py.File(cfg.h5_path, "r") as f:
        n_total, T = f[cfg.traces_key].shape
    avail = n_total - cfg.row_start
    n = avail if cfg.limit_n <= 0 else min(cfg.limit_n, avail)
    T_used = T if cfg.t_len <= 0 else min(cfg.t_len, T)
    return int(n), int(T_used)


def cache_key(cfg, spec: HistSpec, n: int, T: int) -> str:
    st = os.stat(cfg.h5_path)
    ident = {
        "ver": CACHE_VERSION,
        "file": os.path.basename(cfg.h5_path), "size": st.st_size, "mtime_ns": st.st_mtime_ns,
        "n": n, "T": T, "fs": cfg.fs_hz, "cpp": cfg.counts_per_pA,
        "K": spec.K, "J": spec.J, "lo": spec.thr_lo_pA, "hi": spec.thr_hi_pA,
        "tmin": cfg.dwell_min_s, "tmax": cfg.dwell_max_s,
    }
    if cfg.row_start:                       # only added when non-zero, so existing cache keys stay valid
        ident["row_start"] = cfg.row_start
    h = hashlib.sha1(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    stem = os.path.splitext(os.path.basename(cfg.h5_path))[0][:60]
    return f"{stem}__{spec.tag()}__n{n}__{h}"


def _cache_paths(cfg, key: str) -> Tuple[str, str, str]:
    d = cfg.cache_dir
    return os.path.join(d, key + "_hist.npy"), os.path.join(d, key + "_nev.npy"), os.path.join(d, key + ".json")


def _try_load(cfg, key: str):
    ph, pn, pm = _cache_paths(cfg, key)
    if cfg.cache and os.path.isfile(ph) and os.path.isfile(pn) and os.path.isfile(pm):
        return np.load(ph), np.load(pn)
    return None


def _save(cfg, key: str, hist: np.ndarray, nev: np.ndarray, meta: dict) -> None:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    ph, pn, pm = _cache_paths(cfg, key)
    # write hist last-but-meta so a partially written cache is never picked up
    np.save(ph + ".tmp.npy", hist); os.replace(ph + ".tmp.npy", ph)
    np.save(pn + ".tmp.npy", nev); os.replace(pn + ".tmp.npy", pn)
    with open(pm + ".tmp", "w") as f:
        json.dump(meta, f, indent=2)
    os.replace(pm + ".tmp", pm)


def _default_log(msg: str) -> None:
    print(msg, flush=True)


def build_hist_stacks(cfg, specs: Sequence[HistSpec], device: str = "cuda",
                      log=_default_log) -> Dict[HistSpec, Tuple[np.ndarray, np.ndarray]]:
    """Return {spec: (hist, nev)} for every spec, using the disk cache where possible and
    computing all remaining specs together in a single pass over the traces."""
    n, T = _n_rows_and_len(cfg)
    out: Dict[HistSpec, Tuple[np.ndarray, np.ndarray]] = {}
    todo: List[HistSpec] = []
    keys = {}
    for sp in specs:
        keys[sp] = cache_key(cfg, sp, n, T)
        hit = _try_load(cfg, keys[sp])
        if hit is not None:
            log(f"[hist] cache hit  {sp.tag()}  ({keys[sp]})")
            out[sp] = hit
        else:
            todo.append(sp)
    if not todo:
        return out

    dev = torch.device(device)
    luts = {}
    thrs = {}
    stacks = {}
    for sp in todo:
        if sp.J not in luts:
            luts[sp.J] = torch.from_numpy(make_dwell_lut(T, cfg.fs_hz, cfg.dwell_min_s, cfg.dwell_max_s, sp.J)).to(dev)
        thrs[sp] = torch.from_numpy(make_thresholds(sp.thr_lo_pA * cfg.counts_per_pA,
                                                    sp.thr_hi_pA * cfg.counts_per_pA, sp.K)).to(dev)
        stacks[sp] = (np.zeros((n, sp.K, sp.J, sp.J), dtype=np.float32), np.zeros((n, sp.K), dtype=np.int64))
        log(f"[hist] building {sp.tag()}  thr(counts)={thrs[sp][0].item()}..{thrs[sp][-1].item()}  "
            f"stack={stacks[sp][0].nbytes / 1e9:.2f} GB")

    def read_block(r0: int, r1: int) -> np.ndarray:
        with h5py.File(cfg.h5_path, "r") as f:
            return f[cfg.traces_key][cfg.row_start + r0:cfg.row_start + r1, :T]

    blocks = [(r0, min(r0 + cfg.read_rows, n)) for r0 in range(0, n, cfg.read_rows)]
    t0 = time.perf_counter()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, cfg.prefetch)) as ex:
        futs = [ex.submit(read_block, *b) for b in blocks[:cfg.prefetch + 1]]
        nxt = len(futs)
        for bi, (r0, r1) in enumerate(blocks):
            arr = futs[bi].result()
            if nxt < len(blocks):
                futs.append(ex.submit(read_block, *blocks[nxt])); nxt += 1
            futs[bi] = None   # free
            xt = torch.from_numpy(arr).to(dev, non_blocking=False)
            if xt.dtype != torch.int16:
                xt = xt.to(torch.int16)
            for s0 in range(0, xt.shape[0], cfg.hist_bs):
                xb = xt[s0:s0 + cfg.hist_bs]
                for sp in todo:
                    h, ne = grind_hist_gpu(xb, thrs[sp], luts[sp.J], sp.J, k_group=cfg.k_group)
                    a = r0 + s0
                    stacks[sp][0][a:a + xb.shape[0]] = h.cpu().numpy()
                    stacks[sp][1][a:a + xb.shape[0]] = ne.cpu().numpy()
            done = r1
            if (bi % 8 == 0) or r1 == n:
                el = time.perf_counter() - t0
                rate = done / max(el, 1e-9)
                log(f"[hist] {done}/{n} rows  {rate:.1f} rows/s  elapsed {el:.0f}s  eta {(n - done) / max(rate, 1e-9):.0f}s")

    for sp in todo:
        hist, nev = stacks[sp]
        out[sp] = (hist, nev)
        if cfg.cache:
            _save(cfg, keys[sp], hist, nev, {"spec": asdict(sp), "n": n, "T": T, "version": CACHE_VERSION,
                                              "h5": cfg.h5_path, "built_s": time.perf_counter() - t0})
            log(f"[hist] cached {sp.tag()} -> {_cache_paths(cfg, keys[sp])[0]}")
    return out
