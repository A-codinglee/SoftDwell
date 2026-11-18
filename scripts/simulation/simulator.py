# scripts/simulation/simulator.py
"""Python access to the C++ single-channel simulator (3D_Fit_unclamped, light branch)."""
import ctypes
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SIM_ROOT = os.path.normpath(os.path.join(HERE, "..", "..", "external", "3D_Fit_unclamped"))
DEFAULT_LIB = os.path.join(HERE, "build", "libsimdwell.so")


# -------------------------------------------------------------------------
# Simulation config file -> what to sample per training example
# -------------------------------------------------------------------------
def prepare_sim_config(src_path: str, out_path: str, seed: int, holdout_noise: Optional[str] = None,
                       train_noise: Optional[str] = None) -> str:
    """
    Read a simulation JSON (the simulator's own format), make its file paths absolute and
    set the base seed, and write the result to out_path. Returns out_path.

    Relative paths are looked up next to the JSON file first, then in the simulator repo.

    holdout_noise / train_noise split the files of noise.source_paths (+ test_source_paths) anew;
    each is a piece of one file's name:
      holdout_noise: that file is the test set's only noise, all others are for training
      train_noise:   that file is the only training noise, all others are for the test set
    """
    with open(src_path) as f:
        cfg = json.load(f)

    def resolve(p: str) -> str:
        if os.path.isabs(p):
            return p
        for root in (os.path.dirname(os.path.abspath(src_path)), DEFAULT_SIM_ROOT):
            cand = os.path.join(root, p)
            if os.path.isfile(cand):
                return cand
        raise FileNotFoundError(f"{p!r} (from {src_path}) not found next to the config or in {DEFAULT_SIM_ROOT}")

    cfg["input"]["step_response_double"] = resolve(cfg["input"]["step_response_double"])
    # noise.source_paths / noise.test_source_paths (several files, see SampleSpec) are ours; the
    # simulator itself only reads noise.source_path, so that always names one existing file.
    jn = cfg["noise"]
    for key in ("source_paths", "test_source_paths"):
        if key in jn:
            jn[key] = [resolve(p) for p in jn[key]]
    pick = holdout_noise or train_noise
    if holdout_noise and train_noise:
        raise ValueError("give only one of holdout_noise / train_noise")
    if pick:
        files = jn.get("source_paths", []) + jn.get("test_source_paths", [])
        hit = [p for p in files if pick in os.path.basename(p)]
        if len(hit) != 1:
            raise ValueError(f"{pick!r} matches {len(hit)} of the noise files in {src_path}, need exactly 1: "
                             f"{[os.path.basename(p) for p in files]}")
        rest = [p for p in files if p != hit[0]]
        jn["source_paths"], jn["test_source_paths"] = (rest, hit) if holdout_noise else (hit, rest)
        jn.pop("source_path", None)
    if "source_path" not in jn and jn.get("source_paths"):
        jn["source_path"] = jn["source_paths"][0]
    jn["source_path"] = resolve(jn["source_path"])
    cfg["seeding"] = {"mode": "fixed", "seed": int(seed)}

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    return out_path


def _as_range(v, what: str) -> Tuple[float, float]:
    """A plain number or a {min, max} object -> (lo, hi)."""
    if isinstance(v, (int, float)):
        return float(v), float(v)
    if isinstance(v, dict) and "min" in v and "max" in v:
        lo, hi = float(v["min"]), float(v["max"])
        if hi < lo:
            raise ValueError(f"{what}: max < min")
        return lo, hi
    raise ValueError(f"{what} must be a number or {{min, max}}, got {v!r}")


@dataclass
class SampleSpec:
    """
    What is drawn per training example, taken from the simulation JSON:

      model.transition_rates   number = fixed, {min,max} = log-uniform, 0 = no transition
      noise.sigma[_pA]         number or {min,max} (uniform)
      noise.source_paths       spectral_resample only: several noise files, one picked per training /
                               validation example (default: [noise.source_path])
      noise.test_source_paths  spectral_resample only: noise files for the test set, kept out of
                               training (default: the same files as training)
      levels, either
        model.level_groups.<label>.level[_pA]   number or {min,max} (uniform, independent per label), or
        training.baseline[_pA] / training.amplitude[_pA] / training.open_sign:
            closed ("C") level = baseline, open ("O") level = baseline + open_sign * amplitude

    All values are held in raw ADC counts.
    """
    labels: List[str]
    q_lo: np.ndarray
    q_hi: np.ndarray
    sigma: Tuple[float, float]
    level_groups: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    baseline: Optional[Tuple[float, float]] = None
    amplitude: Optional[Tuple[float, float]] = None
    open_sign: float = -1.0
    counts_per_pA: float = 0.0
    # noise.mode: "recorded_noise" (done by the simulator) or "spectral_resample" (scripts.simulation.noise)
    noise_mode: str = "recorded_noise"
    noise_path: str = ""
    noise_paths: List[str] = field(default_factory=list)        # training / validation
    noise_test_paths: List[str] = field(default_factory=list)   # test; empty = same as training
    noise_avg_seg: int = 131072
    noise_avg_overlap: float = 0.5
    noise_generate_factor: int = 2

    @property
    def topology(self) -> str:
        return "".join(self.labels)

    @property
    def num_states(self) -> int:
        return len(self.labels)

    @property
    def all_noise_paths(self) -> List[str]:
        """Training files, then test files: the row order of the noise spectra."""
        return self.noise_paths + self.noise_test_paths

    def noise_sources(self, test: bool) -> range:
        """Rows of all_noise_paths an example may draw its noise from."""
        n = len(self.noise_paths)
        return range(n, n + len(self.noise_test_paths)) if test and self.noise_test_paths else range(n)

    @classmethod
    def from_simulator(cls, sim: "TraceSimulator") -> "SampleSpec":
        cfg, c = sim.config, sim.counts_per_pA      # cfg: pA fields the simulator knows are already counts
        model = cfg["model"]
        labels = [st.get("label", "") for st in model["states"]]
        K = len(labels)

        q_lo, q_hi = np.zeros((K, K)), np.zeros((K, K))
        rates = model["transition_rates"]
        for r in range(K):
            for col in range(K):
                if r != col:
                    q_lo[r, col], q_hi[r, col] = _as_range(rates[r][col], f"transition_rates[{r}][{col}]")
        if (q_lo < 0).any() or ((q_hi > 0) & (q_lo <= 0)).any():
            raise ValueError("transition rate ranges need min > 0 (they are sampled log-uniformly)")

        if "sigma" not in cfg["noise"]:
            raise ValueError("noise.sigma (or noise.sigma_pA) is required")
        jn = cfg["noise"]
        spec = cls(labels=labels, q_lo=q_lo, q_hi=q_hi, sigma=_as_range(jn["sigma"], "noise.sigma"),
                   counts_per_pA=c,
                   noise_mode=jn.get("mode", "recorded_noise"), noise_path=jn["source_path"],
                   noise_paths=list(jn.get("source_paths", [jn["source_path"]])),
                   noise_test_paths=list(jn.get("test_source_paths", [])),
                   noise_avg_seg=int(jn.get("avg_seg", 131072)),
                   noise_avg_overlap=float(jn.get("avg_overlap", 0.5)),
                   noise_generate_factor=int(jn.get("generate_factor", 2)))
        if spec.noise_mode not in ("recorded_noise", "spectral_resample"):
            raise ValueError(f"noise.mode must be 'recorded_noise' or 'spectral_resample', got {spec.noise_mode!r}")
        if spec.sigma[0] <= 0:
            raise ValueError("noise.sigma must be > 0")
        if spec.noise_mode != "spectral_resample" and len(spec.all_noise_paths) > 1:
            raise ValueError("noise.source_paths / test_source_paths need noise.mode 'spectral_resample'")
        if not spec.noise_paths:
            raise ValueError("noise.source_paths is empty")

        def counts(block: dict, key: str, what: str) -> Optional[Tuple[float, float]]:
            if key in block:
                return _as_range(block[key], what)
            if key + "_pA" in block:
                if c <= 0:
                    raise ValueError(f"{what}_pA needs an 'amplifier' block in the config")
                lo, hi = _as_range(block[key + "_pA"], what + "_pA")
                return lo * c, hi * c
            return None

        tr = cfg.get("training", {})
        spec.amplitude = counts(tr, "amplitude", "training.amplitude")
        if spec.amplitude is not None:
            spec.baseline = counts(tr, "baseline", "training.baseline") or (0.0, 0.0)
            spec.open_sign = float(tr.get("open_sign", -1.0))
            if not set(labels) <= {"C", "O"}:
                raise ValueError("training.amplitude needs every state label to be 'C' or 'O'")
        else:
            for lab, grp in model.get("level_groups", {}).items():
                spec.level_groups[lab] = _as_range(grp["level"] if isinstance(grp, dict) else grp,
                                                   f"level_groups.{lab}.level")
            missing = sorted(set(labels) - set(spec.level_groups))
            if missing:
                raise ValueError(f"no level for state label(s) {missing}: set model.level_groups or training.amplitude")
        return spec

    def sample_Q(self, rng: np.random.Generator) -> np.ndarray:
        """(K, K) float64. Chain-adjacent cells are drawn first, in (i,i+1),(i+1,i) order."""
        K = self.num_states
        Q = np.zeros((K, K), dtype=np.float64)
        chain = [c for i in range(K - 1) for c in ((i, i + 1), (i + 1, i))]
        rest = [(r, c) for r in range(K) for c in range(K) if r != c and (r, c) not in chain]
        for r, c in chain + rest:
            lo, hi = self.q_lo[r, c], self.q_hi[r, c]
            if hi <= 0:
                continue
            Q[r, c] = lo if hi == lo else 10.0 ** rng.uniform(np.log10(lo), np.log10(hi))
        return Q

    def sample_levels_sigma(self, rng: np.random.Generator) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Returns (state levels (K,), sigma, aux = [closed/first level, level span, sigma]), raw counts.
        With training.amplitude the first aux column is the closed level, also when the chain starts with
        an open state (OCO).
        """
        def draw(rg):
            return rg[0] if rg[1] == rg[0] else rng.uniform(*rg)

        if self.amplitude is not None:
            base = draw(self.baseline)
            by_label = {"C": base, "O": base + self.open_sign * draw(self.amplitude)}
        else:
            by_label = {lab: draw(rg) for lab, rg in sorted(self.level_groups.items())}
        levels = np.array([by_label[lab] for lab in self.labels], dtype=np.float64)
        sigma = float(draw(self.sigma))
        first = by_label["C"] if self.amplitude is not None else levels[0]
        aux = np.array([first, levels.max() - levels.min(), sigma], dtype=np.float32)
        return levels, sigma, aux


# -------------------------------------------------------------------------
# Fixed-setup config builder (used by generate_hist_h5.py)
# -------------------------------------------------------------------------
@dataclass
class SimConfig:
    """Everything the simulator needs that is fixed for a whole dataset."""
    topology: str = "COC"                     # chain of state labels, e.g. "COC", "COCO", "COCOC"
    levels: Dict[str, float] = field(default_factory=lambda: {"C": 22000.0, "O": 20000.0})  # raw ADC counts
    sampling_rate_hz: float = 100_000.0
    t_max_s: float = 10.0
    noise_sigma: float = 2000.0               # raw ADC counts
    noise_path: str = os.path.join(DEFAULT_SIM_ROOT, "noise_files", "Bath_Noise_Gain_100.bin")
    step_response_path: str = os.path.join(DEFAULT_SIM_ROOT, "step_responses", "step_response_double.bin")
    step_response_dt: float = 1e-5
    seed: int = 42                            # base seed (noise segment choice)

    # grind dwell-histogram backend
    hist_bins: int = 24
    hist_tmin_s: float = 1e-5
    hist_tmax_s: float = 0.1
    grind_min: float = 18000.0
    grind_max: float = 24000.0
    grind_count: int = 24

    # rate sampling (log-uniform), used by sample_chain_Q
    k_min: float = 10.0
    k_max: float = 1_000_000.0

    @property
    def num_states(self) -> int:
        return len(self.topology)

    def state_levels(self) -> np.ndarray:
        return np.array([self.levels[s] for s in self.topology], dtype=np.float64)

    def to_json(self) -> dict:
        K = self.num_states
        return {
            "seeding": {"mode": "fixed", "seed": int(self.seed)},
            "input": {
                "step_response_double": self.step_response_path,
                "step_response_dt": self.step_response_dt,
            },
            "sampling": {"rate_hz": self.sampling_rate_hz},
            "simulation": {"t_max_s": self.t_max_s},
            "model": {
                "initial_state_index": 0,
                "level_groups": {lab: {"level": lvl} for lab, lvl in self.levels.items()},
                "states": [{"label": s} for s in self.topology],
                # placeholder; the actual Q is passed per call
                "transition_rates": [[0] * K for _ in range(K)],
            },
            "noise": {
                "mode": "recorded_noise",
                "source_path": self.noise_path,
                "sigma": self.noise_sigma,
            },
            "analysis": {
                "hist_backend": "grind",
                "hist2d": {"tmin_s": self.hist_tmin_s, "tmax_s": self.hist_tmax_s, "bins": self.hist_bins},
                "grind": {"min": self.grind_min, "max": self.grind_max, "count": self.grind_count},
            },
        }

    def write_json(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2)
        return path


def sample_chain_Q(rng: np.random.Generator, num_states: int, k_min: float, k_max: float) -> np.ndarray:
    """Linear chain 0<->1<->...<->S-1 with log10-uniform adjacent rates. Returns (S, S) float64."""
    Q = np.zeros((num_states, num_states), dtype=np.float64)
    lo, hi = np.log10(k_min), np.log10(k_max)
    for i in range(num_states - 1):
        Q[i, i + 1] = 10.0 ** rng.uniform(lo, hi)
        Q[i + 1, i] = 10.0 ** rng.uniform(lo, hi)
    return Q


# -------------------------------------------------------------------------
# The simulator itself
# -------------------------------------------------------------------------
class TraceSimulator:
    """One simulator instance per process. Not safe to share across forked workers; create it in each."""

    def __init__(self, config_path: str, lib_path: Optional[str] = None):
        lib_path = lib_path or os.environ.get("SIMDWELL_LIB", DEFAULT_LIB)
        if not os.path.isfile(lib_path):
            raise FileNotFoundError(
                f"{lib_path} not found. Build it with: apptainer exec pytorch.sif bash scripts/simulation/build.sh"
            )
        lib = ctypes.CDLL(lib_path)
        vp, dp = ctypes.c_void_p, ctypes.c_void_p
        lib.sd_create.restype = vp
        lib.sd_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.sd_destroy.argtypes = [vp]
        lib.sd_trace_len.restype = ctypes.c_long
        lib.sd_trace_len.argtypes = [vp]
        lib.sd_hist_detectors.restype = ctypes.c_int
        lib.sd_hist_detectors.argtypes = [vp]
        lib.sd_hist_bins.restype = ctypes.c_int
        lib.sd_hist_bins.argtypes = [vp]
        lib.sd_last_error.restype = ctypes.c_char_p
        lib.sd_last_error.argtypes = [vp]
        lib.sd_config_json.restype = ctypes.c_char_p
        lib.sd_config_json.argtypes = [vp]
        lib.sd_counts_per_pA.restype = ctypes.c_double
        lib.sd_counts_per_pA.argtypes = [vp]
        self._has_timings = hasattr(lib, "sd_last_timings")     # absent in libraries built before it existed
        if self._has_timings:
            lib.sd_last_timings.restype = None
            lib.sd_last_timings.argtypes = [dp]
        lib.sd_simulate_trace.restype = ctypes.c_long
        lib.sd_simulate_trace.argtypes = [vp, ctypes.c_int, dp, dp, ctypes.c_uint32, ctypes.c_uint32,
                                          ctypes.c_double, dp, ctypes.c_double, dp, dp]
        lib.sd_simulate_hist.restype = ctypes.c_long
        lib.sd_simulate_hist.argtypes = [vp, ctypes.c_int, dp, dp, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_double, dp, ctypes.c_double, dp, dp, dp]
        self._has_copy_trace = hasattr(lib, "sd_copy_last_trace")   # absent in libraries built before it existed
        if self._has_copy_trace:
            lib.sd_copy_last_trace.restype = None
            lib.sd_copy_last_trace.argtypes = [vp, dp]
        self._has_levels = hasattr(lib, "sd_simulate_levels")       # absent in libraries built before it existed
        if self._has_levels:
            lib.sd_simulate_levels.restype = ctypes.c_long
            lib.sd_simulate_levels.argtypes = [vp, ctypes.c_int, dp, dp, ctypes.c_uint32, dp, dp]
            lib.sd_kernel.restype = ctypes.c_long
            lib.sd_kernel.argtypes = [vp, dp, ctypes.c_long, dp]
        self._lib = lib

        err = ctypes.create_string_buffer(1024)
        self._h = lib.sd_create(config_path.encode(), err, len(err))
        if not self._h:
            raise RuntimeError(f"simulator init failed: {err.value.decode(errors='replace')}")

        self.trace_len = int(lib.sd_trace_len(self._h))
        self.hist_detectors = int(lib.sd_hist_detectors(self._h))
        self.hist_bins = int(lib.sd_hist_bins(self._h))
        # the config as the simulator parsed it: "*_pA" fields it knows are already raw counts
        self.config = json.loads(lib.sd_config_json(self._h).decode())
        self.counts_per_pA = float(lib.sd_counts_per_pA(self._h))   # 0 if no "amplifier" block
        self.last_sigma = float("nan")

    def close(self) -> None:
        if getattr(self, "_h", None):
            self._lib.sd_destroy(self._h)
            self._h = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _prep(self, Q: np.ndarray, levels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        Q = np.ascontiguousarray(Q, dtype=np.float64)
        levels = np.ascontiguousarray(levels, dtype=np.float64)
        K = Q.shape[0]
        if Q.shape != (K, K) or levels.shape != (K,):
            raise ValueError(f"expected Q (K,K) and levels (K,), got {Q.shape} and {levels.shape}")
        return Q, levels, K

    def _noise_args(self, noise, sigma):
        """noise = (int16 array (trace_len,), unit) from scripts.simulation.noise, or None."""
        if noise is None:
            return None, 0.0
        arr, unit = noise
        if arr.dtype != np.int16 or arr.shape != (self.trace_len,) or not arr.flags.c_contiguous:
            raise ValueError(f"noise must be a contiguous int16 array of length {self.trace_len}")
        if not sigma or sigma <= 0:
            raise ValueError("supplied noise needs sigma > 0")
        return arr.ctypes.data, float(unit)

    def _check(self, n: int) -> int:
        if n < 0:
            raise RuntimeError(f"simulation failed: {self._lib.sd_last_error(self._h).decode(errors='replace')}")
        return n

    def last_timings(self):
        """Seconds the last simulate_* call spent in (Gillespie, trace synthesis, histograms), or None."""
        if not self._has_timings:
            return None
        out = (ctypes.c_double * 3)()
        self._lib.sd_last_timings(ctypes.addressof(out))
        return float(out[0]), float(out[1]), float(out[2])

    def simulate_trace(self, Q: np.ndarray, levels: np.ndarray, seed: int, noise_index: int = 0,
                       sigma: Optional[float] = None, noise=None) -> np.ndarray:
        """
        Returns the noisy, filtered trace as int16 (trace_len,). sigma overrides the config's noise
        sigma; noise = (int16 array, unit) replaces the config's noise source with the given noise.
        """
        Q, levels, K = self._prep(Q, levels)
        noise_ptr, noise_unit = self._noise_args(noise, sigma)
        out = np.empty(self.trace_len, dtype=np.int16)
        sigma_used = ctypes.c_float(0.0)
        self._check(self._lib.sd_simulate_trace(
            self._h, K, Q.ctypes.data, levels.ctypes.data,
            seed & 0xFFFFFFFF, noise_index & 0xFFFFFFFF, float(sigma or 0.0), noise_ptr, noise_unit,
            out.ctypes.data, ctypes.addressof(sigma_used)))
        self.last_sigma = float(sigma_used.value)
        return out

    def simulate_hist(self, Q: np.ndarray, levels: np.ndarray, seed: int, noise_index: int = 0,
                      sigma: Optional[float] = None, noise=None) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (grids float32 (D, B, B), n_events int64 (D,)). sigma / noise as in simulate_trace."""
        Q, levels, K = self._prep(Q, levels)
        noise_ptr, noise_unit = self._noise_args(noise, sigma)
        D, B = self.hist_detectors, self.hist_bins
        if D <= 0 or B <= 0:
            raise RuntimeError("config has no usable analysis.grind / analysis.hist2d block")
        grids = np.empty((D, B, B), dtype=np.float32)
        n_events = np.empty(D, dtype=np.int64)
        sigma_used = ctypes.c_float(0.0)
        self._check(self._lib.sd_simulate_hist(
            self._h, K, Q.ctypes.data, levels.ctypes.data,
            seed & 0xFFFFFFFF, noise_index & 0xFFFFFFFF, float(sigma or 0.0), noise_ptr, noise_unit,
            grids.ctypes.data, n_events.ctypes.data, ctypes.addressof(sigma_used)))
        self.last_sigma = float(sigma_used.value)
        return grids, n_events

    def kernel(self) -> np.ndarray:
        """The step response the trace synthesis uses, float32 (K,)."""
        if not self._has_levels:
            raise RuntimeError("libsimdwell.so predates sd_kernel; rebuild it (scripts/simulation/build.sh)")
        out = np.empty(4096, dtype=np.float32)
        K = int(self._lib.sd_kernel(self._h, out.ctypes.data, len(out), None))
        if K > len(out):
            raise RuntimeError(f"kernel of {K} samples is longer than expected")
        return out[:K].copy()

    def simulate_levels(self, Q: np.ndarray, levels: np.ndarray, seed: int):
        """
        Gillespie only, for trace synthesis on the GPU (scripts/hist_pipeline/gpu_synth.py): returns
        (u uint16 (trace_len,), lo, hi), the noise-free current averaged over each sample interval,
        u * (hi - lo) / 65535 + lo. Same seed, same channel states as simulate_trace.
        """
        if not self._has_levels:
            raise RuntimeError("libsimdwell.so predates sd_simulate_levels; rebuild it (scripts/simulation/build.sh)")
        Q, levels, K = self._prep(Q, levels)
        u = np.empty(self.trace_len, dtype=np.uint16)
        lohi = np.empty(2, dtype=np.float64)
        self._check(self._lib.sd_simulate_levels(self._h, K, Q.ctypes.data, levels.ctypes.data, seed & 0xFFFFFFFF,
                                                 u.ctypes.data, lohi.ctypes.data))
        return u, float(lohi[0]), float(lohi[1])

    def simulate_trace_and_hist(self, Q: np.ndarray, levels: np.ndarray, seed: int, noise_index: int = 0,
                                sigma: Optional[float] = None, noise=None):
        """One simulation, both outputs: (trace int16 (trace_len,), grids (D, B, B), n_events (D,))."""
        if not self._has_copy_trace:
            raise RuntimeError("libsimdwell.so predates sd_copy_last_trace; rebuild it (scripts/simulation/build.sh)")
        grids, n_events = self.simulate_hist(Q, levels, seed, noise_index, sigma, noise)
        trace = np.empty(self.trace_len, dtype=np.int16)
        self._lib.sd_copy_last_trace(self._h, trace.ctypes.data)
        return trace, grids, n_events
