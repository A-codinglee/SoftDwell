# scripts/refine_pipeline/resim.py
"""
Re-simulation of a batch at given parameters, for the refinement loop (train_refine_online.py).

theta (the stage-1 output space: log10 chain rates, then the standardized [baseline, amplitude, sigma]) is turned
into the simulator's (Q, levels, sigma) by ThetaToSim, and a pool of CPU processes simulates one trace per row
into a shared int16 buffer (no 2 MB pickles through pipes).

Common random numbers: the Gillespie seed, the noise offset and (spectral noise) the noise draw of a re-simulation
depend only on (seed, stream, index), not on theta. Every iteration of one sample therefore re-simulates with the
same random numbers, so the difference between two iterations' histograms comes from the parameters, not from
fresh noise; and the same sample gets the same re-simulations in every epoch's validation.

Re-simulations always use the training noise files: on a recorded trace the true noise is unknown, and the test
set (possibly held-out noise) is meant to measure exactly that situation.
"""
import time

import numpy as np
import torch

from scripts.hist_pipeline.train_hist_online import make_synth
from scripts.simulation.simulator import SampleSpec, TraceSimulator

# random streams of the re-simulations (train_hist_online uses 0, 1, 2 for the samples themselves)
STREAM_RESIM = {"train": 10, "val": 11, "test": 12}

_W = {}


class ThetaToSim:
    """theta rows -> (Q (B, K, K), levels (B, K), sigma (B,)) in raw counts, clipped to simulable values."""

    def __init__(self, sim_cfg: SampleSpec, cond: dict, n_rates: int):
        if sim_cfg.amplitude is None:
            raise ValueError("re-simulation needs levels given as training.baseline / amplitude in the sim-config")
        if cond["index"] != [0, 1, 2]:
            raise ValueError("re-simulation needs baseline, amplitude and sigma among the stage-1 outputs")
        K = sim_cfg.num_states
        if n_rates != 2 * (K - 1):
            raise ValueError(f"{n_rates} rates for a chain of {K} states")
        self.K, self.n_rates, self.labels = K, n_rates, list(sim_cfg.labels)
        self.open_sign = sim_cfg.open_sign
        chain = [c for i in range(K - 1) for c in ((i, i + 1), (i + 1, i))]
        self.rows = np.array([r for r, _ in chain])
        self.cols = np.array([c for _, c in chain])
        lo = np.array([sim_cfg.q_lo[r, c] for r, c in chain])
        hi = np.array([sim_cfg.q_hi[r, c] for r, c in chain])
        # canonicalization may reverse the chain, so every rate gets the widest range of the config
        self.log_lo, self.log_hi = float(np.log10(lo.min())), float(np.log10(hi.max()))
        self.mean = np.asarray(cond["mean"], dtype=np.float64)
        self.std = np.asarray(cond["std"], dtype=np.float64)
        self.min_amp = 0.05 * float(np.mean(sim_cfg.amplitude))
        self.min_sigma = 0.05 * float(np.mean(sim_cfg.sigma))

    def __call__(self, theta: np.ndarray):
        theta = np.asarray(theta, dtype=np.float64)
        B = theta.shape[0]
        rates = 10.0 ** np.clip(theta[:, :self.n_rates], self.log_lo, self.log_hi)
        Q = np.zeros((B, self.K, self.K))
        Q[:, self.rows, self.cols] = rates
        aux = theta[:, self.n_rates:self.n_rates + 3] * self.std + self.mean       # counts
        base = aux[:, 0]
        amp = np.maximum(aux[:, 1], self.min_amp)
        sigma = np.maximum(aux[:, 2], self.min_sigma)
        level = {"C": base, "O": base + self.open_sign * amp}
        levels = np.stack([level[lab] for lab in self.labels], axis=1)
        return Q, levels, sigma

    def theta_center_scale(self, theta_dim: int):
        """Centre / half-width of each theta component, for the refiner's input normalisation."""
        c = np.zeros(theta_dim, dtype=np.float32)
        s = np.ones(theta_dim, dtype=np.float32)
        c[:self.n_rates] = 0.5 * (self.log_lo + self.log_hi)
        s[:self.n_rates] = max(0.5 * (self.log_hi - self.log_lo), 1e-3)
        return c, s


def _init(config_path, sim_cfg, spectrum_path, seed, out_buf, noise_source=None):
    torch.set_num_threads(1)
    _W["sim"] = TraceSimulator(config_path)
    _W["synth"] = make_synth(spectrum_path, _W["sim"].trace_len)
    _W["cfg"], _W["seed"], _W["out"] = sim_cfg, int(seed), out_buf.numpy()
    _W["noise_source"] = noise_source


def _job(job):
    slot, Q, levels, sigma, stream, idx = job
    t0 = time.time()
    rng = np.random.default_rng(np.random.SeedSequence([_W["seed"], stream, int(idx)]))
    sim_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    noise_index = int(rng.integers(0, 2**32, dtype=np.uint64))
    noise = None
    if _W["synth"] is not None:
        src = _W["cfg"].noise_sources(test=False)
        k = src[int(rng.integers(len(src)))] if len(src) > 1 else src[0]
        if _W["noise_source"] is not None:        # forced file (the draw above still runs: same random numbers)
            k = _W["noise_source"]
        noise = _W["synth"].synthesize(rng, k)
    _W["out"][slot] = _W["sim"].simulate_trace(Q, levels, seed=sim_seed, noise_index=noise_index, sigma=float(sigma),
                                               noise=noise)
    return time.time() - t0


class Resimulator:
    """
    A pool of simulator processes. resim(theta, stream, idx) returns the traces (B, T) int16 as a view into a
    shared CPU buffer (page-locked if possible); the view is overwritten by the next call. noise_source: row of
    the noise spectra every re-simulation uses (default: drawn per sample from the training files).
    """

    def __init__(self, ctx, n_workers: int, config_path: str, sim_cfg: SampleSpec, spectrum_path: str, seed: int,
                 to_sim: ThetaToSim, max_batch: int, trace_len: int, pin=None, noise_source=None):
        self.to_sim, self.max_batch = to_sim, max_batch
        self.buf = torch.zeros((max_batch, trace_len), dtype=torch.int16).share_memory_()
        self.pinned = bool(pin(self.buf)) if pin is not None else False
        self.pool = ctx.Pool(n_workers, initializer=_init,
                             initargs=(config_path, sim_cfg, spectrum_path, seed, self.buf, noise_source))
        self.n_workers = n_workers
        self.wall_s = 0.0           # seconds the caller waited for re-simulations
        self.cpu_s = 0.0            # simulator seconds summed over the workers
        self.n_traces = 0

    def __call__(self, theta, stream: int, idx) -> torch.Tensor:
        theta = theta.detach().float().cpu().numpy() if torch.is_tensor(theta) else np.asarray(theta)
        idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx, dtype=np.int64)
        B = theta.shape[0]
        if B > self.max_batch:
            raise ValueError(f"batch of {B} > max_batch {self.max_batch}")
        Q, levels, sigma = self.to_sim(theta)
        t0 = time.time()
        jobs = [(b, Q[b], levels[b], sigma[b], stream, idx[b]) for b in range(B)]
        self.cpu_s += sum(self.pool.imap_unordered(_job, jobs, chunksize=1))
        self.wall_s += time.time() - t0
        self.n_traces += B
        return self.buf[:B]

    def close(self):
        self.pool.terminate()
        self.pool.join()
