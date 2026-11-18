# scripts/hist_pipeline/gpu_noise.py
"""
Spectral noise for the training samples, made on GPUs instead of in the CPU simulator processes
(train_selfcal_online.py --noise-on-gpu).

Same method as scripts.simulation.noise.SpectralNoise: the recording's spectrum on the synthesis grid,
a uniform random phase per frequency, inverse FFT, first trace_len points, mean removed, scaled to a
standard deviation of NOISE_I16_STD, clipped and rounded to int16. Results are not bit-identical to the
CPU version (different random numbers, GPU arithmetic), but have the same statistics.

Which noise a training sample gets depends only on (seed, sample index):
  recording: drawn from SeedSequence([seed, STREAM_TRAIN, i, 1]) among the training recordings
  phases:    torch generator seeded from SeedSequence([seed, STREAM_TRAIN, i, 2])
The validation and test sets keep the CPU noise, so they stay identical to earlier runs.

Hand-over: a ring of `ring` slots in shared, page-locked CPU memory. Noise for sample i goes into slot
i % ring; seq[slot] = i marks it ready, the simulator sets it back to -1 once its trace is built.
With ring_device the ring is on that GPU instead, for traces synthesized on the GPUs (gpu_synth.py); then the
trainer takes the noise and frees the slots.
"""
import itertools
import threading
import time

import numpy as np
import torch

from scripts.simulation.noise import NOISE_I16_STD


def _seed(seed, stream, i, tag):
    return int(np.random.SeedSequence([seed, stream, i, tag]).generate_state(1, dtype=np.uint64)[0] & (2**63 - 1))


class GPUNoiseProducer:
    def __init__(self, mag, trace_len, devices, train_sources, seed, stream, ring=1024, chunk=16, pin=None,
                 ring_device=None):
        self.mag = np.ascontiguousarray(np.atleast_2d(mag), dtype=np.float32)
        self.nb = self.mag.shape[1]
        self.nfft = 2 * (self.nb - 1)
        self.N = int(trace_len)
        if self.nfft < self.N:
            raise ValueError("spectrum is too short for this trace length")
        if ring % chunk:
            raise ValueError("ring must be a multiple of chunk")
        self.devices, self.src = list(devices), list(train_sources)
        self.seed, self.stream, self.R, self.chunk = seed, stream, ring, chunk
        if ring_device is not None:
            self.ring = torch.zeros((ring, self.N), dtype=torch.int16, device=ring_device)
        else:
            self.ring = torch.zeros((ring, self.N), dtype=torch.int16).share_memory_()
            if pin is not None:
                pin(self.ring)
        self.seq = torch.full((ring,), -1, dtype=torch.int64).share_memory_()
        self.kbuf = torch.zeros(ring, dtype=torch.int64).share_memory_()
        self.stop = False
        self.threads = []
        self.written = [g - ring // chunk for g in range(ring // chunk)]    # last chunk written per slot group

    def source_of(self, i):
        """Row of the spectra (all_noise_paths order) whose noise training sample i gets."""
        if len(self.src) == 1:
            return self.src[0]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.stream, i, 1]))
        return self.src[int(rng.integers(len(self.src)))]

    def make(self, idx, device):
        """Noise of the samples idx on device: (len(idx), N) int16 (steps of SpectralNoise.synthesize)."""
        mag = self._mag[device]
        k = torch.tensor([self.source_of(i) for i in idx], device=device)
        phi = torch.stack([torch.rand(self.nb, generator=torch.Generator(device=device).manual_seed(
            _seed(self.seed, self.stream, i, 2)), device=device, dtype=torch.float32) for i in idx])
        phi *= np.float32(2.0 * np.pi)
        Y = torch.polar(mag[k], phi)                       # amplitude of the sample's recording, random phase
        Y.imag[:, -1] = 0.0                                # Nyquist bin is real
        x = torch.fft.irfft(Y, n=self.nfft)[:, :self.N]
        x -= x.mean(dim=1, keepdim=True)
        x *= NOISE_I16_STD / x.std(dim=1, correction=0, keepdim=True).clamp_min(1e-30)
        return x.clamp_(-32767.0, 32767.0).round_().to(torch.int16), k

    def _run(self, t):
        dev = self.devices[t % len(self.devices)]
        T = len(self.threads_planned)
        G = self.R // self.chunk
        for c in itertools.count(t, T):
            idx = list(range(c * self.chunk, (c + 1) * self.chunk))
            s0 = idx[0] % self.R
            # chunk c - G (another thread's) must have been written into these slots first: an empty slot can
            # also mean that chunk is still to come, and filling it with chunk c would block that thread for
            # good, and with it the simulators waiting for chunk c - G
            t0 = time.time()
            while not self.stop and self.written[c % G] != c - G:
                time.sleep(0.0005)
            while not self.stop and bool((self.seq[s0:s0 + self.chunk] != -1).any()):
                time.sleep(0.0005)                         # slots still in use by simulators
            if self.stop:
                return
            t1 = time.time()
            noise, k = self.make(idx, dev)
            self.ring[s0:s0 + self.chunk].copy_(noise)
            for d in {dev, self.ring.device}:
                if d.type == "cuda":
                    torch.cuda.synchronize(d)              # copy finished before the slots are flagged
            self.kbuf[s0:s0 + self.chunk] = k.cpu()
            self.seq[s0:s0 + self.chunk] = torch.tensor(idx, dtype=torch.int64)
            self.written[c % G] = c
            st = self.stats[t]                             # [chunks, s making + copying, s waiting for slots]
            st[0] += 1
            st[1] += time.time() - t1
            st[2] += t1 - t0

    def start(self, threads_per_device=1):
        self._mag = {d: torch.from_numpy(self.mag).to(d) for d in self.devices}
        self.threads_planned = list(range(len(self.devices) * threads_per_device))
        self.stats = [[0, 0.0, 0.0] for _ in self.threads_planned]
        for t in self.threads_planned:
            th = threading.Thread(target=self._run, args=(t,), daemon=True)
            th.start()
            self.threads.append(th)

    def close(self):
        self.stop = True
        for th in self.threads:
            th.join(timeout=10)

    def thread_stats(self):
        """Per noise thread: (device, chunks made, seconds making + copying them, seconds waiting for free slots)."""
        return [(str(self.devices[t % len(self.devices)]), *self.stats[t]) for t in self.threads_planned]

    def handles(self):
        """What a simulator process needs (shared tensors)."""
        return self.ring, self.seq, self.kbuf


def take_noise(handles, i, stop=None, timeout_s=900.0):
    """Simulator side: wait for sample i's noise. Returns (noise int16 view, unit, recording row, slot)."""
    ring, seq, kbuf = handles
    s = i % ring.shape[0]
    t0 = time.time()
    while int(seq[s]) != i:
        if stop is not None and stop.value:
            return None
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"no GPU noise for sample {i} after {timeout_s:.0f}s")
        time.sleep(0.0005)
    return ring[s].numpy(), NOISE_I16_STD, int(kbuf[s]), s


def release_noise(handles, slot):
    handles[1][slot] = -1
