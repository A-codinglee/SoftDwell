# scripts/simulation/noise.py
"""
"spectral_resample" noise for the light simulator branch, which only ships "recorded_noise".

Same method as Noise.cpp on the simulator's main branch:
  1. Welch-average the magnitude spectrum of the source file (Hann window, per-segment demean).
  2. Per sample: interpolate that spectrum onto an FFT of length >= generate_factor * N,
     give every bin a uniformly random phase, inverse-FFT, keep the first N points, remove the mean.
The one difference: main rounds the FFT length up to a power of two, here it is the next
FFT-friendly length (2,000,000 instead of 2,097,152 for a 10 s trace), which is about twice as fast.
The result is handed to the simulator as a ready-made noise array; the simulator scales it to the
sample's sigma and adds it during trace synthesis.
"""
from typing import Tuple

import numpy as np
import scipy.fft

# The simulator reads noise as int16, so the synthesized noise is stored at this standard
# deviation (in int16 units) and scaled back by sigma / NOISE_I16_STD. Peaks beyond 6.5 sigma clip.
NOISE_I16_STD = 5000.0


def welch_magnitude(source_path: str, avg_seg: int = 131072, avg_overlap: float = 0.5) -> np.ndarray:
    """Averaged magnitude spectrum of an int16 noise file, (avg_seg // 2 + 1,) float64."""
    if avg_seg < 16:
        raise ValueError("avg_seg too small")
    if not 0.0 <= avg_overlap < 1.0:
        raise ValueError("avg_overlap must be in [0,1).")
    src = np.memmap(source_path, dtype=np.int16, mode="r")
    if src.size < avg_seg:
        raise ValueError("Source noise shorter than avg_seg")

    hop = max(1, int(np.floor((1.0 - avg_overlap) * avg_seg)))
    n_segments = (src.size - avg_seg) // hop + 1
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(avg_seg) / (avg_seg - 1)))

    power = np.zeros(avg_seg // 2 + 1, dtype=np.float64)
    for s in range(n_segments):
        seg = np.asarray(src[s * hop: s * hop + avg_seg], dtype=np.float64)
        seg -= seg.mean()
        X = scipy.fft.rfft(seg * win)
        power += X.real ** 2 + X.imag ** 2
    return np.sqrt(power / n_segments)


def target_magnitude(mag: np.ndarray, trace_len: int, generate_factor: int = 2) -> np.ndarray:
    """Interpolate the Welch spectrum onto the synthesis FFT grid. Returns (nfft // 2 + 1,) float32."""
    if generate_factor < 1:
        raise ValueError("generate_factor must be >= 1")
    avg_seg = 2 * (mag.size - 1)
    nfft = scipy.fft.next_fast_len(generate_factor * trace_len, real=True)
    nfft += nfft % 2                                   # keep it even: (nfft // 2 + 1) bins <-> nfft points
    # bin k of the target sits at source-bin position k * avg_seg / nfft (the sampling rate cancels)
    pos = np.arange(nfft // 2 + 1, dtype=np.float64) * (avg_seg / nfft)
    mag_tgt = np.interp(pos, np.arange(mag.size, dtype=np.float64), mag)
    mag_tgt[0] = 0.0                                   # no DC
    return mag_tgt.astype(np.float32)


def source_spectra(paths, trace_len: int, avg_seg: int = 131072, avg_overlap: float = 0.5,
                   generate_factor: int = 2) -> np.ndarray:
    """target_magnitude() of every source file, one row per file: (len(paths), nfft // 2 + 1) float32."""
    return np.stack([target_magnitude(welch_magnitude(p, avg_seg, avg_overlap), trace_len, generate_factor)
                     for p in paths])


class SpectralNoise:
    """
    Per-process synthesizer. mag_tgt comes from target_magnitude() (one source) or
    source_spectra() (one row per source).
    """

    def __init__(self, mag_tgt: np.ndarray, trace_len: int):
        self.mag = np.ascontiguousarray(np.atleast_2d(mag_tgt), dtype=np.float32)
        self.nfft = 2 * (self.mag.shape[1] - 1)
        self.trace_len = int(trace_len)
        if self.nfft < self.trace_len:
            raise ValueError("spectrum is too short for this trace length")
        # Work buffers, allocated once: allocating ~40 MB of fresh arrays per call made the
        # synthesis several times slower inside the generator processes.
        self._phi = np.empty(self.mag.shape[1], dtype=np.float32)
        self._Y = np.empty(self.mag.shape[1], dtype=np.complex64)
        self._x = np.empty(self.nfft, dtype=np.float32)
        self._q = np.empty(self.trace_len, dtype=np.int16)

    def synthesize(self, rng: np.random.Generator, source: int = 0) -> Tuple[np.ndarray, float]:
        """
        Noise with the spectrum of source file `source` (row of mag_tgt).
        Returns (noise int16 (trace_len,), unit): zero-mean noise whose standard deviation is
        `unit` int16 steps. Multiply by sigma / unit to get noise of standard deviation sigma.
        The array is an internal buffer, overwritten by the next call.
        """
        phi, Y = self._phi, self._Y
        rng.random(out=phi, dtype=np.float32)
        phi *= np.float32(2.0 * np.pi)
        np.cos(phi, out=Y.real)
        np.sin(phi, out=Y.imag)
        Y *= self.mag[source]
        Y.imag[-1] = 0.0                               # Nyquist bin is real
        np.fft.irfft(Y, n=self.nfft, out=self._x)
        noise = self._x[: self.trace_len]
        noise -= noise.mean()
        noise *= np.float32(NOISE_I16_STD / max(float(noise.std()), 1e-30))
        np.clip(noise, -32767.0, 32767.0, out=noise)
        np.rint(noise, out=noise)
        np.copyto(self._q, noise, casting="unsafe")
        return self._q, NOISE_I16_STD
