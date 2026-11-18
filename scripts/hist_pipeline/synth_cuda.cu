// scripts/hist_pipeline/synth_cuda.cu
// Trace synthesis on the GPU: the simulator's synthesizeTraceInt16CoreF (Filter.cpp, inv == 1 path) as one
// FIR filter. Built and called by scripts/hist_pipeline/gpu_synth.py.
//
// For each event (level change delta at time t, sample n0 = ceil(t * fs), offset w in [0, 1]) Filter.cpp
// adds delta * ((1 - w) s[m] + w s2[m]) at n0 + m, m >= 0, with s[m] = k[min(m, K-1)], s2[m] = k[min(m+1, K-1)]
// (k the step response, k[0] = 0, k[K-1] = 1). Summed over all events this is
//     y[n] = sum_{m=0}^{K-2} g[m] u[n-m],   g[m] = k[m+1] - k[m],
// with u[n] the noise-free current averaged over the sample interval (n-1, n] (u[n<0] = u[0], the initial
// level). The simulator process computes u (TraceSimulator.simulate_levels), quantized to uint16 between the
// sample's lowest and highest level; here: out[n] = round(y[n] + noise[n] * scale), clamped to int16.
#include <cuda_runtime.h>
#include <stdint.h>

#define TILE 256
#define MAX_TAPS 256

__global__ void synth_traces(const uint16_t* __restrict__ uq, const int16_t* __restrict__ noise,
                             const float* __restrict__ meta, const float* __restrict__ g, int G, int N,
                             int16_t* __restrict__ out) {
    __shared__ float su[TILE + MAX_TAPS];
    __shared__ float sg[MAX_TAPS];
    const int b = blockIdx.y;
    const int n0 = blockIdx.x * TILE;
    const float lo = meta[3 * b], step = (meta[3 * b + 1] - lo) * (1.0f / 65535.0f), scale = meta[3 * b + 2];
    const uint16_t* u = uq + (size_t)b * N;
    for (int i = threadIdx.x; i < G; i += TILE) sg[i] = g[i];
    for (int i = threadIdx.x; i < TILE + G - 1; i += TILE) {     // u[n0 - (G-1) .. n0 + TILE - 1]
        int n = n0 - (G - 1) + i;
        n = n < 0 ? 0 : (n >= N ? N - 1 : n);
        su[i] = lo + (float)u[n] * step;
    }
    __syncthreads();
    const int n = n0 + threadIdx.x;
    if (n >= N) return;
    float y = 0.0f;
    for (int m = 0; m < G; ++m) y += sg[m] * su[threadIdx.x + (G - 1) - m];
    y += (float)noise[(size_t)b * N + n] * scale;
    y = roundf(y);
    out[(size_t)b * N + n] = (int16_t)fminf(fmaxf(y, -32768.0f), 32767.0f);
}

extern "C" int synth_traces_cuda(int device, void* stream, const uint16_t* uq, const int16_t* noise,
                                 const float* meta, const float* g, int G, int B, int N, int16_t* out) {
    if (G < 1 || G > MAX_TAPS) return -1;
    int prev = 0;                                    // called from the trainer's main thread: keep its current device
    cudaError_t e = cudaGetDevice(&prev);
    if (e == cudaSuccess) e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;
    dim3 grid((N + TILE - 1) / TILE, B);
    synth_traces<<<grid, TILE, 0, (cudaStream_t)stream>>>(uq, noise, meta, g, G, N, out);
    e = cudaGetLastError();
    cudaSetDevice(prev);
    return (int)e;
}
