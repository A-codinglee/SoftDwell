// scripts/hist_pipeline/grind_cuda.cu
// Dwell-pair histograms in one pass over the traces; same histograms as grind_hist_fast / the reference
// grind_hist_gpu (Grind.cpp semantics). Built and called by scripts/hist_pipeline/gpu_hist_cuda.py.
//
// A trace's transitions are the positions i in [1, T) where (x[i] > thr) != (x[i-1] > thr), plus a pseudo
// transition at 0 for the start of the first dwell. Every three consecutive transitions (p, m, n) give one
// pair of dwells, a = m - p and b = n - m, filed at [bin(above dwell), bin(below dwell)]; a pair with a dwell
// outside the LUT (bin -1) is dropped, and the last (unterminated) dwell never counts.
//
// Kernel 1: each trace is cut into C chunks; one warp scans one chunk, lane k = threshold k, all lanes reading
// the same samples. A lane counts the triples with all three transitions in its chunk into a shared-memory
// histogram, and records its chunk's transition count, first two and last two transitions.
// Kernel 2: one thread per (trace, threshold) walks the chunks in order and adds the triples that span
// chunks (and the one starting at the pseudo transition).
#include <cuda_runtime.h>
#include <stdint.h>

#define WARPS 8

__device__ __forceinline__ void add_pair(int* h, const int* lut, int J, int p, int m, int n, bool a_above) {
    int a = m - p, b = n - m;
    int by = lut[a_above ? a : b], bx = lut[a_above ? b : a];
    if (by >= 0 && bx >= 0) atomicAdd(h + by * J + bx, 1);
}

__global__ void grind_chunks(const int16_t* __restrict__ x, const int* __restrict__ thr, int thr_row_stride,
                             const int* __restrict__ lut, int T, int K, int J, int C, int L,
                             int* __restrict__ hist, int* __restrict__ summ) {
    extern __shared__ int sh[];                      // K * J * J
    const int JJ = J * J;
    const int blocks_per_row = C / WARPS;
    const int row = blockIdx.x / blocks_per_row;
    const int chunk = (blockIdx.x % blocks_per_row) * WARPS + threadIdx.x / 32;
    const int k = threadIdx.x % 32;
    for (int i = threadIdx.x; i < K * JJ; i += blockDim.x) sh[i] = 0;
    __syncthreads();

    const int16_t* xr = x + (size_t)row * T;
    const int start = chunk * L, end = min(T, start + L);
    if (k < K && start < end) {
        const int t = thr[row * thr_row_stride + k];
        int* h = sh + k * JJ;
        bool cur = (start > 0 ? xr[start - 1] : xr[0]) > t;   // state before the chunk
        int cnt = 0, f0 = -1, f1 = -1, q1 = -1, q2 = -1;       // q1 last, q2 second-last transition
        const int nvec = (end - start + 7) / 8;
        const int4* v = reinterpret_cast<const int4*>(xr + start);   // start and T are multiples of 8
        for (int iv = 0; iv < nvec; ++iv) {
            int4 w = __ldg(v + iv);
            const int16_t* s = reinterpret_cast<const int16_t*>(&w);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int i = start + iv * 8 + j;
                const bool st = s[j] > t;
                if (st != cur && i < end) {
                    // dwell (q1, i) has state cur; dwell (q2, q1) has state !cur
                    if (q2 >= 0) add_pair(h, lut, J, q2, q1, i, !cur);
                    q2 = q1; q1 = i;
                    if (cnt == 0) f0 = i; else if (cnt == 1) f1 = i;
                    ++cnt;
                    cur = st;
                }
            }
        }
        int* o = summ + (((size_t)row * K + k) * C + chunk) * 5;
        o[0] = cnt; o[1] = f0; o[2] = f1; o[3] = q2; o[4] = q1;
    } else if (k < K) {
        int* o = summ + (((size_t)row * K + k) * C + chunk) * 5;
        o[0] = 0;
    }
    __syncthreads();
    int* hr = hist + (size_t)row * K * JJ;
    for (int i = threadIdx.x; i < K * JJ; i += blockDim.x)
        if (sh[i]) atomicAdd(hr + i, sh[i]);
}

__global__ void grind_join(const int16_t* __restrict__ x, const int* __restrict__ thr, int thr_row_stride,
                           const int* __restrict__ lut, int B, int T, int K, int J, int C,
                           int* __restrict__ hist, const int* __restrict__ summ) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= B * K) return;
    const int row = id / K, k = id % K;
    const int t = thr[row * thr_row_stride + k];
    const int16_t* xr = x + (size_t)row * T;
    int* h = hist + (size_t)id * J * J;
    int p2 = -1, p1 = 0;                              // last two transitions before the chunk (0 = pseudo)
    for (int c = 0; c < C; ++c) {
        const int* o = summ + ((size_t)id * C + c) * 5;
        const int cnt = o[0];
        if (cnt == 0) continue;
        const int f0 = o[1];
        // the dwell ending at transition m has the state of x[m - 1]
        if (p2 >= 0) add_pair(h, lut, J, p2, p1, f0, xr[p1 - 1] > t);
        if (cnt >= 2) {
            add_pair(h, lut, J, p1, f0, o[2], xr[f0 - 1] > t);
            p2 = o[3]; p1 = o[4];
        } else {
            p2 = p1; p1 = f0;
        }
    }
}

extern "C" int grind_hist_cuda(int device, void* stream, const int16_t* x, const int* thr, int thr_row_stride,
                               const int* lut, int B, int T, int K, int J, int C, int* hist, int* summ) {
    if (K > 32 || C % WARPS || T % 8) return -1;
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) return (int)e;
    cudaStream_t s = (cudaStream_t)stream;
    const int L = ((T + C - 1) / C + 7) / 8 * 8;     // chunk length, multiple of 8
    const size_t shm = (size_t)K * J * J * sizeof(int);
    grind_chunks<<<B * (C / WARPS), 32 * WARPS, shm, s>>>(x, thr, thr_row_stride, lut, T, K, J, C, L, hist, summ);
    grind_join<<<(B * K + 127) / 128, 128, 0, s>>>(x, thr, thr_row_stride, lut, B, T, K, J, C, hist, summ);
    return (int)cudaGetLastError();
}
