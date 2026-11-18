// Thin C ABI over the 3D_Fit_unclamped simulator core, loaded from Python via ctypes.
// Follows the same call path as the simulator's own src/simulate_main.cpp
// (Config -> Simulator -> Resampler -> Noise -> Filter), plus the grind
// dwell-histogram backend. The simulator sources are used unmodified.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "Config.hpp"
#include "Dwell.hpp"
#include "Event.hpp"
#include "Filter.hpp"
#include "Grind.hpp"
#include "Noise.hpp"
#include "Resampler.hpp"
#include "Simulator.hpp"

struct SdHandle {
    Config cfg;
    Simulator sim;
    std::vector<float> kernel;
    double dt_kernel{0.0}, fs{0.0}, tmax{0.0};
    size_t N{0};
    grind::GrindParams grind;   // B == 0 if the config has no usable "analysis" block
    std::string err;
    std::string config_json;    // config as the simulator sees it (pA fields already converted to counts)
    double counts_per_pA{0.0};  // 0 if the config has no "amplifier" block

    explicit SdHandle(const std::string& path) : cfg(path), sim(cfg) {
        sim::Resampler r(cfg);
        kernel.assign(r.kernel().begin(), r.kernel().end());
        dt_kernel = r.kernel_dt();
        fs   = cfg.sampling_rate_hz();
        tmax = cfg.t_max_s();
        N    = (size_t)std::ceil(tmax * fs);
        if (cfg.get().contains("analysis"))
            grind = grind::build_grind_params(cfg.get().at("analysis"));
        config_json = cfg.get().dump();
        if (cfg.get().contains("amplifier")) counts_per_pA = cfg.counts_per_pA();
    }
};

namespace {

// Per-thread buffers, reused across calls (same convention as the simulator's hot path).
thread_local std::vector<sim::Event> t_events;
thread_local std::vector<int16_t>    t_trace;
thread_local std::vector<Log2DHist>  t_hists;
thread_local std::vector<float>      t_lev_d, t_lev_b;   // sd_simulate_levels' per-sample sums

// Wall time of the stages of the last call on this thread, in seconds: Gillespie, trace synthesis
// (step-response filtering + noise + int16), histograms. For profiling; three clock reads per call.
thread_local double t_stage_s[3] = {0.0, 0.0, 0.0};

inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Gillespie -> filtered, noisy int16 trace in t_trace. Returns the event count.
// sigma > 0 sets this sample's noise sigma (raw ADC counts); otherwise the config's noise block decides.
// noise != nullptr supplies the noise itself (trace_len int16, zero mean, standard deviation noise_unit),
// scaled here to sigma; the config's noise source is then not used at all.
long simulate_into_buffer(SdHandle* h, int K, const double* Q, const double* levels,
                          uint32_t seed, uint32_t noise_index, double sigma,
                          const int16_t* noise, double noise_unit, float* sigma_out)
{
    std::vector<std::vector<double>> Qm((size_t)K, std::vector<double>((size_t)K));
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < K; ++c) Qm[(size_t)r][(size_t)c] = Q[r * K + c];
    const std::vector<double> L(levels, levels + K);

    const double t0 = now_s();
    h->sim.simulateGillespieWithRatesFiltered(Qm, L, seed, t_events);
    const double t1 = now_s();

    const auto mm = std::minmax_element(L.begin(), L.end());
    sim::NoiseParams np;
    if (noise) {
        if (!(sigma > 0.0) || !(noise_unit > 0.0))
            throw std::runtime_error("supplied noise needs sigma > 0 and noise_unit > 0");
        np.scale    = sigma / noise_unit;
        np.seg_mean = 0.0;
        np.off      = 0;
        np.src      = noise;
        np.result   = sim::NoiseResult{ (float)sigma, 0.0 };
    } else {
        np = sim::computeNoiseParams(h->cfg, h->N, noise_index, /*mpi_rank*/ 0, *mm.second - *mm.first,
                                     /*has_sigma_override*/ sigma > 0.0, sigma);
    }
    if (sigma_out) *sigma_out = np.result.sigma;

    sim::synthesizeTraceNoisyInt16F(t_events, L, h->kernel, h->dt_kernel, h->fs, h->tmax, np, t_trace);
    t_stage_s[0] = t1 - t0;
    t_stage_s[1] = now_s() - t1;
    t_stage_s[2] = 0.0;
    return (long)t_events.size();
}

} // namespace

extern "C" {

SdHandle* sd_create(const char* config_path, char* errbuf, int errlen) {
    try {
        return new SdHandle(config_path);
    } catch (const std::exception& e) {
        if (errbuf && errlen > 0) { std::strncpy(errbuf, e.what(), (size_t)errlen - 1); errbuf[errlen - 1] = '\0'; }
        return nullptr;
    }
}

void sd_destroy(SdHandle* h) { delete h; }

long sd_trace_len(const SdHandle* h) { return (long)h->N; }
int  sd_hist_detectors(const SdHandle* h) { return h->grind.B > 0 ? (int)h->grind.thr.size() : 0; }
int  sd_hist_bins(const SdHandle* h) { return h->grind.B; }
const char* sd_last_error(const SdHandle* h) { return h->err.c_str(); }
const char* sd_config_json(const SdHandle* h) { return h->config_json.c_str(); }
double sd_counts_per_pA(const SdHandle* h) { return h->counts_per_pA; }
// out3: seconds spent in Gillespie, trace synthesis, histograms during the last call on this thread.
void sd_last_timings(double* out3) { for (int k = 0; k < 3; ++k) out3[k] = t_stage_s[k]; }

// Q: K*K row-major rates [1/s]; levels: K raw ADC counts; out: trace_len int16.
// Returns the number of emitted events, or -1 on error (see sd_last_error).
long sd_simulate_trace(SdHandle* h, int K, const double* Q, const double* levels,
                       uint32_t seed, uint32_t noise_index, double sigma,
                       const int16_t* noise, double noise_unit, int16_t* out, float* sigma_out)
{
    try {
        const long n = simulate_into_buffer(h, K, Q, levels, seed, noise_index, sigma, noise, noise_unit, sigma_out);
        std::memcpy(out, t_trace.data(), h->N * sizeof(int16_t));
        return n;
    } catch (const std::exception& e) { h->err = e.what(); return -1; }
}

// The trace of the last sd_simulate_trace / sd_simulate_hist call on this thread: trace_len int16.
// Lets a caller get both the histograms and the trace they came from without simulating twice.
void sd_copy_last_trace(const SdHandle* h, int16_t* out) {
    std::memcpy(out, t_trace.data(), h->N * sizeof(int16_t));
}

// Kernel (step response) the trace synthesis uses: writes up to maxlen float32 values, returns its length.
// Also returns the kernel index step 1/(fs*dt_kernel) as float, as Filter.cpp computes it.
long sd_kernel(const SdHandle* h, float* out, long maxlen, float* inv_out) {
    const long K = (long)h->kernel.size();
    for (long k = 0; k < std::min(K, maxlen); ++k) out[k] = h->kernel[(size_t)k];
    if (inv_out) *inv_out = static_cast<float>(1.0 / (h->fs * h->dt_kernel));
    return K;
}

// Gillespie only, for trace synthesis on the GPU (scripts/hist_pipeline/gpu_synth.py). Writes u_out (trace_len
// uint16): the noise-free current averaged over each sample interval (n-1, n], u[0] the initial level, as
// round((u - lo) / (hi - lo) * 65535) with lo / hi the smallest / largest of the K levels (lohi_out[0..1]).
// Filter.cpp's synthesizeTraceInt16CoreF (inv == 1 path) equals sum_m g[m] u[n-m] (g[m] = kernel[m+1] -
// kernel[m]) plus the noise, before rounding; an event's position within its interval, w, is computed with
// the same float steps as there. Returns the event count, -1 on error.
long sd_simulate_levels(SdHandle* h, int K, const double* Q, const double* levels, uint32_t seed,
                        uint16_t* u_out, double* lohi_out)
{
    try {
        std::vector<std::vector<double>> Qm((size_t)K, std::vector<double>((size_t)K));
        for (int r = 0; r < K; ++r)
            for (int c = 0; c < K; ++c) Qm[(size_t)r][(size_t)c] = Q[r * K + c];
        const std::vector<double> L(levels, levels + K);

        const double t0 = now_s();
        h->sim.simulateGillespieWithRatesFiltered(Qm, L, seed, t_events);
        const double t1 = now_s();

        const size_t N = h->N;
        const double fs = h->fs;
        const float inv = static_cast<float>(1.0 / (fs * h->dt_kernel));
        if (inv != 1.0f) throw std::runtime_error("GPU trace synthesis needs step_response_dt == 1 / sampling rate");
        std::vector<float> lev((size_t)K + 1, 0.0f);   // float level per state, lev[K] = 0 for any other index
        for (int k = 0; k < K; ++k) lev[(size_t)k] = static_cast<float>(L[(size_t)k]);
        auto level = [&](int s) { return lev[(s >= 0 && s < K) ? (size_t)s : (size_t)K]; };
        const auto mm = std::minmax_element(L.begin(), L.end());
        const float lo_f = static_cast<float>(*mm.first), hi_f = static_cast<float>(*mm.second);
        const float q = (hi_f > lo_f) ? 65535.0f / (hi_f - lo_f) : 1.0f;

        // Per sample n: d[n] = sum of the level changes in (n-1, n], b[n] = the same weighted with the part of the
        // interval after each change; then u[n] = (level at n-1) + b[n], the level at n-1 a running sum of d
        // (in float, as Filter.cpp's hold). Two adds per event, no branches on the event pattern. The events and
        // buffers are thread_local: plain pointers here, or every access in the loops goes through a TLS lookup.
        if (t_lev_d.size() != N) { t_lev_d.resize(N); t_lev_b.resize(N); }
        float* d = t_lev_d.data();
        float* bw = t_lev_b.data();
        std::fill(d, d + N, 0.0f);
        std::fill(bw, bw + N, 0.0f);
        const sim::Event* ev = t_events.data();
        const size_t n_ev = t_events.size();
        for (size_t ei = 1; ei < n_ev; ++ei) {
            const double t_evt = ev[ei].time;
            const float delta = level(ev[ei].state) - level(ev[ei - 1].state);
            const int64_t n0 = static_cast<int64_t>(std::ceil(t_evt * fs));
            if (n0 >= (int64_t)N) break;
            // the event's offset in the kernel, as Filter.cpp: idx_n0 = n0 * inv + (-t) * fs * inv, in float
            const float idx0 = static_cast<float>((-t_evt) * fs * inv);
            const float idx_n0 = static_cast<float>(n0) * inv + idx0;
            const float w = std::fmin(1.0f, std::fmax(0.0f, idx_n0));
            d[n0] += delta;
            bw[n0] += delta * w;
        }
        float run = level(n_ev ? ev[0].state : 0);
        for (size_t n = 0; n < N; ++n) {            // d[n] := level at n-1 (the running sum is the only serial part)
            const float dn = d[n];
            d[n] = run;
            run += dn;
        }
        for (size_t n = 0; n < N; ++n) {            // quantize, vectorizable
            const float v = std::fmin(65535.0f, std::fmax(0.0f, (d[n] + bw[n] - lo_f) * q));
            u_out[n] = static_cast<uint16_t>(static_cast<int32_t>(v + 0.5f));
        }
        lohi_out[0] = lo_f;
        lohi_out[1] = hi_f;
        t_stage_s[0] = t1 - t0;
        t_stage_s[1] = now_s() - t1;
        t_stage_s[2] = 0.0;
        return (long)t_events.size();
    } catch (const std::exception& e) { h->err = e.what(); return -1; }
}

// Same simulation, but returns the grind dwell-histogram stack instead of the trace.
// grids: D*B*B float32 (row-major [D,B,B]); n_events: D int64.
long sd_simulate_hist(SdHandle* h, int K, const double* Q, const double* levels,
                      uint32_t seed, uint32_t noise_index, double sigma,
                      const int16_t* noise, double noise_unit,
                      float* grids, long long* n_events, float* sigma_out)
{
    try {
        if (h->grind.B <= 0 || h->grind.thr.empty())
            throw std::runtime_error("config has no usable analysis.grind / analysis.hist2d block");
        const long n = simulate_into_buffer(h, K, Q, levels, seed, noise_index, sigma, noise, noise_unit, sigma_out);

        const double tg = now_s();
        grind::compute_grind_hist_stack(t_trace, h->fs, h->grind, t_hists);
        t_stage_s[2] = now_s() - tg;

        const size_t D = h->grind.thr.size();
        const size_t BB = (size_t)h->grind.B * (size_t)h->grind.B;
        if (t_hists.size() != D) throw std::runtime_error("unexpected histogram stack size");
        for (size_t d = 0; d < D; ++d) {
            if (t_hists[d].grid.size() != BB) throw std::runtime_error("unexpected histogram grid size");
            std::memcpy(grids + d * BB, t_hists[d].grid.data(), BB * sizeof(float));
            if (n_events) n_events[d] = t_hists[d].N_events;
        }
        return n;
    } catch (const std::exception& e) { h->err = e.what(); return -1; }
}

} // extern "C"
