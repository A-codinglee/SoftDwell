#!/usr/bin/env python3
# scripts/refine_pipeline/check_recording.py
"""
Before refining a recorded trace: does the simulator reproduce it at all, and is it inside the training ranges?

1. Fit check (posterior predictive). The recording goes through stage 1; R re-simulations at stage 1's estimate
   (training noise files, as the refiner does) give the discrepancy d = mean |H_rec - H_sim| over the 48x12x12
   log10 histograms the refiner sees. The same is done for --n-ref simulated test traces, where also the truth is
   known. If the recording's d sits far above the simulated traces' d(stage 1), the gap is model mismatch (noise,
   filtering, extra states, drift), not parameter error: no refiner can remove it.
   The comparison is made against all reference traces and against the --k-near ones whose stage-1 estimate is
   closest to the recording's (d depends strongly on amplitude / sigma / rates). Per histogram channel it reports
   where the recording deviates more than simulated traces do.
2. Range check. Stage 1's rates / baseline / amplitude / sigma for the recording against the training ranges.
With --hist-run, a one-look train_hist_online model is scored the same way next to stage 1 (its estimate is
re-simulated and compared on the same histograms), on the simulated traces and on the recording.
3. Noise check. The power spectrum of the recording's closed-level stretches (runs of >= --seg points, default 64, within
   min(3 sigma, 0.3 amplitude) of the baseline) is compared in shape (unit variance) with the 5 noise recordings and
   with a re-simulation of the recording treated the same way (so the selection itself is not mistaken for a
   difference). Distance = mean |log10 PSD_a - log10 PSD_b| over 0.1 kHz .. fs/2.

Usage (CPU is enough):
  python3 -m scripts.refine_pipeline.check_recording --stage1-run <selfcal run>/seed1 --trace rec.bin --out <dir>
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import scipy.signal
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline import train_selfcal_online as tso
from scripts.hist_pipeline.train_hist_online import STREAM_TEST, _pool_init, simulate_fixed_set, write_noise_spectra
from scripts.refine_pipeline.resim import STREAM_RESIM, Resimulator, ThetaToSim
from scripts.refine_pipeline.train_refine_online import Stage1, precompute
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config
from scripts.softdwell_pipeline.model import TFExactInceptionResNet


_trapz = getattr(np, "trapezoid", None) or np.trapz


def closed_psd(x, base, amp, sigma, seg, fs):
    """Welch PSD (Hann, seg points) of the closed-level runs of x (counts). Closed dwells are often short, so seg is small (64 points = 1.6 kHz resolution). Returns (f, psd normalized, n_seg, std)."""
    tol = min(3.0 * sigma, 0.3 * amp)
    ok = np.abs(x.astype(np.float64) - base) < tol
    # runs of consecutive closed points, cut into non-overlapping pieces of seg points
    edges = np.flatnonzero(np.diff(np.r_[0, ok.astype(np.int8), 0]))
    pieces = []
    for a, b in zip(edges[::2], edges[1::2]):
        for s in range(a, b - seg + 1, seg):
            pieces.append(x[s:s + seg].astype(np.float64))
    if not pieces:
        return None, None, 0, float("nan")
    P = np.stack(pieces)
    P -= P.mean(axis=1, keepdims=True)
    f, psd = scipy.signal.welch(P, fs=fs, window="hann", nperseg=seg, noverlap=0, axis=1, detrend=False)
    psd = psd.mean(axis=0)
    std = float(P.std())
    return f, psd / _trapz(psd, f), len(pieces), std


class HistNet:
    """
    A train_hist_online.py run (one look, fixed ladder from its sim-config, --predict-conditions) as a second
    estimator: raw traces -> theta in stage 1's space (log10 rates, stage-1-normalized baseline / amplitude / sigma),
    so it is re-simulated and scored exactly like stage 1. Its histograms are built with the GPU code of
    train_selfcal_online at the run's ladder, which gives the simulator's CPU histograms the run was trained on.
    """

    def __init__(self, run_dir, stage1, cond_s1, trace_len, fs, c, device):
        tc = json.load(open(os.path.join(run_dir, "train_config.json")))
        jc = json.load(open(os.path.join(run_dir, "sim_config.json")))
        self.cond = json.load(open(os.path.join(run_dir, "conditions.json")))
        m, g, h2 = tc["model"], jc["analysis"]["grind"], jc["analysis"]["hist2d"]
        if m["name"] != "TFExactInceptionResNet" or self.cond["index"] != [0, 1, 2]:
            raise SystemExit(f"{run_dir}: needs a train_hist_online run with --predict-conditions")
        if m["num_rates"] != stage1.n_rates:
            raise SystemExit(f"{run_dir}: {m['num_rates']} rates, stage 1 has {stage1.n_rates}")
        a = copy.copy(stage1.args)
        a.fixed_k, a.fixed_j, a.fixed_thr_pA = g["count"], h2["bins"], (g["min_pA"], g["max_pA"])
        self.hists = tso.Hists(a, trace_len, fs, c, device)
        self.model = TFExactInceptionResNet(in_channels=m["in_channels"], num_outputs=m["num_outputs"],
                                            use_bn=m["use_bn"], separate_heads=m["separate_heads"],
                                            input_hw=tuple(m["input_hw"])).to(device)
        self.model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
        self.model.eval()
        self.nr = m["num_rates"]
        self.mh, self.sh = np.asarray(self.cond["mean"]), np.asarray(self.cond["std"])
        self.ms, self.ss = np.asarray(cond_s1["mean"]), np.asarray(cond_s1["std"])

    @torch.no_grad()
    def __call__(self, x):
        out = self.model(self.hists.fixed(self.hists.load(x))).float().cpu().numpy().astype(np.float64)
        aux = out[:, self.nr:self.nr + 3] * self.sh + self.mh                  # counts
        th = np.concatenate([out[:, :self.nr], (aux - self.ms) / self.ss], axis=1)
        return torch.from_numpy(th).float()


def file_psd(path, seg, fs, n_points=2_000_000):
    x = np.fromfile(path, dtype="<i2", count=n_points).astype(np.float64)
    P = x[: (x.size // seg) * seg].reshape(-1, seg)
    P -= P.mean(axis=1, keepdims=True)
    f, psd = scipy.signal.welch(P, fs=fs, window="hann", nperseg=seg, noverlap=0, axis=1, detrend=False)
    psd = psd.mean(axis=0)
    return f, psd / _trapz(psd, f)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage1-run", required=True)
    p.add_argument("--trace", required=True, help="raw int16 recording (first trace_len points are used)")
    p.add_argument("--out", required=True)
    p.add_argument("--shift-pA", type=float, nargs="*", default=[0.0], help="pA subtracted from the recording")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--n-ref", type=int, default=256)
    p.add_argument("--reps", type=int, default=4, help="re-simulations per reference trace")
    p.add_argument("--rec-reps", type=int, default=32, help="re-simulations of the recording")
    p.add_argument("--k-near", type=int, default=32)
    p.add_argument("--seg", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--hist-run", default=None,
                   help="a train_hist_online run (same topology) to score the same way, side by side with stage 1")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = os.path.abspath(args.stage1_run)
    config_path = prepare_sim_config(os.path.join(run, "sim_config.json"), os.path.join(args.out, "sim_config.json"),
                                     args.seed)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c, fs = probe.trace_len, sim_cfg.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    raw_cfg = probe.config
    probe.close()
    jcfg = json.load(open(os.path.join(run, "sim_config.json")))
    cond = json.load(open(os.path.join(run, "conditions.json")))
    stage1 = Stage1(run, trace_len, fs, c, device)
    y_dim, nr = stage1.y_dim, stage1.n_rates
    to_sim = ThetaToSim(sim_cfg, cond, nr)
    hist = HistNet(os.path.abspath(args.hist_run), stage1, cond, trace_len, fs, c, device) if args.hist_run else None
    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.out)
    n_cpus = len(os.sched_getaffinity(0))
    workers = args.workers or max(1, n_cpus - 2)
    ctx = mp.get_context("spawn")
    mean, std = np.asarray(cond["mean"]), np.asarray(cond["std"])
    to_pA = lambda th: (th[:, nr:nr + 3].numpy() * std + mean) / c          # baseline, amplitude, sigma in pA
    B = args.batch_size
    t_all = time.time()

    # ---------------- reference: simulated test traces ----------------
    t0 = time.time()
    with ctx.Pool(workers, initializer=_pool_init,
                  initargs=(config_path, sim_cfg, args.seed, spectrum_path, cond, True)) as pool:
        X, y, _, _ = simulate_fixed_set(pool, STREAM_TEST, args.n_ref, (trace_len,), y_dim, torch.int16)
    ref = {}
    for s in range(0, args.n_ref, B):                                      # precompute without ctx (memory)
        out = stage1(X[s:s + B])
        for k, v in zip(("theta0", "ctx", "base", "amp", "h_obs"), out):
            if k != "ctx":
                ref.setdefault(k, []).append(v.float().cpu())
        if hist is not None:
            ref.setdefault("theta_h", []).append(hist(X[s:s + B]))
    ref = {k: torch.cat(v) for k, v in ref.items()}
    del X
    print(f"[Check] {args.n_ref} reference traces simulated + stage 1 in {time.time() - t0:.0f}s ({workers} workers, "
          f"{device})", flush=True)

    resim = Resimulator(ctx, workers, config_path, sim_cfg, spectrum_path, args.seed, to_sim, max(B, args.rec_reps),
                        trace_len)
    R, N, NC = args.reps, args.n_ref, stage1.hist_channels
    d_s1, d_true = torch.zeros(R, N), torch.zeros(R, N)
    ch_s1 = torch.zeros(N, NC)                     # per-channel |H_obs - mean_r H_sim| at stage 1
    d_h = torch.zeros(R, N)                        # the hist-online run's estimate (--hist-run)
    t0 = time.time()
    try:
        for s in range(0, N, B):
            idx = torch.arange(s, min(s + B, N))
            h_obs, base, amp = ref["h_obs"][idx].to(device), ref["base"][idx].to(device), ref["amp"][idx].to(device)
            acc = 0
            for r in range(R):
                stream = STREAM_RESIM["test"] if r == 0 else 100 + r
                hs = stage1.sim_hists(resim(ref["theta0"][idx], stream, idx), base, amp)
                d_s1[r, idx] = (h_obs - hs).abs().mean(dim=(1, 2, 3)).float().cpu()
                acc = acc + hs
                ht = stage1.sim_hists(resim(y[idx].float(), stream, idx), base, amp)
                d_true[r, idx] = (h_obs - ht).abs().mean(dim=(1, 2, 3)).float().cpu()
                if hist is not None:
                    hh = stage1.sim_hists(resim(ref["theta_h"][idx], stream, idx), base, amp)
                    d_h[r, idx] = (h_obs - hh).abs().mean(dim=(1, 2, 3)).float().cpu()
            ch_s1[idx] = (h_obs - acc / R).abs().mean(dim=(2, 3)).float().cpu()
        print(f"[Check] reference re-simulations ({(2 + (hist is not None)) * R * N}) in {time.time() - t0:.0f}s",
              flush=True)
        yr = y[:, :nr].float()
        rae = lambda th: float((th[:, :nr] - yr).abs().add(1e-12).sqrt().mean())
        print(f"[Check] reference traces: rate RAE stage 1 {rae(ref['theta0']):.4f}, median d(stage 1) "
              f"{d_s1.mean(0).median():.5f}, d(truth) {d_true.mean(0).median():.5f}", flush=True)
        if hist is not None:
            print(f"[Check] reference traces: rate RAE hist-online {rae(ref['theta_h']):.4f}, median d(hist-online) "
                  f"{d_h.mean(0).median():.5f}; d(hist) < d(stage 1) in "
                  f"{float((d_h.mean(0) < d_s1.mean(0)).float().mean()) * 100:.0f}% of traces", flush=True)

        # ---------------- the recording ----------------
        raw = np.fromfile(args.trace, dtype="<i2")
        if raw.size < trace_len:
            raise SystemExit(f"{args.trace}: {raw.size} points < {trace_len}")
        report = {"trace": os.path.abspath(args.trace), "stage1_run": run, "n_ref": N, "reps": R,
                  "rec_reps": args.rec_reps, "shifts": []}
        th_ref = ref["theta0"].numpy()
        ref_lvl = to_pA(ref["theta0"])
        rate_lo, rate_hi = to_sim.log_lo, to_sim.log_hi
        rng_b = jcfg["training"]["baseline_pA"]; rng_a = jcfg["training"]["amplitude_pA"]
        rng_s = jcfg["noise"]["sigma_pA"]
        d_s1_m, d_true_m = d_s1.mean(0).numpy(), d_true.mean(0).numpy()
        ch_ref_med = ch_s1.median(dim=0).values.numpy()
        ch_ref_q90 = ch_s1.quantile(0.9, dim=0).numpy()
        rate_names = [nm for i in range(nr // 2) for nm in (f"k{i}->{i + 1}", f"k{i + 1}->{i}")]
        resim_example = None
        for shift in args.shift_pA:
            x = np.clip(raw[:trace_len].astype(np.int32) - int(round(shift * c)), -32768, 32767).astype(np.int16)
            xt = torch.from_numpy(x)[None]
            th0, _, base, amp, h_obs = stage1(xt)
            th0 = th0.cpu()
            Rr = args.rec_reps
            idx = torch.arange(Rr)
            hs = stage1.sim_hists(resim(th0.repeat(Rr, 1), STREAM_RESIM["test"], idx), base.repeat(Rr),
                                  amp.repeat(Rr))
            if resim_example is None:
                resim_example = resim.buf[0].numpy().copy()
            d_rec = (h_obs - hs).abs().mean(dim=(1, 2, 3)).float().cpu().numpy()
            ch_rec = (h_obs[0] - hs.mean(0)).abs().mean(dim=(1, 2)).float().cpu().numpy()
            # self-consistency: one re-simulation as "observed", the others as re-simulations
            d_self = (hs[:1] - hs[1:]).abs().mean(dim=(1, 2, 3)).float().cpu().numpy()

            lvl = to_pA(th0)[0]
            # nearest reference traces by stage-1 estimate (standardized theta, rates + conditions)
            sc = th_ref.std(0) + 1e-9
            near = np.argsort((((th_ref - th0.numpy()) / sc) ** 2).sum(1))[:args.k_near]
            dm = float(d_rec.mean())
            pct_all = float((d_s1_m < dm).mean() * 100)
            pct_near = float((d_s1_m[near] < dm).mean() * 100)
            z_near = float((dm - d_s1_m[near].mean()) / (d_s1_m[near].std() + 1e-12))
            rates = 10 ** th0[0, :nr].numpy()
            flags = []
            for nm, v in zip(rate_names, th0[0, :nr].numpy()):
                if not rate_lo <= v <= rate_hi:
                    flags.append(f"{nm} outside {10 ** rate_lo:g}..{10 ** rate_hi:g} Hz")
            for nm, v, rg in (("baseline", lvl[0], rng_b), ("amplitude", lvl[1], rng_a), ("sigma", lvl[2], rng_s)):
                if not rg["min"] <= v <= rg["max"]:
                    flags.append(f"{nm} {v:.3f} pA outside {rg['min']}..{rg['max']} pA")
            # where does the recording deviate more than simulated traces do?
            ratio = ch_rec / (ch_ref_q90 + 1e-9)
            worst = np.argsort(-ratio)[:6]
            nf = stage1.args.fixed_k
            ch_name = lambda j: (f"fixed ladder thr {j}" if j < nf else f"selfcal ladder thr {j - nf}")

            print(f"\n=== recording, shift {shift:+.2f} pA ===")
            print("  stage 1: " + "  ".join(f"{nm} {v:9.1f} Hz" for nm, v in zip(rate_names, rates))
                  + f" | baseline {lvl[0]:+.3f} amplitude {lvl[1]:.3f} sigma {lvl[2]:.3f} pA "
                    f"(first look baseline {float(base):+.3f}, amplitude {float(amp):.3f})")
            print(f"  ranges: " + ("; ".join(flags) if flags else "all inside the training ranges"))
            print(f"  d(recording, re-sim at stage 1) = {dm:.5f} (rep std {d_rec.std():.5f}, {Rr} reps)")
            print(f"  simulated traces, d(stage 1): median {np.median(d_s1_m):.5f}, 90% {np.quantile(d_s1_m, .9):.5f}, "
                  f"max {d_s1_m.max():.5f}; d(truth) median {np.median(d_true_m):.5f}")
            print(f"  {args.k_near} nearest (by stage-1 estimate): d(stage 1) mean {d_s1_m[near].mean():.5f} "
                  f"+- {d_s1_m[near].std():.5f}, d(truth) mean {d_true_m[near].mean():.5f}; their baseline / amp / "
                  f"sigma {np.median(ref_lvl[near, 0]):+.2f} / {np.median(ref_lvl[near, 1]):.2f} / "
                  f"{np.median(ref_lvl[near, 2]):.2f} pA")
            print(f"  -> recording is above {pct_all:.0f}% of all and {pct_near:.0f}% of the nearest simulated traces "
                  f"(z = {z_near:+.1f} vs the nearest); re-sim vs re-sim (pure noise) {d_self.mean():.5f}")
            print("  channels where the recording deviates most (|H_rec - mean H_sim| / 90% of simulated traces): "
                  + ", ".join(f"{ch_name(j)} {ratio[j]:.1f}x" for j in worst))
            report["shifts"].append({
                "shift_pA": shift, "rates_hz": dict(zip(rate_names, rates.tolist())),
                "baseline_pA": float(lvl[0]), "amplitude_pA": float(lvl[1]), "sigma_pA": float(lvl[2]),
                "range_flags": flags, "d_rec_mean": dm, "d_rec_rep_std": float(d_rec.std()),
                "d_self": float(d_self.mean()), "pct_above_all": pct_all, "pct_above_near": pct_near,
                "z_near": z_near, "d_s1_near_mean": float(d_s1_m[near].mean()),
                "d_true_near_mean": float(d_true_m[near].mean()),
                "channel_ratio": ratio.tolist(), "channel_rec": ch_rec.tolist()})
            if hist is not None:
                th_h = hist(xt)
                hh = stage1.sim_hists(resim(th_h.repeat(Rr, 1), STREAM_RESIM["test"], idx), base.repeat(Rr),
                                      amp.repeat(Rr))
                d_rec_h = (h_obs - hh).abs().mean(dim=(1, 2, 3)).float().cpu().numpy()
                lvl_h, rates_h = to_pA(th_h)[0], 10 ** th_h[0, :nr].numpy()
                d_h_m = d_h.mean(0).numpy()
                print("  hist-online: " + "  ".join(f"{nm} {v:9.1f} Hz" for nm, v in zip(rate_names, rates_h))
                      + f" | baseline {lvl_h[0]:+.3f} amplitude {lvl_h[1]:.3f} sigma {lvl_h[2]:.3f} pA")
                print(f"  d(recording, re-sim at hist-online) = {d_rec_h.mean():.5f} (rep std {d_rec_h.std():.5f}) "
                      f"vs stage 1 {dm:.5f}; on the nearest simulated traces d(hist-online) mean "
                      f"{d_h_m[near].mean():.5f} vs d(stage 1) {d_s1_m[near].mean():.5f}")
                print("  rates hist-online / stage 1: " + "  ".join(f"{nm} {a / b:.2f}x" for nm, a, b in
                                                                  zip(rate_names, rates_h, rates)))
                report["shifts"][-1]["hist"] = {"rates_hz": dict(zip(rate_names, rates_h.tolist())),
                                                "baseline_pA": float(lvl_h[0]), "amplitude_pA": float(lvl_h[1]),
                                                "sigma_pA": float(lvl_h[2]), "d_rec_mean": float(d_rec_h.mean()),
                                                "d_near_mean": float(d_h_m[near].mean())}
            if shift == args.shift_pA[0]:
                rec_state = (x, lvl)
    finally:
        resim.close()
    report["ref"] = {"d_s1_median": float(np.median(d_s1_m)), "d_s1_q90": float(np.quantile(d_s1_m, .9)),
                     "d_true_median": float(np.median(d_true_m)), "channel_median": ch_ref_med.tolist(),
                     "channel_q90": ch_ref_q90.tolist()}

    # ---------------- noise spectrum of the closed level ----------------
    x, lvl = rec_state
    b_c, a_c, s_c = lvl[0] * c, lvl[1] * c, lvl[2] * c
    f, psd_rec, n_rec, std_rec = closed_psd(x, b_c, a_c, s_c, args.seg, fs)
    _, psd_sim, n_sim, std_sim = closed_psd(resim_example, b_c, a_c, s_c, args.seg, fs)
    band = f >= 100
    dist = lambda a, b: float(np.abs(np.log10(a[band]) - np.log10(b[band])).mean())
    files = jcfg["noise"]["source_paths"]
    psd_files = {os.path.basename(pth): file_psd(pth, args.seg, fs)[1] for pth in files}
    print(f"\n=== closed-level noise ({args.seg}-point stretches, shift {args.shift_pA[0]:+.2f} pA) ===")
    print(f"  recording: {n_rec} stretches, std {std_rec / c:.3f} pA (stage-1 sigma {lvl[2]:.3f}); "
          f"its re-simulation: {n_sim} stretches, std {std_sim / c:.3f} pA")
    noise = {"n_rec": n_rec, "std_rec_pA": std_rec / c, "n_sim": n_sim, "std_sim_pA": std_sim / c, "dist": {}}
    if psd_rec is not None and psd_sim is not None:
        print(f"  shape distance recording vs its re-simulation: {dist(psd_rec, psd_sim):.3f} (log10 units)")
        noise["dist"]["resim"] = dist(psd_rec, psd_sim)
        for nm, ps in psd_files.items():
            noise["dist"][nm] = dist(psd_rec, ps)
            print(f"  vs {nm:<45} {dist(psd_rec, ps):.3f}")
        names = list(psd_files)
        between = [dist(psd_files[a], psd_files[b]) for i, a in enumerate(names) for b in names[i + 1:]]
        print(f"  (between the 5 recordings: min {min(between):.3f}, median {np.median(between):.3f}, "
              f"max {max(between):.3f})")
        noise["between_files"] = between
        np.savez(os.path.join(args.out, "psd.npz"), f=f, rec=psd_rec, resim=psd_sim, **psd_files)
    report["noise"] = noise
    if hist is not None:
        report["hist_run"] = os.path.abspath(args.hist_run)
        report["ref"]["d_hist_median"] = float(d_h.mean(0).median())
    np.savez(os.path.join(args.out, "check.npz"), d_s1=d_s1.numpy(), d_true=d_true.numpy(), d_hist=d_h.numpy(),
             theta0_ref=th_ref, y_ref=y.numpy(), ch_s1=ch_s1.numpy())
    json.dump(report, open(os.path.join(args.out, "check.json"), "w"), indent=2)
    print(f"\n[Check] done in {time.time() - t_all:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
