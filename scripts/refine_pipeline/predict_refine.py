#!/usr/bin/env python3
# scripts/refine_pipeline/predict_refine.py
"""
Inference with a trained refinement run (train_refine_online.py) on a recorded trace.

Three estimators, as in the run's test report:
  stage 1     the original stage-1 network (no refinement)
  A           original stage 1 + the phase-A refiner (best_refiner_phaseA.pt; for a run without --finetune-stage1
              this is best_refiner.pt)
  B           fine-tuned stage 1 (best_stage1.pt) + best_refiner.pt         (only for --finetune-stage1 runs)

The recording has no ground truth, so the refinement loop is run --reps times with different re-simulation random
numbers (each copy is one possible run of the loop; their spread shows how much the result depends on the noise of
the re-simulations). Per iteration it reports the median rates over the copies and the discrepancy d the loop sees.
Stop rules as in the test report: fixed (--iters), best-fit (the iterate with the smallest d of its copy).
Finally the chosen estimates are re-scored with --reps fresh re-simulations (random numbers the loop never saw), so
picking the best-fit iterate does not flatter it.

Usage (CPU is enough):
  python3 -m scripts.refine_pipeline.predict_refine --refine-run <refine output>/seed101 --trace rec.bin --out <dir>
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.multiprocessing as mp

from scripts.hist_pipeline import train_selfcal_online as tso
from scripts.hist_pipeline.train_hist_online import write_noise_spectra
from scripts.refine_pipeline.model import SimRefineNet
from scripts.refine_pipeline.resim import STREAM_RESIM, Resimulator, ThetaToSim
from scripts.refine_pipeline.train_refine_online import Stage1
from scripts.simulation.simulator import SampleSpec, TraceSimulator, prepare_sim_config

STREAM_FRESH = 300          # + r: re-scoring streams, disjoint from the loop's (12) and the diagnostics' (100 + r)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--refine-run", required=True, help="the refinement run's seed folder")
    p.add_argument("--trace", required=True, help="raw int16 recording (first trace_len points)")
    p.add_argument("--out", required=True)
    p.add_argument("--shift-pA", type=float, nargs="*", default=[0.0], help="pA subtracted from the recording")
    p.add_argument("--iters", type=int, default=0, help="0 = the run's test_iters")
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    run = os.path.abspath(args.refine_run)
    tc = json.load(open(os.path.join(run, "train_config.json")))
    s1_run, mc = tc["stage1_run"], tc["model"]
    iters = args.iters or tc["refinement"]["test_iters"]
    seed = int(tc.get("seed", 101))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_path = prepare_sim_config(os.path.join(s1_run, "sim_config.json"), os.path.join(args.out, "sim_config.json"),
                                     seed)
    probe = TraceSimulator(config_path)
    sim_cfg = SampleSpec.from_simulator(probe)
    trace_len, c, fs = probe.trace_len, sim_cfg.counts_per_pA, float(probe.config["sampling"]["rate_hz"])
    probe.close()
    cond = json.load(open(os.path.join(s1_run, "conditions.json")))
    stage1 = Stage1(s1_run, trace_len, fs, c, device)
    s1_orig = {k: v.clone() for k, v in stage1.core.state_dict().items()}
    nr, y_dim = stage1.n_rates, stage1.y_dim
    to_sim = ThetaToSim(sim_cfg, cond, nr)
    center, scale = to_sim.theta_center_scale(y_dim)
    refiner = SimRefineNet(stage1.hist_channels, stage1.hist_hw, y_dim, stage1.ctx_dim, center, scale,
                           hidden=mc["hidden"], enc_hidden=mc["enc_hidden"], enc_depth=mc["enc_depth"],
                           kernel_size=mc["kernel_size"], max_step=mc["max_step"]).to(device).eval()
    ft = tc.get("stage1_finetune") is not None
    pa = os.path.join(run, "best_refiner_phaseA.pt")
    variants = [("A", None, pa if ft else os.path.join(run, "best_refiner.pt"))]
    if ft:
        variants.append(("B", os.path.join(run, "best_stage1.pt"), os.path.join(run, "best_refiner.pt")))
    for _, s1p, rp in variants:
        for f in (s1p, rp):
            if f and not os.path.isfile(f):
                raise SystemExit(f"missing {f}")

    spectrum_path = write_noise_spectra(sim_cfg, trace_len, args.out)
    workers = args.workers or max(1, len(os.sched_getaffinity(0)) - 2)
    R = args.reps
    resim = Resimulator(mp.get_context("spawn"), workers, config_path, sim_cfg, spectrum_path, seed, to_sim, R,
                        trace_len)
    mean, std = np.asarray(cond["mean"]), np.asarray(cond["std"])
    rate_names = [nm for i in range(nr // 2) for nm in (f"k{i}->{i + 1}", f"k{i + 1}->{i}")]
    fmt_rates = lambda th: "  ".join(f"{nm} {10 ** v:9.1f}" for nm, v in zip(rate_names, th[:nr]))
    fmt_lvl = lambda th: "b {:+.3f} A {:.3f} s {:.3f}".format(*((th[nr:nr + 3] * std + mean) / c))
    idx = torch.arange(R)

    def disc(theta, base, amp, h_obs, stream):
        hs = stage1.sim_hists(resim(theta, stream, idx), base.expand(R), amp.expand(R))
        return (h_obs - hs).abs().mean(dim=(1, 2, 3)).float().cpu()

    def fresh(theta, base, amp, h_obs):
        """d of each row of theta (R, y_dim) with R fresh re-simulations each -> (R,) mean over them."""
        out = torch.zeros(R)
        for r in range(R):
            out += disc(theta[r:r + 1].expand(R, -1), base, amp, h_obs, STREAM_FRESH + r)
        return out / R

    raw = np.fromfile(args.trace, dtype="<i2")
    if raw.size < trace_len:
        raise SystemExit(f"{args.trace}: {raw.size} points < {trace_len}")
    report = {"refine_run": run, "stage1_run": s1_run, "trace": os.path.abspath(args.trace), "iters": iters,
              "reps": R, "rate_names": rate_names, "shifts": []}
    t0 = time.time()
    try:
        for shift in args.shift_pA:
            x = np.clip(raw[:trace_len].astype(np.int32) - int(round(shift * c)), -32768, 32767).astype(np.int16)
            xt = torch.from_numpy(x)[None]
            print(f"\n=== {os.path.basename(args.trace)}, shift {shift:+.2f} pA | {R} copies of the loop, "
                  f"{iters} iterations; rates in Hz, levels in pA ===", flush=True)
            entry = {"shift_pA": shift, "variants": {}}
            stage1.core.load_state_dict(s1_orig)
            th0, _, base0, amp0, hobs0 = stage1(xt)
            d0 = fresh(th0.cpu().expand(R, -1), base0, amp0, hobs0)
            print(f"  stage 1   {fmt_rates(th0[0].cpu().numpy())} | {fmt_lvl(th0[0].cpu().numpy())} | "
                  f"d {d0.mean():.5f}", flush=True)
            entry["stage1"] = {"theta": th0[0].cpu().tolist(), "d_fresh": float(d0.mean())}
            for name, s1p, rp in variants:
                stage1.core.load_state_dict(torch.load(s1p, map_location=device) if s1p else s1_orig)
                refiner.load_state_dict(torch.load(rp, map_location=device))
                with torch.no_grad():
                    th, ctx, base, amp, h_obs = stage1(xt)
                    ctx, theta = ctx.expand(R, -1), th.expand(R, -1).contiguous()
                    h, inp = refiner.init_state(ctx)
                    traj, ds = [theta.cpu()], []
                    h_obs_r = h_obs.expand(R, -1, -1, -1)
                    for k in range(iters + 1):
                        h_sim = stage1.sim_hists(resim(theta, STREAM_RESIM["test"], idx), base.expand(R),
                                                 amp.expand(R))
                        ds.append((h_obs_r - h_sim).abs().mean(dim=(1, 2, 3)).float().cpu())
                        if k == iters:
                            break
                        h, theta = refiner.step(h, inp, theta, h_obs_r, h_sim)
                        traj.append(theta.cpu())
                traj, ds = torch.stack(traj, 1), torch.stack(ds, 1)              # (R, iters + 1, y), (R, iters + 1)
                print(f"  --- {name}: " + ("fine-tuned stage 1 + refiner" if s1p else "original stage 1 + phase-A refiner"))
                for k in range(iters + 1):
                    med = traj[:, k].median(0).values.numpy()
                    spread = traj[:, k, :nr].std(0).mean().item()
                    print(f"  it{k:<2}{'*' if k == 0 else ' '}  {fmt_rates(med)} | {fmt_lvl(med)} | d {ds[:, k].mean():.5f}"
                          f" | spread over copies {spread:.4f} dec", flush=True)
                best_k = ds.argmin(1)
                th_best = traj[torch.arange(R), best_k]
                th_last = traj[:, -1]
                d_last, d_best = fresh(th_last, base, amp, h_obs), fresh(th_best, base, amp, h_obs)
                d_start = fresh(traj[:, 0], base, amp, h_obs)
                med_l, med_b = th_last.median(0).values.numpy(), th_best.median(0).values.numpy()
                print(f"  fixed {iters}:  {fmt_rates(med_l)} | {fmt_lvl(med_l)}")
                print(f"  best-fit:  {fmt_rates(med_b)} | {fmt_lvl(med_b)} (iterations picked: "
                      f"{np.bincount(best_k.numpy(), minlength=iters + 1).tolist()})")
                print(f"  fresh re-simulations: d(start) {d_start.mean():.5f} -> d(fixed) {d_last.mean():.5f} "
                      f"+- {d_last.std():.5f}, d(best-fit) {d_best.mean():.5f} +- {d_best.std():.5f}")
                print("  change vs stage 1 (median, x): fixed " + "  ".join(
                    f"{nm} {10 ** (a - b):.2f}" for nm, a, b in zip(rate_names, med_l[:nr], th0[0, :nr].cpu().numpy())))
                entry["variants"][name] = {
                    "start_theta": traj[0, 0].tolist(), "d_start_fresh": float(d_start.mean()),
                    "traj_median": traj.median(0).values.tolist(), "d_loop_mean": ds.mean(0).tolist(),
                    "fixed_theta_median": med_l.tolist(), "d_fixed_fresh": float(d_last.mean()),
                    "bestfit_theta_median": med_b.tolist(), "d_bestfit_fresh": float(d_best.mean()),
                    "rates_fixed_hz": dict(zip(rate_names, (10 ** med_l[:nr]).tolist())),
                    "rates_bestfit_hz": dict(zip(rate_names, (10 ** med_b[:nr]).tolist())),
                    "levels_fixed_pA": ((med_l[nr:nr + 3] * std + mean) / c).tolist(),
                    "spread_fixed_dec": traj[:, -1, :nr].std(0).tolist()}
                np.savez(os.path.join(args.out, f"traj_{name}_shift{shift:+.2f}.npz"), traj=traj.numpy(),
                         d=ds.numpy())
            report["shifts"].append(entry)
    finally:
        resim.close()
    json.dump(report, open(os.path.join(args.out, "predict_refine.json"), "w"), indent=2)
    print(f"\n[Predict] {resim.n_traces} re-simulations, {time.time() - t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
