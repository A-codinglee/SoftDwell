# Development route

How the project got from SoftDwell to loop refinement. Until 2026-11-18 the code was not committed step by
step, so git holds only the version of that date (tag `route-2026-11-18`). Most earlier steps are options of
that code, so each can be re-run; the table says how. Every run's `train_config.json` records its settings, and
runs started after 2026-11-18 also record the commit (`"git"` block).

Results folders: `outputs/` is in the project, `woody/` is `/home/woody/mfpb/mfpb102h/SoftDwell_outputs`
(symlink `outputs_woody`). Neither is in git.

| # | Step | Code | How to run it today | Results |
|---|---|---|---|---|
| 1 | SoftDwell | `scripts/softdwell_pipeline/` | `sim_softdwell_online.sh` (online); `train_ram.py` (stored data) | `outputs/softdwell_online/` |
| 2 | Histograms from stored data | `hist_pipeline/train_hist.py`, `simulation/generate_hist_h5.py` | `hist_train.sh` | `outputs/hist_train/` |
| 3 | Histograms online (simulate while training) | `hist_pipeline/train_hist_online.py`, `simulation/simulator.py`, `simulation/simdwell_capi.cpp` | `sim_hist_online.sh` | `outputs/hist_train_online/`, `woody/hist_train_online/COCO_noise5` |
| 4 | Self-calibrating stage 1 | `hist_pipeline/train_selfcal_online.py` | `sim_selfcal_online.sh` | `woody/hist_train_online/COCO_noise5_selfcal`, `COC_noise5_selfcal_all5/cond_selfcal_all5` |
| 5a | Fast GPU histograms | `hist_pipeline/gpu_hist_fast.py` | `--hist-impl fast` (default) | `woody/hist_train_online/COCO_noise5_selfcal_fast`, `_pin` |
| 5b | Noise made on the GPU | `hist_pipeline/gpu_noise.py` | `--noise-on-gpu` | `woody/hist_train_online/COCO_noise5_selfcal_gpunoise` |
| 5c | CUDA histogram kernels, 2 GPUs | `hist_pipeline/gpu_hist_cuda.py`, `grind_cuda.cu` | `--hist-impl cuda --hist-gpus 2 --fixed-hist-on-gpu` | `woody/hist_train_online/speedup_check*`, `speedup_comparison_all` |
| 5d | Training traces synthesized on the GPU | `hist_pipeline/gpu_synth.py`, `synth_cuda.cu` | 5c + `--synth-on-gpu` | `woody/hist_train_online/COC_noise5_selfcal_all5/cond_selfcal_all5_gpu2_synthgpu` |
| 6 | Loop refinement (RAFT-style, re-simulation) | `scripts/refine_pipeline/` | `sim_refine_online.sh --stage1-run <selfcal run>/seed1 [--finetune-stage1]` | `outputs/refine_online/` |

Benchmarks and profiling for step 5: `bench_*.sh`, `profile_*.sh`, `hist_pipeline/bench_*.py`,
`hist_pipeline/profile_*.py`.

## Branches

| Branch | For |
|---|---|
| `main` | code that works; results you keep come from it or from a tagged commit |
| `refine-loop` | refinement experiments (step 6) |
| `softdwell` | SoftDwell work (step 1) |
| `gpu-resim` | not created yet: re-simulation on the GPU, if it is done |

Before submitting a job whose result you will keep: commit, push, and if you report it, tag it
(`git tag <name> && git push --tags`).
