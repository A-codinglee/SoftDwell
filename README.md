# SoftDwell

Neural-network estimation of ion-channel gating rates (plus baseline, amplitude and noise level) from
single-channel current recordings, trained online on simulated traces.

## Layout

| Path | What it holds |
|---|---|
| `scripts/simulation/` | Python wrapper (`simulator.py`) and C API (`simdwell_capi.cpp`) around the 3D_Fit_unclamped simulator; `build.sh` compiles `libsimdwell.so` |
| `scripts/hist_pipeline/` | dwell-time histogram networks trained online: `train_hist_online.py`, the self-calibrating `train_selfcal_online.py`, GPU histogram / noise / trace synthesis code |
| `scripts/refine_pipeline/` | RAFT-style refinement by re-simulation on top of a self-calibrating stage 1 (`train_refine_online.py`), inference on recordings, diagnostics and comparison plots |
| `scripts/softdwell_pipeline/` | SoftDwell network and its online trainer |
| `scripts/plotting/` | result and comparison plots |
| `configs/sim/` | simulator configs (COC, COCO, OCO; noise-file sets) |
| `*.sh` (top level) | Slurm job scripts for the NHR@FAU Alex cluster |

Results, checkpoints, noise files and the container image (`pytorch.sif`) are not in the repository.
How the pipelines developed, how to re-run each step and which branch is for what: [docs/ROUTE.md](docs/ROUTE.md).

## Simulator

The simulator core is a separate repository and is not copied here. Clone it into `external/`:

```bash
git clone -b 3DFit_unclamped_light git@github.com:vatsimanos/3D_Fit_unclamped.git external/3D_Fit_unclamped
git -C external/3D_Fit_unclamped checkout 8a6ec71   # the commit the current results were made with
```

`scripts/simulation/build.sh` builds `libsimdwell.so` from it (set `SIM_ROOT` to use another location); the job
scripts build it on the compute node.

## Running

Every job script runs inside the Apptainer image `pytorch.sif` (definition: `hohd_pytorch.def`), e.g.

```bash
sbatch sim_selfcal_online.sh --sim-config configs/sim/COC_noise5.json --output-root outputs/<name>
sbatch sim_refine_online.sh --stage1-run <selfcal run>/seed1 --output-root outputs/refine_online/<name>
```
