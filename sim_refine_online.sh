#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/refine_online/logs/slurm-%j.out

# Refinement by re-simulation (scripts/refine_pipeline/train_refine_online.py), RAFT-style, on top of a
# trained, frozen self-calibrating network (train_selfcal_online.py):
#   sbatch sim_refine_online.sh --stage1-run <selfcal run>/seed1 --output-root outputs/refine_online/<name>
#   sbatch sim_refine_online.sh --stage1-run ... --output-root ... --iters 4 --test-iters 8
# Every refinement iteration re-simulates every sample on the CPUs, so this run is bound by the CPUs, not the
# GPU: each A100 brings 16 cores, so more GPUs in the job (--gres=gpu:a100:4 --cpus-per-task=64) mostly buy
# cores; the trainer itself uses one GPU, all of them build histograms.
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }
mkdir -p "$PROJECT_ROOT/outputs/refine_online/logs"

echo "Node: $(hostname)"

# Build (uses -march=native, so build on the node that runs it), on the node's local disk.
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec --nv -B "$BUILD_DIR" -B /home/woody "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.refine_pipeline.train_refine_online "$@"
