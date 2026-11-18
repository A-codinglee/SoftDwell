#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train_online/logs/slurm-%j.out

# Histogram pipeline on a refreshing buffer: CPUs simulate in the background while the GPU trains.
# What gets simulated is set by a JSON in configs/sim/:
#   sbatch sim_hist_online.sh                                        # configs/sim/COC.json
#   sbatch sim_hist_online.sh --sim-config configs/sim/COCO.json
#   sbatch sim_hist_online.sh --sim-config configs/sim/COCO.json --predict-conditions   # also baseline / amplitude / sigma
# Further options go the same way (e.g. --epochs 60 --buffer-size 50000); with --export=NONE,
# variables set in front of sbatch do not reach the job.
# More simulation speed: each A100 comes with 16 CPU cores, so request more GPUs for their cores
# and still train on one:
#   sbatch --gres=gpu:a100:4 --cpus-per-task=64 sim_hist_online.sh --train-gpus 1 ...
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NCCL_IB_DISABLE=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }
mkdir -p "$PROJECT_ROOT/outputs/hist_train_online/logs"

echo "Node: $(hostname)"

# Build (uses -march=native, so build on the node that runs it). Built on the node's local disk
# instead of scripts/simulation/build, so a full or failing $HOME cannot break the job here.
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.train_hist_online \
        --sim-config configs/sim/COC.json \
        "$@"
