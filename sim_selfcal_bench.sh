#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train_online/logs/slurm-%j.out

# Same as sim_selfcal_online.sh, plus a GPU utilization log (logs/benchmarks/gpu-util-<job>.csv). For benchmarks.
# Two-stage self-calibrating network (scripts/hist_pipeline/train_selfcal_online.py) on freshly simulated
# traces: CPUs simulate in the background, the GPU builds the histograms and trains. Every sample is used once.
#   sbatch --gres=gpu:a100:4 --cpus-per-task=64 sim_selfcal_online.sh \
#       --sim-config configs/sim/COCO_noise5.json --holdout-noise 2026_02_20 --output-root <dir>
# Each A100 comes with 16 CPU cores; training uses one GPU, the others are requested for their cores.
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

# GPU utilization every 2 s while training (for benchmarks)
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 2 \
    > /home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/gpu-util-${SLURM_JOB_ID}.csv &
SMI_PID=$!
apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.train_selfcal_online "$@"
kill $SMI_PID 2>/dev/null || true
