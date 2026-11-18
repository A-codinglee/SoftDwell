#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train_online/logs/eval-%j.out

# Evaluate a finished run on a larger, freshly simulated test set (no training), then plot it.
#   sbatch eval_hist_online.sh outputs/hist_train_online/COCO/seed1            # 10000 test samples
#   sbatch eval_hist_online.sh outputs/hist_train_online/COCO/seed1 20000
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

RUN_DIR=${1:?usage: sbatch eval_hist_online.sh <run-dir> [n-test]}
N_TEST=${2:-10000}

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }

echo "Node: $(hostname)"

# Build (uses -march=native, so build on the node that runs it). Built on the node's local disk
# instead of scripts/simulation/build, so a full or failing $HOME cannot break the job here.
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.eval_hist_online --run-dir "$RUN_DIR" --n-test "$N_TEST"

MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl" apptainer exec "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.plotting.plot_online_results --run-dir "$RUN_DIR" --eval-dir "$RUN_DIR/eval_test$N_TEST"
