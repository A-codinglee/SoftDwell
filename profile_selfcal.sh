#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/profile-selfcal-%j.out

# Profile the two-look network of train_selfcal_online.py on a compute node (no real training run):
# CPU cost per sample, GPU time per training / validation step for 1-4 GPUs building histograms.
#   sbatch profile_selfcal.sh
#   sbatch profile_selfcal.sh -- --batch-size 64        # options of train_selfcal_online.py after --
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }
echo "Node: $(hostname)"

export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.profile_selfcal --tmp-dir "$BUILD_DIR/profile" "$@"
