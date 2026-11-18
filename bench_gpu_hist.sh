#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/bench-gpuhist-%j.out
# Speed and exactness of the lean GPU histogram routine against the reference (scripts/hist_pipeline/bench_gpu_hist.py).
set -euo pipefail
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT"
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh
apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.bench_gpu_hist --tmp-dir "$BUILD_DIR/bench" "$@"
