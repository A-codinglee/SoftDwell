#!/bin/bash -l
#SBATCH --partition=a100mig
#SBATCH --gres=gpu:a100med:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/predict-%j.out
#
# Rates of recorded traces from a two-look model (scripts/hist_pipeline/predict_selfcal.py); the options
# are passed on, e.g.
#   sbatch predict_selfcal.sh --run-dir <run>/seed1 --trace <file.bin> --check 256 --shift-pA 0 1.2
set -euo pipefail
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$HOME/master/SoftDwell"
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" pytorch.sif bash scripts/simulation/build.sh
apptainer exec --nv -B "$BUILD_DIR" -B /home/woody pytorch.sif python3 -m scripts.hist_pipeline.predict_selfcal "$@"
