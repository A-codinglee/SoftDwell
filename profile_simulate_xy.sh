#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/profile-%j.out

# Profile simulate_xy() on a compute node (no training): time per stage of one training sample,
# alone and with the node as busy as in a real run.
#   sbatch profile_simulate_xy.sh                          # configs/sim/COCO.json
#   sbatch profile_simulate_xy.sh configs/sim/COC.json
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

apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.simulation.profile_simulate_xy \
        --sim-config "${1:-configs/sim/COCO.json}" --predict-conditions \
        --workers 1 16 62 --n 200 --tmp-dir "$BUILD_DIR" "${@:2}"
