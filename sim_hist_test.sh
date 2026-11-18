#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/sim_hist_test/logs/slurm-%j.out

# End-to-end check: build simulator wrapper -> simulate histograms -> train hist pipeline on them.
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }

N_SAMPLES=${N_SAMPLES:-20000}
OUT_ROOT="$PROJECT_ROOT/outputs/sim_hist_test"
H5="${TMPDIR:-$OUT_ROOT}/sim_hist_COC_N${N_SAMPLES}.h5"
mkdir -p "$OUT_ROOT/logs"

echo "Node: $(hostname)"

# 1. Build (uses -march=native, so build on the node that runs it)
apptainer exec "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

# 2. Simulate
apptainer exec "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.simulation.generate_hist_h5 \
        --out "$H5" --n "$N_SAMPLES" --workers "${SLURM_CPUS_PER_TASK:-16}" --topology COC

# 3. Train on the simulated file
export HIST_H5_PATH="$H5"
export HIST_OUTPUT_ROOT="$OUT_ROOT"
export HIST_TOPOLOGY=COC
export HIST_EPOCHS=${HIST_EPOCHS:-30}

apptainer exec --nv "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.hist_pipeline.train_hist
