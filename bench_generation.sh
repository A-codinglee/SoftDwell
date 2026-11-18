#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train_online/logs/bench-%j.out

# Short benchmark of the data generation only (no training): time per stage and scaling with
# the number of worker processes.
#   sbatch bench_generation.sh                       # configs/sim/COCO.json
#   sbatch bench_generation.sh configs/sim/COC.json
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }
echo "Node: $(hostname)"; lscpu | grep -E "Model name|^CPU\(s\)|Thread|NUMA node\(s\)"

apptainer exec "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.simulation.bench_generation \
        --sim-config "${1:-configs/sim/COCO.json}" \
        --workers 1 8 16 32 62 --generate-factor 2 1
