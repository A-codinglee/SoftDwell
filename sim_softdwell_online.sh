#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/softdwell_online/logs/slurm-%j.out

# SoftDwell model (scripts/softdwell_pipeline/train_softdwell_online.py) on freshly simulated traces:
# CPUs simulate raw traces in the background, the GPU(s) build the soft dwell histograms and train.
# Every sample is used once. What gets simulated is set by a JSON in configs/sim/:
#   sbatch sim_softdwell_online.sh                                        # configs/sim/COC.json, K=4, J=24, ConvNeXt V2
#   sbatch sim_softdwell_online.sh --sim-config configs/sim/COCO.json --k 4 --res 24
#   sbatch sim_softdwell_online.sh --backbone inception                  # Inception-ResNet, as train_ram.py
# The SoftDwell layer, not the simulator, is the bottleneck (1 GPU uses ~4 samples/s of the ~40/s per
# core the simulator makes), so more GPUs train faster; all GPUs of the job train (DDP):
#   sbatch --gres=gpu:a100:8 --cpus-per-task=64 sim_softdwell_online.sh ...
# With --export=NONE, variables set in front of sbatch do not reach the job.
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
mkdir -p "$PROJECT_ROOT/outputs/softdwell_online/logs"

echo "Node: $(hostname)"

# Build (uses -march=native, so build on the node that runs it). Built on the node's local disk
# instead of scripts/simulation/build, so a full or failing $HOME cannot break the job here.
export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID:-$$}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

apptainer exec --nv -B "$BUILD_DIR" "$PROJECT_ROOT/pytorch.sif" \
    python3 -m scripts.softdwell_pipeline.train_softdwell_online \
        --sim-config configs/sim/COC.json \
        "$@"
