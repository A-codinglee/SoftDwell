#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1                
#SBATCH --cpus-per-task=32        # Increased to 32 (8*4) to feed 4 GPUs
#SBATCH --time=02:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train/COCO/SNR_5/K24J72/logs/slurm-%j.out

set -euo pipefail

# 1. Environment Setup
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8 

# 2. Project Root
PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT" || { echo "Could not cd to $PROJECT_ROOT"; exit 1; }

# 3. Create Logs
mkdir -p "$PROJECT_ROOT/outputs/hist_train/COCO/SNR_5/K24J72/logs"

# 4. Performance Tuning
#    Pin threads to avoid fighting between processes
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

#    Network/NCCL settings (Safe defaults for single node)
export NCCL_IB_DISABLE=1
export TORCH_SHOW_CPP_STACKTRACES=1

SCRIPT="scripts.hist_pipeline.train_hist"

echo "Node: $(hostname)"
echo "Running module: $SCRIPT on 4 GPUs"

# 5. Execution
#    --nproc_per_node=4 is CRITICAL. Without it, it might run on 1 GPU.
apptainer exec --nv "$PROJECT_ROOT/pytorch.sif" \
    python3 -m "$SCRIPT"