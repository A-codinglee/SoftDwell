#!/bin/bash -l
#SBATCH --job-name=sd_100k_benchmark
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/benchmark/test/logs/slurm-%j.out
#SBATCH --chdir=/home/hpc/ihpc/ihpc134h/master/ionchan_pro
#SBATCH --export=NONE

set -euo pipefail

# --- env & threading ---
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# How many GPUs are allocated on this node (SLURM sets this)
NP=${SLURM_GPUS_ON_NODE:-${SLURM_GPUS:-1}}
# Threads per rank (rough rule: cpus-per-task / NP, min 1)
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-32} / NP ))
if [ "$OMP_NUM_THREADS" -lt 1 ]; then export OMP_NUM_THREADS=1; fi
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# NCCL single-node niceties (disable IB if node doesn't have it)
export NCCL_IB_DISABLE=1

# Debug helpers (optional)
export TORCH_SHOW_CPP_STACKTRACES=1
# export CUDA_LAUNCH_BLOCKING=1

# --- checkpoint/resume policy ---
export OUTPUT_ROOT=/home/hpc/ihpc/ihpc134h/master/ionchan_pro/outputs/benchmark/test
export CKPT_DIR="$OUTPUT_ROOT/checkpoints"
export SPLITS_DIR="$OUTPUT_ROOT/splits"
export RESUME=auto

mkdir -p "$OUTPUT_ROOT" "$CKPT_DIR" "$SPLITS_DIR"
nvidia-smi || true

# --- launch ---
apptainer exec --nv ~/master/ionchan_pro/pytorch.sif \
  torchrun --standalone --nproc_per_node="${NP}" -m scripts.benchmark.train
