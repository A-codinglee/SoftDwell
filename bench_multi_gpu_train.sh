#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/bench-multigpu-%j.out
# Network time per training step on 1 GPU (batch 128 / 32) and on 4 GPUs with gradient exchange.
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
cd $HOME/master/SoftDwell
apptainer exec --nv pytorch.sif python3 -m scripts.hist_pipeline.bench_multi_gpu_train
