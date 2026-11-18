#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --export=NONE
#SBATCH --output=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res/test/slurm-%j.out
#SBATCH --error=/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COC/1/4Det_24Res/test/slurm-%j.err

set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

PROJECT_ROOT=/home/hpc/mfpb/mfpb102h/master/SoftDwell
cd "$PROJECT_ROOT"

OUTDIR="$PROJECT_ROOT/outputs/COC/1/4Det_24Res/test"
mkdir -p "$OUTDIR"

apptainer exec --nv "$PROJECT_ROOT/pytorch.sif" \
python3 -m scripts.softdwell_pipeline.eval_from_ckpt \
  --ckpt "$PROJECT_ROOT/outputs/COC/1/4Det_24Res/best_model.pt" \
  --h5 /home/woody/mfpb/mfpb003h/time_series_datasets/COC_SNR_1_1000_kij_fixed_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_time_series_unclamped_2026_06_30.h5 \
  --outdir "$OUTDIR" \
  --batch-size 8 \
  --num-workers 0 \
  --topology COC