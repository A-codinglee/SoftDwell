#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:05:00
#SBATCH --export=NONE

apptainer exec --nv ~/master/SoftDwell/pytorch.sif \
python3 -m scripts.hist_pipeline.eval_from_ckpt \
  --ckpt /home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train/COC/SNR_1/K24J24/seed1/best_model.pt \
  --h5 /home/woody/mfpb/mfpb003h/time_series_datasets/COC_SNR_1_1000_kij_fixed_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_grind_K24_range_10us_100ms_J24_unclamped_2026_07_01.h5 \
  --outdir /home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/hist_train/COC/SNR_1/K24J24/test \
  --split-mode full