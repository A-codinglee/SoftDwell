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
python3 -m scripts.benchmark.eval_from_ckpt \
  --ckpt /home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_3/best_model.pt \
  --outdir /home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/4Det_24Res_3/infer_2/