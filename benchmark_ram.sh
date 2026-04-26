#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:8 -C a100_80    # 8 GPUs on one node
#SBATCH --nodes=1
#SBATCH --ntasks=1              # ONE parent process; mp.spawn does the rest
#SBATCH --cpus-per-task=64      # enough CPU for preload + 8 loaders
#SBATCH --time=24:00:00
#SBATCH --export=NONE

set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8 
# Threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

export NCCL_IB_DISABLE=1
export TORCH_SHOW_CPP_STACKTRACES=1


# # -------------------------------
# # TMPDIR DATASET COPY
# # -------------------------------
# DATA_SRC="/home/woody/mfpb/mfpb003h/time_series_datasets/COCOC_SNR_4_6_100000_kij_100_100000_samples_1000000_lvl_20000_22000_recStepResp_spectralBathNoise_SUMIN.h5"
# DATA_BASENAME=$(basename "$DATA_SRC")
# DATA_DST="$TMPDIR/${DATA_BASENAME}"

# echo "TMPDIR = $TMPDIR"
# echo "Copying dataset:"
# echo "  from: $DATA_SRC"
# echo "  to  : $DATA_DST"

# SECONDS=0
# cp "$DATA_SRC" "$DATA_DST"
# echo "Copy done in $SECONDS seconds."
# echo "Checking copied file:"
# ls -lh "$DATA_DST"

# # Tell Python to use TMPDIR version (TrainConfig should read this via apply_env_overrides)
# export H5_PATH="$DATA_DST"

EXP_NAME=$1
RES=$2
K=$3
BS=$4

echo "Running Python with: Name=$EXP_NAME | Res=$RES | K=$K | BS=$BS"

# -------------------------------
# RUN TRAINING (multi-GPU, shared RAM)
# -------------------------------

apptainer exec --nv ~/master/SoftDwell/pytorch.sif \
  python3 -m scripts.softdwell_pipeline.train_ram \
    --exp-name "$EXP_NAME" \
    --res "$RES" \
    --k "$K" \
    --bs "$BS"
