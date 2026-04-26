#!/bin/bash

# Usage: ./run.sh <RES> <K> <BS>
# Example: ./run.sh 11 4 32

# 1. Read Arguments (Set defaults if missing)
RES=${1:-11}  # Default 11
K=${2:-4}     # Default 4
BS=${3:-32}   # Default 32

# 2. Auto-Generate Experiment Name
# Example Name: ram_100k_4Det_11Res
EXP_NAME="${K}Det_${RES}Res"

# 3. Setup Directories
BASE_DIR="/home/hpc/mfpb/mfpb102h/master/SoftDwell/outputs/COCO/1p6_3p2"
OUTPUT_ROOT="${BASE_DIR}/${EXP_NAME}"
LOG_DIR="$OUTPUT_ROOT/logs"
CKPT_DIR="$OUTPUT_ROOT/checkpoints"
SPLITS_DIR="${BASE_DIR}/splits"
RESUME=auto

mkdir -p "$OUTPUT_ROOT" "$CKPT_DIR" "$SPLITS_DIR" "$LOG_DIR"

echo "------------------------------------------------"
echo " Experiment : $EXP_NAME"
echo " Parameters : Res=$RES | K=$K | BS=$BS"
echo " Output Root: $OUTPUT_ROOT"
echo " Logs Dir   : $LOG_DIR"
echo " Checkpts   : $CKPT_DIR"
echo " Splits Dir : $SPLITS_DIR"
echo "------------------------------------------------"

export OUTPUT_ROOT
export RESUME
# optionally:
export CKPT_DIR="$CKPT_DIR"
export SPLITS_DIR="$SPLITS_DIR"
export DEBUG_SD=0
export DEBUG_RELU=0
# export CUDA_VISIBLE_DEVICES=1

# 4. Submit to SLURM
# We pass the calculated name and flags to submit.sh
sbatch \
    --job-name="$EXP_NAME" \
    --output="${LOG_DIR}/slurm-%j.out" \
    --export=ALL,OUTPUT_ROOT="$OUTPUT_ROOT",RESUME="$RESUME",CKPT_DIR="$CKPT_DIR",SPLITS_DIR="$SPLITS_DIR",DEBUG_SD="$DEBUG_SD" \
    benchmark_ram.sh "$EXP_NAME" "$RES" "$K" "$BS"