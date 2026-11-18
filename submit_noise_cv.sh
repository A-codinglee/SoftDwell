#!/bin/bash
# Noise cross-validation on configs/sim/COCO_noise5.json (5 noise recordings), one job per run:
#   holdout-<file>: train on the other 4 recordings, test on <file>
#   only-<file>:    train on <file> alone, test on the other 4
# Usage: bash submit_noise_cv.sh [holdout|only|all] [names...]      e.g.  bash submit_noise_cv.sh holdout 2026_02_20
# REUSE=1 bash submit_noise_cv.sh ...   trains from the refreshing buffer (samples drawn several times)
#                                       instead of --no-reuse; results go to COCO_noise5_reuse
# ARCH=selfcal bash submit_noise_cv.sh ...   the two-stage self-calibrating network (sim_selfcal_online.sh,
#                                       always no reuse); results go to COCO_noise5_selfcal
# SUFFIX=_x bash submit_noise_cv.sh ... appended to the results folder and job names, so a repeat of the same
#                                       cases does not overwrite earlier runs (e.g. COCO_noise5_selfcal_x)
# EXTRA="--noise-on-gpu" bash submit_noise_cv.sh ...   further options for the training script
# GPUS=1 bash submit_noise_cv.sh ...    GPUs requested per job (default 4). Training always uses one;
#                                       the others only bring CPU cores for the simulators (16 per GPU)
set -euo pipefail

ROOT=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online
NAMES=(Patch_Noise 2023_02_01 2026_02_20 2026_02_23 2026_02_25)   # pieces of the noise file names

what="${1:-all}"; shift || true
[ $# -gt 0 ] && NAMES=("$@")
DATA_FLAG=--no-reuse; CV_DIR=COCO_noise5; TAG=noreuse
if [ "${REUSE:-0}" = 1 ]; then DATA_FLAG=; CV_DIR=COCO_noise5_reuse; TAG=reuse; fi
GPUS="${GPUS:-4}"
JOB_SCRIPT=sim_hist_online.sh; TIME=12:00:00
RUN_FLAGS="--predict-conditions $DATA_FLAG --train-gpus 1 --n-val 10000 --n-test 10000"
if [ "${ARCH:-}" = selfcal ]; then
    JOB_SCRIPT=sim_selfcal_online.sh; TIME=24:00:00; RUN_FLAGS=""; CV_DIR=COCO_noise5_selfcal; TAG=selfcal
fi
CV_DIR="$CV_DIR${SUFFIX:-}"; TAG="$TAG${SUFFIX:-}"
mkdir -p "$ROOT/logs"
cd "$(dirname "$0")"

submit() {   # $1 = holdout | only, $2 = name
    local flag=--holdout-noise; [ "$1" = only ] && flag=--train-noise
    local out="$ROOT/$CV_DIR/cond_${TAG}_$1-$2"
    mkdir -p "$out/seed1"            # the log goes next to the run's results
    sbatch --job-name="$TAG-$1-$2" --time=$TIME --gres=gpu:a100:$GPUS --cpus-per-task=$((16 * GPUS)) \
        --output="$out/seed1/slurm-%j.out" \
        $JOB_SCRIPT --sim-config configs/sim/COCO_noise5.json $flag "$2" $RUN_FLAGS ${EXTRA:-} \
        --output-root "$out"
}

for n in "${NAMES[@]}"; do
    [ "$what" = only ]    || submit holdout "$n"
    [ "$what" = holdout ] || submit only "$n"
done
