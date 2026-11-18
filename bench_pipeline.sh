#!/bin/bash -l
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/slurm-%j.out
#
# Fixed pipeline benchmark: the current two-look pipeline (GPU noise, page-locked memory, CPU-built
# first-look histograms) on one fixed case (2026-02-20 held out, seed 1, 4 GPUs), 30 epochs.
# Pass a name and any options to test on top of it; everything else stays the same:
#   sbatch bench_pipeline.sh reference                          # the reference itself (run it 3 times)
#   sbatch bench_pipeline.sh streams --cuda-streams
#   sbatch bench_pipeline.sh batch512 --batch-size 512 --steps-per-epoch 39
#   sbatch bench_pipeline.sh ddp4_b512 --train-gpus 4 --batch-size 512 --steps-per-epoch 39
# Results: logs/benchmarks/pipeline/<name>_<job>/ ; summary.json, and the comparison with the reference
# (logs/benchmarks/pipeline/reference.json, made with bench_summary.py --make-reference) at the end of the log.
set -euo pipefail
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NCCL_IB_DISABLE=1

NAME="${1:?give the benchmark a name}"; shift
PROJECT_ROOT=$HOME/master/SoftDwell
BENCH=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/pipeline
OUT="$BENCH/${NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUT/seed1"
cd "$PROJECT_ROOT"
exec > >(tee "$OUT/seed1/slurm-${SLURM_JOB_ID}.out") 2>&1
echo "Node: $(hostname)"
echo "BENCH OPTIONS: $*"

export BUILD_DIR="${TMPDIR:-/tmp}/simdwell_build_${SLURM_JOB_ID}"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BUILD_DIR" pytorch.sif bash scripts/simulation/build.sh

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 2 \
    > "$OUT/seed1/gpu-util-${SLURM_JOB_ID}.csv" &
SMI_PID=$!
top -b -d 10 -c -w 250 -u "$USER" > "$OUT/seed1/cpu-top-${SLURM_JOB_ID}.txt" &    # CPU use of our processes
TOP_PID=$!
apptainer exec --nv -B "$BUILD_DIR" -B /home/woody pytorch.sif \
    python3 -m scripts.hist_pipeline.train_selfcal_online \
        --sim-config configs/sim/COCO_noise5.json --holdout-noise 2026_02_20 --seed 1 \
        --epochs 30 --noise-on-gpu --output-root "$OUT" "$@"
kill $SMI_PID $TOP_PID 2>/dev/null || true

# the trained model is not needed; the metrics, log and test results stay
rm -f "$OUT/seed1/best_model.pt" "$OUT/seed1/"*_test.npy "$OUT/seed1/noise_spectrum.npy"
apptainer exec -B /home/woody pytorch.sif python3 -m scripts.hist_pipeline.bench_summary "$OUT" \
    --reference "$BENCH/reference.json"
