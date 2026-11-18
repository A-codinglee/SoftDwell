#!/bin/bash -l
#SBATCH --job-name=levels-alex
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:15:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/levels-alex-%j.out

# Same CPU-only driver as profile_levels_cpu.sh (generator path of --synth-on-gpu), on an Alex node without
# training, to compare the per-sample cost with milan1 and with the 30 workers inside the training run.
set -euo pipefail
unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PROJECT_ROOT=$HOME/master/SoftDwell; cd "$PROJECT_ROOT"
BASE=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks
OUT=$BASE/levels_alex_${SLURM_JOB_ID}; mkdir -p "$OUT/tmp"
echo "Node: $(hostname); $(grep -m1 'model name' /proc/cpuinfo); cpus: $(taskset -cp $$)"
echo "THP: $(cat /sys/kernel/mm/transparent_hugepage/enabled)"
export BUILD_DIR="$OUT/build" SIMDWELL_LIB="$OUT/build/libsimdwell.so" PYTHONPATH="$PROJECT_ROOT"
mkdir -p "$BUILD_DIR"
apptainer exec -B "$BASE" pytorch.sif bash scripts/simulation/build.sh     # as the training (-march=native)
(sleep 30; grep MHz /proc/cpuinfo | awk '{s+=$4;n++} END {print "mean core clock under load:", s/n, "MHz"}') &
apptainer exec -B "$BASE" pytorch.sif python3 "$BASE/prof_levels/profile_levels.py" \
    --sim-config "${1:-configs/sim/COC_noise5.json}" --tmp-dir "$OUT/tmp" --workers 1 30 --n 1000
