#!/bin/bash -l
#SBATCH --job-name=cpu-prof-levels
#SBATCH --nodelist=milan1
#SBATCH --constraint=hwperf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/cpu-prof-levels-%j.out

# CPU profile of the generator workers of train_selfcal_online.py --synth-on-gpu (trace="levels": Gillespie +
# averaged levels in sd_simulate_levels; noise, synthesis and histograms are on the GPU in that mode), as in
# COC_noise5_selfcal_all5/cond_selfcal_all5_gpu2_synthgpu (COC_noise5, all 5 noise files, 30 gen workers).
# milan1 = Zen3 like Alex; library built with -march=znver3, plus -g / frame pointers for perf.
#   sbatch profile_levels_cpu.sh [configs/sim/COC_noise5.json]
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SIM_CONFIG="${1:-configs/sim/COC_noise5.json}"
PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT"
BASE=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks
DRIVER=$BASE/prof_levels/profile_levels.py
OUT=$BASE/cpu_prof_levels_${SLURM_JOB_ID}
mkdir -p "$OUT/tmp"
echo "Node: $(hostname); $(grep -m1 'model name' /proc/cpuinfo); perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
echo "Output dir: $OUT"

export BUILD_DIR="$OUT/build" SIMDWELL_LIB="$OUT/build/libsimdwell.so" PYTHONPATH="$PROJECT_ROOT"
mkdir -p "$BUILD_DIR"
CXX="g++ -g -fno-omit-frame-pointer" MARCH=-march=znver3 \
    apptainer exec -B "$BASE" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

RUN=(apptainer exec -B "$BASE" "$PROJECT_ROOT/pytorch.sif" python3 "$DRIVER" --sim-config "$SIM_CONFIG"
     --tmp-dir "$OUT/tmp")

echo "=== 1. stage timers, P=1 / 30 / 62 (no profiler)"
"${RUN[@]}" --workers 1 30 62 --n 1000

echo "=== 2. perf record P=1, DWARF call graphs"
perf record -F 1999 --call-graph dwarf,16384 -o "$OUT/perf_P1.data" -- "${RUN[@]}" --workers 1 --n 3000
R=(perf report -i "$OUT/perf_P1.data" --stdio)
"${R[@]}" --no-children --sort dso -g none 2>/dev/null | grep -E '^ +[0-9.]+%' > "$OUT/P1_by_dso.txt"
"${R[@]}" --no-children --sort dso,sym -g none --percent-limit 0.2 2>/dev/null | grep -E '^ +[0-9.]+%' \
    | cut -c1-200 > "$OUT/P1_flat.txt"
"${R[@]}" --children --sort sym -g none --percent-limit 1 2>/dev/null | grep -E '^ +[0-9.]+%' \
    | cut -c1-200 > "$OUT/P1_inclusive.txt"
"${R[@]}" --no-children --sort dso -g caller,2,callee,function,percent --percent-limit 1 2>/dev/null \
    > "$OUT/P1_dso_callchains.txt"
"${R[@]}" --no-children --dsos libsimdwell.so --sort srcline -g none --percent-limit 0.2 2>/dev/null \
    | grep -E '^ +[0-9.]+%' > "$OUT/P1_srcline_libsimdwell.txt"
echo "--- by DSO"; cat "$OUT/P1_by_dso.txt"
echo "--- flat";   head -30 "$OUT/P1_flat.txt"
echo "--- libsimdwell by source line"; head -40 "$OUT/P1_srcline_libsimdwell.txt"

echo "=== 3. perf record P=30 (whole node, flat, no call graphs)"
perf record -a -F 999 -o "$OUT/perf_P30.data" -- "${RUN[@]}" --workers 30 --n 1000
perf report -i "$OUT/perf_P30.data" --stdio --no-children --sort dso,sym --percent-limit 0.3 2>/dev/null \
    | grep -E '^ +[0-9.]+%' | cut -c1-200 > "$OUT/P30_flat.txt"
head -30 "$OUT/P30_flat.txt"

echo "=== 4. perf stat P=1 and P=30"
EV=cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-load-misses,cache-misses,page-faults
perf stat -e $EV -o "$OUT/perf_stat_P1.txt"  -- "${RUN[@]}" --workers 1  --n 1000; cat "$OUT/perf_stat_P1.txt"
perf stat -e $EV -o "$OUT/perf_stat_P30.txt" -- "${RUN[@]}" --workers 30 --n 1000; cat "$OUT/perf_stat_P30.txt"

echo "=== 5. likwid P=1 pinned to core 2"
module load likwid/5.4.1
for g in CPI BRANCH CACHE L2CACHE L3CACHE TLB; do
    echo "--- $g"
    likwid-perfctr -C 2 -g $g -O -o "$OUT/likwid_P1_${g}.csv" -- "${RUN[@]}" --workers 1 --n 800 \
        > "$OUT/likwid_P1_${g}.log" 2>&1 || { echo "likwid $g failed"; tail -5 "$OUT/likwid_P1_${g}.log"; continue; }
    sed -n '/^Metric,/,/^TABLE/p' "$OUT/likwid_P1_${g}.csv" | grep -vE '^(Metric|TABLE),' | head -15
done

echo "=== 6. likwid memory bandwidth at P=30 (whole node)"
for g in MEM1 MEM2 L3CACHE; do
    echo "--- $g"
    likwid-perfctr -c 0-63 -g $g -O -o "$OUT/likwid_P30_${g}.csv" -- "${RUN[@]}" --workers 30 --n 1000 \
        > "$OUT/likwid_P30_${g}.log" 2>&1 || { echo "likwid $g failed"; tail -5 "$OUT/likwid_P30_${g}.log"; continue; }
    grep -iE '^(Memory|L3).*STAT|bandwidth.*STAT|miss.*STAT' "$OUT/likwid_P30_${g}.csv" | head -10
done
echo done
