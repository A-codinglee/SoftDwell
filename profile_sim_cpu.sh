#!/bin/bash -l
#SBATCH --job-name=cpu-prof-sim
#SBATCH --nodelist=milan1
#SBATCH --constraint=hwperf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:45:00
#SBATCH --export=NONE
#SBATCH --output=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/cpu-prof-sim-%j.out

# CPU profile of the simulator path of one training sample (simulate_xy, two-look as in
# train_selfcal_online.py) on the test cluster (needs -C hwperf for counter access).
# milan1 = Zen3, same micro-architecture as Alex, so the library is built with -march=znver3.
#   sbatch profile_sim_cpu.sh                                   # COCO_noise5, holdout 2026_02_20
#   sbatch profile_sim_cpu.sh configs/sim/COC_noise5.json 2026_02_20
# Output: perf report/annotate + likwid groups in $OUT (see the .out file for the path).
set -euo pipefail

unset SLURM_EXPORT_ENV
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SIM_CONFIG="${1:-configs/sim/COCO_noise5.json}"
HOLDOUT="${2:-2026_02_20}"
N_PERF=1500     # samples for perf record (~30 s at P=1)
N_CNT=400       # samples per likwid group at P=1

PROJECT_ROOT=$HOME/master/SoftDwell
cd "$PROJECT_ROOT"
OUT=/home/woody/mfpb/mfpb102h/SoftDwell_outputs/hist_train_online/logs/benchmarks/cpu_prof_sim_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "Node: $(hostname); $(grep -m1 'model name' /proc/cpuinfo); perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"
echo "Output dir: $OUT"

# --- build: same flags as build.sh (-O3, -ffast-math ...) + debug info and frame pointers for perf
export BUILD_DIR="$OUT/build"
export SIMDWELL_LIB="$BUILD_DIR/libsimdwell.so"
mkdir -p "$BUILD_DIR"
CXX="g++ -g -fno-omit-frame-pointer" MARCH=-march=znver3 \
    apptainer exec -B "$OUT" "$PROJECT_ROOT/pytorch.sif" bash scripts/simulation/build.sh

RUN=(apptainer exec -B "$OUT" "$PROJECT_ROOT/pytorch.sif"
     python3 -m scripts.simulation.profile_simulate_xy --sim-config "$SIM_CONFIG" --two-look
     --holdout-noise "$HOLDOUT" --predict-conditions --tmp-dir "$OUT/tmp")
mkdir -p "$OUT/tmp"

echo "=== 1. stage timers, P=1 and P=62 (no profiler attached)"
"${RUN[@]}" --workers 1 62 --n 300

echo "=== 2. perf record P=1, DWARF call graphs"
perf record -F 1999 --call-graph dwarf,16384 -o "$OUT/perf.data" -- "${RUN[@]}" --workers 1 --n $N_PERF
perf report -i "$OUT/perf.data" --no-children --sort dso,sym --percent-limit 0.3 --stdio 2>/dev/null \
    | grep -v '^$' > "$OUT/report_flat.txt"
perf report -i "$OUT/perf.data" --no-children --sort dso,sym --percent-limit 0.3 -g caller,0.5,callee,function,percent \
    --stdio 2>/dev/null > "$OUT/report_callers.txt"
perf report -i "$OUT/perf.data" --children --sort sym --percent-limit 1 -g none --stdio 2>/dev/null \
    | grep -v '^$' > "$OUT/report_inclusive.txt"
for sym in synthesizeTraceInt16CoreF synthesizeTraceNoisyInt16F simulateGillespieWithRatesFiltered \
           simulateGillespieCore sd_; do
    perf annotate -i "$OUT/perf.data" --stdio -l --percent-type=local-period "$sym" 2>/dev/null \
        > "$OUT/annotate_${sym}.txt" || true
    [ -s "$OUT/annotate_${sym}.txt" ] || rm -f "$OUT/annotate_${sym}.txt"
done
# every libsimdwell symbol, line-level (source:line), in case the hot core got inlined under another name
perf report -i "$OUT/perf.data" --no-children --dsos libsimdwell.so --sort srcline --percent-limit 0.2 \
    --stdio 2>/dev/null | grep -v '^$' > "$OUT/report_srcline_libsimdwell.txt"
echo "--- top of flat profile"; head -45 "$OUT/report_flat.txt"
echo "--- top inclusive"; head -40 "$OUT/report_inclusive.txt"
echo "--- libsimdwell by source line"; head -50 "$OUT/report_srcline_libsimdwell.txt"

echo "=== 3. perf stat P=1"
perf stat -e cycles,instructions,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,dTLB-load-misses,cache-misses \
    -o "$OUT/perf_stat_P1.txt" -- "${RUN[@]}" --workers 1 --n $N_CNT
cat "$OUT/perf_stat_P1.txt"

echo "=== 4. likwid-perfctr groups, P=1 pinned to core 2"
module load likwid/5.4.1
likwid-perfctr -a > "$OUT/likwid_groups.txt" 2>&1 || true
for g in CPI BRANCH CACHE L2CACHE L3CACHE FLOPS_DP FLOPS_SP TLB; do
    echo "--- $g"
    likwid-perfctr -C 2 -g $g -O -o "$OUT/likwid_P1_${g}.csv" -- "${RUN[@]}" --workers 1 --n $N_CNT \
        > "$OUT/likwid_P1_${g}.log" 2>&1 || { echo "likwid $g failed"; tail -5 "$OUT/likwid_P1_${g}.log"; continue; }
    grep -A40 'Metric,' "$OUT/likwid_P1_${g}.csv" | grep -vE '^(Event|Metric),' | head -20
done

echo "=== 5. likwid memory bandwidth / L3 at full load (P=62, whole node, not pinned)"
for g in MEM1 MEM2 L3CACHE; do
    echo "--- $g"
    likwid-perfctr -c 0-63 -g $g -O -o "$OUT/likwid_P62_${g}.csv" -- "${RUN[@]}" --workers 62 --n 1200 \
        > "$OUT/likwid_P62_${g}.log" 2>&1 || { echo "likwid $g failed"; tail -5 "$OUT/likwid_P62_${g}.log"; continue; }
    grep -A60 'STAT' "$OUT/likwid_P62_${g}.csv" | grep -iE 'bandwidth|volume|miss|STAT' | head -20
done
echo done
