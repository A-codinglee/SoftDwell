#!/bin/bash
# Build libsimdwell.so from the (unmodified) 3D_Fit_unclamped simulator core + our C wrapper.
# Run inside the training container so it links against the same libstdc++:
#   apptainer exec pytorch.sif bash scripts/simulation/build.sh
# Uses -march=native, so rebuild on the machine type you run on (or set MARCH, e.g. MARCH=-march=znver3).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="${SIM_ROOT:-$HERE/../../external/3D_Fit_unclamped}"
BUILD_DIR="${BUILD_DIR:-$HERE/build}"
CXX="${CXX:-g++}"

[ -f "$SIM_ROOT/src/Simulator.cpp" ] || { echo "Simulator sources not found in $SIM_ROOT (set SIM_ROOT)"; exit 1; }
mkdir -p "$BUILD_DIR"

INC="-I$SIM_ROOT/include -I$SIM_ROOT/include/nlohmann"
BASE="-std=c++17 -O3 -fPIC"
# Same numeric flags the simulator's CMakeLists applies to its hot sources.
FAST="${MARCH:--march=native} -mfma -ffast-math -fno-math-errno -fno-trapping-math -funsafe-math-optimizations -funroll-loops"

for f in Config Resampler Utils; do
    $CXX $BASE $INC -c "$SIM_ROOT/src/$f.cpp" -o "$BUILD_DIR/$f.o"
done
for f in Simulator Filter Noise Dwell Grind; do
    $CXX $BASE $FAST $INC -c "$SIM_ROOT/src/$f.cpp" -o "$BUILD_DIR/$f.o"
done
# with the hot sources' flags too: sd_simulate_levels walks every event and repeats Filter.cpp's float steps
$CXX $BASE $FAST $INC -c "$HERE/simdwell_capi.cpp" -o "$BUILD_DIR/simdwell_capi.o"

$CXX -shared -o "$BUILD_DIR/libsimdwell.so" "$BUILD_DIR"/*.o
echo "built $BUILD_DIR/libsimdwell.so (simulator commit: $(git -C "$SIM_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown))"
