#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00
#SBATCH --job-name=adavn-full

set -euo pipefail
# Submit from the repository root. The compiler/deal.II environment must match
# an existing working GPU build. Explicit paths avoid relying on login modules.
: "${DEAL_II_DIR:?Set DEAL_II_DIR to the deal.II CMake directory used by build-gpu}"
: "${CXX:?Set CXX to the MPI C++ compiler used by build-gpu}"
: "${MPIEXEC:?Set MPIEXEC to the matching mpiexec executable}"
BUILD_DIR=${ADAVN_BUILD_DIR:-build-gpu}
BUILD_DIR=$(realpath -m "$BUILD_DIR")
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
cmake -S . -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -Ddeal.II_DIR="$DEAL_II_DIR" -DCMAKE_CXX_COMPILER="$CXX"
cmake --build "$BUILD_DIR" --target blood_flow adavn_mean_flow benchmark_assembly -j8
ADAVN_MEAN_EXECUTABLE="$BUILD_DIR/adavn_mean_flow" tutorials/05_adavn_full/run.sh mean
ADAVN_BLOOD_FLOW_EXECUTABLE="$BUILD_DIR/blood_flow" tutorials/05_adavn_full/run.sh transient
DATA_DIR=${ADAVN_DATA_DIR:-"$PWD/build/adavn_full"}
"${ADAVN_PYTHON:-python3}" tools/adan/benchmark_full.py --executable "$BUILD_DIR/benchmark_assembly" \
  --parameters "$DATA_DIR/assembly.prm" --output-dir "$DATA_DIR/performance" \
  --launcher "$MPIEXEC" --launcher-arg=--bind-to --launcher-arg=core
