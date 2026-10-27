#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --time=02:00:00
#SBATCH --job-name=adavn-transient

set -euo pipefail
# Submit an already prepared parameter file whose Output directory is fresh.
: "${ADAVN_BLOOD_FLOW_EXECUTABLE:?Set the absolute path to blood_flow}"
: "${ADAVN_PARAMETERS:?Set the absolute path to the prepared transient parameters}"
: "${MPIEXEC:?Set the matching mpiexec executable}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
"$MPIEXEC" --bind-to core --report-bindings -n "$SLURM_NTASKS" \
  "$ADAVN_BLOOD_FLOW_EXECUTABLE" "$ADAVN_PARAMETERS"
