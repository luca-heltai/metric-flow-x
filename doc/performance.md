# Assembly profiling

This is the first measured step for issue #9. Constant parameters now use live
references to their registered map values, removing string lookup from flux and
wave-speed calls while observing parameter reloads. The reload regression checks
the independent density/wave-speed scaling law. The parameter map retains its
nodes for the lifetime of the owning problem.

## Reproduce the assembly measurement

Use the corrected ADAN56 mesh from PR #17 and SI tree data from PR #20. Generated
datasets and benchmark outputs remain outside Git. The input generator records
source revision and SHA-256 hashes of the exact meshes and parameter files.

```sh
cmake -S . -B build-benchmark -GNinja \
  -DCMAKE_CXX_COMPILER=mpicxx -DCMAKE_BUILD_TYPE=Release
cmake --build build-benchmark --target benchmark_assembly -j 4
python3 tools/prepare_assembly_benchmarks.py \
  --adan56-mesh notebooks/56_adnr_new.vtk \
  --network10k-mesh notebooks/Networks-by-svv/trees/network_10000_adnr.vtk \
  --output-dir /tmp/metric-flow-benchmarks
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -n 1 \
  build-benchmark/benchmark_assembly /tmp/metric-flow-benchmarks/adan56.prm 10
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -n 2 \
  build-benchmark/benchmark_assembly /tmp/metric-flow-benchmarks/adan56.prm 10
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 mpirun -n 1 \
  build-benchmark/benchmark_assembly /tmp/metric-flow-benchmarks/network10k.prm 10
```

The generator retains the ADAN56 tutorial's degree 2 and five refinements
(2464 cells, 24671 unknowns). The 10k-tree benchmark explicitly uses degree 1
and zero refinements (20001 cells, 170009 unknowns), retaining the SI network's
topology and outlet data. These are distinct workload sizes, not equal-resolution
physical comparisons.

Each process prepares the same nonzero synthetic state without time integration
or a linear solve. A warm-up precedes three samples of residual and state-Jacobian
assembly. Timing includes ghost preparation and MPI communication; each sample
uses the slowest rank. The `BENCHMARK` JSON line records setup time, ranks, problem
size, compiler, deal.II version, min/median/max samples, residual norm and weighted
checksum. The existing section timer supplies more detailed totals. No physical
output files are written, so these measurements do not include output cost.

## Recorded lnx3 experiment

The baseline uses assembly from `b3046ba` (the build-warning fix, without assembly
changes); the constant-reference source change is `37708a1`. Both use GCC 13.2.0,
the same Open MPI wrapper, deal.II 9.8.0, Release libraries, one OpenMP/OpenBLAS
thread, and identical input files. The machine was shared. Times below are
milliseconds per residual assembly, with no CI timing threshold.

| Case | Ranks | Baseline | Constant references |
|---|---:|---:|---:|
| ADAN56 | 1 | 26.60 | 24.05 |
| ADAN56 | 2 | 12.83 | 11.76 |
| SI 10k, median of three process runs | 1 | 240.71 | 197.30 |

The 10k process medians ranged from 201.06–244.10 ms before and 182.41–209.02 ms
after. An earlier paired sample measured 246.02–309.69 ms, showing that the
improvement is not uniform under host variability. Residual norms and weighted
checksums match each corresponding baseline exactly in the recorded runs.
These are assembly measurements; they do not establish an end-to-end speedup,
a constant Jacobian, or a validated circulation trajectory.

## Remaining campaign

Coordinate ownership with Martin before broader optimization work. No competing
optimization PR was open at the initial inspection. The current change is limited
to constant access and a reproducible measurement tool. Profiling production
trajectories, separating solver/output costs, independent full residual/Jacobian
comparisons, connectivity caches and invalidation, reduced value extraction,
geometry reuse and SIMD decisions remain follow-ups. SIMD or a matrix-free
rewrite should follow measurements of the remaining hotspots. The preliminary
10x estimate is not an acceptance result of this change.
