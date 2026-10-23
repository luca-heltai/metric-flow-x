# Measured performance campaign

The five-intervention campaign is tracked in
[issue #36](https://github.com/luca-heltai/metric-flow-x/issues/36), following the
assembly investigation in issue #9. Each accepted intervention has a separate
pull request based on the preceding intervention. Candidates must demonstrate
an improvement and preserve the measured numerical results.

## Measurement and numerical comparison

`benchmark_assembly` measures setup, residual assembly, the state Jacobian, and
the derivative Jacobian separately. Three timed samples follow a warm-up. Each
sample reports elapsed time per call on the slowest MPI rank. The benchmark
does not solve a circulation trajectory or include physical output costs.

An optional third argument supplies a snapshot filename prefix. After timing,
the benchmark assembles three nonuniform states at different times, including
reversed velocity, and writes every locally owned residual entry and every
stored entry of both Jacobian matrices as hexadecimal doubles. Comparisons use
the same MPI process count and numbering before and after an intervention.
Neither a norm nor a weighted checksum alone is the acceptance criterion.

`tools/compare_assembly.py` runs archived before/after executables and libraries
in alternating order with one OpenMP/OpenBLAS thread and MPI processes bound
to cores. It checks exact snapshot equality and records the hashes and sizes
before removing the large snapshot files. It preserves raw JSON timing records,
command lines, build/input hashes, host and CPU metadata, process logs, and SVG
and PNG comparison plots. Plot error bars show the range of independent process
medians, not a confidence interval. Timing is a local experiment, not a CI gate.

To archive a build, copy `benchmark_assembly` and `libmetric_flow_x.so` into a
directory and add a `revision.txt` describing the source revision and any
uncommitted source hash. Archive before rebuilding. `LD_LIBRARY_PATH` selects
the corresponding library; confirm the selection with `ldd` on Linux. Use the
same measurement driver in both archives.

```sh
python3 tools/prepare_assembly_benchmarks.py \
  --adan56-mesh notebooks/56_adnr_new.vtk \
  --network10k-mesh notebooks/networks_by_svv/trees/network_10000_adnr.vtk \
  --output-dir build/performance_campaign/inputs
python3 tools/compare_assembly.py \
  --before build/performance_campaign/baseline \
  --after build/performance_campaign/candidate1 \
  --inputs build/performance_campaign/inputs/adan56.prm \
           build/performance_campaign/inputs/network10k.prm \
  --ranks 1 2 --runs 5 --repetitions 20 \
  --output build/performance_campaign/step1
```

The reporting script requires matplotlib. The solver and its tests do not.
Keep working artifacts on persistent storage when execution may move between
nodes. Do not combine timings from different machines into a paired comparison.

Full trajectory regressions, finite-difference Jacobian tests, external-pressure
oracles, taper tests and MPI tests complement these synthetic snapshots. Exact
agreement on finite workloads does not establish all possible configurations
or physiological validity. Assembly speedups and end-to-end speedups must be
reported separately.

## Intervention 1: reuse local cell state

The cell residual and cell Jacobian previously extracted distributed DoF values
separately for each component, volume and face. They now extract once per cell
and use deal.II's existing `get_function_values_from_local_dof_values()` path.
No interpolation formula, quadrature, flux, or accumulation order is changed.
The numerical source is commit `6c2e6f8`, compared with main `b5d1b62`; both
archives use the same extended benchmark driver.

The accepted experiment ran on lnx3 with GCC 13.2.0, deal.II 9.8.0, PETSc,
Release (`-O3`, AVX2), and five alternating process pairs per case. Each process
uses three timed samples of 20 residual calls. The ADAN56 workload has degree 2,
2464 cells and 24671 unknowns; the SI network has degree 1, 20001 cells and
170009 unknowns. The preliminary lnx2 data are excluded.

| Case | Before residual (ms) | After residual (ms) | Median paired speedup |
|---|---:|---:|---:|
| adan56_mpi1 | 43.962 | 35.506 | 1.124 |
| adan56_mpi2 | 19.806 | 17.974 | 1.111 |
| network10k_mpi1 | 318.719 | 288.133 | 1.107 |
| network10k_mpi2 | 179.748 | 148.003 | 1.224 |

Times are medians of process medians. The paired speedup is the median of the
five before/after ratios, which differs from the ratio of independently taken
medians. Every residual pair improved. The state-Jacobian paired medians improved
by 1.141x/1.047x for ADAN56 MPI1/MPI2 and 1.053x/1.074x for the SI network.
The derivative Jacobian is unchanged code and serves as a variability control:
its ADAN56 MPI1 samples drifted appreciably, while the other three paired
medians are approximately 1.0. No derivative-Jacobian speedup is claimed.

![Before and after local-state reuse](performance_data/01_local_state/comparison.svg)

The [raw records](performance_data/01_local_state/timings.jsonl),
[environment and binary hashes](performance_data/01_local_state/metadata.json),
[input manifest](performance_data/01_local_state/inputs.json), and
[exact comparison receipts](performance_data/01_local_state/equivalence.json)
are committed alongside the figure. All 54 snapshot pairs matched exactly,
covering 1,179,248,416 bytes of residual and matrix data on each side.
The seven focused Release tests passed: coupling readiness, external pressure,
cell Jacobian, taper, terminal parameters, mesh initializer and constant reload.
These tests include independent analytical and finite-difference oracles.
The [test log](performance_data/01_local_state/release_tests.log) records the
actual selection and result.

Both archives also passed the existing analytic/CSV inflow integration check
at MPI1 and MPI2 through 0.01 s with nonzero flow. Direct comparison between
the archives matched all 32 output CSVs (2372 rows) exactly after removing only
`scaling_timings.csv` and the `wall_time` statistics field. Physical time columns
were retained. The [trajectory receipt](performance_data/01_local_state/trajectory_equivalence.json)
records the compared files and row hashes. This is a short numerical regression,
not a full cardiac-cycle validation or an end-to-end timing claim.
