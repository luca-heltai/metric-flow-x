# Assembly performance

The first five changes reduce repeated assembly work: reuse each cell's local state;
transfer FE vectors in bulk; reuse the derivative Jacobian until setup, mass
or capacitance changes; insert cell matrices in bulk; gather/scatter residual
cell vectors in bulk. A CTest resource lock also prevents concurrent nested
Ninja writers.

The initial baseline and complete stack were rebuilt on **lnx3**, GCC 13.2,
deal.II 9.8/PETSc, Release (`-O2`, AVX2). Three alternating process pairs
measure ADAN56 and the SI 10k network at **MPI1/2/4/8**, with three warmed
samples per process. Median paired speedups are **1.20–1.34x** for the
residual and **1.13–1.26x** for the state Jacobian. Repeated valid-cache
derivative calls take less than 0.001 ms; first assembly is outside that timing.

![Initial baseline versus all five changes](performance_data/assembly.svg)

All **270 complete residual/Jacobian snapshot pairs** and **32 physical CSVs
(2372 rows)** match exactly; the final runtime passes 26 Release regressions.
The CSV gate excludes execution-time diagnostics. At MPI8 final residual/
Jacobian efficiencies are **91.9%/93.4%** on ADAN56 and **85.7%/78.2%** on
the 10k network. Relative efficiency does not improve uniformly. These are
one-node assembly measurements and short trajectories, not full-cycle solver
or multi-node performance results. [Final results](performance_data/results.json)
retain medians, ranges, paired improvements and the measurement configuration.

## Reuse the measurement tools

`tools/prepare_assembly_benchmarks.py` prepares representative inputs. Archive
each `benchmark_assembly`, matching `libmetric_flow_x.so` and `revision.txt`
in separate before/after directories, then run:

```sh
python3 tools/prepare_assembly_benchmarks.py \
  --adan56-mesh /path/to/56_adnr_new.vtk \
  --network10k-mesh /path/to/network_10000_adnr.vtk --output-dir INPUTS
python3 tools/compare_assembly.py --before BEFORE --after AFTER \
  --inputs INPUTS/adan56.prm INPUTS/network10k.prm --ranks 1 2 4 8 \
  --runs 5 --repetitions 20 --output /tmp/assembly-comparison
python3 tools/plot_assembly_results.py \
  /tmp/assembly-comparison/results.json --output /tmp/assembly.svg
```

Plotting requires Matplotlib. Use `--map-by POLICY` to keep placement on
verified idle cores identical before and after.

Raw logs stay in the local output directory. Commit only the final summary
and figure. Exact snapshot comparison is the default; explicit `--rtol` and
`--atol` permit small floating-point errors in later experiments. Set the
criteria before measuring, report actual errors and verify physical trajectory
changes against the configured solver tolerances.

## Five further assembly changes (6–10)

Cache interior-face topology; gather face local state once; skip inactive FE
supports; reuse vessel material stiffness; reuse cell-Jacobian scratch storage.
Three alternating Release/PETSc pairs on lnx3 compare the consolidated first
five changes with this complete series at MPI1/2/4/8, with 20 repetitions and
verified idle cores. ADAN56 residual/Jacobian speedups are **1.43–1.49x /
1.52–1.62x**; the 10k network reaches **1.13–1.17x / 1.29–1.43x**.

All **270 complete snapshots match byte-for-byte** (109,327,968 entries),
under predeclared 64-epsilon local bounds. The cumulative 32-CSV physical
comparison stays strictly below unchanged solver-weight bounds
(`atol=1e-12`, `rtol=1e-7`), maximum normalized error **0.365**.
These are one-node assembly timings and short physical regressions.

![Five further assembly changes](performance_data/06_10_overall/comparison.svg)

[Final results](performance_data/06_10_overall/results.json) retain all pairs,
ranges, source/build provenance and completion details. Timings use revision
`72345a5`; this consolidation additionally includes the separately tested
square-root-law wave-speed derivative correction from PR #48.

For the scalar-IDA single-vessel inflow fixture, compare physical outputs with:

```sh
python3 tools/compare_inflow_trajectories.py --before BEFORE --after AFTER \
  --policy PREDECLARED.json --output REPORT.json
```

Record the policy before measuring. The tool checks unchanged effective inputs,
reads actual solver weights and propagates bounds in SI units. It rejects
unsupported componentwise tolerances and fixtures; no tolerance overrides exist.

Remaining assembly costs are cells/interior faces on ADAN56 and cells,
junctions/boundaries on the network. Full-solver startup, factorization and
output require separate measurements; see #9, #25 and #26.

## 11. Remove unused face-Jacobian evaluations

Remove unused interior-velocity extraction and wave-speed vectors from boundary
and junction Jacobians. Five alternating Release/PETSc pairs on lnx4 use
20 repetitions, three samples and verified idle cores 8/9 at MPI1/MPI2.
The 10k network Jacobian improves **1.053/1.067x**, with all **10 pairs** faster;
target junction scopes improve **1.117/1.108x**. ADAN56 Jacobians measure
**0.982/1.011x**: the MPI1 decrease tracks unchanged cell/interior controls
(**0.970/0.972x**). Tiny ADAN boundary scopes measure **0.953/0.975x**.
No uniform full-Jacobian or residual improvement is claimed.

All **54 snapshots** and **32 trajectory CSVs / 2372 rows** match exactly;
**29 Release regressions** and cache MPI2/MPI4 checks pass.
[Results](performance_data/11_unused_face_jacobian/results.json) retain all pairs,
ranges, controls and provenance.

![Unused face-Jacobian evaluations](performance_data/11_unused_face_jacobian/comparison.svg)
