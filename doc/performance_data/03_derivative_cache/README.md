# Reuse the derivative Jacobian

Baseline `2042e28a647fff293f633f8d6b4472131b983bd9`; candidate
`7908e4f954714aa26157ad58dfb3676166525dd4`. Both binaries were rebuilt and
measured on **lnx2**. This comparison includes the preceding local-state and
bulk-vector improvements. An earlier interrupted pilot against `965e7fd` is
excluded from this report.

The derivative Jacobian contains only cell mass matrices and terminal
capacitances. Reuse its existing matrix after the first assembly. Parameter
reload, sparsity reconstruction, and mass reconstruction invalidate the cache;
the next call uses the original assembly sequence. No additional matrix is
allocated. The combined IDA Jacobian adds from this matrix without modifying it.

## Repeated assembly timings (valid cache)

Five alternating process pairs per workload; 20 residual/derivative repetitions
and two state-Jacobian repetitions in each of three samples per process. The
reported value is the median of process medians; the graph also shows their
range. MPI measurements use the slowest rank, one thread per process, and core
binding. GCC 13.2, deal.II 9.8, PETSc, Release; `toolchain.json` preserves the
actual flags, including deal.II's final `-O2` and AVX2 options.

| Workload | Before derivative (ms) | Before range (ms) | Cached derivative (ms) |
|---|---:|---:|---:|
| adan56_mpi1 | 4.937 | 4.604–5.733 | <0.001 |
| adan56_mpi2 | 3.137 | 3.057–3.286 | <0.001 |
| network10k_mpi1 | 35.659 | 33.499–36.786 | <0.001 |
| network10k_mpi2 | 22.058 | 20.696–22.837 | <0.001 |

![Repeated calls after warm-up; valid derivative cache](comparison.svg)

The benchmark warms up each operation. These are repeated calls with a valid
cache, rather than first calls following invalidation. The after timings reach
the cost of a function call and cache check; their raw nanosecond values are
retained in `timings.jsonl`, but no large speedup ratio is inferred from them.
The node was busy (`host_state_before.json` and `host_state_after.json`);
residual and state-Jacobian timings are controls, with no speedup claimed for
those unchanged paths. This is an assembly result, not an end-to-end solve
speedup claim.

## Numerical and regression gates

- **54 complete snapshot pairs matched exactly**, 1,179,248,416 bytes on each side.
  They cover every owned residual entry and every stored entry of both Jacobian
  blocks at three nonuniform states, including reversed velocity, for both
  meshes with MPI1/MPI2. Hashes and sizes are in `equivalence.json`.
- **32 trajectory CSVs / 2,372 rows matched exactly** for analytic/CSV
  inflow with MPI1/MPI2. Only `scaling_timings.csv` and the `wall_time` field of
  IDA statistics are excluded. Physical times and solver counters remain in the
  comparison. This exercises a short 0.01 s case, not a full cardiac cycle.
- **26 Release tests passed BUILD/RUN/DIFF**, followed by 26/26 successful CTest
  checks. A single Ninja graph built the tests; CTest ran with `-j1`.
- The new cache regression also passed with MPI2 and MPI4, including empty
  ranks. It checks repeated calls at different states, the analytic factor-two
  mass change after scaling geometry, and rebuilt mesh/terminal capacitance.

Inputs differ from the preceding campaign only in their output directory,
which is private to this remote lane. `inputs/manifest.json` records both input
hashes and unchanged mesh hashes. Binary hashes, revisions, and host details are
in `metadata.json`. CI for a published PR has not been run by this report.

The measured candidate was then rebased onto the published bulk-vector report
without changing executable source or tests. `runtime_rebase_equivalence.json`
records the unchanged Git trees and resulting submission commit. Full original
benchmark logs remain under `build/performance_campaign/lnx2_derivative/`; the
committed excerpts start at BENCHMARK and retain the complete TimerOutput table.
`log_excerpt_receipts.json` records both hashes and the extraction line. The exact measured inputs are
archived as deterministic `inputs/*.prm.gz`; for example, extract with
`gzip -dc inputs/adan56.prm.gz > adan56.prm`. Their decompressed SHA256 hashes
match `input_sha256` in `inputs/manifest.json`.
