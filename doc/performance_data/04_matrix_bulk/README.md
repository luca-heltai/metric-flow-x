# Insert each cell Jacobian through the native matrix API

Measured baseline `7908e4f954714aa26157ad58dfb3676166525dd4`; measured
candidate `7bfd570ab83ebc786c50647070ec92a55041c478`. Both archives were
compiled and measured on **lnx3**, including the preceding local-state,
bulk-vector and derivative-cache interventions.

The final cell Jacobian scatter previously called the sparse matrix insertion
interface once per local matrix entry. It now calls
`jacobian_matrix.add(ldofs, cell_matrix)`. The installed PETSc wrapper traverses
the same local rows and columns, filters zero entries as before, and submits
each row in one call. Local volume and face arithmetic, trace insertion and
compression remain unchanged. Both application backends provide the overload;
the runtime measurements and regressions here use PETSc.

## State Jacobian timings

Five alternating process pairs per workload; three samples per process,
with two state-Jacobian repetitions and 20 residual/derivative repetitions
per sample after warm-up. MPI timings use the slowest rank, one thread per
process, and core binding. GCC 13.2, deal.II 9.8, PETSc, Release; the configured
flags end with `-O2` and AVX2 options, as recorded in `toolchain.json`.

| Workload | Before Jacobian (ms) | After Jacobian (ms) | Median paired speedup |
|---|---:|---:|---:|
| adan56_mpi1 | 98.038 | 85.955 | 1.135 |
| adan56_mpi2 | 51.821 | 46.877 | 1.119 |
| network10k_mpi1 | 587.857 | 561.946 | 1.054 |
| network10k_mpi2 | 352.059 | 310.201 | 1.140 |

![State Jacobian before and after native cell-matrix insertion](comparison.svg)

The displayed times are medians of process medians; error bars show the range.
The speedup column instead takes the median of the five paired ratios.
Nineteen of twenty Jacobian pairs improved. Residual and derivative operations
are unchanged controls, retained in `timings.jsonl` and `summary.json`.
The derivative cache makes repeated derivative calls reach nanosecond values;
no improvement is claimed for those controls or for end-to-end solve time.

## Numerical and regression gates

- **54 complete snapshot pairs matched exactly**, covering 1,179,248,416 bytes
  per side. Snapshots include every owned residual entry and every stored entry
  of both Jacobian blocks at three nonuniform states, including reversed
  velocity, for both meshes with MPI1/MPI2.
- Both archives passed analytic/CSV inflow validation with MPI1/MPI2.
  **32 trajectory CSVs / 2,372 rows matched exactly**. The comparison excludes
  only `scaling_timings.csv` and the `wall_time` field in IDA statistics;
  physical times and solver counters remain included. The short 0.01 s case
  is a numerical regression, rather than full cardiac-cycle validation.
- **26 Release tests passed BUILD/RUN/DIFF**, followed by 26/26 successful
  CTest checks. The build used a single Ninja graph with four jobs; CTest ran
  with `-j1`. All test and trajectory work followed the timing phase.

`metadata.json` records the measured binary hashes, revisions and host.
`inputs/manifest.json` records input and mesh hashes. Exact measured parameters
are archived as deterministic `inputs/*.prm.gz`; their decompression hashes
match `input_sha256`. Their only difference from the previous campaign inputs
is the output directory, isolated under this lane.

The runtime patch was rebased onto published intervention 3 at
`bcdc829ec5eff0c9cbde691e35bb78b34f746314`, producing runtime commit
`6a4a3b718ca40070aedcda5ec7f4848c02998888`. The only inherited executable-source
difference is declaration alignment with unchanged tokens and line count.
`runtime_rebase_equivalence.json` records the Git trees and whitespace audit.
The complete changed source passed clang-format 16. The runtime was then rebuilt
in the same build directory, and **all three ELF files were byte-identical to
the measured archive**, as recorded in `formatting_binary_equivalence.json`.

Full original logs remain under `build/performance_campaign/lnx3_matrix`.
Committed benchmark excerpts begin at `BENCHMARK` and retain the complete
TimerOutput table. Only trailing whitespace is removed from excerpts;
`log_excerpt_receipts.json` records original and excerpt hashes. The timing,
summary, metadata and snapshot JSON files are byte-for-byte copies of the
original campaign outputs. CI is outside this local report.
