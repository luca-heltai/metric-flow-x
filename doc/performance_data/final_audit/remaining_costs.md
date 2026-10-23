# Remaining costs and limits after five interventions

This is an evidence review, not a sixth optimization. It incorporates the earlier
analysis note using the completed cumulative audit. Related issues:
[#9](https://github.com/luca-heltai/metric-flow-x/issues/9) (assembly),
[#25](https://github.com/luca-heltai/metric-flow-x/issues/25) (initial conditions),
[#26](https://github.com/luca-heltai/metric-flow-x/issues/26) (MUMPS lifecycle).

## Remaining assembly work

The table divides rounded TimerOutput section totals by call counts in final
candidate MPI1 run 1. Each log has 61 residual and seven Jacobian calls, including
warm-up, with no snapshot evaluations in that run. These approximate per-call
section costs differ from the three-pair medians in `scaling_summary.json`.
Nested scopes overlap; parent and child timers must not be summed blindly.

| Section, approximate ms/call | ADAN56 final | 10k network final |
|---|---:|---:|
| Full residual | 33.6 | 261 |
| Cell residual | 12.1 | 91.8 |
| Interior trace residual | 16.9 | 5.84 |
| Junction residual | 0.42 | 80.2 |
| Boundary residual | 0.38 | 45.7 |
| Ghost-vector update | 0.42 | 2.65 |
| Cell Jacobian block | 49.1 | 293 |
| Interior trace Jacobian block | 33.3 | 8.94 |
| Junction Jacobian block | 0.69 | 150 |
| Boundary Jacobian block | 0.42 | 54.1 |

Evidence: `adan56_mpi1_after_1.log` and `network10k_mpi1_after_1.log`.
On refined ADAN56, interior-face work remains substantial. Verified source still
builds a processed-face set, computes canonical keys, looks up trace maps,
reinitializes face evaluators and extracts values on each call. Their individual
costs versus flux arithmetic require sampling or hardware counters. The earlier
face-cache experiment was rejected for last-bit numerical changes and remains
excluded. On the large network, cell, junction and boundary work instead dominate
the visible section times. Different topology, polynomial degree and refinement
explain why the two workloads need separate profiles; no unmeasured attribution
is assumed.

## Communication and measured scaling

Final residual efficiency at MPI2/4/8 is **101.9%/96.6%/91.9%** for ADAN56 and
**96.0%/90.9%/85.7%** for the 10k network. State-Jacobian efficiency is
**100.8%/96.2%/93.4%** and **83.4%/83.7%/78.2%**, respectively. These are final
three-pair aggregates, normalized by each variant's MPI1 median, rather than
single-run estimates. Above-100% values can reflect cache/timing variability.
The final 10k Jacobian has lower relative efficiency than the baseline (86.9%
at MPI8), even though its absolute latency is lower at every measured rank.
Faster assembly therefore does not imply uniformly higher parallel efficiency.

The ghost timer combines local copies/index handling, compression and imports;
it does not measure communication alone. The accepted bulk-vector intervention
reduced representative 10k MPI1 rounded ghost costs from about 15.9 ms/call in
step 1 to 2.91 ms/call in step 2; these cross-campaign examples support
attribution but the paired intervention data establish acceptance. The final
profile gives about 2.65 ms/call. Remaining ghost imports, compressions, RCR
Allreduce, global setup data/reductions, partition imbalance and memory effects
are plausible limits. Their separate costs and the saturation rank are unknown.
This audit covers one node, not multi-node communication.

## Setup, bootstrap and output

`startup_s` is rank-0 construction/parameter/setup time only. It excludes trace
Newton, IDA consistency correction, integration and physical output; it does not
measure #25's reported expensive bootstrap. Representative final MPI1 profiles
show setup costs of about 0.347 s on ADAN56 (sparsity 0.329 s) and 1.85 s on the
10k network (sparsity 1.62 s). These are nested costs, not additive. Refinement
cycles repeat DoF/sparsity/mass setup and default-state/trace initialization.
Verbosity also enables a finite-difference Jacobian diagnostic before IDA.
Phase-specific timers and iteration counters are still needed.

A real, small 0.01 s final analytic MPI1 regression gives 640 residual calls
costing 0.186 s, 26 Jacobian setup callbacks costing 0.0190 s, 640 linear solve
callbacks costing 0.0101 s, and 11 visualization outputs costing 0.198 s
(~18.0 ms/output), against a 0.569 s process timer. See
`trajectory_analytic1_profile.log`. Visualization is about 35% in this tiny case;
this does not establish ADAN56/10k output cost or confirm #9's historical ~10%
figure. `output_results` allocates/imports two ghosted FE vectors, builds patches
and writes VTU/PVTU/PVD files; CSV and conservation reductions are separate
paths. Output cadence must be controlled in full-solver benchmarks. MPI2 in this
regression selects GMRES, whereas MPI1 selects direct solve: these runs validate
numerics, rather than same-policy solver scaling.

## Actual direct-solver lifecycle

IDA setup constructs one MUMPS wrapper per refreshed Jacobian, then retains it
across RHS callbacks. The installed legacy deal.II header explicitly documents
reuse after the first solve. **640 RHS callbacks are not evidence of 640
factorizations**; there are 26 setup callbacks in the small regression. Runtime
PETSc options can change reuse, and no symbolic/numeric factorization counters
were collected. The first factorization is included in aggregate solve-callback
time; that timer cannot separate factorization from triangular solves.

Trace Newton separately constructs a fresh solver per update with a newly
assembled matrix. Reusing numeric factors across those changed matrices would
generally be invalid. Trace-only/local-block formulations and symbolic reuse
remain #25 questions. #26's explicit initialize/solve contract is a dependency
compatibility follow-up, not an established explanation of current RHS costs.

## Unmeasured limits and stop point

The campaign establishes faster assembly, exact snapshots and short trajectory
regressions. It does not establish full cardiac-cycle speedup, peak memory
reduction, large-network bootstrap/IDA-correction cost, large-network solver or
output balance, isolated MPI costs, runtime MUMPS factorization counts, Trilinos
performance or multi-node scaling. Stop at five accepted interventions and retain
these questions in #9/#25/#26 without adding another optimization candidate.
