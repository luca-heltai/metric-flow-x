# Local validation

Configuration: Linux, deal.II 9.8.0 with SUNDIALS 6.7, PETSc backend,
OpenMPI 5.0.10, DebugRelease. Runs used the Slurm GPU partition's CPU cores.

- Final focused CTest: **16/16 entries passed**, including two build dependency
  entries. Coverage: consistent initialization, CSV output, steady graph,
  wall models and terminal connections; Debug/Release, serial and 2/4 MPI ranks.
- Terminal-connection tests exercise capacitor and noncapacitor venous ports,
  opposite flow orientations, independent compartment balances, analytical
  static pressures/flows, both implicit Jacobians against finite differences,
  recoverable hyperbolicity checks and ranks without cells.
- Python unit suites: **119 checks passed**: full ADAVN/cycles 8, existing ADAN 6,
  inflow comparison 44, ramp outputs 19, assembly comparisons 25, paper runner/
  collection 15, path naming 2.
- Native runtime scripts passed: CSV paths/defaults/disabled output; ten
  stationary cases and two rejection checks; six ADAVN elastic/viscoelastic
  native/IDA cases; IDA progress, step limits and output cadence.
- Ten-cycle simulation: completed to 8 s, all 401 output times present,
  final tolerance ramp factors one. See `cycle_report.json` and `vtk_result.json`.
- Fresh input preparation exactly matches every consumed input digest.
- CPU/MPI benchmark: 1/2/4/8 ranks, three independent repetitions;
  residual norm relative spread 1.8252e-15. Raw logs are archived beside this file.
- clang-format 16.0.6 check passed for changed C++ files; Python compilation,
  shell syntax and staged whitespace checks passed.

The local installation has no PETSc MUMPS support. Parallel full transient
factorization and Trilinos runtime were not measured locally. The CI workflow
retains both backend builds and now compiles the tutorial's mean-flow target
and runs the offline full-network tests. Remote CI status is separate from
these local results.
