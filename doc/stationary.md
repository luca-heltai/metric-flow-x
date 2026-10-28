# Stationary blood flow

`blood_flow` supports three values of `Run mode` in `MetricFlowSystem<1, 3>`:

| Value | Calculation |
|---|---|
| `transient` (default) | Integrate the HDG DAE with IDA. |
| `steady network` | Solve the resistive 0D graph used by the initial-state seed, and output its reconstruction. |
| `steady HDG` | Hold every time derivative at zero and solve the complete HDG equations with `IDACalcIC(use_y_dot)`. |

Use the same executable, mesh, physical parameters and output configuration:

```sh
cmake -S . -B build
cmake --build build --target blood_flow
./build/blood_flow build/parameters/stationary.prm
```

The example selects `steady HDG`. Change `Run mode` to `steady network` and
choose a different output directory to compare the two calculations. Both
freeze inflow and other time-dependent data at `IDA parameters / Initial time`
and write one snapshot per refinement cycle. No time integration is performed;
`Final time` may equal `Initial time`. Startup tolerance ramps, file-output
cadence and the IDASolve work limit do not govern these stationary calculations.
Use a fresh output directory to keep results from different runs separate.

## Resistive 0D calculation

`steady network` runs the same coarse-graph solve used by
`Initial state source = steady network`, regardless of the selected initial
state source. It reconstructs pressure, area and velocity on the mesh and sets
capacitor pressure to its resistive equilibrium. It preserves that reconstruction
without applying the HDG trace correction, and never creates an IDA solver.

This mode requires positive viscosity, admissible geometry and RCR/resistive
outlets anchoring each connected component. It uses reference-geometry tube
resistances. The reconstructed fields need not satisfy the compliant HDG
momentum balance or total-head junction equations. Details are in
[the initialization documentation](initialization.md#the-static-0d-seed).

## Complete stationary HDG calculation

`steady HDG` uses `Initial state source` to prepare its Newton starting point.
The 0D seed is useful when available; `rest` is also supported. It sets
`solution_dot = 0` and invokes the IDA wrapper's `reset()` with correction mode
`use_y_dot`, which calls `IDACalcIC` with `IDA_Y_INIT`. All state components,
including capacitor pressures, may change; derivatives remain fixed. There is
no call to `IDASolve` and no temporal relaxation. Nonlinear nonconvergence is an
error rather than a partially valid stationary output.

The solve uses the configured final IDA relative/absolute tolerances, including
componentwise absolute tolerances. The initial step size supplies only CalcIC's
time scale; zero selects one second. The initial-condition Newton iteration
limit remains configurable under `IDA parameters / Initial condition correction
parameters`. A constant custom weight function avoids the installed wrapper's
componentwise-vector minimum defect on empty MPI ranks.

After CalcIC, the executable checks that time has not advanced, no IDA steps
were taken and all derivatives remain zero. It reassembles the full residual
and the state Jacobian, solves `J delta = F(t0,y,0)`, and requires every component
of `delta` to be smaller than its final tolerance weight
`atol_i + rtol*abs(y_i)`. Nonfinite data and nonpositive areas are rejected before
output. This verifies the nonlinear algebraic solve at its configured accuracy;
it is not an estimate of spatial discretization error or of stability of the
equilibrium under time integration.

## Output and tests

Both modes reuse the VTK/PVTU/PVD output and the vessel/optional terminal CSV
formats. Midpoint probes share `Output vessel CSV file` (default
`vessel_probes.csv`) and use SI units with a vessel-ID column; select them with
`Vessel probe IDs`. See [outputs](outputs.md#vessel-csv-probes).
They do not create transient step histories, mass-conservation histories or
IDA integration statistics. `steady HDG` additionally writes
`stationary_statistics.json` with the snapshot time, zero integration-step
count, zero derivative norm and weighted Newton defect. CSV output describes
the last refinement cycle; VTK snapshots are indexed by refinement cycle.

```sh
python3 tests/test_stationary_modes.py --executable build/blood_flow --mpi
```

The regression checks analytical single-tube graph pressures and flows,
resistive/capacitive outlets, refinement, nonzero initial time, frozen
nonconstant inflow, the HDG residual certificate, and MPI partitions including
empty ranks. Existing transient regressions continue to exercise the default
mode.
