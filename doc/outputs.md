# Outputs

Output is controlled by `Output filename` and `Output directory` in the `MetricFlowSystem<1, 3>` section. An empty output directory writes into the process working directory; otherwise the implementation prefixes generated names with that directory.

## Visualization files

At each IDA output callback, deal.II `DataOut` writes parallel visualization records:

- one `.vtu` piece per MPI rank;
- a `.pvtu` record describing the pieces;
- a `.pvd` time-series file using the configured output base name.

The visualization vectors include the FE-range solution and computed pressure fields, plus `vessel_id` cell data for joining with terminal metadata. The exact piece suffixes are produced by deal.II and include the output cycle and MPI rank.

## Vessel CSV probes

At setup, the implementation opens one CSV file per vessel at its arc-length midpoint. The current filename pattern is:

```text
HDG_IDA_Vessel_<vessel-id>.csv
```

Rows are appended at IDA output times. The header and columns are emitted by `BloodFlowSystem::open_csv_files()`/`BloodFlowSystem::write_csv_row()` in `source/blood_flow_system.cc`; consumers should treat them as implementation output rather than a stable file-format contract.

## Terminal names and effective parameters

`network_terminals.csv` contains one row per terminal with `boundary_id`,
`vessel_id`, `label`, `R1`, `R2`, `C`, and `P_out` (physical values in SI).
The optional input CSV column `label` supplies the name; a missing or blank
label defaults to `boundary_<boundary_id>`. In VTK mode the same default is
used. Boundary and vessel IDs remain the keys, so labels need not be unique.

Each vessel probe also includes a `boundary_<id>_label` column for every
terminal connected to that vessel. A vessel with no terminal retains its five
numeric columns. Probe filenames keep the existing numeric vessel ID, so
arbitrary label text cannot alter output paths. Visualization `vessel_id` cell
data links to the terminal table. Global mass, solver statistics, and timing
summaries retain their aggregate schemas.

All C++ solver CSV writers use Rapidcsv, including probes, terminal metadata,
`mass_conservation.csv`, `ida_statistics.csv`, and `scaling_timings.csv`.
Commas, quotes, and line breaks in text fields are quoted and escaped by the
library. Each record is flushed progressively with bounded memory; double
values use round-trip precision and a locale-independent decimal point.
Statistics and timings append to existing files, writing a header when the
file is absent or empty. Python tools use the Python CSV library.

## Terminal time series

With `Write terminal CSV files = true`, `terminal_<boundary_id>.csv` records
`time_s,pressure_Pa,flow_m3_s,boundary_id,vessel_id,label`. The optional input
label is carried in each record; unnamed boundaries use `boundary_<id>`,
including `boundary_0` at the inlet. Rapidcsv handles quoting and escaping.
Numeric filenames and ID columns remain the stable keys.

## Global diagnostics

The standalone driver opens `mass_conservation.csv` before time integration
and writes volume and boundary-flow diagnostics at the physical output
cadence. It writes one `ida_statistics.csv` summary per solve and the
`scaling_timings.csv` section timings at the end of the run. MPI diagnostics
are collective; rank zero writes the global files.

### Live IDA progress

`print_current_statistics(ida, output_time)` prints an aligned row on rank zero
at every accepted IDA output callback, including the initial state. This cadence
is independent of `Output time interval`, which continues to throttle physical
files and `ida_step_history.csv`. Set `Print current statistics = false` to
suppress the console table without changing the CSV diagnostics or integration.

The table shows requested output time and current internal IDA time separately
(IDA can integrate past an output and interpolate back), last/minimum/maximum
accepted timestep, last BDF order, and elapsed wall time since initialization
preparation began. Times use enough digits to distinguish closely spaced output
callbacks on long runs. Counter columns show **interval/cumulative** pairs:

- `steps`: accepted internal time steps;
- `Newton`: nonlinear iterations reported by IDA;
- `J`: actual application Jacobian callback invocations;
- `R`: native residual evaluations;
- `LS`: actual application linear-solve callback invocations;
- `Krylov`: inner GMRES iterations (zero for the direct solver);
- `err` and `Nfail`: error-test and nonlinear-convergence failures.

Cumulative counts include initial-condition correction work when IDA performs
it. Rejected trial work contributes iterations/evaluations/failures, but not
accepted-step counts or timestep extrema. The initial row shows `-` for all
step sizes because no temporal step has been accepted, even if CalcIC used an
artificial time scale. Wall time includes initialization and output overhead.

The standalone runner observes each accepted step from its residual and output
callbacks, checks consecutive step numbers, and computes exact accepted-step
extrema and BDF-order counts. `ida_internal_steps.csv` retains the accepted-step
trace. The existing columns of `ida_step_history.csv` retain their order and
file-cadence interval semantics; new columns append cumulative work, min/max
accepted step sizes and `step_extrema_available`. All native step and failure
totals remain independently available in `ida_statistics.csv`.

Library clients using only the original output helpers remain supported.
Without a residual/output poll for every step, exact extrema cannot be recovered:
`step_extrema_available` is false and its CSV extrema are zero (console `-`).
Their BDF histogram keeps the previous grouped attribution to the last observed
order. For exact progress, call
`poll_ida_internal_step(ida, false, true)` from each residual and
`poll_ida_internal_step(ida, true, true)` at each output before recording or
printing. The final argument requests an error if an accepted-step observation
is missing. `current_ida_statistics()` supplies the same native snapshot used
by the console and CSV output.

Regression checks compare console rows, callback work and accepted-step extrema
with the native final counters and full step trace, exercise a cap that binds
on an easy equilibrium, and verify that an excessive minimum causes a demanding
startup to fail rather than silently stepping below it:

```sh
python3 tests/test_ida_progress.py --executable build/blood_flow
```

## Logs and parameter receipt

`ParameterAcceptor::initialize()` writes `last_used_parameters.prm` after reading the input. deal.II logging is controlled by `Verbosity (console depth)`, and rank zero is the visible writer in the MPI run.

## Status

The repository contains example images and reference assets. This page documents file-writing behavior.
