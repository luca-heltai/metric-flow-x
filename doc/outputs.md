# Outputs

Output is controlled by `Output filename` and `Output directory` in the `MetricFlowSystem<1, 3>` section. An empty output directory writes into the process working directory; otherwise the implementation prefixes generated names with that directory.

## Visualization files

At each IDA output callback, deal.II `DataOut` writes parallel visualization records:

- one `.vtu` piece per MPI rank;
- a `.pvtu` record describing the pieces;
- a `.pvd` time-series file using the configured output base name.

The visualization vectors include the FE-range solution and computed pressure fields, plus `vessel_id` cell data for joining with terminal metadata. The exact piece suffixes are produced by deal.II and include the output cycle and MPI rank.

## CSV filenames

The following CSV paths are configurable in `MetricFlowSystem<1, 3>`. An empty
filename disables that file. Relative paths are resolved under
`Output directory`; absolute paths are used directly. Missing parent directories are
created automatically. Existing files are not removed when output is disabled.

| Parameter | Default |
| --- | --- |
| `Output vessel CSV file` | `vessel_probes.csv` |
| `Output terminal CSV file` | empty (disabled) |
| `Output network terminals CSV file` | `network_terminals.csv` |
| `Output mass conservation CSV file` | `mass_conservation.csv` |
| `Output IDA statistics CSV file` | `ida_statistics.csv` |
| `Output IDA step history CSV file` | `ida_step_history.csv` |
| `Output IDA internal steps CSV file` | `ida_internal_steps.csv` |
| `Output scaling timings CSV file` | `scaling_timings.csv` |
| `Output flow regime CSV file` | `flow_regime.csv` |

Flow-regime output additionally requires
`Write flow regime diagnostics = true`, as before; clearing its filename preserves console
warnings while suppressing the CSV. Disabling diagnostics files does not disable
IDA step accounting or the live statistics table. Statistics and timing summaries
retain their append behavior; other files are overwritten for each solve.

## Vessel CSV probes

Midpoint probes use one CSV, written by rank zero. Configure the filename and
the optional list of vessel IDs independently:

```text
subsection MetricFlowSystem<1, 3>
  set Output vessel CSV file = probes.csv
  set Vessel probe IDs = 0, 2, 7
end
```

An empty filename disables vessel probes, regardless of the ID list. A nonempty
filename with an empty ID list selects all vessels. Unknown selected IDs are
errors. Paths follow the rules above; no `none` or `all` keywords are needed.

The columns are `time_s,vessel_id,pressure_Pa,flow_m3_s,area_m2,velocity_m_s`,
all in SI units. Each output appends one row per selected midpoint, ordered by
vessel ID. The unique rank owning each midpoint evaluates it; all ranks,
including empty ranks, participate in gathering a bounded snapshot. Only rank
zero opens the file. No `HDG_IDA_Vessel_<id>.csv` files are created. Use a fresh
directory to avoid confusing historical per-vessel files with new output.

## Terminal names and effective parameters

`network_terminals.csv` contains one row per terminal with `boundary_id`,
`vessel_id`, `label`, `R1`, `R2`, `C`, and `P_out` (physical values in SI).
The optional input CSV column `label` supplies the name; a missing or blank
label defaults to `boundary_<boundary_id>`. In VTK mode the same default is
used. Boundary and vessel IDs remain the keys, so labels need not be unique.

Use `vessel_id` to join midpoint probes or visualization cells to this metadata
table for terminal labels. Labels do not become filenames. Global mass, solver
statistics, and timing summaries retain their aggregate schemas.

All C++ solver CSV writers use Rapidcsv, including probes, terminal metadata,
`mass_conservation.csv`, `ida_statistics.csv`, and `scaling_timings.csv`.
Commas, quotes, and line breaks in text fields are quoted and escaped by the
library. Each record is flushed progressively with bounded memory; double
values use round-trip precision and a locale-independent decimal point.
Statistics and timings append to existing files, writing a header when the
file is absent or empty. Python tools use the Python CSV library.

## Terminal time series

Set `Output terminal CSV file = terminals.csv` to write one terminal CSV with
`time_s,pressure_Pa,flow_m3_s,boundary_id,vessel_id,label`. The optional input
label is carried in each record; unnamed boundaries use `boundary_<id>`,
including `boundary_0` at the inlet. Rapidcsv handles quoting and escaping.
Each output appends one row per boundary, ordered by `boundary_id`; the inlet
is included. All ranks evaluate their owned traces, including ranks with no
local terminals, and gather the current snapshot to rank zero. Only rank zero
opens the file, and each snapshot is flushed as one batch. Memory is bounded
by the number of terminals, not by simulation duration. Terminal calculation
and file-write failures are propagated collectively.

`Output terminal CSV file` defaults to empty, disabling terminal output. Relative
paths are resolved under `Output directory`; absolute paths are used directly.
Parent directories are created as needed. The former `Write terminal CSV files`
boolean is removed; replace it with the desired filename in existing inputs.
Disabled output creates no terminal time-series file.
`network_terminals.csv` remains the static outlet metadata table. Use a fresh
output directory to avoid mixing
new output with older `terminal_<boundary_id>.csv` files.

The inflow and ramp comparison tools read both layouts, including comparisons
between historical per-terminal files and a consolidated file. They reject
mixed layouts and duplicate time/boundary rows. The current ramp policy pins
the updated reader checksum; numerical tolerances, gates and stored historical
benchmark results are unchanged. Comparison reports retain the logical table
count and additionally report the actual CSV counts for each input run.

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

## Wall-domain checks and nonlinear convergence

For ADAVN walls, residual and Jacobian callbacks reject trial states outside the
positive-area, hyperbolic wall domain with a recoverable IDA return code. The
`check_wall_domain` timer measures this check. At console depth greater than
one, rejected non-hyperbolic trials identify their vessel and area ratio.

`IDA parameters / Running parameters / Nonlinear convergence coefficient`
controls the native IDA Newton convergence test independently of time-error
acceptance; its default 0.33 preserves the SUNDIALS default. A smaller value can
be useful when deliberately loose startup weights would otherwise also loosen
the nonlinear solve. See the [SUNDIALS IDAS documentation](https://sundials.readthedocs.io/en/latest/idas/Usage/SIM.html).

## Logs and parameter receipt

`ParameterAcceptor::initialize()` writes `last_used_parameters.prm` after reading the input. deal.II logging is controlled by `Verbosity (console depth)`, and rank zero is the visible writer in the MPI run.

## Status

The repository contains example images and reference assets. This page documents file-writing behavior.
