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

## Logs and parameter receipt

`ParameterAcceptor::initialize()` writes `last_used_parameters.prm` after reading the input. deal.II logging is controlled by `Verbosity (console depth)`, and rank zero is the visible writer in the MPI run.

## Status

The repository contains example images and reference assets. This page documents file-writing behavior.
