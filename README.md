# MetricFlowX

MetricFlowX is a standalone deal.II library for the one-dimensional blood-flow
DAE embedded in three-dimensional geometry. It contains the distributed mesh,
DG/HDG discretization, RCR terminal models, junction equations, residual
assembly, and analytic linearizations. It has no ImmersX dependency and does
not contain coupling or TensorProduct code.

The mathematical Problem is

```text
F(t, y, y_dot) = 0
```

where the native state contains cell area and velocity, face-trace area and
velocity, and (when present) RCR capacitor pressures. Pressure is the derived
observable `P(A)`; it is not an additional DAE unknown.

## Public API

```cpp
#include <metric_flow_x/blood_flow_system.h>

MetricFlowX::BloodFlowSystem<1, 3> flow(MPI_COMM_WORLD);
flow.initialize_params("network.prm");
flow.setup();

auto y     = flow.make_state();
auto y_dot = flow.make_state();
auto F     = flow.make_state();

flow.initialize_state(y, 0.0);
flow.initialize_state_derivative(y_dot, 0.0);
flow.assemble_residual(0.0, y, y_dot, F);
flow.assemble_state_jacobian(0.0, y, y_dot);      // dF/dy
flow.assemble_derivative_jacobian(0.0, y, y_dot); // dF/dy_dot
```

`triangulation()`, `dof_handler()`, `finite_element()`, `constraints()`,
`mpi_communicator()`, ownership index sets, differential/algebraic masks,
component DoFs, and area/velocity extractors expose the discretization needed
by an external multiphysics application. Constitutive access is provided by
`vessel_properties()`, `pressure()`, `pressure_derivative()`, and
`wave_speed()`.

An optional prescribed surrounding pressure can be supplied with
`set_external_pressure_provider()`. The provider receives time, physical
point, vessel id, and the incident centerline `CellId`, and returns pressure
in the same units as the network data. MetricFlowX uses
`p_internal = p_tube(A) + p_external(t, x, vessel_id)` in momentum fluxes,
RCR pressure relations, junction total-head equations, and pressure output.
The default provider is zero; the provider is external data and is not
linearized as part of either native Jacobian. It is called during residual
assembly and pressure output, but not during Jacobian assembly.

Embedding applications that own a pressure field can call
`add_external_pressure_residual(t, y, provider, destination)`. This adds only
the external-pressure part of the native residual, using additive semantics;
it does not install or replace the system's provider. The candidate state is
used for the state-dependent numerical-flux branch, while the supplied
pressure remains independent of the native flow unknowns.

The state layout is deliberately not split into separate area and velocity
vectors. Component values are identified with
`BloodFlowSystem<1, 3>::Component`, preserving the native distributed layout.

## Standalone application

The `blood_flow` executable reads a parameter file, delegates SUNDIALS IDA
time integration to its standalone `BloodFlowIDARunner`, and writes the
established VTU/PVD and CSV outputs. The library Problem owns physics and
spatial assembly; applications embedding it can own time integration and call
the residual/Jacobian API directly.

## Container execution

The [container guide](doc/containers.md) describes the downloadable launcher
for Docker, Podman, and Apptainer, the compiled library SDK, and publication
of versioned images. A container runtime is the only required host dependency.
Images become available after the container publication workflow succeeds.

## Building

MetricFlowX uses an out-of-source CMake build and requires deal.II 9.5 or newer:

```bash
cmake -S . -B build-metric-flow-x \
  -DDEAL_II_DIR=/path/to/deal.II \
  -DCMAKE_BUILD_TYPE=Debug
cmake --build build-metric-flow-x
```

The installed package exports the target `MetricFlowX::metric_flow_x`:

```bash
cmake --install build-metric-flow-x --prefix /path/to/install
```

```cmake
find_package(MetricFlowX CONFIG REQUIRED)
target_link_libraries(application PRIVATE MetricFlowX::metric_flow_x)
```

To build and install both variants in one single-config build tree (Ninja or
Unix Makefiles), use a deal.II installation providing both Debug and Release:

```bash
cmake -S . -B build-metric-flow-x -DCMAKE_BUILD_TYPE=DebugRelease
cmake --build build-metric-flow-x -j
ctest --test-dir build-metric-flow-x --output-on-failure
cmake --install build-metric-flow-x --prefix /path/to/install
```

In this mode, `metric_flow_x` and `blood_flow` are Release targets;
`metric_flow_x_debug` and `blood_flow_debug` are Debug targets. Tests run
against both libraries. Ordinary `Debug` and `Release` builds retain the
unsuffixed target names.

The installed `MetricFlowX::metric_flow_x` target provides both imported
configurations and selects the library matching the consumer's build type.
Configure ImmersX with `-DCMAKE_BUILD_TYPE=Debug` or `Release` and
`-DCMAKE_PREFIX_PATH=/path/to/install`; both builds can use the same prefix.
Consumers building both variants as separate targets in a single-config tree
can link explicitly to `MetricFlowX::metric_flow_x_Debug` and
`MetricFlowX::metric_flow_x_Release` instead of relying on the global build type.
ImmersX's combined `DebugRelease` build currently needs this explicit selection
([ImmersX #266](https://github.com/luca-heltai/immersx/issues/266)); separate
ImmersX Debug and Release builds already select the correct variant.

Headers are installed below `include/metric_flow_x/`, libraries below `lib/`,
and the standalone executable below `bin/`.

## Large network datasets

The generated vascular-network datasets are distributed as GitHub Release
assets on [the datasets release](https://github.com/luca-heltai/metric-flow-x/releases/tag/datasets-final-version-v1)
so they do not inflate the Git checkout. The network archive contains the
corrected SI solver meshes and the original source trees; the cerebral archive
is about 2.7 MB (8.7 MB extracted).

From the repository root, run:

```bash
./scripts/download_datasets.sh
```

The script downloads both `metric_flow_x_datasets_si_v1.tar.gz` and
`cerebral_datasets_v2.tar.gz` from release `datasets-final-version-v1` in
`luca-heltai/metric-flow-x`. It verifies pinned SHA-256 checksums before
extracting each archive into its expected `notebooks/` paths. The network archive
ships corrected SI meshes directly. No post-extraction
conversion or Python dependency is needed. Dataset contents, units and hashes
are recorded in [the release manifest](datasets/datasets_si_v1.json).
The cerebral archive adds:

- `notebooks/cerebral_data/cco_parietal_hc_w_data.vtk`
- `notebooks/cerebral_data/standard_cco.vtk`

All downloaded data paths are ignored by Git. `curl`, `tar`, and either
`shasum` or `sha256sum` are required.

To extract into another directory, pass it as the first argument:

```bash
./scripts/download_datasets.sh /path/to/dataset-root
```

## Paper experiment inputs

The [reproduction guide](doc/reproducibility.md) describes the canonical
current-code parameter templates, all seven paper meshes, and the 43-case
experiment matrix. List the available cases with
`python3 scripts/reproduce_paper.py --list`. Each prepared or executed case
records its input checksums and configuration in a separate run directory.
These inputs support new controlled runs; archived paper results have not
been verified by this workflow.

## Testing

```bash
ctest --test-dir build-metric-flow-x --output-on-failure
```

The suite covers residual and Jacobian finite differences, mass assembly,
junctions, VTK input, and a coupling-readiness test that exercises state
creation, DAE masks, component extraction, split Jacobians, and constitutive
observables without using ImmersX.

The formulation and derivation remain documented in `doc/math.md` and
`latex/blood_flow.tex`.

Terminal RCR and inflow CSV input use the vendored header-only
[rapidcsv v9.07](https://github.com/d99kris/rapidcsv/tree/v9.07) dependency;
see `third_party/rapidcsv/README.md` for provenance and its BSD 3-Clause license.
No extra package or network download is needed to build the CSV reader.

## License

This project is licensed under the MIT License (see `LICENSE.md`).


### svVascularize units

The solver consumes SI geometry, area and flow. Convert trusted tree archives
using their recorded `unit_system`; missing or unsupported length/time units
require explicit `--length-unit` and `--time-unit` overrides. The converter
requires Python 3 and NumPy, without svVascularize or Matplotlib.

To regenerate the 10000-terminal tree with its existing RCR/material assumptions:

```sh
python3 notebooks/networks_by_svv/convert_to_standard_vtk.py \
  notebooks/networks_by_svv/trees/network_10000.tree.npz \
  notebooks/networks_by_svv/trees/network_10000_adnr.vtk \
  --rcr --young-modulus 225000 --diastolic-pressure 10000 \
  --perfusion-pressure 10000 --reference-inflow 8.178e-7
```

Pressure and wall-law arguments are SI. Centimetre/second trees require lengths
and radii scaled by 0.01 and design flow by 1e-6 before deriving areas, wall
thickness, resistances and capacitances. The interior boundary marker is 300655,
within the unsigned 32-bit range and above the outlet IDs in these datasets.

The release ships SI solver meshes directly. Download the current archive again
to replace an older extraction; there is no legacy repair path. Original source
trees retain their recorded CGS metadata. See the dataset manifest for the units
and role of each file. Generated network files stay outside Git. Correct units
alone do not validate inflow, wall parameters, near-sonic flow or physiological
suitability.

Project-owned file and directory names use lowercase letters and underscores.
Conventional metadata and tool filenames retain their standard spelling:
`README.md`, `LICENSE.md`, `CITATION.cff`, `CMakeLists.txt`, `Doxyfile`, and `.clang-format`.

### Terminal RCR defaults and CSV overrides

Existing parameter files keep `Terminal parameter source = VTK`: terminal R1,
R2, C and P_out are read from the mesh exactly as before. To migrate explicitly,
set the following in `MetricFlowSystem<1, 3>`:

```text
set Terminal parameter source = Parameters
set Default terminal R1 = 2e7
set Default terminal R2 = 3e7
set Default terminal C = 0
set Default terminal P_out = 0
set Terminal parameters CSV file = terminals.csv
```

CSV columns are selected by their case-sensitive names, in any order:

- `boundary_id` is required and must uniquely identify an actual terminal.
- `R1`, `R2`, and `C` are optional. Each missing column or empty/whitespace-only
  cell uses the corresponding parameter-file default. At least one of these
  column names, or `label`, must be present, so a different kind of data file
  is not silently accepted as terminal configuration.
- `label` is an optional terminal name. A missing column or blank cell uses
  `boundary_<boundary_id>` (for example, `boundary_2`), matching the existing
  numeric terminal identity. Labels may contain spaces, commas, quotes, or
  quoted line breaks; they are metadata and need not be unique. A file with
  only `boundary_id,label` can name terminals while using all RCR defaults.
- Additional named columns are ignored; they may contain arbitrary metadata or
  measurements, including quoted commas and line breaks.

For example, with the defaults above:

```{csv-table}
:header-rows: 1

label,R2,boundary_id,C
outlet_b,4e7,2,
```

Terminal 2 is named `outlet_b` and receives R1=2e7, R2=4e7, C=0. Omitted terminal IDs use all defaults;
with no CSV configured, every terminal uses the defaults. Headers must be
nonempty and unique, and every data row must have the same number of cells as
the header (use explicit empty cells for defaults). Nonempty numeric cells must
contain a complete valid number; `NaN`, infinity and invalid final RCR values
are rejected. Duplicate or unknown boundary IDs are errors.

Output `network_terminals.csv` records the effective `boundary_id`,
`vessel_id`, `label`, `R1`, `R2`, `C`, and `P_out` for each terminal. Vessel
probe files carry a `boundary_<id>_label` column for each terminal connected to
that vessel. Existing probe filenames keep their numeric vessel IDs; labels
never become filesystem paths. Visualization files include `vessel_id` cell
data, which links them to the terminal table. Global mass and timing summaries
retain their aggregate columns.

All solver CSV output uses Rapidcsv for quoting and escaping. Records are
written progressively, with round-trip precision for double values and a
locale-independent decimal point.

Resistances are in Pa*s/m3, compliance in m3/Pa, and P_out in Pa. P_out remains
the global default in Parameters mode; CSV rows do not override it. VTK mode
preserves existing per-terminal P_out values. No units, pressure datum or flow
sign convention are inferred from unused columns.

This is a constant-parameter table, with one row per terminal, not a time-series
reader. Future prescribed pressure/flow series should have a separate schema
with `boundary_id` and `time_s`, explicit units and driving-variable selection,
strictly increasing times per terminal, and explicit interpolation, coverage
and periodicity policies. External-file adapters can map column names and units
to that contract while retaining other columns as unused data.

Parameters mode requires geometry, vessel fields and boundary IDs in VTK, but
no R1/R2/C/P_out arrays. Actual graph endpoints identify terminals, excluding
junctions and inlet boundary 0. CSV IDs must identify these terminals uniquely;
unreadable/malformed files, duplicate/unknown IDs, nonfinite or invalid physical
values cause an error. A configured CSV requires Parameters mode. Each C>0
terminal gets one capacitor unknown across MPI ranks; C=0 gets none and uses
the series resistance R1+R2 (PR #18).

Geometry and arrays now share one VTK reader per rank. On lnx3 with GCC 13.2,
deal.II 9.8.0 and the 20001-cell, degree-1 SI 10k tree, the measured setup time
fell from 3.697 s to 1.340 s in a paired run. Whole-process peak RSS, including
subsequent assembly, was 657904/656936 KiB; this is not a measurement of memory
per initialization phase. Nonzero-state residual norms/checksums agree, and
CSV/legacy residuals and both Jacobian blocks are regression-tested with one
and two ranks. Rank-zero reading/distribution remains a profiling follow-up;
each rank currently loads one dataset.

### Tabulated inflow and terminal traces

In `MetricFlowSystem<1, 3>`, set `Inflow CSV file` to a CSV containing the
case-sensitive columns `time_s` and `flow_m3_s`, in any order. Additional named
columns are ignored and may hold arbitrary metadata. Headers must be nonempty
and unique; every row must match the header width. The shared rapidcsv reader
accepts quoted fields, quoted commas/newlines, UTF-8 BOM and CRLF. Required
fields must contain complete, finite numeric values in seconds and m3/s, with
strictly increasing times. At least two samples are required. Piecewise linear
interpolation replaces the analytic inlet only when this file is configured.
Both ends of the simulation interval must be covered; extrapolation is an error.
Paths are relative to the run directory, as for the existing mesh path.

`Inflow CSV period` defaults to zero (no repetition). A positive value explicitly
repeats one complete cycle; the file must span exactly that period and have
matching endpoint flows. No cardiac period or flow-sign convention is inferred.
Only inlet flow is prescribed. Any cardiac pressure history remains a comparison
signal.

Set `Write terminal CSV files = true` to export `terminal_<boundary_id>.csv` in
the output directory. The fields are `time_s,pressure_Pa,flow_m3_s,boundary_id,vessel_id,label`.
`label` comes from the terminal parameter CSV and defaults to `boundary_<id>`;
the inlet uses `boundary_0`. Rapidcsv quotes and escapes arbitrary label text.
Pressure and velocity/area come from the boundary trace, including prescribed
external pressure. Flow is outward: negative at a conventionally oriented inlet
and positive at an outlet. Each boundary ID must identify exactly one terminal;
only its owning MPI rank writes the file. Existing midpoint probe CSVs retain
their original names and units.

The PETSc direct path uses MUMPS when available and native serial LU otherwise.
Native serial LU reorders for a nonzero diagonal before factorization.
Parallel direct runs require PETSc built with MUMPS. Set `Use direct solver = false`
for parallel GMRES with block Jacobi; local ILU shifts address zero pivots in DAE
subdomain blocks and can be overridden through PETSc options. This also applies
to trace initialization. A missing PETSc MUMPS backend must never produce an
unchanged solution while reporting a successful direct solve.

Validation uses a synthetic triangular waveform at one and two ranks, compares
CSV with the equivalent analytic inlet, checks nonzero prescribed flow, and
compares terminal pressure/flow within solver tolerance. It does not establish
MOX pressure datum, valve sign, cardiac period, RCR compatibility or a complete
physiological cardiac cycle.
