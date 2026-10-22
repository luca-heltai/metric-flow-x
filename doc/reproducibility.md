# Reproducing the paper experiments

The canonical inputs for new paper runs live in
[`reproducibility/parameters/`](../reproducibility/parameters/). They were
imported from the manuscript `2026_1d_blood_flow`, adapted to the current
parameter parser and connected to pinned SI datasets.
[`manifest.json`](../reproducibility/manifest.json) records source-file
checksums, adaptations, mesh digests and the complete proposed run matrix.
These are inputs for a new, controlled run of the paper experiments; archived
results have not been checked or certified by this workflow.

## Download all paper inputs

```bash
./scripts/download_datasets.sh
# Or keep the datasets outside the checkout:
./scripts/download_datasets.sh /path/to/datasets
```

The downloader verifies each archive's pinned SHA-256 before extraction. It
retrieves the existing SI network and cerebral archives, plus
`paper_inputs_v1.tar.gz` with the MMS, bifurcation, 37-segment and liver inputs.
All seven canonical meshes are therefore available through this one command.
The runner independently checks each selected mesh against its manifest digest.

Geometry, material properties and terminal circuits are provided by the VTK
input. Units are SI: metres, seconds, kilograms, pascals, square metres for area
and cubic metres per second for flow. Plotting in mmHg, dyn/cm² or ml/s needs
an explicit output conversion. The generator inventory and regeneration
contracts are in
[`reproducibility/generators.md`](../reproducibility/generators.md).

## Prepare and run a case

List the experiment identifiers without requiring datasets or an executable:

```bash
python3 scripts/reproduce_paper.py --list
```

Prepare the small bifurcation case, inspect the generated configuration, then
run it in a **new** output root:

```bash
python3 scripts/reproduce_paper.py bifurcation --prepare-only \
  --dataset-root /path/to/datasets --output-root /tmp/paper-prepared

python3 scripts/reproduce_paper.py bifurcation \
  --executable /path/to/build/blood_flow \
  --dataset-root /path/to/datasets --output-root /path/to/paper-runs
```

The Python runner uses only the standard library. Native execution requires a
compiled `blood_flow` application and its runtime libraries. Select the
executable explicitly to reuse an existing merged build. The checked-in
templates remain portable; the runner writes absolute dataset/output paths to
the materialized parameter file so it can run from an independent directory.

Each run gets its own directory with `parameters.prm`, `run.json`,
`solver.log` and `results/`. The receipt records the workflow checkout revision and dirty
state, parameter/template/manifest/mesh and executable hashes, command, MPI
count, platform, thread environment, start/end times and exit code. A prepared
case is recorded as `prepared`; only a zero solver exit status is recorded as
`completed`. Neither status means agreement with a reference curve. Preserve
the build's dependency versions and compiler configuration alongside the run
when making the final paper archive. When reusing a separately built executable,
pass `--executable-revision SHA` to record its known source revision separately;
the workflow checkout revision does not establish the executable's source.

Existing case directories are rejected. This prevents timing CSV append mode
from mixing repeated runs. To repeat an experiment, choose a fresh output
root. There is deliberately no overwrite switch.

For a short infrastructure check, an explicit override is recorded separately:

```bash
python3 scripts/reproduce_paper.py bifurcation \
  --executable /path/to/build/blood_flow \
  --dataset-root /path/to/datasets --output-root /tmp/paper-smoke \
  --final-time 0.001
```

This shortened run is not a paper simulation or scientific comparison.

## Experiment matrix

| Suite | Canonical cases | Configuration |
| --- | --- | --- |
| `mms` | `mms-p1-n4` through `mms-p3-n64` | p=1,2,3; 4,8,16,32,64 cells; separate output for each of 15 runs |
| `benchmarks` | `bifurcation`, `network37` | p3; final times 30 s and 20 s |
| `mesh` | `mesh37-r0` through `mesh37-r3` | 37 segments; p3; refinements 0–3 |
| `time` | `time37-r1-bdf1` through `time37-r2-bdf5` | p3; refinements 1,2; maximum BDF order 1–5 |
| `applications` | `liver`, `cerebral` | 7999 and 19999 segments; p2; final time 2 s |
| `scaling` | `network50k-np4` through `network100k-np12` | 50001/100001 segments; p3; 4,6,8,10,12 MPI ranks |

Run an entire supported suite sequentially, stopping on the first solver failure:

```bash
python3 scripts/reproduce_paper.py --suite benchmarks \
  --executable /path/to/build/blood_flow --output-root /path/to/benchmark-runs
```

`--suite all --prepare-only` prepares all 43 configurations. The 15 MMS
configurations are candidates only: execution is blocked because the current
solver does not implement their exact boundary data. Consequently `--suite all`
also refuses execution before starting any run. The remaining suites contain
28 runnable configurations, including the costly scaling cases. For MPI
runs the runner uses `mpirun -np N`; `--mpi-launcher` chooses another compatible
launcher executable and `--mpi` explicitly overrides the manifest count.
Distributed direct solves require a build with PETSc/MUMPS support. Record
machine and core allocation when reporting scaling timings; timing values
depend on hardware, MPI placement and dependency versions.

## Configuration decisions before submission

The manifest exposes differences that the paper authors must settle before
freezing a submission dataset:

- MMS uses final time 0.1 s from the manuscript, instead of the imported
  parameter file's 1 s. Its source expressions use literal SI coefficients,
  so they do not require per-vessel constants in `FunctionParser`. The RHS is
  assembled, and its exact state at t=0 agrees with the implemented rest state.
  However, `Exact solution` is used for error calculation only: the current
  boundary assembly prescribes inflow plus a characteristic compatibility row
  and RCR/reflection outlets, not manufactured Dirichlet traces. The string
  `Dirichlet` in the imported candidate does not implement such a boundary.
  These 15 candidates may be inspected with `--prepare-only` but cannot run
  through this workflow until a separate solver change defines the MMS boundary
  contract. A convergence claim from the current candidate would be incorrect.
- The imported cerebral input retains maximum BDF order 2; the manuscript
  describes orders through 5. The current pinned cerebral v2 mesh is used.
- Liver retains the supplied parameter tolerances 1e-6 and 1e-7; the manuscript
  table lists 1e-8 and 1e-9.
- The synthetic cases use the corrected SI release meshes. These are new
  runs, not replays with the archived paper meshes. The historical filenames
  “50k” and “100k” describe 50001 and 100001 computational segments.
- The mesh/time sweeps are explicit new configurations derived from the
  supplied 37-segment input. They do not recover unrecorded historical choices.

After author review, update the canonical templates or matrix, rerun, and
freeze the resulting receipts with the new raw outputs. The collection below
produces portable plot inputs. Selection of final figures and quantitative
comparison with archived paper results remain author work; image-only plots
or a successful short run do not establish scientific reproduction.

## Collect new outputs for plotting

```bash
python3 scripts/collect_paper_outputs.py /path/to/paper-runs \
  --output /path/to/new-plot-inputs --vessels 0 1 2 \
  --start 28.6 --end 29.7
```

The collector reads only cases whose receipt says `completed`. It writes
`samples.csv` with explicit SI column names, a `runs.csv` index, unmodified
diagnostic CSV copies and `collection.json` with every source digest, selected
vessels and time window. The sample time range is inclusive. Sample times
are preserved without interpolation or resampling. Vessel probes represent
the solver's midpoint sampling rule; they do not reconstruct junction traces.
Omit `--vessels` to collect every available vessel. For large networks select
specific vessels to keep the plot table manageable. Missing diagnostic CSVs
are listed explicitly; no counters or timings are reconstructed from images.

The vessel probe writer already converts physical SI state to historical
display units. The collector applies the documented reverse conversions:
`P_dynpcm2 / 10`, `Q_cm3ps / 1e6`, `A_cm2 / 1e4`, `U_cmps / 100`.
Terminal CSVs, when enabled, already use SI and are distinct from midpoint
probes. For scaling and BDF studies use `scaling_timings.csv` and
`ida_statistics.csv` from each run; do not sum unrelated timing sections or
replace missing IDA counters with numbers from a plot.

Existing postprocessing tools have narrower contracts:

| Tool | Input and purpose |
| --- | --- |
| `tools/collect_convergence.py REPORT --json OUT` | Parses the immutable historical formatted convergence report; does not parse new runner logs or validate their values. |
| `tools/analyze_periodicity.py TRACES --period 0.827` | Requires a `time` column and explicitly named `trace_*` columns; construct these from the selected vessel/quantity before applying it. |
| `tools/analyze_y_junction_residuals.py RESIDUALS` | Requires exported named raw residual columns; state samples cannot substitute for residuals. |
| `tools/generate_network_diagram.py INPUT.vtk --svg OUT.svg --json OUT.json` | Deterministic topology diagram and graph metadata from a VTK input. |

The manuscript's `tikz_plots.tex` reads reduced tables in its `plot_data/`
directory. Its existing Python plotting scripts contain manuscript-specific
paths and comparisons; the collector provides portable new-run samples
without importing or checking those archived comparisons.
