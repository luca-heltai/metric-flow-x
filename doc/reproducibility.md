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

## Run through a container

Use the container launcher through the runner so the receipt transitions to
`completed` or `failed` and the collector can discover the outputs:

```bash
python3 scripts/reproduce_paper.py bifurcation --runtime docker \
  --image ghcr.io/luca-heltai/metric-flow-x:main \
  --dataset-root datasets --output-root paper-runs
```

The published `main` image becomes available after the container publication
workflow is merged and succeeds. Use an explicitly published preview tag when
testing before that point. `--runtime podman` and `--runtime apptainer` use
the same launcher contract. A container engine must be installed on the host;
the application and its compiled libraries are supplied by the image.
This mode requires `scripts/run_container.sh` from the container distribution
change. It forwards the manifest rank count with `--ranks` and records the
image reference. If an explicit `@sha256:...` reference is supplied, that digest
is recorded too; an image tag is not reported as a resolved immutable digest.

By default the checkout is the shared mount root. Both datasets and run output
must lie underneath it. If they are elsewhere, pass `--container-workdir` with
an explicit common parent directory. The runner never chooses the filesystem
root automatically. The launcher mounts that directory at its original absolute
path as well as `/work`, so prepared absolute parameter paths remain usable.
Preparing inputs and separately invoking the standalone launcher is still
possible, but the prepared receipt is not updated by that separate invocation.

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

`--suite all --prepare-only` prepares all 43 configurations. `--suite all`
executes the full matrix, including the costly scaling cases. MMS execution
requires a build containing the manufactured Dirichlet boundary support:

```bash
python3 scripts/reproduce_paper.py --suite mms \
  --executable /path/to/build/blood_flow \
  --executable-revision FULL_SOURCE_COMMIT --output-root /path/to/mms-runs
```

Record `git -C /path/to/build-source rev-parse HEAD` and the source dirty state
when building the executable. Supply that **full 40-character commit** from
the build record; the current workflow checkout revision is not a substitute
for a reused binary's provenance. The manifest pins the minimum supported
commit. The runner accepts that exact commit even in a release/shallow checkout;
for a later commit it requires Git history proving ancestry. Fetch the relevant
commits into this checkout if history is missing. A missing, older or unknown
revision fails before any suite run starts. The same explicit source attestation
is required for MMS container runs; obtain it from the image's build provenance
or `org.opencontainers.image.revision` label. This is a user-supplied attestation,
not automatic certification of a binary's source. `--prepare-only` needs no
executable attestation.

For MPI
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
  The manufactured `Dirichlet` boundary support uses components 0 and 1 of
  `Exact solution` at both endpoints; this is why old executable revisions
  cannot be used for MMS runs. Boundary functional tests establish that contract;
  they do not establish the paper's convergence table.
- The imported cerebral input retains maximum BDF order 2; the manuscript
  describes orders through 5. The current pinned cerebral v2 mesh is used.
- Liver retains the supplied parameter tolerances 1e-6 and 1e-7; the manuscript
  table lists 1e-8 and 1e-9.
- The synthetic cases use the corrected SI release meshes. These are new
  runs, not replays with the archived paper meshes. The historical filenames
  “50k” and “100k” describe 50001 and 100001 computational segments.
- The imported large-network inputs retain `Gamma (Total pressure factor) = 0`,
  which equates static pressure at junctions. The paper's total-head formula
  corresponds to gamma=1. The selected junction model must be reconciled by
  the authors before freezing final runs.
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
For completed MMS cases, `mms_errors.csv` extracts the four reported A/U L2/H1
errors and final time from `solver.log`, alongside polynomial degree and cell
count. It preserves the solver's displayed precision (currently three decimal
places in scientific notation), reports a missing error block as an error,
and does not infer convergence rates or compare against archived values.

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
