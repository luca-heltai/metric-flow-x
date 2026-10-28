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

## Submit experiments with Slurm

The submission driver runs CPU/MPI experiments on one reserved compute node.
Replace the uppercase placeholders and paths with values for your system:

```bash
python3 scripts/submit_paper_gpu.py --scaling-only \
  --node COMPUTE_NODE --machine MACHINE_PROFILE --partition PARTITION \
  --env /path/to/spack-environment.sh \
  --description 'Hardware and operating system used for these runs' \
  --output-root "$PWD/paper-runs/campaign-01"
```

The environment script must provide CMake, Ninja, OpenMPI 5, Spack and a
compatible deal.II installation with PETSc and MUMPS. `MACHINE_PROFILE` is a
label for the results; every run independently records the execution hostname
and hardware. The hardware description may be omitted when that profile already
contains `machine.tex` in the manuscript checkout.

To reuse a freshly compiled executable, add `--executable /path/to/blood_flow`
and `--executable-revision FULL_BUILD_COMMIT`. The first job then snapshots the
binary and its application library, records their hashes and runs a small
output-scheduling check instead of compiling again. Subsequent jobs use the
frozen copies and verify their hashes before every run. The reused binary may
use either supported linear-algebra backend; its source revision is supplied
by the person who built it.

Choose a fresh directory on storage shared by the submission host and compute
node. Missing or modified datasets are downloaded and verified before any job
is submitted; compute nodes do not need Internet access. Keep this checkout
unchanged while jobs are pending or running: each job checks the source revision
and tracked-file hashes against the submission snapshot.

With `--scaling-only`, the driver submits three jobs with `afterok` dependencies:

1. **Build**: compile a fresh Release executable, record source revision and
   patch, binary digest and linked libraries. Every solver run verifies the
   executable against that build receipt.
2. **Scaling**: run the 50k and 100k sweeps sequentially on the same exclusive
   node, with one thread per rank. The default counts are 1,2,4,8,16,32,64;
   select a smaller set with `--ranks 1 2 4 8 16`. The reservation requests as
   many CPU cores as the largest selected count. Choose counts that fit the
   physical cores available on the selected node.
3. **Collection**: validate and export the scaling tables and provenance into
   `paper-output/scaling/MACHINE_PROFILE/` within the campaign directory.

Omitting `--scaling-only` adds a 35-element output array between scaling and
collection. It covers MMS, benchmarks, mesh/time sweeps, applications and field
outputs for both large networks. MMS and the three-cell bifurcation use one MPI
rank; the other output runs use four. The collector also writes portable sample
tables into `plot-inputs/`. Raw VTU/PVD fields and vessel CSVs remain in `outputs/`.

All jobs are pinned to the node selected by `--node`. No accelerator is requested:
these are CPU/MPI measurements. The generated launcher uses OpenMPI 5 with local
PRRTE resource discovery and binds each rank to one core within the allocation.
Check that this launch method and the generated Slurm resource options match
your scheduler configuration; inspect the scripts with `--dry-run` before a
first submission on another system. Thread limits cover OpenMP, BLAS and deal.II.

Scaling sets `Write solution output = false`, disables progress printing and
per-step diagnostic CSVs. It skips VTU, vessel probes, pressure reconstruction
at output callbacks and mass/flow diagnostics while retaining the original IDA
callback cadence and solver settings. Final `ida_statistics.csv` records
`wall_time` for `solve_dae`, including IDA initialization and excluding mesh/setup
and MPI startup. `initialization_statistics.json` records preparation and
initialization times; its `ida_setup_and_ic_wall_s` contribution can be subtracted
when reporting integration only. `scaling_timings.csv` records individual timed
sections. These final summaries are written after integration.

The runner also records `process_wall_seconds` in `run.json`. This includes MPI
startup, setup and final reporting and should not replace solve time. Separate
field-output runs have no role in scaling measurements.

Use `--smoke` for a recorded 0.001 s workflow check, which skips manuscript
export. `--time D-HH:MM:SS` changes the scaling/output/collection limits.
`submission.json` retains submitted job IDs even if a later submission fails;
failed prerequisites prevent dependent jobs from running. The scripts regenerate
current-code data; the provenance differences below still apply, and figures
are not automatically compared or rendered.

## Experiment matrix

| Suite | Canonical cases | Configuration |
| --- | --- | --- |
| `mms` | `mms-p1-n4` through `mms-p3-n64` | p=1,2,3; 4,8,16,32,64 cells; separate output for each of 15 runs |
| `benchmarks` | `bifurcation`, `network37` | p3; final times 30 s and 20 s |
| `mesh` | `mesh37-r0` through `mesh37-r3` | 37 segments; p3; refinements 0–3 |
| `time` | `time37-r1-bdf1` through `time37-r2-bdf5` | p3; refinements 1,2; maximum BDF order 1–5 |
| `applications` | `liver`, `cerebral` | 7999 and 19999 segments; p2; final time 2 s |
| `fields` | `network50k-output`, `network100k-output` | Same large networks with solution output, 4 MPI ranks |
| `scaling` | `network50k-np1` through `network100k-np64` | 50001/100001 segments; p3; 1,2,4,8,16,32,64 MPI ranks; no solution output |

The runner accepts `--machine MACHINE_PROFILE` and records the actual hostname,
hardware, thread limits, executable hash and source revision in every `run.json`.
Set `--executable-revision "$(git rev-parse HEAD)"` only for a binary built from
that revision; `--build-provenance FILE` additionally verifies its SHA256 and
revision against a build receipt. `--mpi-arg=--option` adds launcher options,
including for a single rank. `process_wall_seconds` includes MPI startup, setup
and final reporting; use solver statistics for solve timings.

To restrict the scaling sweep to the physical cores available on a smaller
machine, use `--suite scaling --ranks 1 2 4 8 16`. This selects canonical
experiments without changing their process counts or numerical settings.
Set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`,
`BLIS_NUM_THREADS` and `DEAL_II_NUM_THREADS` to `1` for timing runs.

Run an entire supported suite sequentially, stopping on the first solver failure:

```bash
python3 scripts/reproduce_paper.py --suite benchmarks \
  --executable /path/to/build/blood_flow --output-root /path/to/benchmark-runs
```

`--suite all --prepare-only` prepares all 49 configurations. `--suite all`
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

## Finalize machine-specific scaling tables

Pass the manuscript checkout with `--paper-root /path/to/manuscript`; the
default is `paper/` relative to the software checkout. The current
manuscript actually plots the 50k and 100k strong-scaling tables; the old
8k/30k tables are unused and are not regenerated. The companion paper branch
removes those two unused LaTeX table loads.

After a completed canonical campaign, export its results with:

```bash
python3 scripts/finalize_paper_output.py /path/to/campaign-01 \
  --machine MACHINE_PROFILE --paper-root /path/to/manuscript \
  --description 'Hardware and operating system used for these runs'
```

The input may be a `submit_paper_gpu.py` campaign root, or the output root of
`reproduce_paper.py --suite scaling`. The destination is
`<manuscript>/scaling/MACHINE_PROFILE/`. Existing hardware-only profiles are
accepted; profiles containing results are never overwritten, and the historical
reference profile is protected. A new
profile requires `--description 'LaTeX hardware and OS description'`.
For an intentionally smaller sweep, pass the same `--ranks 1 2 4 8 16` to
both the runner and finalizer.

The export contains:

- `plot_data/scaling_50k.csv` and `scaling_100k.csv`, with the exact column names
  used by the manuscript, sorted by MPI count. Each value comes from the named
  accumulated section in `scaling_timings.csv`; no averaging, last-row selection
  or substitution of process wall time is performed.
- `50k_vessels/runs/`, `100k_vessels/runs/` and, when supplied, `8k_vessels/runs/`
  and `cerebral/runs/`, preserving receipts, parameters, logs and raw outputs.
- `provenance/` with available build/submission metadata, and `finalization.json`
  with source hashes, the measured execution host, executable revision/hash,
  solve/initialization/integration times, speedups and parallel efficiencies
  relative to the smallest measured MPI count, and remaining manuscript work.

The finalizer rejects failed, prepared, smoke, incomplete, duplicate or mixed
scaling runs. A sweep must use one executable, one host and the same numerical
inputs for each network, with solution/progress/per-step diagnostic output
disabled and explicit single-thread limits. All requested counts must be present
for both networks. Source revision for a directly reused binary remains a user
attestation unless a build receipt was supplied; the binary hash is always
recorded by the runner.

For a direct run on a machine with fewer available cores, first compile the
executable from the revision being recorded, then run:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 DEAL_II_NUM_THREADS=1
python3 scripts/reproduce_paper.py --suite scaling --machine MACHINE_PROFILE \
  --ranks 1 2 4 8 16 --executable /path/to/fresh-build/blood_flow \
  --executable-revision "$(git rev-parse HEAD)" --output-root /path/to/campaign-01
python3 scripts/finalize_paper_output.py /path/to/campaign-01 \
  --machine MACHINE_PROFILE --ranks 1 2 4 8 16 --paper-root /path/to/manuscript \
  --description 'Hardware and operating system used for these runs'
```

Exporting scaling tables does not complete the other manuscript figures.
The pressure/network snapshots and mass-conservation images require rendering
new field outputs; the cerebral IDA discussion and its historical BDF5 filename
must be reconciled with the current cerebral input (BDF2). The finalizer neither
copies historical measurements/images nor relabels BDF2 results as BDF5. It reports
`scaling_tables_ready: true` and `manuscript_profile_complete: false` explicitly.
Before selecting a new full profile, generate the application figures and
review the measured numerical claims and software versions in the text. Older
manuscript checkouts also need the unused `scaling_8k.csv` and `scaling_30k.csv`
loads removed from `tikz_plots.tex`.
