# Single-vessel manufactured solution (MMS)

## Current paper experiment

The supported MMS input is
[`reproducibility/parameters/mms.prm.in`](../../reproducibility/parameters/mms.prm.in).
The [paper manifest](../../reproducibility/manifest.json) defines independent
runs for polynomial degrees 1–3 and successive mesh refinements. The template
provides numerical vessel constants in its exact fields and forcing, Dirichlet
traces at both ends, native consistent initialization, BDF5 and uniform scalar
absolute/relative tolerances of `1e-12`. Startup tolerance relaxation is disabled
so it cannot contaminate the measured errors.

Prepare the current sweep without launching a solver:

```bash
scripts/download_datasets.sh .
python3 scripts/reproduce_paper.py --suite mms --prepare-only
```

For execution, supply a `blood_flow` executable and its full source revision;
see [the reproduction guide](../reproducibility.md) for build provenance,
serial/MPI execution and output collection. Use a fresh output root for each
preparation or execution. The runner records the effective parameters and mesh
checksums for every case. Current errors and convergence rates must be obtained
from those new runs before the paper is finalized.

## Removed ARKode-derived inputs

The historical p1/p2/p3 tutorial templates and the old ARKode parameter cases
have been removed. They used vessel constants and initial data unavailable
through the current parameter interface; a section rename to IDA was insufficient
to make them reproducible. They remain accessible in Git history.

[`tools/generate_mms_expressions.py`](../../tools/generate_mms_expressions.py)
and the symbolic JSON in tutorial 01 retain mathematical derivations for the
historical profiles. They are reference artifacts, not alternative runnable
paper inputs. In particular, symbolic vessel constants in those expressions
cannot be passed directly through the current user FunctionParser constant map.

## Earlier reference evidence

The supplied report is retained as
[`reference/convergence.txt`](../../tutorials/01_single_vessel_mms/reference/convergence.txt).
Its adjacent JSON records the original SHA-256 and preserves all displayed
errors and rates. These tables describe that earlier report and do not assert
that the current paper experiment reproduces its values. `DoFs` denotes the
report's total degrees of freedom.

Regenerate the reference JSON or table without a solver:

```bash
python3 tools/collect_convergence.py \
  tutorials/01_single_vessel_mms/reference/convergence.txt \
  --json /tmp/convergence.json
python3 tools/collect_convergence.py \
  tutorials/01_single_vessel_mms/reference/convergence.txt \
  --format markdown --output /tmp/convergence.md
```

### Polynomial degree 3

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 2.621e-08 | -- | 1.386e-06 | -- | 2.557e-04 | -- | 1.895e-02 | -- |
| 2 | 96 | 1.676e-09 | 3.97 | 1.796e-07 | 2.95 | 1.657e-05 | 3.95 | 2.438e-03 | 2.96 |
| 3 | 192 | 1.054e-10 | 3.99 | 2.274e-08 | 2.98 | 1.050e-06 | 3.98 | 3.075e-04 | 2.99 |
| 4 | 384 | 6.598e-12 | 4.00 | 2.855e-09 | 2.99 | 6.602e-08 | 3.99 | 3.855e-05 | 3.00 |
| 5 | 768 | 4.299e-13 | 3.94 | 3.574e-10 | 3.00 | 4.267e-09 | 3.95 | 4.825e-06 | 3.00 |

### Polynomial degree 2

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 40 | 2.729e-07 | -- | 8.754e-06 | -- | 2.554e-03 | -- | 1.295e-01 | -- |
| 2 | 80 | 3.502e-08 | 2.96 | 2.289e-06 | 1.94 | 3.328e-04 | 2.94 | 3.343e-02 | 1.95 |
| 3 | 160 | 4.409e-09 | 2.99 | 5.816e-07 | 1.98 | 4.238e-05 | 2.97 | 8.438e-03 | 1.99 |
| 4 | 320 | 5.523e-10 | 3.00 | 1.463e-07 | 1.99 | 5.343e-06 | 2.99 | 2.116e-03 | 2.00 |
| 5 | 640 | 6.908e-11 | 3.00 | 3.668e-08 | 2.00 | 6.707e-07 | 2.99 | 5.297e-04 | 2.00 |

### Polynomial degree 1

| Cycle | DoFs | A L2 | A L2 rate | A H1 | A H1 rate | U L2 | U L2 rate | U H1 | U H1 rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 32 | 2.197e-06 | -- | 3.316e-05 | -- | 1.926e-02 | -- | 5.766e-01 | -- |
| 2 | 64 | 5.714e-07 | 1.94 | 1.758e-05 | 0.92 | 5.021e-03 | 1.94 | 2.982e-01 | 0.95 |
| 3 | 128 | 1.441e-07 | 1.99 | 8.960e-06 | 0.97 | 1.286e-03 | 1.97 | 1.506e-01 | 0.99 |
| 4 | 256 | 3.609e-08 | 2.00 | 4.514e-06 | 0.99 | 3.256e-04 | 1.98 | 7.549e-02 | 1.00 |
| 5 | 512 | 9.030e-09 | 2.00 | 2.264e-06 | 1.00 | 8.193e-05 | 1.99 | 3.778e-02 | 1.00 |

The JSON provenance records the TXT checksum and identifies the report as the
source of every displayed string. The reference numbers are retained from the
source report.


See [the output reference](../outputs.md) for current CSV units and
[the transient policy](../transient_parameters.md) for the active input inventory.
