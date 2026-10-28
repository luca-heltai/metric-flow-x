# Canonical paper inputs

The editable parameter sources are `parameters/*.prm.in` in this directory.
`manifest.json` defines the experiment matrix, mesh checksums and provenance of
the inputs imported from the manuscript directory `2026_1d_blood_flow`.
Changes for future paper runs belong here; generated `parameters.prm` files
and manuscript copies are not a second source of configuration.

See [the reproduction guide](../doc/reproducibility.md) for downloading,
preparing, running and recording experiments. These inputs target the current
code and SI dataset releases. They do not certify archived paper results.

The current [transient parameter policy](../doc/transient_parameters.md) applies
the startup controls from PR #80 to future paper runs, including initialization
benchmarks. MMS retains uniform scalar tolerances. Earlier numerical results
must be regenerated with these inputs before the paper is finalized.

Use `scripts/submit_paper_gpu.py` for Slurm campaigns, passing the partition, machine
profile label and environment script for your system. Node selection is left
to the scheduler unless `--node` is specified.
Use `scripts/finalize_paper_output.py` to export a completed campaign into a
manuscript machine profile. See the reproduction guide for portable examples,
provenance checks, resources, timing definitions and the separate field-output stage.
