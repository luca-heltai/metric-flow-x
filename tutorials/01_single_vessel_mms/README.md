# Single-vessel manufactured-solution evidence

Current runnable paper inputs are maintained in
[`reproducibility/parameters/mms.prm.in`](../../reproducibility/parameters/mms.prm.in).
Use `scripts/reproduce_paper.py --suite mms` for the degree/refinement sweep;
see [the reproduction guide](../../doc/reproducibility.md).

The old p1/p2/p3 parameter templates have been removed together with the legacy
ARKode cases. They depended on obsolete vessel-constant and initial-condition
interfaces and did not provide reproducible MMS runs with the current solver.

`reference/convergence.txt` retains the supplied report;
`reference/convergence.json` records its displayed error/rate strings and SHA-256
provenance. These are earlier reference evidence, not results of current runs.
`reference/mms_expressions.json` and `tools/generate_mms_expressions.py` retain
symbolic derivations for the historical profiles. They are mathematical reference
artifacts, not parameter inputs for the current paper experiment.
