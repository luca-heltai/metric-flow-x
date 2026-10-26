---
orphan: true
---

# historical junction-model derivation and inputs

The early DG junction model, characteristic/Riemann boundary experiments, finite-difference Jacobian checks and accompanying derivation.

These are historical research materials, not current solver inputs or newly reproduced results. Parameter syntax, units, solver settings and plot labels are preserved as recorded; no new scientific validation or equivalence to a current IDA run is claimed.

## Resolution against current main

The current solver already assembles junction mass/head/characteristic equations in the global HDG/IDA system. Current cell, boundary, interior and junction Jacobian tests replace the old monolithic DG test. Replaying the earlier driver or its unqualified includes would remove current MPI, standalone-library and initialization support.

The branch changes only this documentation directory. The current library, applications, configured parameter templates, tests and main manuscript remain byte-for-byte unchanged.

## Provenance

- Original branch: `blood_flow_model_with_junction`.
- Original head: [`47ce4862`](https://github.com/luca-heltai/metric-flow-x/tree/47ce48629b3dafe0422316084bc8c904624ddd14).
- Original incremental base: `93d8d36cb89e9ece7021d57461d75b3ff267f4af`.
- Rebased onto main: `456e5bf906b6cc2ecc904b9354ff60b48297d2e4`.
- [File mapping and SHA-256 hashes](provenance.json) identify every retained file, the original commit series and the conflict decisions.
- `original/` preserves the selected manuscript, parameter and output bytes. Only file and directory names were normalized to the current repository convention.
- The complete pre-rebase history, including superseded source code, is recoverable from the pre-rebase Git bundle saved during this audit.

The archive is deliberately outside `parameters/` and `latex/`: historical settings are not configured as runnable examples and old drafts do not replace the current manuscript.

## Files retained

| Original path | Archived path |
|---|---|
| `latex/arterial_tree_bifurcation.png` | [arterial_tree_bifurcation.png](original/latex/arterial_tree_bifurcation.png) |
| `latex/blood_flow.tex` | [blood_flow.tex](original/latex/blood_flow.tex) |
| `parameters/constant.prm` | [constant.prm](original/parameters/constant.prm) |
| `parameters/jacobian_const.prm` | [jacobian_const.prm](original/parameters/jacobian_const.prm) |
| `parameters/jacobian_fd.prm` | [jacobian_fd.prm](original/parameters/jacobian_fd.prm) |
| `parameters/linear.prm` | [linear.prm](original/parameters/linear.prm) |
| `parameters/mass_integration.prm` | [mass_integration.prm](original/parameters/mass_integration.prm) |
| `parameters/mass_integration_velocity.prm` | [mass_integration_velocity.prm](original/parameters/mass_integration_velocity.prm) |
| `parameters/parameters_time.prm` | [parameters_time.prm](original/parameters/parameters_time.prm) |
| `parameters/quadratic.prm` | [quadratic.prm](original/parameters/quadratic.prm) |
| `parameters/visualization_parameters.pvsm` | [visualization_parameters.pvsm](original/parameters/visualization_parameters.pvsm) |
| `parameters/working_parameters/Hll_parameters1.prm` | [hll_parameters1.prm](original/parameters/working_parameters/hll_parameters1.prm) |
| `parameters/working_parameters/Hll_parameters2.prm` | [hll_parameters2.prm](original/parameters/working_parameters/hll_parameters2.prm) |
| `parameters/working_parameters/hll_parameters3.prm` | [hll_parameters3.prm](original/parameters/working_parameters/hll_parameters3.prm) |
| `parameters/working_parameters/parameters_1.prm` | [parameters_1.prm](original/parameters/working_parameters/parameters_1.prm) |
| `parameters/working_parameters/parameters_2.prm` | [parameters_2.prm](original/parameters/working_parameters/parameters_2.prm) |
| `parameters/working_parameters/parameters_3.prm` | [parameters_3.prm](original/parameters/working_parameters/parameters_3.prm) |
| `parameters/working_parameters/parameters_4.prm` | [parameters_4.prm](original/parameters/working_parameters/parameters_4.prm) |
