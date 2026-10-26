---
orphan: true
---

# historical ARKode carotid results and solver settings

The three ARKode single-vessel/common-carotid commits from April 2026.

These are historical research materials, not current solver inputs or newly reproduced results. Parameter syntax, units, solver settings and plot labels are preserved as recorded; no new scientific validation or equivalence to a current IDA run is claimed.

## Resolution against current main

The source changes silence debug output, use the callback step number and replace solve_ode_incrementally with solve_ode in the old ARKode driver. That driver no longer exists: the current application uses IDA and its own tested output scheduler. These changes have no current ARKode call site to port. ARKode settings cannot be relabeled as IDA settings without a new numerical validation.

The branch changes only this documentation directory. The current library, applications, configured parameter templates, tests and main manuscript remain byte-for-byte unchanged.

## Provenance

- Original branch: `junction-arkode`.
- Original head: [`396ed297`](https://github.com/luca-heltai/metric-flow-x/tree/396ed2975925dc20a401d063b14304d9e5d3db3b).
- Original incremental base: `c94747cc29d312027d2b3a059711e8dafc0c2407`.
- Rebased onto main: `456e5bf906b6cc2ecc904b9354ff60b48297d2e4`.
- [File mapping and SHA-256 hashes](provenance.json) identify every retained file, the original commit series and the conflict decisions.
- `original/` preserves the selected manuscript, parameter and output bytes. Only file and directory names were normalized to the current repository convention.
- The complete pre-rebase history, including superseded source code, is recoverable from the pre-rebase Git bundle saved during this audit.

The archive is deliberately outside `parameters/` and `latex/`: historical settings are not configured as runnable examples and old drafts do not replace the current manuscript.

## Files retained

| Original path | Archived path |
|---|---|
| `parameters/benchmark-parameters/common-carotid-artery/commom_carotid_artery.prm` | [commom_carotid_artery.prm](original/parameters/benchmark_parameters/common_carotid_artery/commom_carotid_artery.prm) |
| `parameters/benchmark-parameters/common-carotid-artery/pressure_at_Lby2.png` | [pressure_at_lby2.png](original/parameters/benchmark_parameters/common_carotid_artery/pressure_at_lby2.png) |
| `parameters/benchmark-parameters/common-carotid-artery/pressure_difference.csv` | [pressure_difference.csv](original/parameters/benchmark_parameters/common_carotid_artery/pressure_difference.csv) |
| `parameters/benchmark-parameters/common-carotid-artery/pressure_difference.png` | [pressure_difference.png](original/parameters/benchmark_parameters/common_carotid_artery/pressure_difference.png) |
| `parameters/benchmark-parameters/common-carotid-artery/radius_change_mm.png` | [radius_change_mm.png](original/parameters/benchmark_parameters/common_carotid_artery/radius_change_mm.png) |
| `parameters/benchmark-parameters/common-carotid-artery/volume_at_lby2.png` | [volume_at_lby2.png](original/parameters/benchmark_parameters/common_carotid_artery/volume_at_lby2.png) |
