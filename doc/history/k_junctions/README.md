---
orphan: true
---

# historical K-vessel derivation and benchmark material

The six incremental arbitrary-K junction commits after junction-rcr, including the manuscript, single-vessel inputs and result images.

These are historical research materials, not current solver inputs or newly reproduced results. Parameter syntax, units, solver settings and plot labels are preserved as recorded; no new scientific validation or equivalence to a current IDA run is claimed.

## Resolution against current main

The vector-based K-vessel junction state and generic junction solver already exist in include/metric_flow_x/junction_solver.h. Current assemble_trace_junction_equations and assemble_jacobian_trace_junction_block implement the mass, total-head and characteristic equations for K incident vessels. Replacing them with the historical DG driver would regress the current HDG/IDA API.

The branch changes only this documentation directory. The current library, applications, configured parameter templates, tests and main manuscript remain byte-for-byte unchanged.

## Provenance

- Original branch: `K-branches`.
- Original head: [`161caa25`](https://github.com/luca-heltai/metric-flow-x/tree/161caa25447a0fe6f66d3d28546206f02ee7cdb8).
- Original incremental base: `2c3775b994b3ca12dbe8baff48d2e4399957cf34` (`junction-rcr`).
- Rebased onto main: `456e5bf906b6cc2ecc904b9354ff60b48297d2e4`.
- [File mapping and SHA-256 hashes](provenance.json) identify every retained file, the original commit series and the conflict decisions.
- `original/` preserves the selected manuscript, parameter and output bytes. Only file and directory names were normalized to the current repository convention.
- The complete pre-rebase history, including superseded source code, is recoverable from the pre-rebase Git bundle saved during this audit.

The archive is deliberately outside `parameters/` and `latex/`: historical settings are not configured as runnable examples and old drafts do not replace the current manuscript.

## Files retained

| Original path | Archived path |
|---|---|
| `latex/blood_flow.bib` | [blood_flow.bib](original/latex/blood_flow.bib) |
| `latex/blood_flow.tex` | [blood_flow.tex](original/latex/blood_flow.tex) |
| `parameters/benchmark-parameters/common_carotid_artery.prm` | [common_carotid_artery.prm](original/parameters/benchmark_parameters/common_carotid_artery.prm) |
| `parameters/benchmark-parameters/single-vessel(tpoint1).png` | [single_vessel_tpoint1_.png](original/parameters/benchmark_parameters/single_vessel_tpoint1_.png) |
| `parameters/benchmark-parameters/single-vessel(tpoint5).png` | [single_vessel_tpoint5_.png](original/parameters/benchmark_parameters/single_vessel_tpoint5_.png) |
| `parameters/benchmark-parameters/single-vessel(tpoint94).png` | [single_vessel_tpoint94_.png](original/parameters/benchmark_parameters/single_vessel_tpoint94_.png) |
| `parameters/benchmark-parameters/single-vessel-mu(t1point5).png` | [single_vessel_mu_t1point5_.png](original/parameters/benchmark_parameters/single_vessel_mu_t1point5_.png) |
| `parameters/benchmark-parameters/single-vessel-mu(tpoint1).png` | [single_vessel_mu_tpoint1_.png](original/parameters/benchmark_parameters/single_vessel_mu_tpoint1_.png) |
| `parameters/benchmark-parameters/single-vessel-mu.prm` | [single_vessel_mu.prm](original/parameters/benchmark_parameters/single_vessel_mu.prm) |
| `parameters/benchmark-parameters/single-vessel.prm` | [single_vessel.prm](original/parameters/benchmark_parameters/single_vessel.prm) |
| `parameters/benchmark-parameters/single_vessel.prm` | [single_vessel_dac0a41c.prm](original/parameters/benchmark_parameters/single_vessel_dac0a41c.prm) |
| `parameters/benchmark-parameters/single_vessel_mu.prm` | [single_vessel_mu_55b6b7c3.prm](original/parameters/benchmark_parameters/single_vessel_mu_55b6b7c3.prm) |
| `parameters/for_rcr_outlet/junction_0.prm` | [junction_0.prm](original/parameters/for_rcr_outlet/junction_0.prm) |
| `parameters/for_rcr_outlet/junction_1.prm` | [junction_1.prm](original/parameters/for_rcr_outlet/junction_1.prm) |
| `parameters/for_rcr_outlet/junction_2.prm` | [junction_2.prm](original/parameters/for_rcr_outlet/junction_2.prm) |
