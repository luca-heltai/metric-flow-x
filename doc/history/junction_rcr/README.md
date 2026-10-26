---
orphan: true
---

# historical RCR junction inputs and output artifacts

The two incremental RCR-outlet commits after the junction-model branch.

These are historical research materials, not current solver inputs or newly reproduced results. Parameter syntax, units, solver settings and plot labels are preserved as recorded; no new scientific validation or equivalence to a current IDA run is claimed.

## Resolution against current main

Current RCR circuits are part of the coupled IDA residual, with capacitor states and a tested zero-capacitance limit. The old driver changes are superseded; their inputs and plots are retained as historical evidence, not copied into current configured examples.

The branch changes only this documentation directory. The current library, applications, configured parameter templates, tests and main manuscript remain byte-for-byte unchanged.

## Provenance

- Original branch: `junction-rcr`.
- Original head: [`2c3775b9`](https://github.com/luca-heltai/metric-flow-x/tree/2c3775b994b3ca12dbe8baff48d2e4399957cf34).
- Original incremental base: `47ce48629b3dafe0422316084bc8c904624ddd14` (`blood_flow_model_with_junction`).
- Rebased onto main: `456e5bf906b6cc2ecc904b9354ff60b48297d2e4`.
- [File mapping and SHA-256 hashes](provenance.json) identify every retained file, the original commit series and the conflict decisions.
- `original/` preserves the selected manuscript, parameter and output bytes. Only file and directory names were normalized to the current repository convention.
- The complete pre-rebase history, including superseded source code, is recoverable from the pre-rebase Git bundle saved during this audit.

The archive is deliberately outside `parameters/` and `latex/`: historical settings are not configured as runnable examples and old drafts do not replace the current manuscript.

## Files retained

| Original path | Archived path |
|---|---|
| `parameters/energy.png` | [energy.png](original/parameters/energy.png) |
| `parameters/for_rcr_outlet/junction_0.prm` | [junction_0.prm](original/parameters/for_rcr_outlet/junction_0.prm) |
| `parameters/for_rcr_outlet/junction_1.prm` | [junction_1.prm](original/parameters/for_rcr_outlet/junction_1.prm) |
| `parameters/mass_animation.mp4` | [mass_animation.mp4](original/parameters/mass_animation.mp4) |
