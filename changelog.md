# Changelog

## [Unreleased]

### Added

- [PR #97](https://github.com/luca-heltai/metric-flow-x/pull/97) Add `Coupling output directory`, `Coupling output fields` and `--coupling-output-directory` to write only selected flow fields at each output time after one layout checkpoint, and `BloodFlowSystem::save_coupling_fields()` and `load_coupling_field()` to write and read them on a loaded object.
- [PR #97](https://github.com/luca-heltai/metric-flow-x/pull/97) Add `BloodFlowSystem::load_checkpoint_state()` to replay the time and state pair of another checkpoint of the same history into a loaded object, checking the mesh and finite element association without parameter parsing or system setup.

- Add self-contained native transient checkpoint/restart, fixed-output history, and composable mesh serialization hooks using deal.II save/load and SolutionTransfer.

- [PR #93](https://github.com/luca-heltai/metric-flow-x/pull/93) Add optional physical-flux transmission across ordinary interior faces, preserving stationary pressure steps with distinct side traces.

- [PR #90](https://github.com/luca-heltai/metric-flow-x/pull/90) Configure the native minimum positive area through numerical parameters, with the existing `1e-10 m^2` default and collective setup validation.
- [PR #88](https://github.com/luca-heltai/metric-flow-x/pull/88) Add explicit sealed endpoints to in-memory networks, with zero outward flow and the outgoing characteristic for either endpoint orientation.

### Fixed

- Select the unified SolutionTransfer API on deal.II 9.7 and newer to avoid checkpoint compilation deprecation warnings, retaining the distributed API for older versions.

- [PR #92](https://github.com/luca-heltai/metric-flow-x/pull/92) Orient HLL characteristic speeds and their state and external-pressure derivatives with each outward face normal.

- [PR #89](https://github.com/luca-heltai/metric-flow-x/pull/89) Evaluate spatial external pressure at each area support point in nodal pressure output, preserving providers that are constant on each cell.
