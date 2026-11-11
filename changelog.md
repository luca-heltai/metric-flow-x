# Changelog

## [Unreleased]

### Added

- [PR #N](https://github.com/luca-heltai/metric-flow-x/pull/N) Add optional physical-flux transmission across ordinary interior faces, preserving stationary pressure steps with distinct side traces.

- [PR #90](https://github.com/luca-heltai/metric-flow-x/pull/90) Configure the native minimum positive area through numerical parameters, with the existing `1e-10 m^2` default and collective setup validation.
- [PR #88](https://github.com/luca-heltai/metric-flow-x/pull/88) Add explicit sealed endpoints to in-memory networks, with zero outward flow and the outgoing characteristic for either endpoint orientation.

### Fixed

- [PR #92](https://github.com/luca-heltai/metric-flow-x/pull/92) Orient HLL characteristic speeds and their state and external-pressure derivatives with each outward face normal.

- [PR #89](https://github.com/luca-heltai/metric-flow-x/pull/89) Evaluate spatial external pressure at each area support point in nodal pressure output, preserving providers that are constant on each cell.
