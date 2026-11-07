# Changelog

## [Unreleased]

### Added

- Configure the native minimum positive area through numerical parameters, with the existing `1e-10 m^2` default and collective setup validation.
- [PR #88](https://github.com/luca-heltai/metric-flow-x/pull/88) Add explicit sealed endpoints to in-memory networks, with zero outward flow and the outgoing characteristic for either endpoint orientation.

### Fixed

- [PR #89](https://github.com/luca-heltai/metric-flow-x/pull/89) Evaluate spatial external pressure at each area support point in nodal pressure output, preserving providers that are constant on each cell.
