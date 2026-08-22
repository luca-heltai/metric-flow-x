# MetricFlow-X

This repository contains a C++/deal.II implementation of an MPI-parallel implicit solver for one-dimensional **navier-stokes flow models** defined on **metric networks embedded in three-dimensional space**. The space discretization is based on an HDG-type (hybridized) monolithic system, which is then solved in time via the SUNDIALS Differential-Algebraic equation solver (IDA).

## Authors

[Devi Raksha](https://github.com/devi-raksha) and [Luca Heltai](https://github.com/luca-heltai)

## Overview

- **Geometry:** 1D network geometry can be defined through VTK files, embedded in `spacedim = 3`.
- **Cell fields:** cross-sectional area `A` and mean axial velocity `U`, represented with discontinuous Galerkin finite elements of arbitrary order.
- **Hybridized fields:** face traces `A_hat` and `U_hat`; terminal RCR capacitor pressures are appended to the global unknown vector.
- **Time integration:** SUNDIALS **IDA** (`SUNDIALS::IDA`), with cell and capacitor rows treated as differential and trace rows as algebraic.
- **Numerical fluxes:** `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS` fluxes are supported via parameter files.
- **Newton linear solves:** with `Use direct solver = true`, PETSc uses `SparseDirectMUMPS` when PETSc is selected and Trilinos uses `TrilinosWrappers::SolverDirect` otherwise. With it set to `false`, the implementation uses GMRES with an ILU preconditioner.

## Prerequisites

A configured deal.II installation (deal.II 9.8.0 or newer), CMake 3.23 or newer, a C++ compiler, and the MPI/linear-algebra support provided by deal.II are required. The selected deal.II build must provide PETSc or Trilinos; the source rejects configurations with neither backend. Doxygen and the Python packages in `doc/requirements.txt` are only needed to build the developer documentation.

## Configure and build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

CMake creates the `metric_flow_x` executable from `apps/metric_flow_x.cc`, builds the shared `test_library`, mirrors `parameters/` into `build/parameters/`, and configures the tests. Parameter templates ending in `.prm.in` are expanded while configuring; for example, `parameters/aortic.prm.in` becomes `build/parameters/aortic.prm` with the source-tree path substituted into the mesh setting.

## Run

The executable requires an existing parameter-file path for a normal run. After a successful configure, the configured aortic example can be invoked as:

```bash
./build/metric_flow_x build/parameters/aortic.prm
```

Command-line modes are:

```text
./build/metric_flow_x --help
./build/metric_flow_x --print-parameters
./build/metric_flow_x --validate-parameters build/parameters/aortic.prm
```

`--print-parameters` prints the schema registered by the existing
`MetricFlowSystem` and its existing parameter defaults. `--validate-parameters`
parses the given file against that schema without running the simulation. After running a successful simulation, a `last_used_parameters.prm` file is written to the current working directory. Missing parameter files are rejected.

See [Configuration](doc/configuration.md) for the registered parameter groups and [Outputs](doc/outputs.md) for the files written by the current implementation.

## Tests

CTest discovers the test executables configured by deal.II's `DEAL_II_PICKUP_TESTS()` macro. Run the configured build-tree tests with:

```bash
ctest --test-dir build --output-on-failure
```

## Repository map

- `apps/`: application entry points; currently `metric_flow_x.cc`.
- `include/`, `source/`: the `MetricFlowSystem` implementation, parameter handling, assembly, solvers, and VTK utilities.
- `parameters/`: input meshes, parameter files/templates, and example/reference assets. The configured copies used by a build are under `build/parameters/`.
- `tests/`: deal.II test sources and expected output files.
- `doc/`: this documentation skeleton and the Doxygen/Sphinx configuration.
- `latex/`: the repository's mathematical manuscript source.
- `scripts/`: helper scripts, including documentation serving/building and test/formatting helpers.

## Citing

If you use this software, please cite it as **MetricFlow-X solver**. The
canonical bibliography is `bibliography/references.bib`,
and its metadata policy and unresolved-key list are documented in
[`bibliography/README.md`](bibliography/README.md). The documentation reference
page is [References](doc/references.md).

## License

This project is licensed under the MIT License (see `LICENSE.md`).

## Funding

> - **Project:** dealii-X
> - **Grant agreement number:** 101172493
> - **DOI:** [10.3030/101172493](https://doi.org/10.3030/101172493)
> - **Programme:** Horizon Europe
> - **Funder:** EuroHPC JU