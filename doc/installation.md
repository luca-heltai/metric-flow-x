# Installation and build/run

## Requirements

The top-level `CMakeLists.txt` requires CMake 3.23 or newer and searches for deal.II 9.5.0 or newer. The deal.II installation must expose at least one supported linear-algebra backend: PETSc or Trilinos. The application uses MPI through deal.II. A C++ compiler compatible with the selected deal.II installation is also required.

To build this documentation site, additionally install Doxygen and the Python requirements in `doc/requirements.txt` (Sphinx, Furo, Breathe, Exhale, MyST, Mermaid, and BibTeX integrations).

## Configure and build

From the repository root:

```bash
cmake -S . -B build
cmake --build build
```

CMake builds the `test_library` shared library and an executable for each source file in `apps/`; the current application target is `metric_flow_x`. It also copies `parameters/` into `build/parameters/`. `.prm.in` files are configured, so `aortic.prm.in` is available as `build/parameters/aortic.prm` after configuration.

The audit recorded successful configure and build commands in a checkout with deal.II 9.8.0-rc1. A different installation may select a different backend or expose different optional features.

## Run an example

Pass a configured parameter file explicitly:

```bash
./build/metric_flow_x build/parameters/aortic.prm
```

The aortic template points to `parameters/aortic.vtk` using a source-directory substitution performed by CMake. Output and IDA settings are in the same parameter file.


## Build-tree versus source-tree parameters

Use files under `build/parameters/` for a configured build. They include generated paths and mirror the source inputs at configure time. Re-run CMake after adding or changing `.prm.in` files when you need their configured copies refreshed.
