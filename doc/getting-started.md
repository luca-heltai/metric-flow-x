# Getting started

The shortest supported path is:

```bash
cmake -S . -B build
cmake --build build
./build/metric_flow_x build/parameters/aortic.prm
```

The first two commands configure and build the current CMake targets. CMake copies and configures parameter files into `build/parameters/`; the aortic `.prm.in` template therefore becomes `build/parameters/aortic.prm`. The final command is the application invocation form; a completed runtime solve depends on the local deal.II, MPI, mesh, and solver environment and is not asserted by this page.

```{include} installation.md
```

```{include} configuration.md
```

```{include} reference/parameter-reference.md
```

```{include} outputs.md
```

```{include} testing.md
```

```{include} documentation.md
```

Use [Installation and build/run](installation.md) for prerequisites, [Configuration](configuration.md) for registered entries, and [Testing](testing.md) for CTest.
