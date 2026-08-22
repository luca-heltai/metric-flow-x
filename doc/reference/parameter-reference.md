# Parameter reference

This page is a compact reference for entries registered by the current `MetricFlowSystem<1, 3>` source. It is intentionally not an exhaustive transcription of every historical file under `parameters/`.

## `MetricFlowSystem<1, 3>`

| Entry | Registered meaning |
|---|---|
| `Finite element degree` | Cell DG polynomial degree. |
| `Problem constants` | `ParsedTools::Constants` group. |
| `Output filename` | Visualization output base name. |
| `Output directory` | Output directory prefix. |
| `Use direct solver` | Direct PETSc/MUMPS or Trilinos solver when true; GMRES+ILU when false. |
| `Number of refinement cycles` | Number of cycles in the run loop. |
| `Number of global refinement` | Initial global refinement count. |
| `Theta (penalty parameter)` | Interior penalty parameter. |
| `Theta Boundary (stability parameter)` | Boundary stability parameter. |
| `Gamma (Total pressure factor)` | Junction total-pressure factor. |
| `Verbosity (console depth)` | deal.II console/file log depth. |
| `Numerical flux type` | `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS`. |
| `Use Riemann Invariants` | Boundary-treatment switch. |
| `Outlet boundary condition type` | Outlet boundary mode, such as `RCR`. |
| `Vtk file path for mesh input` | Input legacy VTK network path. |

## `Metric Flow Parameters`

The source registers these named constants:

| Parameter-file entry | Source symbol |
|---|---|
| `Density (rho)` | `rho` |
| `Viscosity coefficient (mu)` | `mu` |
| `Profile constant for friction term (xi)` | `xi` |
| `Tube law exponent (m)` | `m` |
| `Reflection coefficient at outflow boundary (Rt)` | `Rt` |

## `IDA parameters`

The application enters an `IDA parameters` subsection and delegates registration to deal.II's `SUNDIALS::IDA` parameter object. Repository examples use `Initial time`, `Final time`, `Time interval between each output`, nested `Error control` tolerances, and nested `Running parameters` for step-size, BDF-order, and nonlinear-iteration controls. The installed deal.II version is authoritative for additional IDA entries.

See [Configuration](../configuration.md) for context and [the aortic template](../../parameters/aortic.prm.in) for a source-tree example. The configured file used by a build is `build/parameters/aortic.prm`.
