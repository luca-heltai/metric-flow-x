# Vessel partitioning and system ownership

`Mesh partitioning = graph` partitions the network once during mesh creation,
before the distributed mesh and solver state are created. It requires a deal.II
build with METIS. Builds without METIS default to `zorder`; explicitly selecting
`graph` in a parallel run reports the missing dependency. A supplied custom mesh
initializer retains control of partitioning.

The graph groups descendants of each straight input vessel (coarse mesh cell).
Vessels that fit within the nominal cells-per-rank target remain intact. Larger
refined vessels are divided into contiguous pieces in their one-dimensional
refinement-tree order. A coarse input cell is indivisible without refinement.
This differs from keeping p4est sibling pairs together: a refined single vessel
can use all ranks when there are enough active cells.

Vertex weights count FE unknowns, `2*(degree+1)+4` per active cell, plus one per
local capacitor pressure. They approximate storage and assembly load; they do
not predict nonlinear iterations or direct-factorization fill. Shared mesh
vertices produce graph edges with `Partition junction weight` (default 4).
Peripheral resistor links produce edges with `Partition terminal connection
weight` (default 1). These weights express communication priorities, not physical
conductances. Setting both to one treats all graph adjacencies equally.

METIS uses a fixed seed, four candidate cuts and minimum subdomain connectivity.
Only rank zero runs the partitioner and broadcasts labels. The contiguous option
is enabled when the complete graph is connected. A part connected through an
arteria-vein peripheral link need not be connected along the physical vessels.
Disconnected input graphs and indivisible heavy vessels limit attainable
connectivity and balance. `Partition imbalance tolerance` defaults to 0.03 and
bounds the requested maximum load relative to the mean, not each part's minimum.
If METIS leaves a part empty despite there being enough graph units, a
unit is transferred from the heaviest part containing more than one unit. The
removal minimizes the donor's remaining component count, then unit weight;
this repair cannot increase the maximum load. Empty ranks remain necessary
when there are fewer active cells than ranks.

Geometry is not an objective in this first implementation; anatomical proximity
alone does not imply vascular or algebraic adjacency.

## Coupled system numbering

All unknowns belong to one hp-enabled `DoFHandler`. Both elements in the
`hp::FECollection` have five components:

- components 0 and 1 retain `FE_DGQ(degree)` for cell area and velocity;
- components 2 and 3 retain `FE_DGQ(1)` for the face traces;
- component 4 is `FE_Nothing` on interior cells, at inlets, and at outlets
  without a capacitor; it is `FE_DGQ(0)` on RCR terminal cells with `C > 0`.

`distribute_dofs()` supplies the system numbering, ownership and geometric
ghost indices. State vectors, FE evaluation, matrices and visualization use
the same indices. Callers use `cell->get_dof_indices(indices)` and
`cell->get_fe()` for local dimensions and shape functions. The previous
`system_dof_index()` and `get_system_dof_indices()` conversion API is removed.
`component_dofs(Component::capacitor_pressure)` selects the locally owned
terminal pressures through a standard component mask.

A capacitor belongs to its terminal cell's owner. `Reflection`, `Dirichlet`
and `C = 0` outlets allocate no component-4 DoFs. A serial input cell with two
capacitive endpoints is refined once before partitioning, since two independent
terminal pressures require two DG0 cells. Custom mesh initializers must provide
at most one capacitor terminal per cell; this condition is checked collectively.
Visualization exports `Pc`, extended by zero on `FE_Nothing` cells, as a standard
FE field. Assembly selects `hp::FEValues` and `hp::FEFaceValues` for each active
element and sizes local vectors and matrices from the current cell.

Ghost sets include distant terminal-link endpoints. Capacitor derivatives are
read locally during assembly; the former all-reduction of all capacitor time
derivatives is unnecessary. Empty ranks still participate in all mesh and vector
collectives. The native MUMPS wrapper's separate restriction on empty ranks
continues to apply when there are fewer cells than ranks.

`benchmark_assembly` reports owned cells, ghosts, full system DoFs and capacitor
DoFs per rank, together with residual and Jacobian timings. Compare `graph` and
`zorder` on identical inputs and MPI layouts, then verify native trajectories:
partition metrics alone do not establish a simulation speedup. Global system
indices change with the partition and capacitor ownership; compare numerical
states by vessel/cell and component, not by raw global row number.

## Validation of the hp terminal field

With deal.II 9.8.0, GCC 12.3.0 and OpenMPI 5.0.10:

- PETSc Debug/Release passes 112/112 CTest entries (104 tests and eight build
  dependencies). Trilinos Release passes the 12/12 focused entries (nine tests
  and three build dependencies), including serial, MPI2 and MPI4.
- The hp regression checks the native fifth-component indices, ownership,
  evaluation on ghost cells, zero extension through `FE_Nothing`, empty ranks,
  repeated setup, degrees one/two, and splitting two capacitive endpoints.
  The terminal-link test also exercises refined degree-two cells and both
  implicit Jacobian blocks against finite differences.
- All 14 stationary graph/HDG application cases and two input rejection checks
  pass. Direct inspection of ten serial VTU outputs confirms that `Pc` is zero
  without capacitance and on interior cells, and constant on terminal cells.
- A rest-state RCR transient completes five IDA steps with the existing
  four-component exact solution: area L2 error is zero and the other physical
  field errors are below `8e-18`.
- The MPI2 assembly benchmark reports 129 native DoFs on 16 cells, owned as
  `[64, 65]`, with capacitor counts `[0, 1]`. This checks the migrated tool;
  it is not a performance comparison.

clang-format 16.0.6, whitespace and path-naming checks pass. Sphinx builds the
updated documentation with warnings treated as errors. Hosted CI on other
library versions is separate from these local results.

## Reference ADAVN comparison

These archived measurements were made at commit `7ebd35b`, before the hp
terminal-field representation. They describe the vessel partitioner and the
earlier capacitor layout; they are not fresh validation or timing results for
the hp implementation.

The [32-rank receipt](performance_data/partitioning/adavn32.json) records a run
on the same 8,804-cell anatomical input used by the full-network tutorial:

| Metric | Z-order | Weighted graph |
| --- | ---: | ---: |
| Owned cells per rank | 275–276 | 225–285 |
| Maximum full-system DoFs / mean | 1.0266 | 1.0298 |
| Total ghost cells | 1,001 | 513 |
| Vascular components summed over all parts | 309 | 124 |
| Maximum vascular components in one part | 69 | 11 |
| Peripheral connections crossing ranks | 1,662 | 569 |
| Maximum neighboring ranks | 27 | 10 |
| Maximum capacitor DoFs in one rank | 115 | 63 |

Both columns use the new local capacitor ownership. Before this change all
1,576 capacitor DoFs belonged to one rank. The graph balances full-system DoFs,
so its maximum cell count alone can exceed the requested 3% DoF imbalance.
All 32 graph parts are connected when peripheral links count as graph edges;
they need not be connected through vessels alone.

The native-MUMPS transient completed ten cycles (8 s), with 1,005 IDA steps,
6,036 nonlinear iterations and 401 output times. All 3,208 selected probe rows
agree with the archived Z-order trajectory to roundoff: maximum pressure
difference 5.46e-11 Pa. This validates the selected probes, not every field at
every quadrature point. At the final time, all 88,040 exported endpoint samples of area, velocity,
traces and pressure are identical at VTU Float32 precision. Benchmark residual
norms are identical. Timings were
measured on a shared node and are not controlled speedup measurements.

The capacitor-ownership regression was first added to the original terminal
connection test and run against the base revision at two ranks. It failed with
`Capacitor must be owned by its terminal cell.` The updated test also checks
both implicit Jacobian blocks against finite differences, native consistent
initialization and distant linked terminals, in serial and at 2/4 ranks.
The partition regression covers deterministic labels, preservation of small
vessels, splitting of large refined vessels and the repair of empty parts.

Local validation used deal.II 9.8.0, GCC 12.3.0 and OpenMPI 5.0.10. PETSc
Debug/Release passed 98/98 CTest entries (94 tests and four build dependencies).
The focused Trilinos Release selection passed 8/8 entries (six tests and two
build dependencies), including serial, MPI2 and MPI4. clang-format 16.0.6,
whitespace and path-naming checks passed. Hosted CI on other dependency versions
is separate from these local results.
