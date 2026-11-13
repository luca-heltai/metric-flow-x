# API

The generated C++ API reference is available in the [Library Reference](api/library_root).
It is generated from the current headers and source files by Doxygen, Breathe,
and Exhale.


## An externally prepared network

Prepare the final `flow.triangulation()` before distributing flow DoFs, then
call the collective `setup(const NetworkPhysics &)`. This path reads no mesh,
VTK attributes or terminal CSV. It retains the mesh's material ids, boundary ids,
orientation, refinement and partition. The caller supplies the usual native
vertex-adjacent ghost layer. The existing `setup(mesh_initializer)` still starts
from a file's physical data and is appropriate when retaining an imported mesh.

`NetworkPhysics::vessels` maps compact native material ids `0,...,N-1` to
`VesselProperties`. Keep external physical identifiers separately; do not narrow
a 64-bit physical id into `types::material_id`. Every locally relevant flow
cell must have an entry. The coefficient tables must be identical on every
process, including processes with no cells. Setup checks this collectively.
These tables contain no vertex coordinates or cell connectivity. The prepared
mesh remains distributed and setup does not construct a replicated coarse graph.
The existing global reductions compute straight-vessel arclength bounds for
native taper interpolation across partitions.

The coefficient value types support deal.II's `Utilities::pack`,
`Utilities::unpack` and MPI exchanges. A distributed mesh owner can therefore
prepare vessel coefficients at their physical owners and exchange only these
records to build the replicated native tables. This communication does not
require the physical graph's coordinates or interior connectivity. These are
in-process coefficient records, not a versioned persistent network format.

For example, after preparing a single straight `0.1 m` vessel with material id
zero, inflow boundary zero and terminal boundary one:

```cpp
using Flow = MetricFlowX::BloodFlowSystem<1, 3>;
Flow::NetworkPhysics physics;
auto &v = physics.vessels[0];
v.r_d = v.r_in = v.r_out = 0.003;       // m
v.a0 = v.a_d = dealii::numbers::PI * v.r_d * v.r_d;
v.E = 5.e5;                           // Pa
v.h_wall = 0.0003;                    // m
v.p_d = v.p0 = 0.;                    // Pa
v.L = 0.1;                           // m
v.dynamic_viscosity = 0.004;          // Pa s
physics.terminals.emplace(1, Flow::RCRPhysics{0., 1.e8, 0., 0., "terminal"});
flow.setup(physics);
auto y = flow.make_state(), ydot = flow.make_state();
flow.initialize_consistent_state(y, ydot, 0.);
```

These synthetic coefficients are not an organ calibration. The example uses
`Wall model = Square root`, `Outlet boundary condition type = RCR`, and
`Initial state source = rest`. ADAVN input uses the existing `WallLaw` inside
`VesselProperties`; its type must agree with the configured wall model and its
reference state must have a finite positive tangent. No radius or physical
coefficient is clipped during transfer. The transient solver's area-domain
requirements still apply to the supplied reference and evolved states.

`NetworkPhysics::terminals` supplies native `RCRPhysics`, keyed by the mesh's
terminal boundary ids. Resistance `R1`/`R2` is in Pa s/m³, capacitance `C` in
m³/Pa and `P_out` in Pa. `R1` and `C` are nonnegative; `R2` is positive. Each
entry must identify exactly one degree-one endpoint. Boundary zero is reserved
for inflow and 255 for junctions. A connected venous endpoint can use face zero;
its native outward sign is retained.

Use `physics.sealed_terminals.insert(boundary_id)` for an imaging endpoint that
must have exactly zero outward flow. Every non-inflow endpoint must belong to
exactly one of `terminals` and `sealed_terminals`; boundary ids 0 and 255 remain
reserved. A sealed endpoint cannot participate in `terminal_connections`. It
preserves the outgoing characteristic for either endpoint orientation and adds
no capacitor pressure, resistance, or downstream reservoir. For example, to
replace terminal one in the example above with a closed end, erase it from
`physics.terminals`, insert one into `physics.sealed_terminals`, then call
`flow.setup(physics)`. A prescribed nonzero inflow into an otherwise sealed tube
changes its stored volume, so its startup need not be steady.

`terminal_connections` supplies `{boundary0, boundary1, resistance}` records for
the same native terminal resistors used by `Terminal connections CSV file`.
The resistance must be positive and both endpoints must be declared terminals.
A linked endpoint without a capacitor requires `R1=0`. These links require RCR
boundaries and elastic walls. They do not introduce a second terminal model or
an extra reservoir leak. See [initialization](initialization.md) for their
pressure and flow conventions.

This path supports the native rest seed and subsequent consistency preparation.
`Initial state source = steady network` requires the original coarse hydraulic
graph and is rejected for in-memory setup. Supplying a distributed mesh does not
implicitly collect that graph to enable the optional warm start. The external
application can initialize the native pair and advance its existing IDA adapter
after all flow and coupling DoFs have been prepared.

## Wall reference area

Every wall law is evaluated relative to the tapered diastolic area

$$
a_d(x)=\pi r(x)^2,\qquad r(x)=r_{\rm in}+\xi(x)\,(r_{\rm out}-r_{\rm in}),
$$

where $\xi\in[0,1]$ is the arc-length coordinate along the vessel. Without taper
data ($r_{\rm in}$ or $r_{\rm out}$ not positive) the vessel value `a_d` is used.
The square-root law gives the transmural pressure $p_0+p_d$ at $A=a_d(x)$; the
ADAVN laws take $a_d(x)$ as the area of zero wall strain. A coupled wall is
therefore in the flow model's reference configuration at $A=a_d(x)$. This
configuration is not stress free when the law's pressure there is nonzero, for
example when $p_0+p_d\neq0$ in the square-root law.

`a0` is read from the input, validated as positive and archived in checkpoints,
but it does not enter the flow equations. Couplings should not use it as the
wall reference.

The reference area is available on any set-up object, including one restored by
`load_checkpoint()`:

```cpp
const double a = flow.reference_area(cell, x); // cell of flow.dof_handler()

auto area = flow.make_state();
flow.interpolate_reference_area(area);
// Cell area and area trace DoFs hold a_d; all other DoFs are zero.
```

The vector has the layout of `state()`, so the area components of the two can
be compared entry by entry, for example to form
$\sqrt{A/\pi}-\sqrt{a_d/\pi}$.

## A sparse external-pressure derivative

For a pressure field owned by a coupled application, use
`visit_external_pressure_jacobian(t, y, basis, visitor)` to assemble its
cross-Jacobian. The native flow state `y` is held fixed. The basis callback
returns the nonzero derivatives of pressure at a `PressureEvaluationPoint`
with respect to the application's global pressure coefficients. The visitor
receives additive `(native_row, pressure_column, value)` entries.

For example, one pressure coefficient per compact native vessel id has the
following basis:

```cpp
using Flow = MetricFlowX::BloodFlowSystem<1, 3>;
using Index = dealii::types::global_dof_index;
Flow::ExternalPressureBasisProvider basis =
  [](const Flow::PressureEvaluationPoint &point) {
    return Flow::ExternalPressureBasis{{point.vessel_id, 1.}};
  };
std::vector<std::tuple<Index, Index, double>> entries;
flow.visit_external_pressure_jacobian(
  t, y, basis, [&](Index row, Index column, double value) {
    entries.emplace_back(row, column, value);
  });
```

Each row belongs to the calling rank's native flow partition. Columns belong
to the caller's coefficient space; repeated entries must be summed. The
collected entries can supply a rectangular sparsity pattern and matrix without
probing every global coefficient. A finite element provider can instead return
the nonzero shape values and global coefficient indices on the requested cell,
including any cell-local projection.

This is a collective operation. The provider only needs locally visible cells.
For distant connected terminals, MetricFlowX evaluates the stencil on the
native endpoint owner and exchanges it through the existing terminal layout.
Cell, ordinary-face, boundary, junction and connected-capacitor contributions
use the same signs and numerical pressure flux as
`add_external_pressure_residual`.

HLL pressure weights depend on the candidate native state. Reassemble this
matrix when that state changes. It is separate from
`add_external_pressure_state_jacobian`, which differentiates with respect to
native flow variables while holding prescribed pressures fixed. Neither
operation installs a pressure provider or adds a time-derivative term.
