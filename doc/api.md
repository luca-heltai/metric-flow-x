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
