# Configuration

The application uses deal.II's `ParameterAcceptor`. `BloodFlowSystem<1, 3>::initialize_params()` reads the file passed on the command line. For compatibility, its registered parameter section remains named `MetricFlowSystem<1, 3>`; it contains an `IDA parameters` subsection and the `Metric Flow Parameters` constants section. Example files under `parameters/` are the best source for complete syntax; files from older experiments can contain legacy sections and should not be assumed interchangeable.

## System parameters

The following entries are registered by `BloodFlowSystem` in `include/metric_flow_x/blood_flow_system.h` and `source/blood_flow_system.cc`:

| Entry | Purpose |
|---|---|
| `Finite element degree` | Polynomial degree for the cell DG field. |
| `Problem constants` | Constants parameter group used by the model. |
| `Output filename` | Base name for generated visualization output. |
| `Output directory` | Directory prefix for generated output; empty means the current working directory. |
| `Output time interval` | Positive finite values set the interval between written outputs; zero writes at every IDA callback. Positive intervals must be representable at the configured time scale. |
| `Use direct solver` | Select MUMPS, serial UMFPACK for PETSc without MUMPS, or the Trilinos direct path; `false` selects GMRES+ILU. |
| `Number of refinement cycles` | Number of solve/refinement cycles in `run()`. |
| `Number of global refinement` | Initial global mesh refinement count. |
| `Theta (penalty parameter)` | Interior penalty/stability parameter used by assembly. |
| `Theta Boundary (stability parameter)` | Boundary stability parameter. |
| `Gamma (Total pressure factor)` | Total-pressure factor used in junction coupling. |
| `Verbosity (console depth)` | deal.II log depth; rank zero controls the visible log. |
| `Print current statistics` | Print an aligned IDA progress row at every accepted output callback (default `true`), independently of file-output cadence. |
| `Numerical flux type` | Exactly `HLL`, `HLL_HDG`, or `LAX_FRIEDRICHS`. Other strings are rejected. |
| `Use Riemann Invariants` | Select the configured boundary treatment using Riemann invariants. |
| `Outlet boundary condition type` | `RCR` or `Reflection` for characteristic boundaries; `Dirichlet` prescribes area and velocity from the first two `Exact solution` components at every physical endpoint, including the inlet, and allocates no capacitor unknowns. |
| `Use junction mesh` | Select the mesh construction mode for network junctions. |
| `Vtk file path for mesh input` | Legacy VTK network mesh read during setup. |

The `Metric Flow Parameters` group currently registers the source names `rho`, `mu`, `xi`, `m`, and `Rt`, displayed in parameter files as `Density (rho)`, `Viscosity coefficient (mu)`, `Profile constant for friction term (xi)`, `Tube law exponent (m)`, and `Reflection coefficient at outflow boundary (Rt)`.

For PETSc builds without MUMPS, serial direct solves use deal.II's pivoting
UMFPACK solver on a solver-owned copy of the assembled operator. Its ordinary
factorization controls leave the original Jacobian, initial state and IDA error
weights unchanged. This path requires deal.II with UMFPACK; otherwise select the
iterative solver. Parallel direct solves still require PETSc with MUMPS.

## IDA parameters

`BloodFlowSystem` delegates the nested `IDA parameters` group to deal.II's `SUNDIALS::IDA` parameter support. Current repository examples use:

- `Initial time`, `Final time`, and `Time interval between each output`;
- `Error control/Absolute error tolerance` and `Relative error tolerance`;
- `Running parameters/Initial step size`, `Minimum step size`, `Maximum step size`, `Maximum number of nonlinear iterations`, and `Maximum order of BDF`.

The standalone driver applies `Minimum step size` and `Maximum step size`
through native `IDASetMinStep` / `IDASetMaxStep` during every IDA setup, before
initial-condition correction or integration. Both must be finite and nonnegative;
zero disables the corresponding bound. A positive maximum must be at least the
minimum. The existing deal.II minimum parameter is the single source of its
value; `Maximum step size` is added by the executable in the same subsection.
Previously the wrapper did not forward the minimum, so parameter files that
relied on smaller automatic steps may now fail correctly. Set an appropriately
small minimum, or zero to leave it unrestricted. IDA may truncate its last step
below the minimum to reach a forced stop such as the final time or ramp cutoff.
These integration bounds are not applied to either stationary mode.

```text
subsection MetricFlowSystem<1, 3>
  set Print current statistics = true
  subsection IDA parameters
    subsection Running parameters
      set Minimum step size = 0
      set Maximum step size = 0.001
    end
  end
end
```

Additional IDA entries may be accepted by the installed deal.II version. Consult the generated `last_used_parameters.prm` written after initialization and the matching deal.II API for the full backend-specific set. Do not copy the older `ARKOde parameters` subsection from historical sample files into a new IDA run without checking it: the current application constructs `SUNDIALS::IDA`.

## Example

The configured aortic template is a useful starting point:

```text
subsection MetricFlowSystem<1, 3>
  set Numerical flux type = HLL
  set Use direct solver = true
  set Vtk file path for mesh input = .../parameters/aortic.vtk
  subsection IDA parameters
    set Initial time = 0
    set Final time = 1.1
  end
end
```

Use the exact spelling and subsection nesting shown in the parameter files. Parameter defaults in C++ and values in examples are not a scientific calibration.


## Component tolerances and sonic-regime diagnostics

The existing scalar IDA tolerance remains the default. For SI states, enable
`Use componentwise absolute tolerances = true` in `MetricFlowSystem<1, 3>`.
`Area absolute tolerance fraction` (default 1e-3) multiplies the local reference
area for both A and A_hat; tapered cells use the smaller endpoint reference area.
`Velocity absolute tolerance` defaults to 1e-4 m/s for U and U_hat, and
`Capacitor pressure absolute tolerance` defaults to 1 Pa. All scales must be
finite and positive. The IDA relative tolerance and algebraic-error-control
setting remain configurable through their existing parameters.

Set `Write flow regime diagnostics = true` to write `flow_regime.csv` alongside
other outputs. Each row records SI time, vessel ID and the sampled maximum
absolute U/c across cell quadrature points and face traces. Values are reduced
across MPI ranks before rank zero writes them. The configurable
`Mach warning threshold` defaults to 0.8; warnings occur on upward crossings.
The same option warns about mesh a_d values differing from pi*r_d^2 by more
than 2 percent. Diagnostics use the existing physical-output cadence.

These controls improve scaling and expose approach to the sonic fold. They do
not establish that a subsonic trace solution exists beyond that fold, and a
lower BDF order alone is not a sonic-flow remedy. The softened-wall reproduction
in issue #14 still requires a separate physical/numerical assessment. Short,
abrupt vessel constrictions require geometric assessment in addition to the
area/radius consistency check.
