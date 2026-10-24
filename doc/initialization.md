# Initial conditions and a fast startup interval

The executable prepares a consistent pair `(y0, ydot0)` before calling the IDA
wrapper. It checks the native residual collectively and uses IDA's `none`
correction mode only after this check succeeds. `Initial condition policy = IDA`
retains an explicit IDA-correction path and honors the selected IDA correction
mode. Selecting `none` also requires native pair preparation. Failure to prepare
an admissible, consistent pair stops the run.

```text
subsection MetricFlowSystem<1, 3>
  set Initial state source = steady network
  set Initial condition policy = native
  set Initial consistency tolerance = 0.01
  set Initial consistency maximum iterations = 30
  set Check Jacobian at startup = false
end
```

`Initial state source = rest` remains the default physical seed. `steady network`
changes that seed, so its early trajectory need not agree with a simulation
started at rest. Both use the same native consistency checks. Console verbosity
no longer enables the expensive finite-difference Jacobian diagnostic; enable
`Check Jacobian at startup` explicitly when needed.

## The static 0D seed

Before mesh refinement, retain the original graph's vertex and coarse-cell
identities. At the configured initial time, solve a sparse resistive network for
nodal pressures and oriented edge flows. For the implemented friction profile,

$$
R_e=\frac{2(\xi+2)\mu}{\pi}\int_e\frac{ds}{r(s)^4},\qquad
Q_e=\frac{p_i-p_j}{R_e}.
$$

The fixed reference geometry is the same `a_d` or linear-radius taper used by
the 1D model. For `xi = 2`, this is Poiseuille's `8 mu L / (pi r^4)`. The tapered
integral is analytic. Resistances have units Pa·s/m³, pressures Pa, and flows
m³/s. Configured inlet flow is evaluated at `t0`, including tabulated inflow.
Each connected component must have a pressure reference through an RCR terminal;
cycles, parallel edges and separately anchored components are supported.

This mode requires positive viscosity and resistive RCR terminal conditions.
Use the existing VTK or parameter/CSV terminal source. The steady outlet relation
is `p = P_out + (R1+R2) Q`; initialize `Pc = P_out + R2 Q` when `C > 0`, and
allocate no capacitor state when `C = 0`. Missing anchors, invalid resistances,
nonfinite data and an inadmissible pressure-to-area inverse are errors.

Interpolate pressure according to accumulated resistance, invert the configured
compliant tube law including the current external pressure, and set nodal
velocities to `Q/A`. Preserve branch discontinuities and the native orientation.
The graph uses input coarse identities and straight-cell geometry; an external
mesh owner must preserve this correspondence.

This rigid pressure-continuity graph is a warm start. It does not solve the full
compliant HDG equations or the nonlinear total-head junction conditions.

## Native consistency and the first step

With cell and capacitor states fixed, solve the actual algebraic equations.
Their Jacobian separates into small face/junction blocks, including duplicate
trace-continuity rows. The correction factors these blocks instead of pinning
all differential rows and factorizing a full-system matrix. Backtracking keeps
areas positive and requires a decreasing residual.

Each equation is scaled by its trace-Jacobian sensitivity to the configured
final IDA state weights, `atol_j + rtol*abs(y_j)`. Thus flow, pressure and
continuity equations are checked in their own units. The default consistency
threshold is 0.01 of this weighted scale; it is separate from the startup ramp.

Reconstruct cell derivatives from the existing inverse mass blocks and capacitor
derivatives from `C*Pc_dot = Q - (Pc-P_out)/R2`. Algebraic derivatives are zero;
they do not enter the DAE residual. Check the complete `F(t0,y0,ydot0)` again,
including the mass-inverted differential defect over the initial step divided
by the final IDA state weights. Both scaled maxima must satisfy the configured
consistency threshold. Nonfinite states, derivatives, residuals and weights are
rejected collectively, including ranks without cells.

Library clients can call `initialize_consistent_state(y, ydot, t0)` directly.
`initialize_state` alone prepares a state; it does not supply a consistent pair.

## Looser tolerances during discarded startup

The optional ramp changes error weights while keeping the IDA history. The
configured IDA relative and absolute tolerances are the final values. Absolute
multipliers also apply to componentwise tolerances.

```text
subsection MetricFlowSystem<1, 3>
  set Use componentwise absolute tolerances = true
  set Tolerance ramp enabled = true
  set Tolerance ramp initial relative factor = 1000
  set Tolerance ramp initial absolute factor = 100
  set Tolerance ramp hold duration = 0.1
  set Tolerance ramp end time = 1.0
end
```

Hold duration is measured from `t0`; end time is an absolute simulation time.
After the hold, multipliers decrease geometrically to one. They must be finite
and at least one. The end must follow the hold, precede or equal final time, and
lie on the **IDA callback grid**, configured by `Time interval between each
output`. This is distinct from the file-output cadence. A stop at the cutoff
prevents an internal step with loose controls from crossing into the final
interval. Subsequent integration uses the original final controls.

Use componentwise tolerances for mixed area/velocity/pressure units; a large
scalar absolute tolerance can be inappropriate for small vessel areas. Set the
ramp end before the physical analysis window, leaving time for startup errors
to decay. Tight final local error controls do not by themselves remove a
previously accumulated transient. Validate the final window against a uniformly
tight run with the same initial state and forcing.

The adapter installs one custom error-weight function before initial IDA setup.
It neither resets IDA nor repeats `CalcIC` at ramp changes. This also avoids the
live scalar/vector tolerance-setter defect confirmed in installed SUNDIALS 6.7.
An enabled ramp requires the wrapper's differential-component setup hook:
`use_y_diff` for initialization/restart, or ignoring algebraic error terms.
Unsupported hook configurations are rejected explicitly. The default ramp is
disabled and leaves the built-in tolerance callbacks unchanged.

`initialization_statistics.json` reports preparation, IDA setup/IC, the first IDA
output-callback interval, initial-correction callback counts, native consistency
norms and ramp counters. Phase times are measured on rank zero. The first
interval excludes the initial output write; it is not the first internal BDF
step. Existing `ida_internal_steps.csv` and `ida_step_history.csv` record step
sizes and solver work. Whole-process measurements must also include mesh input,
setup, output and teardown.

Related design discussion: [#25](https://github.com/luca-heltai/metric-flow-x/issues/25),
component scaling [#14](https://github.com/luca-heltai/metric-flow-x/issues/14),
linear-solver lifecycle [#26](https://github.com/luca-heltai/metric-flow-x/issues/26).
