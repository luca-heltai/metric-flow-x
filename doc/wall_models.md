# Optional ADAVN wall models

MetricFlowX implements the arterial and venous constitutive laws in equations
(3) and (6) of [Müller et al. (2023), *An anatomically detailed arterial-venous network model.
Cerebral and coronary circulation*](https://doi.org/10.3389/fphys.2023.1162391).
The elastic square-root law remains the default. Wall viscosity is a separate,
optional Kelvin–Voigt contribution. All quantities below use SI units.

## Configuration and input contract

Select the law in the existing system subsection:

```text
subsection MetricFlowSystem<1, 3>
  set Wall model = ADAVN arterial
  set Viscoelastic wall = true
  set Initial condition policy = native
end
```

`Wall model` accepts `Square root`, `ADAVN arterial`, `ADAVN venous`, and
`ADAVN mixed`. `Viscoelastic wall` defaults to `false`. Its coefficients are
required only when it is enabled; enabling it with `Square root` is an error.
`ADAVN venous` also accepts dural sinuses. `ADAVN mixed` selects the law per
vessel from `adavn_wall_type`.

The VTK retains its existing required mesh, vessel and terminal arrays, including
`a_d`, `h_wall`, `E`, `p_d` and `p0`. For ADAVN, `a_d` is $A_{ref}$ and `h_wall`
is $h_{ref}$; the legacy `E`, `p_d` and `p0` do not enter the new pressure law.
The external-pressure provider is still added to the transmural pressure.
Optional taper changes the local reference area. ADAVN additionally requires
these scalar `CELL_DATA` arrays:

| Array | Required for | Meaning / units |
|---|---|---|
| `adavn_wall_type` | Mixed; optional otherwise | 1: artery, 2: vein, 3: dural sinus |
| `adavn_p_ref` | All ADAVN | Reference transmural pressure, Pa |
| `adavn_E_e`, `adavn_E_c` | Arteries | Elastin and collagen coefficients, Pa |
| `adavn_epsilon_0`, `adavn_epsilon_r` | Arteries | Activation strain and width, dimensionless |
| `adavn_venous_K` | Veins and sinuses | Stiffness, Pa |
| `adavn_venous_m`, `adavn_venous_n` | Veins and sinuses | Exponents, dimensionless |
| `adavn_K_m` | Viscoelastic ADAVN | Arteries: Pa s; veins/sinuses: Pa s m |

A homogeneous selector rejects contradictory `adavn_wall_type` values. Missing
arrays, incorrect tuple counts, nonfinite coefficients, nonpositive reference
area, invalid moduli/exponents, negative viscosity, and inconsistent coefficients
on cells sharing a vessel ID produce exceptions naming the model and input file.
Arteries require $E_e>0$, $E_c\geq0$, $h_{ref}>0$, $\epsilon_0\geq0$ and
$\epsilon_r>0$; veins require $K>0$, $m>0$ and $n\leq0$.
A nonpositive $A\,dp_{el}/dA$ is reported as a non-hyperbolic wall state.
Arrays for the other tissue type may contain zero placeholders in a mixed file.
Use a distinct vessel ID when wall properties change along a vessel.
Reference area and arterial wall thickness must also agree on cells sharing an
ID; use the supported taper arrays for a varying local reference area.

Viscoelastic runs support the default `Initial condition policy = native`.
This solves for algebraic traces and differential-state time derivatives
simultaneously while keeping the differential seed fixed; see
[native viscoelastic consistency](initialization.md#native-viscoelastic-consistency).
`Initial condition policy = IDA` remains available as an alternative.
`Initial state source = steady network` reconstructs reference-geometry graph
pressures using the positive-tangent ADAVN branch connected to the reference
state. Pressures outside the bracketed hyperbolic branch are errors.
The graph-only `Run mode = steady network` uses the same reconstruction; it is
a resistive approximation, not an equilibrium of the full compliant HDG model.
The rest state seeds the terminal capacitor pressure from the selected law.

Runnable, small synthetic examples are configured by CMake:

```sh
./build/blood_flow build/parameters/wall_arterial.prm
./build/blood_flow build/parameters/wall_venous.prm
./build/blood_flow build/parameters/wall_mixed.prm
```

These examples test the implementation; their coefficients and topology are
not physiological validation datasets. To run them elastically, set
`Viscoelastic wall = false`; the native initialization policy is then available.
The complete ADAN arterial converter in [PR #75](https://github.com/luca-heltai/metric-flow-x/pull/75)
writes the arterial fields above. Large source CSV/VTK files remain external
release assets; this change adds no ADAN dataset to Git.

## Arterial law

With $R_{ref}=\sqrt{A_{ref}/\pi}$, define

$$
\epsilon=\sqrt{A/A_{ref}}-1,\qquad
\chi=(\epsilon-\epsilon_0)/\epsilon_r,
$$

$$
f(\epsilon)=E_e\epsilon+E_c\epsilon_r\log(1+e^\chi),\qquad
p_{tm}=p_{ref}+\frac{\pi R_{ref}h_{ref}}{A}f(\epsilon)
       +\nu(A)\dot A,\qquad
\nu(A)=\frac{K_m}{2\sqrt{AA_{ref}}}.
$$

The code evaluates the softplus and logistic functions without overflowing
$e^\chi$. In particular, $p_{el}(A_{ref})$ need not equal $p_{ref}$: the
collagen softplus has a small, nonzero value at zero strain. It is not subtracted.
For $s=\sqrt{A/A_{ref}}$, $C=\pi R_{ref}h_{ref}$ and
$\sigma=(1+e^{-\chi})^{-1}$,

$$
f'=E_e+E_c\sigma,\quad
f''=E_c\sigma(1-\sigma)/\epsilon_r,
$$

$$
p'_{el}=\frac{C}{A^2}\left[-f+\frac{s}{2}f'\right],\qquad
p''_{el}=\frac{C}{A^3}\left[2f-\frac54sf'+\frac14s^2f''\right].
$$

The paper uses $p_{ref}=10^4$ Pa, $\epsilon_0=0.25$,
$\epsilon_r=0.05$, $E_e=4\cdot10^5W_e$ Pa,
$E_c=10^8W_c$ Pa, and $K_m=3.6\cdot10^4W_m$ Pa s.
For radius in cm, $(W_e,W_c,W_m)$ is $(0.85,0.05,0.10)$ above 0.18 cm,
$(0.65,0.20,0.15)$ from 0.07 to 0.18 cm, and $(0.45,0,0.55)$ below
0.07 cm. The source supplies
$h_{ref}/R_{ref}=0.2802e^{-5.053R_{ref,cm}}+0.1324e^{-0.1114R_{ref,cm}}$.
These are dataset-generation rules, not hidden solver defaults: the VTK carries
the actual coefficients.

## Venous and dural-sinus law

For $\alpha=A/A_{ref}$,

$$
p_{tm}=p_{ref}+K(\alpha^m-\alpha^n)+\nu(A)\dot A,\qquad
\nu(A)=\frac{K_m}{A_{ref}\sqrt A}.
$$

The negative-power term permits a compliant, collapsed regime for veins while
the positive-power term stiffens a distended vessel. The paper uses
$(m,n)=(10,-1.5)$ for veins and $(0.5,0)$ for dural sinuses,
$p_{ref}=666.666$ Pa,
$K=\rho c_{ref}^2/(m-n)$ and $K_m=70.898\,R_{ref}$ Pa s m.
The venous and arterial $K_m$ values have **different dimensions** and must not
be copied between tissue types.

$$
p'_{el}=\frac K A(m\alpha^m-n\alpha^n),\qquad
p''_{el}=\frac K {A^2}
 [m(m-1)\alpha^m-n(n-1)\alpha^n].
$$

## Characteristics and implicit discretization

For either elastic law the wave speed and characteristic primitive are

$$
c(A)=\sqrt{\frac A\rho p'_{el}(A)},\qquad
\Psi(A)=\int_{A_{ref}}^A\frac{c(a)}a\,da,\qquad
\Psi'(A)=\frac{c(A)}A.
$$

Boundary and junction compatibility uses $U\pm\Psi$ instead of the
square-root-specific $U\pm4(c-c_{ref})$. The integral uses eight-point Gaussian
quadrature in $\log(A/A_{ref})$, split into intervals no wider than 0.25.
The analytic Jacobians differentiate the pressure, characteristic relation and
state-dependent numerical-flux speeds. The legacy flux linearization is preserved
when `Square root` is selected.

For viscosity, let $q_v=\nu(A)\dot A$. MetricFlowX adds its weak axial gradient
to the velocity rows of $F=M\dot y-R(y)$:

$$
F_{v,U}(\varphi)=-\int_K\frac{q_v}\rho\partial_s\varphi\,ds
 +\sum_{\partial K}\frac{n\widehat q_v}\rho\varphi.
$$

On an ordinary interior face, $\widehat q_v$ is the arithmetic mean of the two
incident cell stresses. This is a shared conservative stress flux. At boundaries
and junctions the stress is extrapolated from the incident cell. It enters the
RCR pressure relation and junction total-head continuity alongside the elastic
pressure. Characteristics and stabilization use the elastic wave speed; traces
remain algebraic and no trace time derivatives are introduced.

Both implicit blocks include the exact viscous derivatives

$$
\frac{\partial q_v}{\partial A}=-\frac{\nu(A)}{2A}\dot A,\qquad
\frac{\partial q_v}{\partial\dot A}=\nu(A).
$$

Consequently $\partial F/\partial\dot y$ depends on the current area and includes
cell-area columns in velocity and pressure-closure rows. It is rebuilt for every
viscoelastic Jacobian assembly. Pressure VTK/CSV output uses both $y$ and
$\dot y$; library callers must use `compute_pressure(y, ydot, p, t)` for this
model. The scalar `pressure(A, vessel_id)` API returns the elastic part only.

This is an extension of MetricFlowX's DG/HDG spatial discretization, not a port
of the paper's finite-volume discretization. Tests check constitutive derivatives,
characteristic derivatives, full state/rate Jacobians for all three fluxes,
mixed junctions, pressure output and rejected inputs. Short IDA examples are
smoke tests. On the tested deal.II 9.8.0-rc1 build, IDA's `use_y_diff`
initial-condition correction fails inside the upstream non-block `N_Vector`
minimum operation when a rank owns no DoFs. Native initialization avoids
`IDACalcIC` and is tested with empty ranks. The separate built-in vector-tolerance
minimum issue still requires scalar tolerances or the existing custom weight
callback for empty-rank integration; see [initialization](initialization.md).

The paper's full closed-loop results additionally require its heart,
valves, microcirculation, venous topology and parameter calibration. Those
components, spatial/time convergence and quantitative waveform reproduction
are outside the evidence provided by these tests.

## Sources and licensing

- **Constitutive equations and physiological coefficients:** Müller, L. O.,
  Watanabe, S. M., Toro, E. F., Feijóo, R. A., and Blanco, P. J. (2023).
  [*An anatomically detailed arterial-venous network model.
  Cerebral and coronary circulation*](https://doi.org/10.3389/fphys.2023.1162391), *Frontiers in
  Physiology* **14**, 1162391, especially equations (3)–(6) and the associated
  tissue-coefficient discussion. The article is available under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); attribution is
  provided here. Its supplementary CSV provenance/license accompanies the
  external dataset assets in PR #75.
- **Implementation, synthetic fixtures and this documentation:** original
  MetricFlowX contributions under the repository's [MIT license](https://github.com/luca-heltai/metric-flow-x/blob/main/LICENSE.md).
  No source code from another solver or HemoLab anatomical surfaces is included.
  The article's open license does not grant rights to restricted anatomical
  source assets.
