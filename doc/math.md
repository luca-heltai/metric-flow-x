# Mathematics

This page collects an explanatory summary of the blood-flow model represented by the current source.

Where intended mathematics differs from implementation details, the source and maintainer decisions take precedence.

## Governing equations

Mass (area):
$$
\frac{\partial A}{\partial t} + \nabla\cdot\left(b\,A\,U\right) = 0
$$

Momentum:
$$
\frac{\partial U}{\partial t} + \nabla\cdot\left(b\left(\frac{U^2}{2} + \frac{P(A)}{\rho}\right)\right) + \eta_c\,U = 0
$$

## Tube law and wave speed

Pressure–area law:
$$
P(A) = p_0 + \mu\left[\left(\frac{A}{A_0}\right)^m - 1\right],
\quad
\frac{dP}{dA} = \mu\,m\,\frac{A^{\,m-1}}{A_0^{\,m}} = \mu\,m\,\frac{1}{A_0}\left(\frac{A}{A_0}\right)^{m-1}.
$$

Wave speed:
$$
c = \sqrt{\frac{A}{\rho}\frac{dP}{dA}}
= \sqrt{\frac{\mu\,m}{\rho}\left(\frac{A}{A_0}\right)^{m}}.
$$

(Here $\rho$ is fluid density, $\eta_c$ a friction coefficient, $b$ a geometric weighting, and $p_0,A_0,\mu,m$ tube-law constants.)

## Prescribed external pressure

The implemented tube-law pressure is the baseline vessel/transmural pressure
$$
p_{\mathrm{tube}}(A) = p_0 + \frac{\beta}{a_d}
\left(\sqrt{A}-\sqrt{a_d}\right) + p_d,
\qquad \beta=\frac{4\sqrt{\pi}}{3}Eh_{\mathrm{wall}}.
$$
Here $a_d$ is the local diastolic area (interpolated from vessel-end radii
when taper data is present); $a_0$ and $r_d$ are input properties but do not
enter this current law. In particular,
$p_{\mathrm{tube}}(a_d)=p_0+p_d$.

`BloodFlowSystem` can add a prescribed external/surrounding pressure through
`set_external_pressure_provider()`. The physical pressure used in the
momentum flux is
$$
p_{\mathrm{internal}}(A,t,x,\mathrm{vessel})
 = p_{\mathrm{tube}}(A) + p_{\mathrm{external}}(t,x,\mathrm{vessel}).
$$
Thus a positive supplied pressure increases physical pressure and momentum
flux. The supplied pressure is independent of the flow state, so it does not
change $dp/dA$, wave speed, the state Jacobian through a pressure derivative,
or the derivative Jacobian.

The provider is sampled at volume and face quadrature points, junction
locations, boundary points, and cell centers for pressure output. It
contributes to volume momentum fluxes, numerical momentum fluxes, RCR trace
pressure equations, junction total-head equations, and pressure output. It
deliberately does not enter mass fluxes, Riemann invariants/wave speeds, RCR
capacitor dynamics, or inflow/reflection characteristic relations. A future
coupled caller therefore needs only a deterministic function of time, physical
point, and vessel id; MetricFlowX's global DoF numbering and any external
coupling state remain outside this interface.

The provider is evaluated during residual assembly and pressure output. It is
not evaluated during either Jacobian assembly: with the provider held fixed,
its contribution has zero derivative with respect to the native flow state
and no dependence on $\dot y$.
