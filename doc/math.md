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
