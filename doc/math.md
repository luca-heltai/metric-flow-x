# Mathematical formulation

MetricFlowX solves for cross-sectional area $A$ and mean axial velocity $U$
on an oriented network of line segments embedded in three-dimensional space.
Pressure is derived from area through the vessel wall law. The method and
its numerical examples are described in {cite}`devi2026hdg`.

## Equations along a vessel

Let $s$ denote arc length and $\boldsymbol b$ the unit tangent in the direction
of increasing $s$. Positive flow is $Q=AU$ in that direction. On each vessel,

$$
\partial_t\boldsymbol w+\partial_s\boldsymbol F(\boldsymbol w,s)
=\boldsymbol S(\boldsymbol w,s),\qquad
\boldsymbol w=\begin{pmatrix}A\\U\end{pmatrix},
$$

$$
\boldsymbol F=\begin{pmatrix}AU\\U^2/2+p/\rho\end{pmatrix},\qquad
\boldsymbol S=\begin{pmatrix}0\\-\eta U/A\end{pmatrix},\qquad
\eta=\frac{2(\xi+2)\pi\mu}{\rho}.
$$

Here $\rho$ is density, $\mu$ is **dynamic viscosity**, and $\xi$ controls
the velocity profile in the friction term. With SI network data, area is in
m², velocity in m/s, pressure in Pa, density in kg/m³ and viscosity in Pa·s.

The embedded flux is $\boldsymbol F\otimes\boldsymbol b$; its intrinsic
divergence along the centerline equals $\partial_s\boldsymbol F$. All
volume integrals use arc length. Each endpoint has outward tangent
$\boldsymbol n=\sigma\boldsymbol b$, with $\sigma=-1$ at the start and
$+1$ at the end of an oriented cell. The implemented geometry uses straight
line cells; there is no dedicated gravitational source term.

## Pressure, taper and wave speed

The implemented elastic tube law is

$$
p_{\mathrm{tube}}(A)=p_0+p_d+
\frac{\beta}{A_d}\left(\sqrt A-\sqrt{A_d}\right),\qquad
\beta=\frac{4\sqrt\pi}{3}E h_w.
$$

$E$ is Young's modulus, $h_w$ wall thickness and $A_d$ the diastolic area.
In particular, $p_{\mathrm{tube}}(A_d)=p_0+p_d$. These vessel properties
come from the VTK input. The input properties `a_0` and `r_d` do not replace
`a_d` in this law.

For taper data with positive endpoint radii, the radius is interpolated
linearly between vessel endpoints and $A_d=\pi r_d(s)^2$. Cell-volume
assembly uses the value at the cell center; trace evaluation uses the value
at the face. Without taper data, the vessel's `a_d` is used.

$$
\frac{\partial p}{\partial A}=\frac{\beta}{2A_d\sqrt A},\qquad
c(A)=\sqrt{\frac A\rho\frac{\partial p}{\partial A}}
=\sqrt{\frac{\beta}{2\rho A_d}}\,A^{1/4}.
$$

The characteristic speeds are $U-c$ and $U+c$. At fixed vessel properties,
the invariants are $W_\pm=U\pm4(c-c_0)$, where $c_0=c(A_d)$.
The registered parameter `m` must remain at $1/2$: it enters a wave-speed
derivative, while the pressure law itself is fixed to the square root.
Changing `m` does not select a different constitutive law.

## Cell fields and face traces

On each cell $K$, both components of $\boldsymbol w_h$ are discontinuous
polynomials of the selected degree. An ordinary interface carries one
trace pair $\widehat{\boldsymbol w}_h=(\widehat A,\widehat U)$.
For a polynomial test function $\boldsymbol v$, the cell residual is

$$
(\partial_t\boldsymbol w_h,\boldsymbol v)_K
-(\boldsymbol F(\boldsymbol w_h),\partial_s\boldsymbol v)_K
+\sum_{F\subset\partial K}\widehat{\boldsymbol f}_{K,F}\cdot
\boldsymbol v|_F
-(\boldsymbol S(\boldsymbol w_h),\boldsymbol v)_K=0.
$$

$\widehat{\boldsymbol f}_{K,F}$ is the outward numerical flux between the
cell state and its trace. At an ordinary interior interface, the sum of the
two incident outward numerical fluxes is zero, giving two algebraic
equations for the shared trace. The code provides `HLL`, `HLL_HDG` and
`LAX_FRIEDRICHS`; the manuscript examples use `HLL`.

For `HLL`, $\boldsymbol f_L$ and $\boldsymbol f_R$ are the physical fluxes
projected with the orientation arguments supplied to the flux routine. The
current routine computes its two speed estimates from the axial states:

$$
\overline U=(U_L+U_R)/2,\qquad \overline c=(c_L+c_R)/2,\qquad
s_L=\overline U-\overline c,\quad s_R=\overline U+\overline c.
$$

It uses the following branches (these averaged speeds are not the min/max
envelope of the two states' characteristic speeds):

$$
\widehat{\boldsymbol f}=\begin{cases}
\boldsymbol f_L,&s_L\geq0,\\
\boldsymbol f_R,&s_R\leq0,\\
\dfrac{s_R\boldsymbol f_L-s_L\boldsymbol f_R
+s_Ls_R(\boldsymbol w_R-\boldsymbol w_L)}{s_R-s_L},&s_L<0<s_R.
\end{cases}
$$

## Junctions

A junction with $k$ incident vessels carries **one trace pair per branch**:
areas need not agree between vessels of different sizes. With $\sigma_i$
the outward orientation of branch $i$, the $2k$ algebraic equations are

$$
\sum_{i=1}^k\sigma_i\widehat A_i\widehat U_i=0,\qquad
H_i=H_1\quad(i=2,\ldots,k),\qquad
H_i=\frac\gamma2\widehat U_i^2+\frac{p_i(\widehat A_i)}\rho,
$$

together with one outgoing-characteristic compatibility equation per branch,

$$
\widehat U_i+4\sigma_i(\widehat c_i-c_{0,i})
=U_i+4\sigma_i(c_i-c_{0,i}).
$$

`Gamma (Total pressure factor)` is $\gamma$. The total-head closure written
in the manuscript corresponds to $\gamma=1$; $\gamma=0$ instead equates
static pressure, so the selected value must be recorded with each run.
The conditions conserve trace flow and do not add
empirical junction-loss terms. Two different vessels meeting at a node use
this closure even when there are only two branches.

## Inflow and terminal circuits

At a subcritical inlet, $\widehat A\widehat U=Q_{\mathrm{in}}(t)$ is
paired with outgoing characteristic compatibility. Inflows can be analytic
functions or tabulated waveforms; see [Configuration](configuration.md).

At an RCR outlet, $Q=\widehat A\widehat U$ and

$$
p(\widehat A)=P_c+R_1Q,\qquad
C\dot P_c-Q+\frac{P_c-P_{\mathrm{out}}}{R_2}=0.
$$

The pressure relation and outgoing characteristic fix the terminal trace;
$P_c$ is an additional differential unknown when $C>0$. For $C\leq0$,
the resistive branch carries no capacitor unknown and imposes

$$
p(\widehat A)=P_{\mathrm{out}}+(R_1+R_2)Q.
$$

Terminal resistance, capacitance and outlet pressure come from the VTK data.
The 37-artery example uses $R_1=0$.

## One differential–algebraic system

Cell coefficients, traces and capacitor pressures form one distributed state
$y$. Cell and capacitor equations contain time derivatives; transmission,
junction and boundary trace equations are algebraic:

$$
F(t,y,\dot y)=0,\qquad
J_{\mathrm{IDA}}=\frac{\partial F}{\partial y}
+\alpha\frac{\partial F}{\partial\dot y}.
$$

The standalone `blood_flow` application uses SUNDIALS IDA with adaptive
step size and variable BDF order (up to the configured maximum of five).
It assembles analytic state and derivative Jacobians and solves the global
Newton system including all blocks, without static condensation of cell
unknowns. Embedding applications can call the residual and Jacobian API and
own time integration themselves.

## Prescribed surrounding pressure

The library additionally supports a prescribed surrounding pressure through
`set_external_pressure_provider()`. It uses

$$
p(A,t,x,i)=p_{\mathrm{tube}}(A)+p_{\mathrm{external}}(t,x,i)
$$

in momentum fluxes, terminal pressure relations, junction head equations and
pressure output. The provider receives time, position, vessel ID and the
incident centerline `CellId`. It is sampled during residual assembly and
pressure output, and held fixed during native Jacobian assembly. It does
not alter the tube-law wave speed or capacitor dynamics.

`add_external_pressure_residual(t, y, provider, destination)` adds this
contribution without replacing the installed provider. These interfaces
support external multiphysics callers; the manuscript's standalone examples
use zero prescribed surrounding pressure.
