# Examples from the manuscript

These figures come from the manuscript by Devi, Heltai and Müller
{cite}`devi2026hdg`, currently in preparation. They show the numerical
questions addressed by the method: pulse propagation, conservation across
many junctions, and adaptive integration of the global DAE.

The images are archived manuscript results, not a validation of the current
development revision. Their source paths and checksums are recorded in the
[figure manifest](_static/paper/provenance.json). The code repository is the
authoritative location for the parameter files used to prepare future
controlled runs.

## Pulse propagation in 37 arteries

The 37-artery benchmark compares arterial pressure and flow waveforms with
a finite-volume reference and, at selected locations, experimental data.
The example below shows the aortic arch pressure over one cardiac cycle.
Reflected waves and the transmission of the pulse through junctions make
this a more demanding test than propagation in an isolated uniform vessel.

```{figure} _static/paper/arteries37_pressure.png
:alt: Aortic arch pressure over one cardiac cycle, comparing the HDG curve with the finite-volume reference.

Aortic arch II pressure in the 37-artery network, as plotted in the
manuscript. Time is in seconds; this archived plot retains pressure in
CGS units (dyn/cm², with 1 mmHg approximately 1,333.22 dyn/cm²). The
manuscript configuration uses degree three, maximum BDF order three and a
20 s simulation. [Download the original plot](_static/paper/arteries37_pressure.pdf).
```

## Network volume balance in a liver tree

The liver example contains 7,999 vessels. Its degree-two approximation uses
two global refinements and maximum BDF order two, over a 2 s interval.
Junction mass conservation couples the branches; a network-level diagnostic
compares the net boundary flow with the change of blood volume,

$$
V(t)=\sum_K\int_K A_h(s,t)\,\mathrm ds,\qquad
\frac{\mathrm dV}{\mathrm dt}
=Q_{\mathrm{in}}-\sum_k Q_{\mathrm{out},k}.
$$

```{figure} _static/paper/liver_mass_balance.png
:alt: Net boundary flow and finite-difference volume derivative overlap, with cumulative relative mass error on a separate axis.

Archived liver simulation: boundary flow and the finite-difference estimate
of the volume derivative use the left axis (mL/s). The right axis shows the
cumulative relative volume-balance error. This diagnostic includes temporal
sampling and quadrature effects; it is distinct from the algebraic
junction constraints.
```

The dimensionless error displayed on the right axis is

$$
\varepsilon_{\mathrm{cum}}(t)=
\frac{\left|V(t)-V(0)-\int_0^t
(Q_{\mathrm{in}}-\sum_k Q_{\mathrm{out},k})\,\mathrm d\tau\right|}{V(t)}.
$$

## Adaptive integration of a cerebral tree

The cerebral example uses a CCO-generated arteriolar tree with 19,999 vessels
and 10,000 terminals. Its reference geometry and physiological data are
described in the manuscript. The approximation uses degree two and one
cell per vessel, with resistive terminal conditions.

```{figure} _static/paper/cerebral_mean_pressure.png
:alt: Cerebral vascular tree colored by pressure averaged over the second cardiac cycle, with a pressure color scale.
:width: 80%

Archived mean arterial pressure over the second cardiac cycle, 1–2 s.
The image's color scale is in mmHg. The SI solver represents pressure in Pa;
presentation-unit conversions belong to postprocessing.
```

```{figure} _static/paper/cerebral_ida.png
:alt: IDA step size, BDF order and cumulative solver counts over logarithmic time; the step size grows after startup.

Adaptive IDA integration in the archived cerebral run. During startup, the
step size is small; subsequently IDA changes both step size and BDF order.
The plot reports internal accepted steps, rather than only the times at
which visualization files are written.
[Download the original plot](_static/paper/cerebral_ida.pdf).
```

The manuscript also studies manufactured-solution convergence, mesh
refinement, BDF-order choices and MPI scaling on synthetic 50k/100k trees.
Those timing results describe the recorded machine and solver environment;
they are not performance guarantees for another system.
