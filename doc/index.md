# MetricFlowX

**Blood flow on vascular networks, from individual vessels to large trees.**

MetricFlowX computes pulse propagation in compliant vessels represented by
centerlines in three-dimensional space. Cross-sectional area and axial
velocity evolve along each vessel; junction constraints and terminal
circuits couple the network into one implicit differential–algebraic system.

```{figure} _static/paper/liver_pressure.png
:alt: Pressure distribution in the branching liver vascular tree from the manuscript.
:width: 70%

Liver network with 7,999 vessels: pressure at 2 s in the manuscript's
simulation, using degree-two elements, two global refinements and BDF of
maximum order two. See [Examples from the manuscript](results.md) for the
mass balance and figure provenance.
```

## Explore the model

- [Mathematical formulation](math.md): conservation equations, elastic tube
  law, numerical fluxes, branch traces, junctions and RCR outlets.
- [Optional ADAVN wall models](wall_models.md): arterial, venous and
  viscoelastic laws, input requirements and validation limits.
- [Examples from the manuscript](results.md): arterial pulse waveforms,
  conservation of network volume and adaptive time integration.
- [Getting started](getting_started.md): build the standalone `blood_flow`
  application and select a network input.
- [Library API](api.md): embed the native residual and analytic Jacobians
  in another application.

## Why a differential–algebraic formulation?

Area and velocity inside a vessel satisfy evolution equations. Face traces
must simultaneously satisfy transmission, boundary and junction constraints.
RCR outlets add capacitor-pressure evolution equations. MetricFlowX assembles
all these unknowns into a distributed system and uses SUNDIALS IDA for
adaptive BDF integration. The implementation solves the full Newton system,
without statically condensing the cell unknowns.

The software provides an MPI-parallel C++ library based on deal.II and a
standalone executable. Vessel properties and terminal circuits come from
VTK network inputs. See the [source repository](https://github.com/luca-heltai/metric-flow-x)
for parameter files, dataset downloads and development.

## Cite this work

Raksha Devi, Luca Heltai and Lucas O. Müller.
*An HDG-type discretization of 1D embedded blood flow in complex networks
with Riemann-coupled junctions*. **In preparation, 2026.**
{cite}`devi2026hdg`

The figures on this site illustrate the manuscript's archived simulations.
They have not been recomputed with the current development revision.
The [references page](references.md) provides the bibliographic entry.

```{toctree}
:hidden:
:maxdepth: 1

overview
getting_started
build_and_run
containers
math
wall_models
initialization
stationary
results
performance
performance_campaign
tutorials
reproducibility
api
references
```
