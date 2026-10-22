# References

## Manuscript

Raksha Devi, Luca Heltai and Lucas O. Müller.
*An HDG-type discretization of 1D embedded blood flow in complex networks
with Riemann-coupled junctions*. In preparation, 2026.
{cite}`devi2026hdg`

```bibtex
@unpublished{devi2026hdg,
  author = {Devi, Raksha and Heltai, Luca and M{\"u}ller, Lucas O.},
  title = {An {HDG}-type discretization of {1D} embedded blood flow in complex
           networks with {Riemann}-coupled junctions},
  year = {2026},
  note = {In preparation}
}
```

The [software repository](https://github.com/luca-heltai/metric-flow-x)
contains the executable, library and parameter files. The manuscript has no
public DOI or preprint URL yet.

## Related literature

The canonical bibliography in [`bibliography/references.bib`](../bibliography/references.bib) contains the literature currently associated with the repository's numerical context. It includes work on one-dimensional blood-flow models, discontinuous Galerkin and hybridizable DG methods, approximate Riemann solvers, and blood-flow benchmarks. The metadata policy and unresolved keys are recorded in [`bibliography/README.md`](../bibliography/README.md).

The complete bibliography is rendered below.

```{bibliography}
:style: plain
:all:
```

Repository-specific implementation references:

- The executable entry point is `apps/blood_flow.cc`.
- The model and IDA callbacks are implemented in `source/blood_flow_system.cc` and declared in `include/metric_flow_x/blood_flow_system.h`.
- The implemented equations are explained in [Mathematical formulation](math.md).
