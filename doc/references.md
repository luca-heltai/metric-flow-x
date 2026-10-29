# References

## Manuscript

Raksha Devi, Luca Heltai and Lucas O. Müller.
*An HDG-type discretization of 1D embedded blood flow in complex networks
with Riemann-coupled junctions*. Submitted to arXiv, 2026; public identifier pending.
{cite}`devi2026hdg`

```bibtex
@unpublished{devi2026hdg,
  author = {Devi, Raksha and Heltai, Luca and M{\"u}ller, Lucas O.},
  title = {An {HDG}-type discretization of {1D} embedded blood flow in complex
           networks with {Riemann}-coupled junctions},
  year = {2026},
  note = {Submitted to arXiv; announcement and public identifier pending}
}
```

The [software repository](https://github.com/luca-heltai/metric-flow-x)
contains the executable, library and parameter files.
[MetricFlowX 1.0.0](https://github.com/luca-heltai/metric-flow-x/releases/tag/v1.0.0)
accompanies this submission. The manuscript is undergoing arXiv technical and
moderation checks and has no public DOI or preprint URL yet. A temporary
submission number is not an arXiv identifier and must not be used as one.
Software citation metadata is available in
[`CITATION.cff`](https://github.com/luca-heltai/metric-flow-x/blob/v1.0.0/CITATION.cff).

## Related literature

The canonical bibliography in [`bibliography/references.bib`](https://github.com/luca-heltai/metric-flow-x/blob/v1.0.0/bibliography/references.bib) contains the literature associated with the repository's numerical context. It includes work on one-dimensional blood-flow models, discontinuous Galerkin and hybridizable DG methods, approximate Riemann solvers, and blood-flow benchmarks. The metadata policy and manuscript bibliography provenance are recorded in [`bibliography/README.md`](https://github.com/luca-heltai/metric-flow-x/blob/v1.0.0/bibliography/README.md).

The complete bibliography is rendered below.

```{bibliography}
:style: plain
:all:
```

Repository-specific implementation references:

- The executable entry point is `apps/blood_flow.cc`.
- The model and IDA callbacks are implemented in `source/blood_flow_system.cc` and declared in `include/metric_flow_x/blood_flow_system.h`.
- The implemented equations are explained in [Mathematical formulation](math.md).
