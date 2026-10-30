# References

## Manuscript

Raksha Devi, Luca Heltai and Lucas O. Müller.
*An HDG-type discretization of 1D embedded blood flow in complex networks
with Riemann-coupled junctions*. [arXiv:2610.40080](https://arxiv.org/abs/2610.40080),
2026. DOI: [10.48550/arXiv.2610.40080](https://doi.org/10.48550/arXiv.2610.40080).
{cite}`devi2026hdg`

```bibtex
@misc{devi2026hdg,
  author = {Devi, Raksha and Heltai, Luca and M{\"u}ller, Lucas O.},
  title = {An {HDG}-type discretization of {1D} embedded blood flow in complex
           networks with {Riemann}-coupled junctions},
  year = {2026},
  eprint = {2610.40080},
  archivePrefix = {arXiv},
  primaryClass = {math.NA},
  doi = {10.48550/arXiv.2610.40080},
  url = {https://arxiv.org/abs/2610.40080}
}
```

The [software repository](https://github.com/luca-heltai/metric-flow-x)
contains the executable, library and parameter files.
[MetricFlowX 1.0.0](https://github.com/luca-heltai/metric-flow-x/releases/tag/v1.0.0)
accompanies this preprint. Current software citation metadata, including the
public arXiv identifier and DOI, is available in
[`CITATION.cff`](https://github.com/luca-heltai/metric-flow-x/blob/main/CITATION.cff).

## Related literature

The canonical bibliography in [`bibliography/references.bib`](https://github.com/luca-heltai/metric-flow-x/blob/main/bibliography/references.bib) contains the literature associated with the repository's numerical context. It includes work on one-dimensional blood-flow models, discontinuous Galerkin and hybridizable DG methods, approximate Riemann solvers, and blood-flow benchmarks. The metadata policy and manuscript bibliography provenance are recorded in [`bibliography/README.md`](https://github.com/luca-heltai/metric-flow-x/blob/main/bibliography/README.md).

The complete bibliography is rendered below.

```{bibliography}
:style: plain
:all:
```

Repository-specific implementation references:

- The executable entry point is `apps/blood_flow.cc`.
- The model and IDA callbacks are implemented in `source/blood_flow_system.cc` and declared in `include/metric_flow_x/blood_flow_system.h`.
- The implemented equations are explained in [Mathematical formulation](math.md).
