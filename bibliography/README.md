# Bibliography and citation metadata

`references.bib` is the canonical bibliography for MetricFlowX. It combines
entries from the historical `doc/references.bib` and `latex/metric_flow.bib`
with the manuscript bibliography supplied in `paper/blood_flow.bib` for the
1.0.0 release. A copy of that submission bibliography, with trailing whitespace removed, is retained in
[`arxiv_submission.bib`](arxiv_submission.bib). Missing entries are copied
from this source; existing canonical entries retain their metadata.

## Manuscript status

The preprint by Raksha Devi, Luca Heltai and Lucas O. Müller is available as
[arXiv:2610.40080](https://arxiv.org/abs/2610.40080), submitted on 29 October
2026, with DOI [10.48550/arXiv.2610.40080](https://doi.org/10.48550/arXiv.2610.40080).
The `devi2026hdg` entry uses the public arXiv metadata and includes the preprint
identifier, subject class, DOI and URL. `CITATION.cff` records software version
1.0.0 and the preprint as the preferred citation.

## Metadata policy

- Preserve source-backed author names, titles and bibliographic fields.
- Do not invent missing papers, DOI values or public preprint identifiers.
- Citation keys are case-sensitive and must be unique in the canonical file.
- A DOI is recorded only when present in the supplied source metadata. The
  checker explicitly verifies its historical core DOI; it does not claim
  external verification of every entry.

The submission bibliography resolves all previously listed keys:
`MR1873203`, `MR2657217`, `MR4260434`, `MR4570554` and `formaggia2003one`, as
well as the additional citations used by the tracked manuscript.

## Use

Use `bibliography/references.bib` for documentation and new manuscript tooling.
The historical manuscript keeps its original bibliography input. Run
`python3 tools/check_bibliography.py` before changing citation keys.
