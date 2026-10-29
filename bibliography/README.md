# Bibliography and citation metadata

`references.bib` is the canonical bibliography for MetricFlowX. It combines
entries from the historical `doc/references.bib` and `latex/metric_flow.bib`
with the manuscript bibliography supplied in `paper/blood_flow.bib` for the
1.0.0 release. A copy of that submission bibliography, with trailing whitespace removed, is retained in
[`arxiv_submission.bib`](arxiv_submission.bib). Missing entries are copied
from this source; existing canonical entries retain their metadata.

## Manuscript status

The maintainer supplied an arXiv receipt confirming submission. The
`devi2026hdg` entry therefore says “Submitted to arXiv; announcement and public
identifier pending.” No public arXiv identifier or DOI has been supplied;
neither is inferred from the temporary submission number. `CITATION.cff`
records software version 1.0.0 and the manuscript as the preferred citation.

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
