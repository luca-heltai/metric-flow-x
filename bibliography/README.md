# Bibliography and citation metadata

`references.bib` is the canonical bibliography for this repository. It was
consolidated from the tracked `doc/references.bib` and
`latex/metric_flow.bib` files. Entries are included only when their metadata is
already present in one of those source files.

## Metadata policy

- Do not infer, enrich, or correct bibliographic metadata from memory or from
  an unreviewed external lookup.
- Preserve the source spelling and fields unless a malformed control character
  or escape has an unambiguous intended representation in the other tracked
  bibliography. The canonical file uses the unambiguous `Breu{\\ss}` and
  `M{\\"u}ller` forms present in the LaTeX source.
- A DOI is recorded only when it already occurs in source-backed metadata. The
  checker maintains an explicit allowlist for the one source-backed core DOI;
  it does not claim that entries without a DOI have been externally verified.
- Citation keys are case-sensitive. The canonical file must contain at most
  one entry for each key.
- Missing keys are recorded below rather than being filled with guessed
  papers. Their presence does not mean that the corresponding metadata has
  been resolved.

## Unresolved keys

The following keys were requested by the manuscript/documentation work but are
absent from both tracked source bibliographies. No paper metadata has been
invented for them:

- `MR1873203`
- `MR2657217`
- `MR4260434`
- `MR4570554`
- `formaggia2003one`

The current tracked TeX/Markdown citation scan does not identify citations for
these absent keys. They remain an explicit follow-up list because uploaded or
otherwise unavailable TeX material may refer to them.

## Proposed maintainer message

> **WP-12: establish canonical bibliography and software citation.** Please
> provide source-backed metadata for `MR1873203`, `MR2657217`, `MR4260434`,
> `MR4570554`, and `formaggia2003one` if they are required by unavailable
> manuscript material. Confirm maintainer names for `CITATION.cff`; do not add
> guessed papers, DOI values, authors, or a license assertion.

## Use

Use `bibliography/references.bib` for new documentation or manuscript tooling.
The tracked manuscript still has its historical `latex/metric_flow.bib` input;
this canonical file does not silently rewrite that manuscript input. Run
`python3 tools/check_bibliography.py` before adding or changing citation keys.
