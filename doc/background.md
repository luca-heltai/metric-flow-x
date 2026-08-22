# Background

Blood-flow models reduce the three-dimensional fluid/structure problem to variables along vessel centerlines. In this repository, each vessel is represented by one-dimensional line cells embedded in three-dimensional coordinates, and the input network is read from a legacy VTK unstructured-grid file.

The state is expressed through cross-sectional area and mean velocity. A pressure/area tube law and a wave-speed calculation appear in the source implementation and in the explanatory [Mathematics](math.md) page. Junctions and terminal boundaries add network coupling; the current code includes an RCR outlet path with capacitor-pressure unknowns.

The numerical design is motivated by discontinuous Galerkin and hybridized DG methods: cell fields can be discontinuous, while face traces provide the shared coupling variables. In the current implementation, these unknown blocks are assembled in one distributed system and passed to SUNDIALS IDA as a differential-algebraic problem.

## Reading the repository responsibly

- `source/` and `include/` describe current executable behavior.
- `parameters/` contains examples and historical/reference inputs; an example value is not a calibration.
- `latex/metric_flow.tex` is a manuscript source and is not rewritten by this documentation work.
- `bibliography/references.bib` records literature associated with DG, HDG, Riemann solvers, and one-dimensional blood flow.
