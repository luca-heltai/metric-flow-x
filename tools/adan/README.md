# Complete ADAN input workflow

**Sources and licenses:** the default uses the [ADAVN 2023 supplementary CSV](https://doi.org/10.3389/fphys.2023.1162391)
under the publisher's **CC BY 4.0** notice. The optional [HeMoLab atlas](http://hemolab.lncc.br/adan-web/)
has **separate restrictive terms**; written permission is required for
modification/redistribution. No surfaces or surface-derived outputs are committed.
The new programs and documentation use the repository's **MIT** license.

Read [Complete ADAN: sources, licenses and mathematical reconstruction](../../doc/adan_complete.md)
before using the data. It documents all source links, conversion assumptions,
geometry modes, wall-law equations and terminal adaptation.

From the repository root, with Python 3.10 or newer:

```sh
python3 tools/adan/download_sources.py
python3 tools/adan/recover_geometry.py
python3 tools/adan/build_vtk.py
python3 tools/adan/check_conversion.py
```

The default creates `build/adan/adan_complete.vtk`: a **nonanatomical metric
graph** with all 4,041 arterial source segments and exact source lengths.
It contains 8,082 straight cells, 7,745 points and 1,471 outlets.
The nonlinear/viscoelastic wall coefficients are included for future use;
the active solver still uses its existing elastic law.

Each command accepts `--data-dir PATH`. The optional anatomical commands and
permission requirements are in the documentation. Source downloads and generated
files remain local and ignored by Git.
