# Hepatic network import tools

See [the workflow documentation](../../doc/hepatic_datasets.md) for dataset
sources, reuse terms, commands, units, assumptions and validation.

- `download_sources.py`: checked downloads and optional ZIP extraction.
- `build_vtk.py`: one DICOM/NIfTI vascular mask to an SI MetricFlowX line network.
- `check_conversion.py`: independent output/provenance consistency checks.
- `import_smoke/`: optional native C++ import and array-order check.

Install `requirements.txt` in a virtual environment. Wall and terminal
coefficients must be supplied explicitly. Keep source data and converted
networks out of Git, including data carrying IRCAD's no-derivatives terms.
