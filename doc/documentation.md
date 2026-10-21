# Documentation tooling

The documentation site combines Markdown pages with a generated C++ API reference:

1. Doxygen scans `source/`, `include/`, and `README.md` and emits XML.
2. Breathe and Exhale consume that XML from `build/docs/doxygen/xml`.
3. Sphinx/MyST renders the Markdown pages and generated API tree into `build/docs/site`.

## Build the site

From the repository root, with Doxygen available:

```bash
./scripts/build_doc.sh
```

The script creates or uses the repository `env/` virtual environment, installs `doc/requirements.txt`, generates Doxygen XML, and runs a Sphinx HTML build with warnings treated as errors. It does not alter source behavior or the manuscript.

Serve an existing site with:

```bash
./scripts/serve_doc.sh [PORT]
```


## Configuration

- `doc/Doxyfile` controls Doxygen input and XML generation.
- `doc/conf.py` enables MyST, Breathe, Exhale, Mermaid, and BibTeX and points Breathe at `build/docs/doxygen/xml`.
- `doc/requirements.txt` pins the supported package ranges.
- `bibliography/references.bib` supplies the canonical bibliography data.

The generated `doc/api/` RST tree is a build artifact consumed by Sphinx. It is not hand-authored API documentation.

A full documentation build requires local Doxygen and Python dependencies. Those tools are not assumed to be installed merely because the C++ project configures successfully. The build status should be reported from the command actually run.
