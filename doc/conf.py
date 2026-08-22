import os
import sys
import io
import glob

project = "MetricFlow-X"
author = "Raksha Devi and Luca Heltai"
html_baseurl = "https://luca-heltai.github.io/metric-flow-x/"
default_role = "any"

extensions = [
    "sphinx.ext.mathjax",
    "breathe",
    "exhale",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.bibtex",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "html",
    "_static/**",
]

# Suppress specific warnings that are benign for this repository build.
suppress_warnings = [
    'doxygenfunction',
    # README is included at the documentation root but retains repository
    # relative links such as doc/configuration.md for GitHub readers.
    'myst.xref_missing',
]

html_theme = "furo"
html_title = project
html_logo = "logo.png"
html_favicon = "favicon.png"
html_theme_options = {
    "top_of_page_buttons": ["view", "edit"],
}

sys.path.insert(0, os.path.abspath(".."))

breathe_projects = {
    project: os.path.abspath("../build/docs/doxygen/xml"),
}
breathe_default_project = project
bibtex_bibfiles = ["../bibliography/references.bib"]
bibtex_reference_style = "label"

exhale_args = {
    "containmentFolder": "./api",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library Reference",
    "contentsDirectives": False,
    "doxygenStripFromPath": "..",
    "createTreeView": False,
    "exhaleExecutesDoxygen": False,
}

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
]

mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
)


def _strip_operator_doxygenfunctions(api_dir):
    """Remove doxygenfunction directives that reference operator overloads.

    Exhale sometimes generates `.. doxygenfunction::` entries for operator
    overloads that Doxygen does not expose; these cause warnings. This helper
    removes those directives and their option lines from generated RST files in
    `doc/api` before Sphinx checks them.
    """
    if not os.path.isdir(api_dir):
        return
    pattern = os.path.join(api_dir, "*.rst")
    for path in glob.glob(pattern):
        try:
            with io.open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()
        except OSError:
            continue
        out = []
        skip_options = False
        changed = False
        for line in lines:
            if line.lstrip().startswith(".. doxygenfunction::") and "operator" in line:
                changed = True
                skip_options = True
                continue
            if skip_options:
                if line.lstrip().startswith(":"):
                    continue
                skip_options = False
            out.append(line)
        if changed:
            try:
                with io.open(path, "w", encoding="utf-8") as fh:
                    fh.writelines(out)
            except OSError:
                # best-effort; continue
                pass


def remove_operator_doxygenfunctions(app, env, docnames):
    api_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "api")
    _strip_operator_doxygenfunctions(api_dir)


def mark_generated_api_root_orphan(app):
    """Mark Exhale's linked API root as intentionally outside the toctree."""
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "api", "library_root.rst")
    try:
        with io.open(path, "r", encoding="utf-8") as fh:
            contents = fh.read()
        if not contents.startswith(":orphan:"):
            with io.open(path, "w", encoding="utf-8") as fh:
                fh.write(":orphan:\n\n" + contents)
    except OSError:
        pass


def setup(app):
    app.connect("builder-inited", mark_generated_api_root_orphan)
    app.connect("env-before-read-docs", remove_operator_doxygenfunctions)


mermaid_version = "11.4.1"
