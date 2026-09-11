"""Sphinx configuration for the qsarkit documentation.

Builds on Read the Docs and locally with the same settings. Autodoc pulls
API reference straight from the NumPy-style docstrings in the package, so
the documentation and the code cannot drift apart.
"""

from __future__ import annotations

import os
import sys
from datetime import date

# Make the package importable without installing it (RTD installs it, but a
# local `make html` from a clean checkout should work too).
sys.path.insert(0, os.path.abspath("../.."))
# `demo_data` lives next to this file and provides the dataset every doctest
# in the documentation starts from.
sys.path.insert(0, os.path.abspath("."))

# -- Project information ------------------------------------------------------

project = "qsarkit"
author = "Frederico Schmitt Kremer"
copyright = f"{date.today().year}, {author}"

try:
    from qsarkit import __version__ as release
except ImportError:  # pragma: no cover - docs build without the package
    release = "0.2.0"
version = ".".join(release.split(".")[:2])

# -- General configuration ----------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # NumPy-style docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",  # signatures from annotations, not docstrings
    "sphinx_copybutton",
    "sphinx_design",
    "myst_parser",               # lets us include README.md and write MD pages
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
nitpicky = False

# -- Autodoc / autosummary ----------------------------------------------------

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "no-value": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
autodoc_class_signature = "mixed"
# Heavy optional dependencies are not installed on the RTD builder; mocking
# them lets autodoc import every module without dragging in ~2 GB of wheels.
autodoc_mock_imports = [
    "torch",
    "transformers",
    "gensim",
    "shap",
    "lime",
    "xgboost",
    "lightgbm",
    "mhfp",
    "umap",
]

# -- Doctests -----------------------------------------------------------------

# Every example in this documentation is executable and is executed, both by
# `make doctest` here and by `pytest tests/docs` in the test suite. The two
# runners share one definition of the demo dataset (docs/source/demo_data.py)
# so an example cannot pass in one and fail in the other.
# `sphinx-build -b doctest` checks the examples written in these .rst pages,
# which use explicit `.. doctest::` directives. It does NOT check the
# autodoc-rendered docstrings: Sphinx merges every plain `>>>` block in a
# document into a single shared namespace, so two classes whose examples both
# bind `X` clobber each other and report failures that no user would ever hit.
# The docstrings are checked instead by `pytest tests/docs`, which runs each
# one isolated with its own globals -- the semantics a reader actually gets.
doctest_test_doctest_blocks = ""

doctest_global_setup = """
import numpy as np
from demo_data import (
    DEMO_SMILES, DEMO_Y, DEMO_LABELS, demo_mols,
    NAMED_SMILES, named_mols, demo_fingerprints, demo_descriptors,
)
np.set_printoptions(precision=3, suppress=True)
"""

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
# Class-level annotations (`support_: NDArray`) exist for mypy and are also
# described in each class's NumPy `Attributes` section, so without this
# autodoc documents every attribute twice and emits ~100 duplicate-object
# warnings. `use_ivar` renders the Attributes section as field-list entries
# on the class instead, which merges the two. It requires every docstring to
# use only standard NumPy section headers -- a custom `Algorithm` or
# `Methods` heading opens a nested rST section that swallows the rest of the
# docstring, and trailing-underscore attribute names then parse as broken
# references.
napoleon_use_ivar = True

# -- Intersphinx --------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

# `:doi:`12345`` renders as a resolvable DOI link. Every algorithm in this
# package cites its source, so this shortcut earns its keep.
extlinks = {
    "doi": ("https://doi.org/%s", "doi:%s"),
    "arxiv": ("https://arxiv.org/abs/%s", "arXiv:%s"),
    "pubmed": ("https://pubmed.ncbi.nlm.nih.gov/%s", "PMID:%s"),
}

# -- HTML output --------------------------------------------------------------

html_theme = "furo"
html_title = f"qsarkit {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/fredericokremer/qsarkit-learn",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "navigation_with_keys": True,
}
html_last_updated_fmt = "%Y-%m-%d"

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "linkify", "substitution"]
myst_heading_anchors = 3
suppress_warnings = ["myst.header"]
