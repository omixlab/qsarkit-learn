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
    # Only reached if autodoc cannot import the package at all, in which
    # case the build is already broken; the literal is a last resort and
    # is deliberately vague rather than a stale exact version.
    release = "unknown"
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
    "sphinx.ext.githubpages",
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

# -- Mathematics --------------------------------------------------------------

# Formulas appear throughout: Q^2 variants, the Golbraikh-Tropsha criteria,
# Tanimoto, ECE, leverage. MathJax 3 renders them in HTML.
#
# `mathjax3_config` loads the AMS packages (align, cases) that multi-line
# derivations need, and registers the macros used repeatedly so each
# docstring writes \Tanimoto rather than spelling the definition out.
mathjax3_config = {
    "tex": {
        "packages": {"[+]": ["ams", "boldsymbol"]},
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "macros": {
            "Tanimoto": r"T",
            "Rsq": r"R^{2}",
            "Qsq": [r"Q^{2}_{\mathrm{#1}}", 1],
            "RMSE": r"\mathrm{RMSE}",
            "ECE": r"\mathrm{ECE}",
            "argmax": r"\operatorname*{arg\,max}",
            "argmin": r"\operatorname*{arg\,min}",
        },
    },
    "options": {
        # Do not typeset inside code blocks: a literal backslash in a SMARTS
        # pattern or a regex must stay literal.
        "ignoreHtmlClass": "highlight|no-mathjax",
        "processHtmlClass": "math|tex2jax_process",
    },
}

# The LaTeX (PDF) builder needs the same macros declared, or a formula that
# renders in HTML silently breaks the PDF that Read the Docs also builds.
latex_elements = {
    "preamble": r"""
\usepackage{amsmath}
\usepackage{amssymb}
\newcommand{\Tanimoto}{T}
\newcommand{\Rsq}{R^{2}}
\newcommand{\Qsq}[1]{Q^{2}_{\mathrm{#1}}}
\newcommand{\RMSE}{\mathrm{RMSE}}
\newcommand{\ECE}{\mathrm{ECE}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
""",
    "papersize": "a4paper",
    "pointsize": "10pt",
    # Unicode used in the prose (superscripts, dashes, Greek) needs a font
    # that has the glyphs; the default LaTeX font does not.
    "fontpkg": r"\usepackage{lmodern}",
}
latex_documents = [
    (
        master_doc,
        "qsarkit.tex",
        "qsarkit documentation",
        author,
        "manual",
    )
]

# Numbered equations, so a formula can be referred to from the prose.
math_number_all = False
math_eqref_format = "Eq. {number}"
numfig = True

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
    "source_repository": "https://github.com/omixlab/qsarkit-learn",
    "source_branch": "main",
    # The sources moved to docs-sphinx/; docs/ is now the built output.
    # A wrong value here 404s the "Edit this page" link on every page.
    "source_directory": "docs-sphinx/source/",
    "navigation_with_keys": True,
}
html_last_updated_fmt = "%Y-%m-%d"

# -- MyST ---------------------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist", "linkify", "substitution"]
myst_heading_anchors = 3
suppress_warnings = ["myst.header"]
