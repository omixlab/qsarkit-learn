"""Lazy-import helper for optional, heavy dependencies (torch, transformers, ...).

A few qsarkit features rest on large third-party packages -- learned molecular
embeddings, SHAP/LIME attribution, UMAP projections. Modules that need one
import it lazily through :func:`require` inside ``__init__``/``fit`` rather
than at module import time, so that ``import qsarkit`` never pulls in
torch/transformers unless the corresponding feature is actually used.
"""

from __future__ import annotations

import importlib
from typing import Any

from qsarkit.base.exceptions import OptionalDependencyError

# Maps an import name to the pip extra that provides it (see pyproject.toml).
_EXTRA_FOR_MODULE = {
    "torch": "nlp",
    "transformers": "nlp",
    "gensim": "embeddings",
    "shap": "explainability",
    "lime": "explainability",
    "xgboost": "boosting",
    "lightgbm": "boosting",
    "umap": "embedding_viz",
    "kaleido": "reporting",
    "reportlab": "reporting",
    "skops": "persistence",
    "skops.io": "persistence",
    "imblearn": "balancing",
}


def require(module_name: str) -> Any:
    """Import and return ``module_name``, raising a helpful error if absent.

    Parameters
    ----------
    module_name:
        Fully qualified module name, e.g. ``"torch"`` or ``"pdfminer.high_level"``.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    OptionalDependencyError
        If the module is not installed. The error message names the pip
        extra (``qsarkit[extra]``) that installs it.

    Examples
    --------
    >>> from qsarkit.base import require
    >>> require("numpy").__name__
    'numpy'

    A missing dependency raises an error naming the extra that provides
    it, rather than an ``ImportError`` the caller has to interpret:

    >>> require("nonexistent_package_xyz")
    Traceback (most recent call last):
        ...
    qsarkit.base.exceptions.OptionalDependencyError: This feature requires...

    Call it inside ``__init__`` or ``fit``, never at module import time --
    that is what keeps ``import qsarkit`` from pulling in PyTorch.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        top_level = module_name.split(".")[0]
        extra = _EXTRA_FOR_MODULE.get(module_name) or _EXTRA_FOR_MODULE.get(top_level)
        raise OptionalDependencyError(top_level, extra) from exc
