"""Run the documentation's examples as part of the ordinary test suite.

Sphinx can execute the ``>>>`` blocks in ``docs-sphinx/source`` via ``make doctest``,
but a documentation build is not something anyone runs before pushing. Wiring
the same examples into pytest means a change that breaks a documented example
fails CI like any other regression, so the docs cannot quietly rot.

The namespace injected here mirrors ``doctest_global_setup`` in
``docs-sphinx/source/conf.py``; both import from ``docs-sphinx/source/demo_data.py``, so
there is exactly one definition of the demo dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DOCS_SOURCE = Path(__file__).resolve().parents[2] / "docs-sphinx" / "source"


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace):
    """Populate every docs doctest with the shared demo dataset."""
    if str(DOCS_SOURCE) not in sys.path:
        sys.path.insert(0, str(DOCS_SOURCE))

    import numpy as np

    import demo_data

    np.set_printoptions(precision=3, suppress=True)
    doctest_namespace["np"] = np
    for name in demo_data.__all__:
        doctest_namespace[name] = getattr(demo_data, name)
