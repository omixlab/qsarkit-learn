"""Robust, pickle-free persistence for QSAR models and their pipelines.

``pickle`` is the obvious way to save a model and the wrong one for a
model you intend to keep: it embeds the exact class layout of every
object, so a file written under one scikit-learn release can fail to load
under the next, and loading one executes arbitrary code.

This package writes a directory bundle instead -- JSON metadata beside a
`skops <https://skops.readthedocs.io>`_ representation of the estimator,
which stores parameters as data and refuses to reconstruct types that were
not explicitly trusted.

Examples
--------
>>> import numpy as np, os, tempfile
>>> from sklearn.linear_model import Ridge
>>> from qsarkit.persistence import ModelMetadata, load_model, save_model
>>> rng = np.random.default_rng(0)
>>> X, y = rng.normal(size=(40, 5)), rng.normal(size=40)
>>> path = save_model(
...     Ridge().fit(X, y),
...     os.path.join(tempfile.mkdtemp(), "demo"),
...     metadata=ModelMetadata(name="demo", endpoint="pIC50 (-log10 M)"),
... )
>>> bundle = load_model(path)
>>> bundle.metadata.endpoint
'pIC50 (-log10 M)'
>>> bool(np.allclose(bundle.predict(X), Ridge().fit(X, y).predict(X)))
True

References
----------
- skops documentation, "Secure persistence with skops":
  https://skops.readthedocs.io/en/stable/persistence.html
- scikit-learn, "Model persistence":
  https://scikit-learn.org/stable/model_persistence.html
"""

from qsarkit.persistence._bundle import (
    BUNDLE_SUFFIX,
    ModelBundle,
    inspect_bundle,
    load_model,
    save_model,
)
from qsarkit.persistence._metadata import (
    BUNDLE_FORMAT_VERSION,
    ModelMetadata,
    environment_summary,
)

__all__ = [
    "ModelBundle",
    "ModelMetadata",
    "save_model",
    "load_model",
    "inspect_bundle",
    "environment_summary",
    "BUNDLE_FORMAT_VERSION",
    "BUNDLE_SUFFIX",
]
