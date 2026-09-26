"""qsarkit - a focused, open-source Python library for QSAR modeling.

qsarkit covers the QSAR workflow proper: curating structures, turning them
into features, fitting and validating a model, defining where it applies,
and interpreting what it learned.

.. code-block:: text

    structures + measured activities -> chemical curation
      -> molecular representation -> QSAR modeling
      -> validation -> applicability domain -> uncertainty
      -> SAR interpretation -> reporting

It deliberately stops there. Literature mining, database retrieval,
docking and de-novo design are separate disciplines with separate failure
modes; bundling them would make the library broad rather than trustworthy.

Subpackages are imported lazily: ``import qsarkit`` is cheap, and the
heavy optional dependencies (PyTorch, transformers, gensim, ...) are only
loaded when you actually touch the feature that needs them.

Examples
--------
>>> import qsarkit
>>> from qsarkit.chemistry import MolecularStandardizer
>>> from rdkit import Chem
>>> standardizer = MolecularStandardizer()
>>> mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
>>> Chem.MolToSmiles(standardizer.transform([mol])[0])
'CC(=O)Oc1ccccc1C(=O)O'

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
  Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
- RDKit: Open-source cheminformatics. https://www.rdkit.org
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
  Python." J. Mach. Learn. Res., 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, List

__version__ = "0.8.0"

# Every public subpackage. Kept explicit (rather than scanned from the
# filesystem) so that `dir(qsarkit)` and tab-completion are stable and
# a typo in a subpackage name fails loudly.
_SUBPACKAGES = (
    "applicability",
    "base",
    "chemistry",
    "chemspace",
    "cluster",
    "data_quality",
    "explainability",
    "feature_selection",
    "functional",
    "metrics",
    "model_selection",
    "models",
    "neighbors",
    "persistence",
    "representation",
    "reporting",
    "sar",
    "transform",
    "uncertainty",
    "utils",
    "validation",
)

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from qsarkit import (
        applicability,
        base,
        chemistry,
        chemspace,
        cluster,
        data_quality,
        explainability,
        feature_selection,
        functional,
        metrics,
        model_selection,
        models,
        neighbors,
        persistence,
        representation,
        reporting,
        sar,
        transform,
        uncertainty,
        utils,
        validation,
    )


def __getattr__(name: str) -> Any:
    """Import a subpackage on first attribute access (PEP 562).

    Parameters
    ----------
    name : str
        Attribute being looked up on the ``qsarkit`` module.

    Returns
    -------
    module
        The imported subpackage.

    Raises
    ------
    AttributeError
        If ``name`` is not a qsarkit subpackage.
    """
    if name in _SUBPACKAGES:
        module = importlib.import_module(f"qsarkit.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module 'qsarkit' has no attribute {name!r}")


def __dir__() -> List[str]:
    """Expose subpackages to ``dir()`` and interactive completion."""
    return sorted([*globals().keys(), *_SUBPACKAGES])


__all__ = ["__version__", *_SUBPACKAGES]
