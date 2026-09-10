"""Base classes shared by every molecule-consuming component in qsarkit.

Every chemistry and representation component accepts
``Iterable[rdkit.Chem.Mol]`` as its primary input and follows the
scikit-learn estimator/transformer protocol (``fit`` / ``transform`` /
``fit_transform``), so they compose naturally with
``sklearn.pipeline.Pipeline`` and the rest of ``qsarkit.transform`` /
``qsarkit.model_selection``.

Examples
--------
Subclassing is the whole contract: implement ``_transform`` on a list of
molecules, and inherit input validation, ``fit``, ``fit_transform`` and
scikit-learn compatibility.

>>> import numpy as np
>>> from rdkit import Chem
>>> from qsarkit.base import MoleculeTransformer
>>> class HeavyAtomCount(MoleculeTransformer):
...     def _transform(self, mols):
...         return np.array([[m.GetNumHeavyAtoms()] for m in mols], dtype=float)
>>> HeavyAtomCount().fit_transform([Chem.MolFromSmiles("CCO")]).tolist()
[[3.0]]

Because it is a genuine scikit-learn transformer, it drops into a
pipeline and survives ``clone``:

>>> from sklearn.base import clone
>>> clone(HeavyAtomCount())
HeavyAtomCount()

References
----------
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
  Journal of Machine Learning Research, 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
- RDKit: Open-source cheminformatics. https://www.rdkit.org
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Optional

from sklearn.base import BaseEstimator, TransformerMixin

from qsarkit.base.exceptions import InvalidMoleculeError


def ensure_mol_list(mols: Iterable[Any]) -> List[Any]:
    """Materialize an ``Iterable[Mol]`` into a list, validating entries.

    Parameters
    ----------
    mols:
        Iterable of ``rdkit.Chem.Mol`` objects. ``None`` entries are allowed
        through (representing molecules that failed an earlier parsing
        step) and are left untouched so callers can decide how to handle
        them positionally.

    Returns
    -------
    list
        The materialized list.

    Raises
    ------
    InvalidMoleculeError
        If an entry is neither ``None`` nor an RDKit ``Mol``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.base import ensure_mol_list
    >>> len(ensure_mol_list(Chem.MolFromSmiles(s) for s in ("CCO", "CCN")))
    2

    ``None`` is allowed through, because a molecule that failed an
    earlier parsing step must keep its position -- dropping it here
    would silently shift every downstream label by one:

    >>> ensure_mol_list([None])
    [None]

    Anything else is a programming error and is reported as one:

    >>> ensure_mol_list(["CCO"])
    Traceback (most recent call last):
        ...
    qsarkit.base.exceptions.InvalidMoleculeError: Element 0 is not an rdkit.Chem.Mol...
    """
    from rdkit import Chem

    out = []
    for i, m in enumerate(mols):
        if m is not None and not isinstance(m, Chem.Mol):
            raise InvalidMoleculeError(
                f"Element {i} is not an rdkit.Chem.Mol (got {type(m)!r})."
            )
        out.append(m)
    return out


class MoleculeTransformer(BaseEstimator, TransformerMixin, ABC):
    """Abstract base for stateless/stateful molecule -> X transformers.

    Subclasses implement :meth:`_transform` and operate on
    ``Iterable[rdkit.Chem.Mol]``. ``fit`` is a no-op by default (most
    chemistry transformers are stateless), but subclasses that need to
    learn parameters from data (e.g. a fingerprint vocabulary or a
    Mol2Vec embedding model) should override it.

    Examples
    --------
    See the module docstring for a complete subclass. Input validation
    is inherited, so a subclass never has to check its own arguments:

    >>> import numpy as np
    >>> from qsarkit.base import MoleculeTransformer
    >>> class RingCount(MoleculeTransformer):
    ...     def _transform(self, mols):
    ...         return np.array([m.GetRingInfo().NumRings() for m in mols])
    >>> RingCount().transform(["not a molecule"])
    Traceback (most recent call last):
        ...
    qsarkit.base.exceptions.InvalidMoleculeError: Element 0 is not an rdkit.Chem.Mol...
    """

    def fit(
        self, mols: Iterable[Any], y: Optional[Iterable[Any]] = None
    ) -> "MoleculeTransformer":
        """Default no-op fit. Override in stateful subclasses."""
        return self

    @abstractmethod
    def _transform(self, mols: List[Any]) -> Any:
        """Perform the actual transformation on a materialized list of Mols."""

    def transform(self, mols: Iterable[Any]) -> Any:
        """Validate input and dispatch to :meth:`_transform`."""
        mol_list = ensure_mol_list(mols)
        return self._transform(mol_list)


class MoleculeToMoleculeTransformer(MoleculeTransformer, ABC):
    """Base for transformers that map Mol -> Mol (standardization, curation, ...).

    Identical to :class:`MoleculeTransformer` in behaviour; the separate
    type documents that ``transform`` returns molecules rather than a
    feature matrix, so these can be chained with each other.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.base import MoleculeToMoleculeTransformer
    >>> class StripStereo(MoleculeToMoleculeTransformer):
    ...     def _transform(self, mols):
    ...         out = []
    ...         for m in mols:
    ...             copy = Chem.Mol(m)
    ...             Chem.RemoveStereochemistry(copy)
    ...             out.append(copy)
    ...         return out
    >>> mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    >>> Chem.MolToSmiles(StripStereo().transform([mol])[0])
    'CC(N)C(=O)O'
    """


class FittableMoleculeTransformer(MoleculeTransformer, ABC):
    """Base for transformers with learned state (embeddings, vocabularies).

    Adds an :attr:`is_fitted` flag and a ``_check_is_fitted`` guard, so
    calling ``transform`` before ``fit`` raises a clear error instead of
    producing silently meaningless features.

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from qsarkit.base import FittableMoleculeTransformer
    >>> class MeanCentredSize(FittableMoleculeTransformer):
    ...     def fit(self, mols, y=None):
    ...         self.mean_ = np.mean([m.GetNumHeavyAtoms() for m in mols])
    ...         self._is_fitted = True
    ...         return self
    ...     def _transform(self, mols):
    ...         self._check_is_fitted()
    ...         return np.array([m.GetNumHeavyAtoms() - self.mean_ for m in mols])
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1")]
    >>> transformer = MeanCentredSize()
    >>> transformer.is_fitted
    False
    >>> transformer.transform(mols)
    Traceback (most recent call last):
        ...
    qsarkit.base.exceptions.ModelNotFittedError: MeanCentredSize must be fitted...
    >>> transformer.fit(mols).transform(mols).tolist()
    [-1.5, 1.5]
    """

    def __init__(self) -> None:
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        from qsarkit.base.exceptions import ModelNotFittedError

        if not self._is_fitted:
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling transform()."
            )
