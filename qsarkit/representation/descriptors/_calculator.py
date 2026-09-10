"""Facade combining several named descriptor blocks into one feature matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
import numpy.typing as npt

from qsarkit.base import MoleculeTransformer
from qsarkit.representation.descriptors._base import BaseDescriptorTransformer
from qsarkit.representation.descriptors._constitutional import ConstitutionalDescriptors
from qsarkit.representation.descriptors._fragments import FragmentDescriptors
from qsarkit.representation.descriptors._lipinski import LipinskiDescriptors
from qsarkit.representation.descriptors._physicochemical import PhysicochemicalDescriptors
from qsarkit.representation.descriptors._rdkit_descriptors import RDKitDescriptors

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["DescriptorCalculator"]

#: Registry of block-name -> zero-argument constructor. ``"3d"`` is
#: resolved lazily inside :meth:`DescriptorCalculator._members` (rather
#: than imported at module level) since it is the only block whose
#: constructor performs no import-time work but whose *use* commonly
#: implies a heavier RDKit conformer-generation dependency chain best kept
#: out of the hot import path.
_BLOCK_FACTORIES: Dict[str, Type[BaseDescriptorTransformer]] = {
    "constitutional": ConstitutionalDescriptors,
    "physicochemical": PhysicochemicalDescriptors,
    "lipinski": LipinskiDescriptors,
    "fragments": FragmentDescriptors,
    "rdkit_all": RDKitDescriptors,
}


class DescriptorCalculator(MoleculeTransformer):
    """Facade combining several named descriptor blocks into one design matrix.

    QSAR feature engineering rarely uses a single descriptor family in
    isolation; this transformer selects and concatenates several of the
    named blocks in :mod:`qsarkit.representation.descriptors` (and,
    optionally, arbitrary user-supplied ``MoleculeTransformer`` instances)
    behind a single ``fit``/``transform`` call, mirroring
    :class:`qsarkit.representation.fingerprints.FingerprintCombiner` for
    descriptor blocks.

    Parameters
    ----------
    blocks : sequence of str, default ("physicochemical", "lipinski", "constitutional", "fragments")
        Names of built-in blocks to include, in order. Valid names are
        ``"constitutional"``, ``"physicochemical"``, ``"lipinski"``,
        ``"fragments"``, ``"rdkit_all"`` and ``"3d"``.
    extra_transformers : sequence of (str, MoleculeTransformer), optional
        Additional named transformers appended after the built-in blocks
        (e.g. a fingerprint, or a custom descriptor transformer). Each must
        implement ``fit``/``transform`` over ``Iterable[Mol]`` and
        ``get_feature_names_out()``.

    Attributes
    ----------
    n_features_out_ : int
        Total width of the concatenated output, set after ``fit``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import DescriptorCalculator
    >>> calc = DescriptorCalculator(blocks=["physicochemical", "lipinski"])
    >>> X = calc.fit_transform([Chem.MolFromSmiles("CCO")])
    >>> X.shape[1] == len(calc.get_feature_names_out())
    True

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley-VCH. https://doi.org/10.1002/9783527628766
    - scikit-learn ``FeatureUnion`` documentation:
      https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html
    """

    def __init__(
        self,
        blocks: Sequence[str] = (
            "physicochemical",
            "lipinski",
            "constitutional",
            "fragments",
        ),
        extra_transformers: Optional[Sequence[Tuple[str, MoleculeTransformer]]] = None,
    ) -> None:
        self.blocks = blocks
        self.extra_transformers = extra_transformers

    def _members(self) -> List[Tuple[str, MoleculeTransformer]]:
        members: List[Tuple[str, MoleculeTransformer]] = []
        for name in self.blocks:
            if name == "3d":
                from qsarkit.representation.descriptors._3d import Descriptors3D

                members.append((name, Descriptors3D()))
                continue
            factory = _BLOCK_FACTORIES.get(name)
            if factory is None:
                raise ValueError(
                    f"Unknown descriptor block {name!r}. Valid blocks are "
                    f"{sorted([*_BLOCK_FACTORIES, '3d'])!r}."
                )
            members.append((name, factory()))
        if self.extra_transformers:
            members.extend(self.extra_transformers)
        if not members:
            raise ValueError("DescriptorCalculator needs at least one block.")
        names = [name for name, _ in members]
        if len(set(names)) != len(names):
            raise ValueError(f"Block/transformer names must be unique, got {names!r}.")
        return members

    def fit(self, mols: Any, y: Optional[Any] = None) -> "DescriptorCalculator":
        """Fit every member block on the same molecules.

        Parameters
        ----------
        mols : Iterable[rdkit.Chem.Mol]
        y : array-like, optional
            Forwarded to each member's ``fit``.

        Returns
        -------
        DescriptorCalculator
            self.
        """
        from qsarkit.base import ensure_mol_list

        members = self._members()
        mol_list = ensure_mol_list(mols)
        for _, transformer in members:
            transformer.fit(mol_list, y)
        self.n_features_out_ = sum(
            int(np.asarray(t.get_feature_names_out()).shape[0]) for _, t in members
        )
        return self

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        members = self._members()
        blocks = [np.asarray(t.transform(mols), dtype=np.float64) for _, t in members]
        return np.asarray(np.concatenate(blocks, axis=1), dtype=np.float64)

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return prefixed feature names from every member block.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names, formatted ``"<block_name>__<feature>"``.
        """
        names: List[str] = []
        for block_name, transformer in self._members():
            for feature_name in transformer.get_feature_names_out():
                names.append(f"{block_name}__{feature_name}")
        return np.asarray(names, dtype=object)
