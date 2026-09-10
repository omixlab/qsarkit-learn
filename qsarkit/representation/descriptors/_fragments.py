"""SMARTS-based functional-group fragment counts (``rdkit.Chem.Fragments``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["FragmentDescriptors"]


class FragmentDescriptors(BaseDescriptorTransformer):
    """SMARTS-based functional-group fragment counts.

    ``rdkit.Chem.Fragments`` defines roughly 85 hand-curated SMARTS
    substructure counters (``fr_Al_COOH`` aliphatic carboxylic acid,
    ``fr_halogen``, ``fr_benzene``, ``fr_epoxide``, ...), each counting how
    many times a specific functional group occurs in the molecule. They
    are the interpretable, medicinal-chemistry counterpart to a hashed
    fingerprint and are frequently used as toxicophore/reactivity alerts
    or as an interpretable QSAR feature block.

    Parameters
    ----------
    fragment_names : sequence of str, optional
        Names of the ``fr_*`` functions to compute (must exist on
        ``rdkit.Chem.Fragments``). ``None`` (default) computes every
        ``fr_*`` function RDKit registers, in alphabetical order.
    missing_value : float, default nan
        Value substituted when a fragment counter raises for a given
        molecule.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import FragmentDescriptors
    >>> fd = FragmentDescriptors(fragment_names=["fr_benzene", "fr_halogen"])
    >>> fd.fit_transform([Chem.MolFromSmiles("c1ccccc1Cl")]).tolist()
    [[1.0, 1.0]]

    References
    ----------
    - Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org
    - RDKit ``rdkit.Chem.Fragments`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html
    - Ertl, P. (2017). "An Algorithm to Identify Functional Groups in
      Organic Molecules." J. Cheminform., 9, 36.
      https://doi.org/10.1186/s13321-017-0225-z
    """

    def __init__(
        self,
        fragment_names: Optional[Sequence[str]] = None,
        missing_value: float = float("nan"),
    ) -> None:
        self.fragment_names = fragment_names
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        from rdkit.Chem import Fragments

        catalogue = {
            name: getattr(Fragments, name)
            for name in dir(Fragments)
            if name.startswith("fr_") and callable(getattr(Fragments, name))
        }
        if self.fragment_names is None:
            return [(name, catalogue[name]) for name in sorted(catalogue)]
        missing = [n for n in self.fragment_names if n not in catalogue]
        if missing:
            raise ValueError(
                f"Unknown fragment descriptor name(s): {missing!r}. See "
                "rdkit.Chem.Fragments for valid 'fr_*' names."
            )
        return [(name, catalogue[name]) for name in self.fragment_names]
