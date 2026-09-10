"""The full catalogue of RDKit 2-D/topological descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Optional, Sequence, Tuple

from qsarkit.representation.descriptors._base import BaseDescriptorTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["RDKitDescriptors"]


class RDKitDescriptors(BaseDescriptorTransformer):
    """Every descriptor registered in ``rdkit.Chem.Descriptors._descList``.

    RDKit registers on the order of 200 2-D/topological/electronic
    descriptors -- molecular weight and LogP, TPSA, connectivity and shape
    indices (Chi, Kappa), BCUT eigenvalues, VSA bins (PEOE_VSA, SMR_VSA,
    SlogP_VSA), fragment counts, ring/heteroatom counts, and more -- in a
    single lookup table, ``Descriptors._descList``. This transformer
    exposes that entire catalogue, or a user-selected subset of it, as one
    dense, named feature block. A handful of these descriptors (notably
    ``Ipc`` on large fused-ring systems) can raise or overflow to
    ``inf``/``nan`` on some molecules; both cases are substituted with
    ``missing_value``.

    Parameters
    ----------
    descriptor_names : sequence of str, optional
        Names of the descriptors to compute (must be keys of
        ``rdkit.Chem.Descriptors._descList``). ``None`` (default) computes
        every registered descriptor, in the order RDKit registers them.
    missing_value : float, default nan
        Value substituted when a descriptor raises or returns a
        non-finite value for a given molecule.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.descriptors import RDKitDescriptors
    >>> rd = RDKitDescriptors(descriptor_names=["MolWt", "TPSA"])
    >>> rd.fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 2)
    >>> list(rd.get_feature_names_out())
    ['MolWt', 'TPSA']

    References
    ----------
    - Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org
    - RDKit documentation, "List of Available Descriptors":
      https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics." Wiley-VCH. https://doi.org/10.1002/9783527628766
    """

    def __init__(
        self,
        descriptor_names: Optional[Sequence[str]] = None,
        missing_value: float = float("nan"),
    ) -> None:
        self.descriptor_names = descriptor_names
        self.missing_value = missing_value

    def _descriptor_functions(self) -> List[Tuple[str, Callable[["Mol"], float]]]:
        from rdkit.Chem import Descriptors

        catalogue = dict(Descriptors._descList)
        if self.descriptor_names is None:
            return list(catalogue.items())
        missing = [n for n in self.descriptor_names if n not in catalogue]
        if missing:
            raise ValueError(
                f"Unknown RDKit descriptor name(s): {missing!r}. See "
                "rdkit.Chem.Descriptors._descList for valid names."
            )
        return [(name, catalogue[name]) for name in self.descriptor_names]
