"""Morgan / circular (ECFP, FCFP) fingerprints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import numpy.typing as npt

from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class MorganFingerprint(BaseFingerprintTransformer):
    """Morgan (circular) fingerprints, i.e. ECFP and FCFP.

    Iteratively hashes the circular atom environment of every atom up to
    ``radius`` bonds and folds the resulting identifiers into ``n_bits``.
    With ``use_features=False`` the atom invariants are connectivity based
    (ECFP flavour); with ``use_features=True`` they are the Gobbi-style
    pharmacophoric feature invariants (FCFP flavour). The ECFP diameter
    naming convention corresponds to ``2 * radius`` (``radius=2`` -> ECFP4).

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the folded fingerprint.
    radius : int, default 2
        Maximum circular environment radius, in bonds.
    use_counts : bool, default False
        If ``True`` return per-bit occurrence counts (``uint32``) instead of
        a binary vector (``uint8``).
    use_features : bool, default False
        Use pharmacophoric (FCFP) rather than connectivity (ECFP) atom
        invariants.
    use_chirality : bool, default False
        Include chiral tags in the atom invariants.
    use_bond_types : bool, default True
        Include bond orders when hashing environments.

    Attributes
    ----------
    n_features_out : int
        Equal to ``n_bits``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import MorganFingerprint
    >>> mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
    >>> X = MorganFingerprint(n_bits=64, radius=2).fit_transform(mols)
    >>> X.shape
    (2, 64)

    References
    ----------
    - Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
      J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    - Morgan, H. L. (1965). "The Generation of a Unique Machine Description
      for Chemical Structures." J. Chem. Doc., 5(2), 107-113.
      https://doi.org/10.1021/c160017a018
    - RDKit ``rdFingerprintGenerator`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
    """

    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        use_counts: bool = False,
        use_features: bool = False,
        use_chirality: bool = False,
        use_bond_types: bool = True,
    ):
        self.n_bits = n_bits
        self.radius = radius
        self.use_counts = use_counts
        self.use_features = use_features
        self.use_chirality = use_chirality
        self.use_bond_types = use_bond_types

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    @property
    def _feature_prefix(self) -> str:  # type: ignore[override]
        flavour = "FCFP" if self.use_features else "ECFP"
        return f"{flavour}{2 * int(self.radius)}"

    def _generator(self) -> Any:
        from rdkit.Chem import rdFingerprintGenerator

        invariants = (
            rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            if self.use_features
            else None
        )
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=int(self.radius),
            fpSize=int(self.n_bits),
            includeChirality=bool(self.use_chirality),
            useBondTypes=bool(self.use_bond_types),
            atomInvariantsGenerator=invariants,
        )

    def _fingerprint(self, mol: "Mol") -> npt.NDArray[np.float64]:
        generator = self._generator()
        if self.use_counts:
            return generator.GetCountFingerprintAsNumPy(mol)  # type: ignore[no-any-return]
        return generator.GetFingerprintAsNumPy(mol)  # type: ignore[no-any-return]

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        # Build the generator once per batch rather than once per molecule.
        generator = self._generator()
        encode = (
            generator.GetCountFingerprintAsNumPy
            if self.use_counts
            else generator.GetFingerprintAsNumPy
        )
        out = np.zeros((len(mols), self.n_features_out), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            out[i] = encode(mol)
        return out


class FeatureMorganFingerprint(MorganFingerprint):
    """FCFP-flavoured Morgan fingerprint (pharmacophoric atom invariants).

    Thin convenience subclass of :class:`MorganFingerprint` with
    ``use_features=True``; feature types are donor, acceptor, aromatic,
    halogen, basic and acidic, as defined by the Gobbi & Poppinger
    pharmacophore typing rules used by RDKit.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the folded fingerprint.
    radius : int, default 2
        Maximum circular environment radius, in bonds.
    use_counts : bool, default False
        Return counts rather than bits.
    use_chirality : bool, default False
        Include chiral tags in the atom invariants.
    use_bond_types : bool, default True
        Include bond orders when hashing environments.

    References
    ----------
    - Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
      J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    - Gobbi, A. & Poppinger, D. (1998). "Genetic Optimization of
      Combinatorial Libraries." Biotechnol. Bioeng., 61(1), 47-54.
      https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z
    - RDKit ``rdFingerprintGenerator`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
    """

    def __init__(
        self,
        n_bits: int = 2048,
        radius: int = 2,
        use_counts: bool = False,
        use_chirality: bool = False,
        use_bond_types: bool = True,
    ):
        super().__init__(
            n_bits=n_bits,
            radius=radius,
            use_counts=use_counts,
            use_features=True,
            use_chirality=use_chirality,
            use_bond_types=use_bond_types,
        )
