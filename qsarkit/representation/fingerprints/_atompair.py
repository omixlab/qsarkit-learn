"""Atom-pair and topological-torsion fingerprints (Carhart / Nilakantan)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import numpy.typing as npt

from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class AtomPairFingerprint(BaseFingerprintTransformer):
    """Carhart atom-pair fingerprint.

    Each feature is the triplet ``(atom type i, topological distance,
    atom type j)`` for every pair of atoms whose shortest-path distance lies
    between ``min_distance`` and ``max_distance``. Atom types encode element,
    number of heavy-atom neighbours and number of pi electrons. The triplets
    are hashed into ``n_bits``.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the folded fingerprint.
    min_distance : int, default 1
        Minimum topological (bond) distance between paired atoms.
    max_distance : int, default 30
        Maximum topological (bond) distance between paired atoms.
    use_counts : bool, default True
        Return per-bit occurrence counts. Atom pairs were defined as a
        counted descriptor in the original publication, so counts are the
        default here (unlike the other fingerprints in this module).
    use_chirality : bool, default False
        Include chiral tags in the atom types.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import AtomPairFingerprint
    >>> AtomPairFingerprint(n_bits=64).fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 64)

    References
    ----------
    - Carhart, R. E., Smith, D. H. & Venkataraghavan, R. (1985).
      "Atom Pairs as Molecular Features in Structure-Activity Studies:
      Definition and Applications." J. Chem. Inf. Comput. Sci., 25(2), 64-73.
      https://doi.org/10.1021/ci00046a002
    - RDKit ``rdFingerprintGenerator.GetAtomPairGenerator`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
    """

    _feature_prefix = "AtomPair"

    def __init__(
        self,
        n_bits: int = 2048,
        min_distance: int = 1,
        max_distance: int = 30,
        use_counts: bool = True,
        use_chirality: bool = False,
    ):
        self.n_bits = n_bits
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.use_counts = use_counts
        self.use_chirality = use_chirality

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _generator(self) -> Any:
        from rdkit.Chem import rdFingerprintGenerator

        return rdFingerprintGenerator.GetAtomPairGenerator(
            minDistance=int(self.min_distance),
            maxDistance=int(self.max_distance),
            includeChirality=bool(self.use_chirality),
            fpSize=int(self.n_bits),
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


class TopologicalTorsionFingerprint(BaseFingerprintTransformer):
    """Nilakantan topological-torsion fingerprint.

    Enumerates every linear path of ``torsion_size`` consecutively bonded
    non-hydrogen atoms (four by default, i.e. a torsion) and hashes the
    ordered tuple of their atom types into ``n_bits``. Torsions complement
    atom pairs by encoding short-range, shape-relevant connectivity.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the folded fingerprint.
    torsion_size : int, default 4
        Number of atoms in each hashed path.
    use_counts : bool, default True
        Return per-bit occurrence counts (the original formulation is a
        counted descriptor).
    use_chirality : bool, default False
        Include chiral tags in the atom types.

    References
    ----------
    - Nilakantan, R., Bauman, N., Dixon, J. S. & Venkataraghavan, R. (1987).
      "Topological Torsion: A New Molecular Descriptor for SAR Applications.
      Comparison with Other Descriptors."
      J. Chem. Inf. Comput. Sci., 27(2), 82-85.
      https://doi.org/10.1021/ci00054a008
    - RDKit ``rdFingerprintGenerator.GetTopologicalTorsionGenerator``
      documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
    """

    _feature_prefix = "TopTorsion"

    def __init__(
        self,
        n_bits: int = 2048,
        torsion_size: int = 4,
        use_counts: bool = True,
        use_chirality: bool = False,
    ):
        self.n_bits = n_bits
        self.torsion_size = torsion_size
        self.use_counts = use_counts
        self.use_chirality = use_chirality

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _generator(self) -> Any:
        from rdkit.Chem import rdFingerprintGenerator

        return rdFingerprintGenerator.GetTopologicalTorsionGenerator(
            torsionAtomCount=int(self.torsion_size),
            includeChirality=bool(self.use_chirality),
            fpSize=int(self.n_bits),
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
