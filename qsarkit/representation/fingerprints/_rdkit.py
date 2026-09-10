"""Path/subgraph based RDKit fingerprints: RDKit FP, Pattern FP, Layered FP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import numpy.typing as npt

from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class RDKitFingerprint(BaseFingerprintTransformer):
    """The RDKit topological (Daylight-like) path fingerprint.

    Enumerates all linear and branched subgraphs of the molecule with a
    number of bonds between ``min_path`` and ``max_path``, hashes each one
    and sets ``n_bits_per_hash`` bits per subgraph in an ``n_bits``-wide
    vector. This is RDKit's default general-purpose substructure
    fingerprint and is closely related to the classic Daylight fingerprint.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the folded fingerprint.
    min_path : int, default 1
        Minimum number of bonds in the hashed subgraphs.
    max_path : int, default 7
        Maximum number of bonds in the hashed subgraphs.
    n_bits_per_hash : int, default 2
        Number of bits set per hashed subgraph.
    use_counts : bool, default False
        Return per-bit occurrence counts instead of a binary vector.
    branched_paths : bool, default True
        Include branched subgraphs in addition to linear paths.
    use_hs : bool, default True
        Include information about the number of hydrogens on each atom.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import RDKitFingerprint
    >>> RDKitFingerprint(n_bits=128).fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 128)

    References
    ----------
    - RDKit documentation, "RDKit fingerprint":
      https://www.rdkit.org/docs/RDKit_Book.html#rdkit-fingerprints
    - RDKit ``rdFingerprintGenerator.GetRDKitFPGenerator`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
    - Daylight Theory Manual, "Fingerprints":
      https://www.daylight.com/dayhtml/doc/theory/theory.finger.html
    """

    _feature_prefix = "RDKitFP"

    def __init__(
        self,
        n_bits: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        n_bits_per_hash: int = 2,
        use_counts: bool = False,
        branched_paths: bool = True,
        use_hs: bool = True,
    ):
        self.n_bits = n_bits
        self.min_path = min_path
        self.max_path = max_path
        self.n_bits_per_hash = n_bits_per_hash
        self.use_counts = use_counts
        self.branched_paths = branched_paths
        self.use_hs = use_hs

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _generator(self) -> Any:
        from rdkit.Chem import rdFingerprintGenerator

        return rdFingerprintGenerator.GetRDKitFPGenerator(
            minPath=int(self.min_path),
            maxPath=int(self.max_path),
            useHs=bool(self.use_hs),
            branchedPaths=bool(self.branched_paths),
            numBitsPerFeature=int(self.n_bits_per_hash),
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


class PatternFingerprint(BaseFingerprintTransformer):
    """RDKit pattern fingerprint, designed as a substructure-search screen.

    The pattern fingerprint enumerates small atom/bond patterns and is built
    so that ``fp(query) & fp(target) == fp(query)`` whenever ``query`` is a
    substructure of ``target``. That screening property makes it a strong
    similarity descriptor for substructure-driven SAR, at the cost of being
    much denser than ECFP-type fingerprints.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the fingerprint.
    tautomeric : bool, default False
        Use the tautomer-insensitive variant
        (``Chem.PatternFingerprint(..., tautomerFingerprints=True)``), which
        makes the screen invariant to common tautomeric shifts.

    Notes
    -----
    This fingerprint is binary only: RDKit does not expose a count variant,
    so no ``use_counts`` parameter is offered.

    References
    ----------
    - RDKit Book, "Pattern fingerprints":
      https://www.rdkit.org/docs/RDKit_Book.html#pattern-fingerprints
    - Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org
    """

    _feature_prefix = "PatternFP"

    def __init__(self, n_bits: int = 2048, tautomeric: bool = False):
        self.n_bits = n_bits
        self.tautomeric = tautomeric

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        from rdkit import Chem, DataStructs

        bit_vector = Chem.PatternFingerprint(
            mol,
            fpSize=int(self.n_bits),
            tautomerFingerprints=bool(self.tautomeric),
        )
        array = np.zeros(int(self.n_bits), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return array


class LayeredFingerprint(BaseFingerprintTransformer):
    """RDKit layered fingerprint (substructure fingerprint with atom layers).

    Like :class:`RDKitFingerprint` it hashes subgraphs, but each subgraph is
    described through several independent "layers" (pure topology, bond
    order, atom types, ring membership, ...), selected by ``layer_flags``.
    Combining layers yields a fingerprint that degrades gracefully between
    a pure-topology and a fully atom-typed description.

    Parameters
    ----------
    n_bits : int, default 2048
        Width of the fingerprint.
    min_path : int, default 1
        Minimum number of bonds in the hashed subgraphs.
    max_path : int, default 7
        Maximum number of bonds in the hashed subgraphs.
    layer_flags : int, default 0xFFFFFFFF
        Bitmask selecting which layers contribute (RDKit default: all).
    branched_paths : bool, default True
        Include branched subgraphs in addition to linear paths.

    References
    ----------
    - RDKit Book, "Layered fingerprints":
      https://www.rdkit.org/docs/RDKit_Book.html#layered-fingerprints
    - Landrum, G. RDKit: Open-source cheminformatics. https://www.rdkit.org
    """

    _feature_prefix = "LayeredFP"

    def __init__(
        self,
        n_bits: int = 2048,
        min_path: int = 1,
        max_path: int = 7,
        layer_flags: int = 0xFFFFFFFF,
        branched_paths: bool = True,
    ):
        self.n_bits = n_bits
        self.min_path = min_path
        self.max_path = max_path
        self.layer_flags = layer_flags
        self.branched_paths = branched_paths

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        from rdkit import Chem, DataStructs

        bit_vector = Chem.LayeredFingerprint(
            mol,
            layerFlags=int(self.layer_flags),
            minPath=int(self.min_path),
            maxPath=int(self.max_path),
            fpSize=int(self.n_bits),
            branchedPaths=bool(self.branched_paths),
        )
        array = np.zeros(int(self.n_bits), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return array
