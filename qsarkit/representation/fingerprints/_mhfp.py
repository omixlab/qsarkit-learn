"""MinHashed fingerprints: MHFP6 and MAP4."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np

from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class _MHFPBackend:
    """Uniform façade over the two available MinHash/MHFP implementations.

    The reference implementation is the ``mhfp`` package by Probst &
    Reymond; RDKit ships a C++ port of it as
    ``rdkit.Chem.rdMHFPFingerprint.MHFPEncoder``. The two expose the same
    algorithm under different method names, so this adapter normalizes them
    and lets the transformers work with either backend.

    Parameters
    ----------
    n_permutations : int
        Number of MinHash permutations.
    seed : int
        Permutation seed.

    Attributes
    ----------
    name : str
        ``"mhfp"`` when the reference package is installed, ``"rdkit"``
        when falling back to RDKit's native port.

    References
    ----------
    - Probst, D. & Reymond, J.-L. (2018). "A Probabilistic Molecular
      Fingerprint for Big Data Settings." J. Cheminform., 10, 66.
      https://doi.org/10.1186/s13321-018-0321-8
    - Reference implementation: https://github.com/reymond-group/mhfp
    - RDKit ``rdkit.Chem.rdMHFPFingerprint`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdMHFPFingerprint.html
    """

    def __init__(self, n_permutations: int, seed: int):
        self.n_permutations = int(n_permutations)
        self.seed = int(seed)
        try:
            from mhfp.encoder import MHFPEncoder  # type: ignore[import-not-found]

            self._encoder = MHFPEncoder(self.n_permutations, self.seed)
            self.name = "mhfp"
        except ImportError:
            from rdkit.Chem import rdMHFPFingerprint

            self._encoder = rdMHFPFingerprint.MHFPEncoder(
                self.n_permutations, self.seed
            )
            self.name = "rdkit"

    def encode_mol(
        self,
        mol: "Mol",
        radius: int,
        min_radius: int,
        rings: bool,
        isomeric: bool,
        kekulize: bool,
    ) -> np.ndarray:
        """MinHash vector of the circular-shingle set of ``mol``."""
        if self.name == "mhfp":
            hashes = self._encoder.encode_mol(
                mol,
                radius=radius,
                rings=rings,
                kekulize=kekulize,
                min_radius=min_radius,
            )
        else:
            hashes = self._encoder.EncodeMol(
                mol,
                radius=radius,
                isomeric=isomeric,
                kekulize=kekulize,
                min_radius=min_radius,
                rings=rings,
            )
        return np.asarray(list(hashes), dtype=np.uint32)

    def secfp_mol(
        self,
        mol: "Mol",
        radius: int,
        min_radius: int,
        rings: bool,
        isomeric: bool,
        kekulize: bool,
        length: int,
    ) -> np.ndarray:
        """Folded binary SECFP vector of ``mol``."""
        from rdkit import DataStructs

        array = np.zeros(int(length), dtype=np.uint8)
        if self.name == "mhfp":
            folded = self._encoder.secfp_from_mol(
                mol,
                length=length,
                radius=radius,
                rings=rings,
                kekulize=kekulize,
                min_radius=min_radius,
            )
            array[:] = np.asarray(folded, dtype=np.uint8)
            return array

        bit_vector = self._encoder.EncodeSECFPMol(
            mol,
            radius=radius,
            rings=rings,
            isomeric=isomeric,
            kekulize=kekulize,
            min_radius=min_radius,
            length=length,
        )
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return array

    def from_strings(self, shingles: Sequence[str]) -> np.ndarray:
        """MinHash vector of an arbitrary set of string shingles."""
        if self.name == "mhfp":
            hashes = self._encoder.from_string_array(list(shingles))
        else:
            hashes = self._encoder.FromStringArray(list(shingles))
        return np.asarray(list(hashes), dtype=np.uint32)


def _mhfp_encoder(n_permutations: int, seed: int) -> _MHFPBackend:
    """Build an :class:`_MHFPBackend` (``mhfp`` if installed, else RDKit)."""
    return _MHFPBackend(n_permutations, seed)


class MHFPFingerprint(BaseFingerprintTransformer):
    """MHFP6 - MinHashed fingerprint of circular substructure SMILES.

    The molecule is decomposed into the set of canonical SMILES of all
    circular substructures ("shingles") up to ``radius`` bonds; that set is
    then compressed with MinHash into ``n_permutations`` 32-bit hash values.
    The MinHash vector approximates the Jaccard distance between shingle
    sets, which makes MHFP6 a strong similarity and virtual-screening
    descriptor for large, diverse libraries.

    Two output modes are available:

    ``fold=False`` (default)
        Return the raw ``uint32`` MinHash vector of length
        ``n_permutations``. This is the representation used for
        MinHash-LSH nearest-neighbour search.
    ``fold=True``
        Return the folded binary SECFP variant of width ``n_bits``, which
        is directly usable as a design matrix for scikit-learn models.

    Parameters
    ----------
    n_permutations : int, default 2048
        Number of MinHash permutations (length of the unfolded vector).
    radius : int, default 3
        Maximum circular substructure radius ("6" in MHFP6 is the diameter).
    min_radius : int, default 1
        Minimum circular substructure radius.
    n_bits : int, default 2048
        Width of the folded binary output when ``fold=True``.
    fold : bool, default False
        Emit the folded binary SECFP vector instead of the MinHash vector.
    rings : bool, default True
        Include whole-ring SMILES as additional shingles (folded mode).
    isomeric : bool, default False
        Keep stereochemistry in the shingle SMILES.
    kekulize : bool, default True
        Kekulize substructures before writing their SMILES.
    seed : int, default 42
        Seed of the MinHash permutation set; fixing it makes the transformer
        deterministic and comparable across runs.

    Notes
    -----
    ``use_counts`` is not offered: a MinHash sketch is a set summary and has
    no meaningful multiplicity.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import MHFPFingerprint
    >>> fp = MHFPFingerprint(n_permutations=64)
    >>> fp.fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 64)

    References
    ----------
    - Probst, D. & Reymond, J.-L. (2018). "A Probabilistic Molecular
      Fingerprint for Big Data Settings." J. Cheminform., 10, 66.
      https://doi.org/10.1186/s13321-018-0321-8
    - Broder, A. Z. (1997). "On the Resemblance and Containment of
      Documents." Proc. Compression and Complexity of Sequences, 21-29.
      https://doi.org/10.1109/SEQUEN.1997.666900
    - Reference implementation: https://github.com/reymond-group/mhfp
    - RDKit ``rdkit.Chem.rdMHFPFingerprint`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.rdMHFPFingerprint.html
    """

    _feature_prefix = "MHFP"

    def __init__(
        self,
        n_permutations: int = 2048,
        radius: int = 3,
        min_radius: int = 1,
        n_bits: int = 2048,
        fold: bool = False,
        rings: bool = True,
        isomeric: bool = False,
        kekulize: bool = True,
        seed: int = 42,
    ):
        self.n_permutations = n_permutations
        self.radius = radius
        self.min_radius = min_radius
        self.n_bits = n_bits
        self.fold = fold
        self.rings = rings
        self.isomeric = isomeric
        self.kekulize = kekulize
        self.seed = seed

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits) if self.fold else int(self.n_permutations)

    @property
    def _value_dtype(self) -> Any:
        return np.uint8 if self.fold else np.uint32

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        encoder = _mhfp_encoder(int(self.n_permutations), int(self.seed))
        return self._encode(encoder, mol)

    def _encode(self, encoder: "_MHFPBackend", mol: "Mol") -> np.ndarray:
        # Route through the _MHFPBackend facade (not the raw encoder) so the
        # call works identically whether the ``mhfp`` package or RDKit's
        # native port is backing it.
        if self.fold:
            return encoder.secfp_mol(
                mol,
                radius=int(self.radius),
                min_radius=int(self.min_radius),
                rings=bool(self.rings),
                isomeric=bool(self.isomeric),
                kekulize=bool(self.kekulize),
                length=int(self.n_bits),
            )
        return encoder.encode_mol(
            mol,
            radius=int(self.radius),
            min_radius=int(self.min_radius),
            rings=bool(self.rings),
            isomeric=bool(self.isomeric),
            kekulize=bool(self.kekulize),
        )

    def _transform(self, mols: List[Any]) -> np.ndarray:
        encoder = _mhfp_encoder(int(self.n_permutations), int(self.seed))
        out = np.zeros((len(mols), self.n_features_out), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            out[i] = self._encode(encoder, mol)
        return out


class MAP4Fingerprint(BaseFingerprintTransformer):
    """MAP4 - MinHashed atom-pair fingerprint of radius 2.

    MAP4 merges the two ideas behind :class:`AtomPairFingerprint` and
    :class:`MHFPFingerprint`: for every pair of atoms ``(i, j)`` and every
    radius ``r <= radius``, the canonical SMILES of the circular
    substructures around ``i`` and ``j`` are paired with their topological
    distance ``d`` into a shingle ``"smiles_a|d|smiles_b"`` (the two SMILES
    sorted lexicographically so the shingle is order independent). The
    resulting shingle set is compressed by MinHash exactly as in MHFP.
    The authors show MAP4 performs well across both small molecules and
    peptides/macrocycles, where circular-only fingerprints degrade.

    Parameters
    ----------
    n_permutations : int, default 2048
        Number of MinHash permutations (output width when ``fold=False``).
    radius : int, default 2
        Maximum circular substructure radius around each atom of the pair.
    max_distance : int, optional
        Only pair atoms whose topological distance is at most this value.
        ``None`` (default) uses all pairs, as in the reference
        implementation.
    n_bits : int, default 2048
        Width of the folded binary output when ``fold=True``.
    fold : bool, default False
        Fold the MinHash vector modulo ``n_bits`` into a binary vector.
    isomeric : bool, default False
        Keep stereochemistry in the shingle SMILES.
    seed : int, default 42
        Seed of the MinHash permutation set.

    Notes
    -----
    Deviation: the reference implementation (``map4``) is not on PyPI as a
    maintained wheel, so the shingling step is re-implemented here directly
    on top of RDKit (``FindAtomEnvironmentOfRadiusN`` +
    ``PathToSubmol``), and the MinHash step reuses the MHFP encoder from
    the ``mhfp`` package when present or RDKit's native port otherwise. The
    shingle grammar follows Capecchi et al. exactly.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import MAP4Fingerprint
    >>> MAP4Fingerprint(n_permutations=32).fit_transform(
    ...     [Chem.MolFromSmiles("CCO")]
    ... ).shape
    (1, 32)

    References
    ----------
    - Capecchi, A., Probst, D. & Reymond, J.-L. (2020). "One Molecular
      Fingerprint to Rule Them All: Drugs, Biomolecules, and the Metabolome."
      J. Cheminform., 12, 43. https://doi.org/10.1186/s13321-020-00445-4
    - Probst, D. & Reymond, J.-L. (2018). "A Probabilistic Molecular
      Fingerprint for Big Data Settings." J. Cheminform., 10, 66.
      https://doi.org/10.1186/s13321-018-0321-8
    - Reference implementation: https://github.com/reymond-group/map4
    """

    _feature_prefix = "MAP4"

    def __init__(
        self,
        n_permutations: int = 2048,
        radius: int = 2,
        max_distance: Optional[int] = None,
        n_bits: int = 2048,
        fold: bool = False,
        isomeric: bool = False,
        seed: int = 42,
    ):
        self.n_permutations = n_permutations
        self.radius = radius
        self.max_distance = max_distance
        self.n_bits = n_bits
        self.fold = fold
        self.isomeric = isomeric
        self.seed = seed

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits) if self.fold else int(self.n_permutations)

    @property
    def _value_dtype(self) -> Any:
        return np.uint8 if self.fold else np.uint32

    def shingles(self, mol: "Mol") -> List[str]:
        """Return the MAP4 shingle set of one molecule.

        Parameters
        ----------
        mol : rdkit.Chem.Mol
            Molecule to decompose.

        Returns
        -------
        list of str
            Shingles of the form ``"<smiles_a>|<distance>|<smiles_b>"``.
        """
        from rdkit import Chem

        n_atoms = mol.GetNumAtoms()
        if n_atoms < 2:
            return []
        distances = Chem.GetDistanceMatrix(mol)
        max_distance = (
            float("inf") if self.max_distance is None else float(self.max_distance)
        )

        # Cache the circular-substructure SMILES per (atom, radius).
        env_smiles: dict = {}

        def substructure_smiles(atom_idx: int, radius: int) -> str:
            key = (atom_idx, radius)
            cached = env_smiles.get(key)
            if cached is not None:
                return cached
            bonds = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            if not bonds:
                smi = mol.GetAtomWithIdx(atom_idx).GetSymbol()
            else:
                atom_map: dict = {}
                submol = Chem.PathToSubmol(mol, bonds, atomMap=atom_map)
                root = atom_map.get(atom_idx, -1)
                smi = Chem.MolToSmiles(
                    submol,
                    rootedAtAtom=root,
                    canonical=True,
                    isomericSmiles=bool(self.isomeric),
                )
            env_smiles[key] = smi
            return smi

        out: List[str] = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                distance = distances[i][j]
                if distance > max_distance or not np.isfinite(distance):
                    continue
                for radius in range(1, int(self.radius) + 1):
                    smi_i = substructure_smiles(i, radius)
                    smi_j = substructure_smiles(j, radius)
                    first, second = sorted((smi_i, smi_j))
                    out.append(f"{first}|{int(distance)}|{second}")
        return out

    def _encode(self, encoder: "_MHFPBackend", mol: "Mol") -> np.ndarray:
        from qsarkit.representation.fingerprints._base import fold_on_bits

        shingles = self.shingles(mol)
        if not shingles:
            return np.zeros(self.n_features_out, dtype=self._value_dtype)
        hashes = encoder.from_strings(shingles)
        if self.fold:
            return fold_on_bits(
                [int(h) % int(self.n_bits) for h in hashes], int(self.n_bits)
            )
        return hashes

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        encoder = _mhfp_encoder(int(self.n_permutations), int(self.seed))
        return self._encode(encoder, mol)

    def _transform(self, mols: List[Any]) -> np.ndarray:
        encoder = _mhfp_encoder(int(self.n_permutations), int(self.seed))
        out = np.zeros((len(mols), self.n_features_out), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            out[i] = self._encode(encoder, mol)
        return out
