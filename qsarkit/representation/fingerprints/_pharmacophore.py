"""2D pharmacophore fingerprints (Gobbi & Poppinger feature definitions)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from qsarkit.representation.fingerprints._base import (
    BaseFingerprintTransformer,
    fold_on_bits,
)

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class PharmacophoreFingerprint(BaseFingerprintTransformer):
    """Gobbi 2D pharmacophore fingerprint.

    Atoms are typed into pharmacophoric classes (hydrogen-bond donor,
    acceptor, positive/negative ionizable, aromatic, lipophilic) using the
    Gobbi & Poppinger SMARTS definitions shipped with RDKit
    (``rdkit.Chem.Pharm2D.Gobbi_Pharm2D``). Every 2- and 3-point
    combination of typed atoms, binned by topological distance, becomes one
    bit of a very sparse 39 972-bit vector.

    Parameters
    ----------
    n_bits : int, optional, default 2048
        If given, the sparse key space is modulo-folded onto ``n_bits``
        columns, which keeps the output dense and machine-learning ready.
        Pass ``None`` to return the full, unfolded 39 972-bit vector.

    Notes
    -----
    Deviation from the original specification: the Gobbi signature is
    defined over 39 972 keys, which is impractical as a dense design matrix
    for typical QSAR datasets. By default this transformer therefore folds
    the set bits modulo ``n_bits`` (the standard RDKit folding strategy),
    accepting bit collisions in exchange for a compact representation. Set
    ``n_bits=None`` to recover the exact, unfolded key vector.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import PharmacophoreFingerprint
    >>> fp = PharmacophoreFingerprint(n_bits=256)
    >>> fp.fit_transform([Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")]).shape
    (1, 256)

    References
    ----------
    - Gobbi, A. & Poppinger, D. (1998). "Genetic Optimization of
      Combinatorial Libraries." Biotechnol. Bioeng., 61(1), 47-54.
      https://doi.org/10.1002/(SICI)1097-0290(199824)61:1<47::AID-BIT9>3.0.CO;2-Z
    - RDKit ``rdkit.Chem.Pharm2D`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.Pharm2D.html
    - RDKit Book, "2D pharmacophore fingerprints":
      https://www.rdkit.org/docs/RDKit_Book.html
    """

    _feature_prefix = "Pharm2D"

    def __init__(self, n_bits: Optional[int] = 2048):
        self.n_bits = n_bits

    @property
    def n_features_out(self) -> int:
        if self.n_bits is not None:
            return int(self.n_bits)
        from rdkit.Chem.Pharm2D import Gobbi_Pharm2D

        return int(Gobbi_Pharm2D.factory.GetSigSize())

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D

        signature = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        on_bits = list(signature.GetOnBits())
        if self.n_bits is None:
            array = np.zeros(signature.GetNumBits(), dtype=np.uint8)
            array[on_bits] = 1
            return array
        return fold_on_bits(on_bits, int(self.n_bits))
