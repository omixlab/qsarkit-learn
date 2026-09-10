"""MACCS structural keys."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

#: RDKit emits 167 bits: the 166 public MACCS keys plus an unused bit 0.
MACCS_N_BITS = 167


class MACCSKeysFingerprint(BaseFingerprintTransformer):
    """166 public MACCS structural keys (RDKit implementation, 167 bits).

    Each bit is a hand-curated SMARTS substructure query ("has a carbonyl",
    "has 4 nitrogens", ...). Unlike the hashed fingerprints in this module
    MACCS keys are directly interpretable, which makes them useful for
    explainable QSAR and for coarse similarity screening.

    Parameters
    ----------
    drop_unused_bit : bool, default False
        RDKit returns 167 bits, index 0 being an always-off placeholder so
        that key ``k`` lands at index ``k``. Set to ``True`` to drop it and
        return exactly the 166 defined keys.

    Notes
    -----
    RDKit implements the 166 *public* MACCS key definitions; a handful of
    keys that require proprietary MDL features are approximated, as
    documented in ``rdkit.Chem.MACCSkeys``. The fingerprint has no
    ``n_bits``/``radius``/``use_counts`` parameters because the key set is
    fixed and binary by definition.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import MACCSKeysFingerprint
    >>> MACCSKeysFingerprint().fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 167)

    References
    ----------
    - Durant, J. L., Leland, B. A., Henry, D. R. & Nourse, J. G. (2002).
      "Reoptimization of MDL Keys for Use in Drug Discovery."
      J. Chem. Inf. Comput. Sci., 42(6), 1273-1280.
      https://doi.org/10.1021/ci010132r
    - RDKit ``rdkit.Chem.MACCSkeys`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Chem.MACCSkeys.html
    """

    def __init__(self, drop_unused_bit: bool = False):
        self.drop_unused_bit = drop_unused_bit

    @property
    def n_features_out(self) -> int:
        return MACCS_N_BITS - 1 if self.drop_unused_bit else MACCS_N_BITS

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        from rdkit import DataStructs
        from rdkit.Chem import MACCSkeys

        bit_vector = MACCSkeys.GenMACCSKeys(mol)
        array = np.zeros(MACCS_N_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return array[1:] if self.drop_unused_bit else array

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        start = 1 if self.drop_unused_bit else 0
        return np.asarray(
            [f"MACCS_{i}" for i in range(start, MACCS_N_BITS)], dtype=object
        )
