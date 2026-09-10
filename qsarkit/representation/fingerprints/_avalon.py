"""Avalon fingerprints (lazy ``rdkit.Avalon`` toolkit binding)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from qsarkit.base import require
from qsarkit.representation.fingerprints._base import BaseFingerprintTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol


class AvalonFingerprint(BaseFingerprintTransformer):
    """Avalon substructure fingerprint.

    The Avalon cheminformatics toolkit enumerates a fixed, hand-designed
    set of feature classes (paths, rings, atom pairs at short distances,
    augmented atoms, ...) and hashes them into a folded bit vector. In the
    original benchmark it matched or outperformed contemporary path- and
    circular-based fingerprints on similarity searching.

    Parameters
    ----------
    n_bits : int, default 512
        Width of the folded fingerprint. 512 is the size used in the
        original publication and the RDKit default.
    use_counts : bool, default False
        Return per-feature occurrence counts
        (``pyAvalonTools.GetAvalonCountFP``) instead of a binary vector.
    is_query : bool, default False
        Generate the query flavour of the fingerprint, used when the
        molecule is a substructure query rather than a full structure.
    bit_flags : int, default 15761407
        Feature-class bitmask; the default is Avalon's
        ``avalonSSSBits`` similarity setting used by RDKit.

    Raises
    ------
    OptionalDependencyError
        If the RDKit build does not include the Avalon toolkit bindings
        (``rdkit.Avalon.pyAvalonTools``). Conda/PyPI RDKit wheels ship it,
        but minimal or source builds may omit it.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.representation.fingerprints import AvalonFingerprint
    >>> AvalonFingerprint(n_bits=256).fit_transform([Chem.MolFromSmiles("CCO")]).shape
    (1, 256)

    References
    ----------
    - Gedeck, P., Rohde, B. & Bartels, C. (2006). "QSAR - How Good Is It in
      Practice? Comparison of Descriptor Sets on an Unbiased Cross Section
      of Corporate Data Sets." J. Chem. Inf. Model., 46(5), 1924-1936.
      https://doi.org/10.1021/ci050413p
    - RDKit ``rdkit.Avalon.pyAvalonTools`` documentation:
      https://www.rdkit.org/docs/source/rdkit.Avalon.pyAvalonTools.html
    """

    _feature_prefix = "Avalon"

    #: RDKit's default Avalon similarity bit flags (``avalonSimilarityBits``).
    DEFAULT_BIT_FLAGS = 15761407

    def __init__(
        self,
        n_bits: int = 512,
        use_counts: bool = False,
        is_query: bool = False,
        bit_flags: int = DEFAULT_BIT_FLAGS,
    ):
        self.n_bits = n_bits
        self.use_counts = use_counts
        self.is_query = is_query
        self.bit_flags = bit_flags

    @property
    def n_features_out(self) -> int:
        return int(self.n_bits)

    def _fingerprint(self, mol: "Mol") -> np.ndarray:
        from rdkit import DataStructs

        avalon = require("rdkit.Avalon.pyAvalonTools")
        n_bits = int(self.n_bits)
        if self.use_counts:
            count_vector = avalon.GetAvalonCountFP(
                mol,
                nBits=n_bits,
                isQuery=bool(self.is_query),
                bitFlags=int(self.bit_flags),
            )
            array = np.zeros(n_bits, dtype=np.uint32)
            for bit, count in count_vector.GetNonzeroElements().items():
                array[bit] = count
            return array

        bit_vector = avalon.GetAvalonFP(
            mol,
            nBits=n_bits,
            isQuery=bool(self.is_query),
            bitFlags=int(self.bit_flags),
        )
        array = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(bit_vector, array)
        return array
