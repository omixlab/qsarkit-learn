"""Shared plumbing for the RDKit-backed fingerprint transformers.

Every fingerprint in :mod:`qsarkit.representation.fingerprints` is a
stateless :class:`~qsarkit.base.MoleculeTransformer`: it consumes
``Iterable[rdkit.Chem.Mol]`` and returns a dense ``numpy.ndarray`` of shape
``(n_molecules, n_features_out)``. Subclasses only have to say how wide they
are (:attr:`BaseFingerprintTransformer.n_features_out`) and how to encode a
single molecule (``_fingerprint``); batching, ``None`` handling, dtype
selection and ``get_feature_names_out`` are provided here.

References
----------
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
  Journal of Machine Learning Research, 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
- RDKit fingerprint documentation:
  https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import numpy.typing as npt

from qsarkit.base import MoleculeTransformer

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["BaseFingerprintTransformer", "fold_on_bits"]


class BaseFingerprintTransformer(MoleculeTransformer):
    """Abstract base for dense, fixed-width molecular fingerprints.

    Notes
    -----
    ``None`` entries in the input (molecules that failed an earlier parsing
    or standardization step) are encoded as an all-zero row so that the
    output stays positionally aligned with the input, mirroring the
    convention used by :class:`qsarkit.chemistry.MolecularStandardizer`.
    Output arrays are always ``float64`` so that fingerprints compose
    directly with downstream scikit-learn estimators/scalers without an
    implicit cast; the intermediate RDKit bit/count vectors are generated in
    their natural ``uint8``/``uint32`` dtype for efficiency and cast on
    assignment into the output matrix.

    References
    ----------
    - RDKit documentation, "Fingerprinting and Molecular Similarity":
      https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
    - scikit-learn transformer API:
      https://scikit-learn.org/stable/developers/develop.html
    """

    #: Prefix used to build feature names (overridden by subclasses).
    _feature_prefix: str = "fp"

    @property
    @abstractmethod
    def n_features_out(self) -> int:
        """Width of the produced feature matrix."""

    @abstractmethod
    def _fingerprint(self, mol: "Mol") -> npt.NDArray[np.float64]:
        """Encode one RDKit ``Mol`` as a 1-D array of length ``n_features_out``."""

    def _transform(self, mols: List[Optional["Mol"]]) -> npt.NDArray[np.float64]:
        width = self.n_features_out
        out = np.zeros((len(mols), width), dtype=np.float64)
        for i, mol in enumerate(mols):
            if mol is None:
                continue
            out[i] = self._fingerprint(mol)
        return out

    def get_feature_names_out(
        self, input_features: Optional[Sequence[str]] = None
    ) -> npt.NDArray[np.object_]:
        """Return ``n_features_out`` bit/count names.

        Parameters
        ----------
        input_features : sequence of str, optional
            Ignored; present for scikit-learn API compatibility.

        Returns
        -------
        numpy.ndarray
            Array of ``str`` names, one per output column.

        Examples
        --------
        >>> from qsarkit.representation.fingerprints import MACCSKeysFingerprint
        >>> len(MACCSKeysFingerprint().get_feature_names_out())
        167
        """
        return np.asarray(
            [f"{self._feature_prefix}_{i}" for i in range(self.n_features_out)],
            dtype=object,
        )


def fold_on_bits(on_bits: Sequence[int], n_bits: int) -> npt.NDArray[np.float64]:
    """Fold a sparse list of set bit indices into a dense ``n_bits`` vector.

    Parameters
    ----------
    on_bits : sequence of int
        Indices of the set bits in a (possibly very large) sparse bit vector.
    n_bits : int
        Width of the folded output.

    Returns
    -------
    numpy.ndarray
        ``float64`` array of length ``n_bits`` holding 0/1 values.

    Notes
    -----
    Modulo folding is the standard RDKit approach for reducing sparse
    fingerprints (e.g. the 39 972-bit Gobbi 2D pharmacophore keys) to a
    machine-learning friendly width; it trades bit collisions for a fixed,
    dense representation.

    Examples
    --------
    Bits 0, 5 and 10 fold to 0, 1 and 2 modulo 4:

    >>> fold_on_bits([0, 5, 10], 4).tolist()
    [1.0, 1.0, 1.0, 0.0]

    Collisions are silent, which is the cost of folding -- bits 1 and 5
    land on the same output position:

    >>> fold_on_bits([1, 5], 4).tolist()
    [0.0, 1.0, 0.0, 0.0]

    References
    ----------
    - RDKit documentation, "Fingerprinting and Molecular Similarity":
      https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
    """
    vector = np.zeros(n_bits, dtype=np.float64)
    for bit in on_bits:
        vector[bit % n_bits] = 1.0
    return vector
