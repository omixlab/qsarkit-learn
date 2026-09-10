"""Fixed-width molecular fingerprint transformers.

Every class here is a stateless :class:`qsarkit.base.MoleculeTransformer`
that maps ``Iterable[rdkit.Chem.Mol]`` to a dense ``(n_molecules,
n_features)`` ``float64`` array, is scikit-learn compatible (``fit`` /
``transform`` / ``fit_transform`` / ``get_params`` / ``clone``), and exposes
``get_feature_names_out()``.

References
----------
- RDKit documentation, "Fingerprinting and Molecular Similarity":
  https://www.rdkit.org/docs/GettingStartedInPython.html#fingerprinting-and-molecular-similarity
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
  J. Mach. Learn. Res., 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
"""

from __future__ import annotations

from qsarkit.representation.fingerprints._atompair import (
    AtomPairFingerprint,
    TopologicalTorsionFingerprint,
)
from qsarkit.representation.fingerprints._avalon import AvalonFingerprint
from qsarkit.representation.fingerprints._base import (
    BaseFingerprintTransformer,
    fold_on_bits,
)
from qsarkit.representation.fingerprints._combiner import FingerprintCombiner
from qsarkit.representation.fingerprints._maccs import MACCSKeysFingerprint
from qsarkit.representation.fingerprints._mhfp import MAP4Fingerprint, MHFPFingerprint
from qsarkit.representation.fingerprints._morgan import (
    FeatureMorganFingerprint,
    MorganFingerprint,
)
from qsarkit.representation.fingerprints._pharmacophore import PharmacophoreFingerprint
from qsarkit.representation.fingerprints._rdkit import (
    LayeredFingerprint,
    PatternFingerprint,
    RDKitFingerprint,
)

__all__ = [
    "BaseFingerprintTransformer",
    "fold_on_bits",
    "MorganFingerprint",
    "FeatureMorganFingerprint",
    "RDKitFingerprint",
    "PatternFingerprint",
    "LayeredFingerprint",
    "AtomPairFingerprint",
    "TopologicalTorsionFingerprint",
    "MACCSKeysFingerprint",
    "AvalonFingerprint",
    "PharmacophoreFingerprint",
    "MHFPFingerprint",
    "MAP4Fingerprint",
    "FingerprintCombiner",
]
