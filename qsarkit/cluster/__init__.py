"""Cheminformatics clustering and diversity picking with a scikit-learn API.

All estimators here follow the scikit-learn clustering protocol
(``fit``/``fit_predict``/``labels_``) and operate on fingerprint matrices
using Tanimoto (Jaccard) distance, which is the chemically appropriate
metric for sparse binary fingerprints.
"""

from qsarkit.cluster._butina import ButinaClustering
from qsarkit.cluster._pickers import (
    HierarchicalClustering,
    MaxMinPicker,
    SphereExclusionClustering,
)

__all__ = [
    "ButinaClustering",
    "SphereExclusionClustering",
    "MaxMinPicker",
    "HierarchicalClustering",
]
