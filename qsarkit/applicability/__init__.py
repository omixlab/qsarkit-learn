"""Applicability domain estimation (OECD validation principle 3).

Every estimator shares one interface: ``fit(X)``, ``score_samples(X)``
(larger = further outside), ``predict(X)`` (True = inside) and
``decision_function(X)`` (positive = inside).

Examples
--------
>>> import numpy as np
>>> from qsarkit.applicability import LeverageAD
>>> X = np.random.RandomState(0).normal(size=(50, 3))
>>> ad = LeverageAD().fit(X)
>>> bool(ad.predict(np.zeros((1, 3)))[0])
True

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
  Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
- Sahigara, F. et al. (2012). "Comparison of Different Approaches to
  Define the Applicability Domain of QSAR Models." Molecules, 17(5),
  4791-4810. https://doi.org/10.3390/molecules17054791
"""

from qsarkit.applicability._analyzer import ADAnalyzer
from qsarkit.applicability._domains import (
    BaseApplicabilityDomain,
    BoundingBoxAD,
    ConvexHullAD,
    DistanceToModelAD,
    EnsembleAD,
    IsolationForestAD,
    KernelDensityAD,
    KNNApplicabilityDomain,
    LeverageAD,
    PCABoundingBoxAD,
    RangeAD,
    TanimotoSimilarityAD,
)

__all__ = [
    "BaseApplicabilityDomain",
    "LeverageAD",
    "DistanceToModelAD",
    "KNNApplicabilityDomain",
    "RangeAD",
    "BoundingBoxAD",
    "PCABoundingBoxAD",
    "ConvexHullAD",
    "TanimotoSimilarityAD",
    "KernelDensityAD",
    "IsolationForestAD",
    "EnsembleAD",
    "ADAnalyzer",
]
