"""Descriptor selection for QSAR models.

Molecular descriptor sets routinely contain hundreds of correlated,
constant or uninformative columns. Selecting among them improves both
generalization and the mechanistic interpretability OECD principle 5
asks for. Every selector is scikit-learn compatible with ``get_support()``.

Examples
--------
>>> from sklearn.datasets import make_regression
>>> from qsarkit.feature_selection import VarianceFilter
>>> X, y = make_regression(n_samples=50, n_features=8, random_state=0)
>>> VarianceFilter().fit(X, y).get_support().shape
(8,)

References
----------
- Guyon, I. & Elisseeff, A. (2003). "An Introduction to Variable and
  Feature Selection." J. Mach. Learn. Res., 3, 1157-1182.
  https://jmlr.org/papers/v3/guyon03a.html
- Kursa, M. B. & Rudnicki, W. R. (2010). "Feature Selection with the
  Boruta Package." J. Stat. Softw., 36(11), 1-13.
  https://doi.org/10.18637/jss.v036.i11
"""

from qsarkit.feature_selection._boruta import BorutaSelector
from qsarkit.feature_selection._correlation import CorrelationFilter
from qsarkit.feature_selection._mutual_info import MutualInformationSelector
from qsarkit.feature_selection._rfe import RFESelector
from qsarkit.feature_selection._variance import VarianceFilter

__all__ = [
    "VarianceFilter",
    "CorrelationFilter",
    "MutualInformationSelector",
    "RFESelector",
    "BorutaSelector",
]
