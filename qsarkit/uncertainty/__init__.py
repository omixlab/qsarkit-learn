"""Uncertainty quantification for QSAR predictions.

Two complementary families:

- **Conformal prediction** (:class:`ConformalRegressor`,
  :class:`ConformalClassifier`) gives distribution-free intervals or
  label sets with a guaranteed coverage rate, assuming only that the
  data are exchangeable.
- **Model-based estimators** (:class:`EnsembleUncertainty`,
  :class:`GaussianProcessUncertainty`, :class:`MCDropoutUncertainty`,
  :class:`QuantileRegressionUncertainty`) give a per-sample standard
  deviation, useful for ranking and acquisition functions.

:class:`UncertaintyCalibration` checks whether either kind actually means
what it claims.

Examples
--------
>>> import numpy as np
>>> from sklearn.ensemble import RandomForestRegressor
>>> from qsarkit.uncertainty import ConformalRegressor
>>> rng = np.random.RandomState(0)
>>> X = rng.normal(size=(200, 4))
>>> y = X[:, 0] * 2 + rng.normal(scale=0.3, size=200)
>>> cp = ConformalRegressor(
...     RandomForestRegressor(n_estimators=20, random_state=0),
...     alpha=0.1, random_state=0,
... ).fit(X, y)
>>> lower, upper = cp.predict_interval(X)
>>> bool(np.all(upper >= lower))
True

References
----------
- Vovk, V., Gammerman, A. & Shafer, G. (2005). "Algorithmic Learning in a
  Random World." Springer. https://doi.org/10.1007/b106715
- Norinder, U. et al. (2014). "Introducing Conformal Prediction in
  Predictive Modeling." J. Chem. Inf. Model., 54(6), 1596-1603.
  https://doi.org/10.1021/ci5001168
- Scalia, G. et al. (2020). "Evaluating Scalable Uncertainty Estimation
  Methods for Deep Learning-Based Molecular Property Prediction."
  J. Chem. Inf. Model., 60(6), 2697-2717.
  https://doi.org/10.1021/acs.jcim.9b00975
"""

from qsarkit.uncertainty._calibration import UncertaintyCalibration
from qsarkit.uncertainty._conformal import (
    ConformalClassifier,
    ConformalPredictor,
    ConformalRegressor,
)
from qsarkit.uncertainty._estimators import (
    BaseUncertaintyEstimator,
    EnsembleUncertainty,
    GaussianProcessUncertainty,
    MCDropoutUncertainty,
    QuantileRegressionUncertainty,
)

__all__ = [
    "ConformalRegressor",
    "ConformalClassifier",
    "ConformalPredictor",
    "BaseUncertaintyEstimator",
    "EnsembleUncertainty",
    "MCDropoutUncertainty",
    "GaussianProcessUncertainty",
    "QuantileRegressionUncertainty",
    "UncertaintyCalibration",
]
