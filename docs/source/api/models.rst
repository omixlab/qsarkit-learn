Models
======

.. currentmodule:: qsarkit.models

QSAR regressors and classifiers behind one scikit-learn-compatible
facade, plus the algorithms that are specifically chemometric: PLS with
VIP scores, a Tanimoto-kernel Gaussian process, and consensus ensembles.

The facades
-----------

:class:`QSARRegressor` and :class:`QSARClassifier` dispatch on a name, so
comparing backends is a one-word change:

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> for name in ("rf", "svm", "pls", "ridge"):
   ...     model = QSARRegressor(name, random_state=0).fit(X, y)
   ...     print(f"{name:6} {model.score(X, y):.3f}")
   rf     0.952
   svm    0.880
   pls    0.900
   ridge  0.944

.. warning::

   Those are *training* scores, shown only to demonstrate the interface.
   A model scored on its own training data tells you nothing about
   whether it generalizes, and on 24 compounds with 256 features every
   one of these will look good. Compare backends on a held-out set or by
   cross-validation — see :doc:`validation`.

Any estimator, not just the menu
--------------------------------

``name`` also accepts anything following the scikit-learn ``fit``/
``predict`` protocol — XGBoost, LightGBM, CatBoost, or your own wrapper —
as an instance or a class:

.. doctest::

   >>> from sklearn.linear_model import Ridge
   >>> model = QSARRegressor(Ridge(alpha=2.0)).fit(X, y)
   >>> type(model.estimator_).__name__, model.estimator_.alpha
   ('Ridge', 2.0)

The instance you pass is cloned, never mutated, so one configured
template can seed several models:

.. doctest::

   >>> template = Ridge(alpha=1.0)
   >>> _ = QSARRegressor(template).fit(X, y)
   >>> hasattr(template, "coef_")
   False

Passing a *class* lets the facade build it, which is what ``model_args``
and ``model_params`` are for:

.. doctest::

   >>> model = QSARRegressor(Ridge, model_params={"alpha": 5.0}).fit(X, y)
   >>> model.estimator_.alpha
   5.0

``fit_params``, ``predict_params`` and ``predict_proba_params`` reach
arguments that belong to the call rather than the constructor — XGBoost's
``eval_set``, LightGBM's ``callbacks``, CatBoost's ``verbose``, or a
plain ``sample_weight``:

.. doctest::

   >>> import numpy as np
   >>> weights = np.linspace(0.5, 1.5, len(y))
   >>> model = QSARRegressor(Ridge, fit_params={"sample_weight": weights}).fit(X, y)
   >>> model.predict(X).shape
   (24,)

Because the facade is a real estimator, a custom backend still composes
with the rest of scikit-learn:

.. doctest::

   >>> from sklearn.base import is_regressor
   >>> from sklearn.model_selection import GridSearchCV
   >>> is_regressor(QSARRegressor(Ridge))
   True
   >>> search = GridSearchCV(
   ...     QSARRegressor(Ridge),
   ...     {"model_params": [{"alpha": 0.1}, {"alpha": 10.0}]},
   ...     cv=3,
   ... ).fit(X, y)
   >>> sorted(search.best_params_["model_params"])
   ['alpha']

PLS and VIP
-----------

Partial least squares is the chemometric workhorse: it handles more
descriptors than compounds, which is the normal QSAR situation and the
regime where ordinary regression fails.

.. doctest::

   >>> from qsarkit.models import PLSRegressor
   >>> pls = PLSRegressor(n_components=3).fit(X, y)
   >>> pls.predict(X).shape
   (24,)
   >>> pls.vip_scores_.shape
   (256,)

VIP scores above 1 mark the variables carrying the fit — the standard
threshold for descriptor selection in a PLS model:

.. doctest::

   >>> int((pls.vip_scores_ > 1.0).sum())
   58

Gaussian processes with uncertainty
-----------------------------------

.. doctest::

   >>> from qsarkit.models import GaussianProcessQSAR
   >>> gp = GaussianProcessQSAR(random_state=0).fit(X, y)
   >>> mean, std = gp.predict(X, return_std=True)
   >>> mean.shape, std.shape
   ((24,), (24,))

The kernel is chosen from the data: Tanimoto for non-negative
fingerprint-like input, RBF otherwise. A Tanimoto kernel is not positive
semi-definite on signed descriptors, so hardcoding it would produce a
model that silently fails to converge.

Consensus
---------

.. doctest::

   >>> from qsarkit.models import ConsensusModel
   >>> ensemble = ConsensusModel([
   ...     ("rf", QSARRegressor("rf", random_state=0)),
   ...     ("ridge", QSARRegressor("ridge")),
   ... ]).fit(X, y)
   >>> ensemble.predict(X).shape
   (24,)

API
---

.. automodule:: qsarkit.models
   :members:
   :show-inheritance:

References
----------

- Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
  :doi:`10.1023/A:1010933404324`
- Wold, S., Sjostrom, M. & Eriksson, L. (2001). "PLS-Regression."
  Chemom. Intell. Lab. Syst., 58(2), 109-130.
  :doi:`10.1016/S0169-7439(01)00155-1`
- Chong, I.-G. & Jun, C.-H. (2005). "Performance of Some Variable
  Selection Methods When Multicollinearity Is Present." Chemom. Intell.
  Lab. Syst., 78(1-2), 103-112. :doi:`10.1016/j.chemolab.2004.12.011`
- Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
  Neural Netw., 18(8), 1093-1110. :doi:`10.1016/j.neunet.2005.07.009`
