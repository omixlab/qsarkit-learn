Uncertainty
===========

.. currentmodule:: qsarkit.uncertainty

Conformal prediction, ensemble and Gaussian-process uncertainty, and the
calibration diagnostics that tell you whether an error bar means
anything.

An applicability domain answers "should I trust this prediction at all";
uncertainty answers "how wrong is it likely to be". They are different
questions and a serious model reports both.

Conformal prediction
--------------------

Conformal prediction is the only method here with a distribution-free
coverage guarantee: set ``alpha=0.2`` and, given exchangeable data, 80%
of intervals contain the truth — regardless of the underlying model.

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.model_selection import RandomSplitter
   >>> from qsarkit.uncertainty import ConformalRegressor
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> train, test = next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> conformal = ConformalRegressor(
   ...     QSARRegressor("rf", random_state=0), alpha=0.2, random_state=0
   ... ).fit(X[train], y[train])
   >>> lower, upper = conformal.predict_interval(X[test])
   >>> lower.shape, upper.shape
   ((6,), (6,))

``evaluate`` checks the guarantee held:

.. doctest::

   >>> result = conformal.evaluate(X[test], y[test])
   >>> result["coverage"], result["expected_coverage"]
   (1.0, 0.8)
   >>> round(result["mean_width"], 1)
   4.2

Coverage of 1.0 against an expected 0.8 is not a bug — with six test
compounds the empirical coverage can only take seven values, and the
guarantee is marginal, not conditional. The width is the number that
should worry you here: an interval 4.2 log units wide on data spanning
3.5 log units is honest about the model knowing very little, which is the
correct conclusion from 18 training compounds.

The split-conformal construction spends part of the training set on
calibration, which is what ``calibration_size`` controls. Pay it: an
uncalibrated interval is a decoration.

Ensemble and Gaussian-process uncertainty
-----------------------------------------

.. doctest::

   >>> from qsarkit.uncertainty import EnsembleUncertainty
   >>> ensemble = EnsembleUncertainty(
   ...     QSARRegressor("rf", random_state=0), random_state=0
   ... ).fit(X[train], y[train])
   >>> mean, sigma = ensemble.predict_uncertainty(X[test])
   >>> mean.shape, sigma.shape
   ((6,), (6,))

These have no coverage guarantee — the spread of an ensemble is a proxy
for uncertainty, not a measurement of it — which is exactly why the
calibration diagnostics below matter.

Is the error bar meaningful?
----------------------------

.. doctest::

   >>> from qsarkit.uncertainty import UncertaintyCalibration
   >>> calibration = UncertaintyCalibration(n_bins=3)
   >>> report = calibration.report(y[test], mean, sigma)
   >>> round(report["ence"], 1)
   0.3
   >>> round(report["spearman_error_correlation"], 2)
   -0.14

The two numbers answer different questions, and they disagree here. ENCE
(expected normalized calibration error) should be near 0, and 0.3 says the
*scale* of the predicted σ is roughly right. The Spearman correlation
between σ and the actual error should be *positive* — a model ought to be
least certain where it is most wrong — and at −0.14 it is slightly
anti-correlated instead. So the error bars are about the right size on
average while carrying almost no information about which predictions to
distrust, which is the more common failure and the harder one to notice.

Both are invisible if you only look at RMSE:

.. doctest::

   >>> round(report["rmse"], 3)
   0.483

.. note::

   On a :class:`~qsarkit.models.QSARRegressor` wrapping a forest,
   :class:`~qsarkit.uncertainty.EnsembleUncertainty` reuses that forest's own
   trees. Before version 0.8.0 the facade did not expose ``estimators_``, so
   it silently refitted a bagging ensemble *of forests* instead: slower, and
   worse calibrated (ENCE 2.9 rather than 0.3 on this example).

API
---

.. automodule:: qsarkit.uncertainty
   :members:
   :show-inheritance:

References
----------

- Vovk, V., Gammerman, A. & Shafer, G. (2005). "Algorithmic Learning in
  a Random World." Springer. :doi:`10.1007/b106715`
- Papadopoulos, H. et al. (2002). "Inductive Confidence Machines for
  Regression." ECML 2002, 345-356. :doi:`10.1007/3-540-36755-1_29`
- Norinder, U. et al. (2014). "Introducing Conformal Prediction in
  Predictive Modeling." J. Chem. Inf. Model., 54(6), 1596-1603.
  :doi:`10.1021/ci5001168`
- Levi, D. et al. (2022). "Evaluating and Calibrating Uncertainty
  Prediction in Regression Tasks." Sensors, 22(15), 5540.
  :doi:`10.3390/s22155540`
- Hirschfeld, L. et al. (2020). "Uncertainty Quantification Using Neural
  Networks for Molecular Property Prediction." J. Chem. Inf. Model.,
  60(8), 3770-3780. :doi:`10.1021/acs.jcim.0c00502`
