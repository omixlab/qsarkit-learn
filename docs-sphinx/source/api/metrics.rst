Metrics
=======

.. currentmodule:: qsarkit.metrics

QSAR-specific performance measures. R² alone does not establish that a
model predicts — these are the statistics the QSAR literature and the
OECD guidance actually ask for.

Regression
----------

.. doctest::

   >>> from qsarkit.metrics import ccc, q2_f1, qsar_regression_report, rmse
   >>> from qsarkit.models import QSARRegressor
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> y_pred = QSARRegressor("rf", random_state=0).fit(X, y).predict(X)
   >>> round(rmse(y, y_pred), 2)
   0.24
   >>> round(ccc(y, y_pred), 3)
   0.972

Q²F1–F3 differ in what they compare the model against. The choice
matters: an external Q² computed against the *training* mean flatters a
model whose test set happens to be centred differently.

.. doctest::

   >>> round(q2_f1(y, y_pred, y), 3)
   0.952

The Golbraikh–Tropsha criteria
------------------------------

Five conditions a predictive QSAR model must satisfy simultaneously.
They exist because a high R² is achievable by a model with a systematic
slope error, which the regression-through-origin terms catch.

.. doctest::

   >>> from qsarkit.metrics import golbraikh_tropsha_criteria
   >>> result = golbraikh_tropsha_criteria(y, y_pred)
   >>> result["passed"]
   True
   >>> [k for k in sorted(result) if k.startswith("criterion")]
   ['criterion_1_q2', 'criterion_2_r2', 'criterion_3_r0', 'criterion_4_slope', 'criterion_5_delta_r0']

Criterion 1 needs a cross-validated Q², which this call was not given, so
it reports ``None`` rather than quietly passing:

.. doctest::

   >>> result["criterion_1_q2"] is None, result["q2_available"]
   (True, False)

The full report
---------------

.. doctest::

   >>> report = qsar_regression_report(y, y_pred)
   >>> sorted(report)
   ['average_r2m', 'ccc', 'delta_r2m', 'golbraikh_tropsha', 'mae', 'q2_f2', 'r2', 'rmse']

Classification and virtual screening
------------------------------------

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.metrics import bedroc, enrichment_factor, roc_auc
   >>> scores = np.linspace(1.0, 0.0, 100)
   >>> labels = np.zeros(100); labels[:10] = 1        # actives ranked first
   >>> round(roc_auc(labels, scores), 3)
   1.0
   >>> round(enrichment_factor(labels, scores, fraction=0.1), 2)
   10.0

ROC AUC weights every rank equally, which is the wrong emphasis for
screening: only the top of the list will ever be tested. BEDROC applies
an exponential weight so early recognition dominates.

.. doctest::

   >>> round(bedroc(labels, scores, alpha=20.0), 3)
   1.0

API
---

.. automodule:: qsarkit.metrics
   :members:
   :show-inheritance:

References
----------

- Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" J. Mol. Graph.
  Model., 20(4), 269-276. :doi:`10.1016/S1093-3263(01)00123-1`
- Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
  Definition of the Q2 Parameter for QSAR Validation." J. Chem. Inf.
  Model., 49(7), 1669-1678. :doi:`10.1021/ci900115y`
- Roy, K. et al. (2012). "Comparative Studies on Some Metrics for
  External Validation of QSPR Models." J. Chem. Inf. Model., 52(2),
  396-408. :doi:`10.1021/ci200520g`
- Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
  Methods: Good and Bad Metrics for the Early Recognition Problem."
  J. Chem. Inf. Model., 47(2), 488-508. :doi:`10.1021/ci600426e`
- Lin, L. I. (1989). "A Concordance Correlation Coefficient to Evaluate
  Reproducibility." Biometrics, 45(1), 255-268.
  :doi:`10.2307/2532051`
