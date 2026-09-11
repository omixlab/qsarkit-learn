Validation
==========

.. currentmodule:: qsarkit.validation

Cross-validation with QSAR-appropriate reporting.

OECD principle 4 asks for goodness-of-fit, robustness *and* predictivity
— three different things. R² on the training set is goodness-of-fit only,
and is the number most likely to be quoted and least likely to mean
anything.

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.validation import CrossValidator
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> report = CrossValidator(n_splits=3, random_state=0).evaluate(
   ...     QSARRegressor("rf", random_state=0), X, y)
   >>> round(report["q2"], 3)
   0.607
   >>> round(report["rmse_cv"], 3)
   0.673

Compare with the training R² of 0.952 from :doc:`models`. The gap is the
part that would not have survived contact with new compounds.

The report includes the out-of-fold predictions, so you can compute any
further statistic — or plot them — without re-running anything:

.. doctest::

   >>> report["y_pred_cv"].shape
   (24,)
   >>> report["method"], report["n_splits"]
   ('kfold', 3)

Choosing a scheme
-----------------

.. doctest::

   >>> loo = CrossValidator(method="loo").evaluate(QSARRegressor("ridge"), X, y)
   >>> loo["n_splits"]
   24

``n_splits`` is reported from the splitter, not from the constructor
argument — leave-one-out on 24 compounds is 24 splits, and a report that
claimed "5" would be wrong.

``"kfold"``
   The default. Cheap and adequate for a first look.
``"repeated_kfold"``
   Averages over several partitions, which matters on small datasets
   where a single k-fold estimate is dominated by the luck of the split.
``"loo"``
   Maximum training data per fold, but a famously high-variance estimate
   of generalization error, and expensive.
``"leave_group_out"``
   The one to use when compounds come in groups — a scaffold series, an
   assay batch, a source publication. Leaving out a whole group is the
   only way to avoid scoring the model on its own analogues.

.. warning::

   Cross-validating a random split still measures interpolation. For an
   honest figure, cross-validate over scaffold-aware folds — see
   :doc:`model_selection` — or report both and let the gap speak.

API
---

.. automodule:: qsarkit.validation
   :members:
   :show-inheritance:

References
----------

- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models," ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Gramatica, P. (2007). "Principles of QSAR Models Validation: Internal
  and External." QSAR Comb. Sci., 26(5), 694-701.
  :doi:`10.1002/qsar.200610151`
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" J. Mol. Graph.
  Model., 20(4), 269-276. :doi:`10.1016/S1093-3263(01)00123-1`
- Varma, S. & Simon, R. (2006). "Bias in Error Estimation When Using
  Cross-Validation for Model Selection." BMC Bioinformatics, 7, 91.
  :doi:`10.1186/1471-2105-7-91`
