Validation
==========

.. currentmodule:: qsarkit.validation

Model validation against OECD principle 4, which asks for three distinct
things — goodness-of-fit, robustness and predictivity. A single :math:`R^2`
addresses only the first, and it is the number most likely to be quoted.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Answers
   * - :class:`CrossValidator`
     - *Does it predict?* Cross-validated :math:`Q^2`, with out-of-fold
       predictions.
   * - :class:`YScrambling`
     - *Is the fit real?* Could the model score this well on permuted
       labels?
   * - :class:`BootstrapValidator`
     - *How precise is the score?* An out-of-bag interval rather than a
       point estimate.
   * - :class:`ExternalValidator`
     - *Does it predict on compounds it never saw?* The full QSAR metric
       set plus the Golbraikh-Tropsha criteria.

Cross-validation
----------------

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.validation import CrossValidator
   >>> X, y = demo_fingerprints(512), DEMO_Y
   >>> model = QSARRegressor("rf", random_state=0)
   >>> report = CrossValidator(n_splits=5, random_state=0).evaluate(model, X, y)
   >>> round(report["q2"], 1), round(report["rmse_cv"], 1)
   (0.7, 0.6)

The report carries the out-of-fold predictions, so any further statistic or
plot needs no refitting:

.. doctest::

   >>> report["y_pred_cv"].shape
   (24,)
   >>> report["method"], report["n_splits"]
   ('kfold', 5)

``n_splits`` comes from the splitter, not the constructor argument —
leave-one-out on 24 compounds is 24 splits, and a report claiming 5 would
be wrong:

.. doctest::

   >>> CrossValidator(method="loo").evaluate(QSARRegressor("ridge"), X, y)["n_splits"]
   24

``"repeated_kfold"`` averages over several partitions, which matters on
small datasets where one k-fold estimate is dominated by the luck of the
split. ``"leave_group_out"`` is the one to use when compounds come in
groups — a scaffold series, an assay batch, a source publication.

Robustness: y-scrambling
------------------------

Refit the model on randomly permuted activities. If the scrambled models
score anywhere near the real one, the apparent performance came from the
model's flexibility relative to the dataset size, not from a
structure-activity relationship.

.. doctest::

   >>> from qsarkit.validation import YScrambling
   >>> scramble = YScrambling(n_iterations=50, random_state=0).run(model, X, y)
   >>> round(scramble["real_score"], 3)
   0.953
   >>> round(scramble["mean_scrambled_score"], 3)
   0.839
   >>> scramble["p_value"] < 0.05
   True

.. danger::

   Read those numbers again. The model fits **randomly permuted
   activities** to :math:`R^2 = 0.84` on average, against 0.95 on the real
   ones. The p-value clears 0.05, but the honest reading is that most of
   this model's apparent fit is capacity: 24 compounds described by 512
   features will fit almost anything.

   This is exactly the failure y-scrambling exists to expose, and it is
   invisible in the training :math:`R^2` that would otherwise be reported.

The p-value can never be exactly zero — a permutation test cannot
distinguish "very unlikely" from "impossible", so reporting 0 would claim
more than was measured:

.. doctest::

   >>> small = YScrambling(n_iterations=20, random_state=0).run(model, X, y)
   >>> round(small["p_value"], 4) == round(1 / 21, 4)
   True

:meth:`YScrambling.plot` shows the scrambled distribution with the real
score marked, which is more informative than either number alone:

.. doctest::

   >>> scrambler = YScrambling(n_iterations=20, random_state=0)
   >>> _ = scrambler.run(model, X, y)
   >>> type(scrambler.plot()).__name__
   'Figure'

Precision: the bootstrap
------------------------

A single cross-validated :math:`Q^2` is one number with no error bar.
Resampling the training set and scoring out-of-bag gives the spread:

.. doctest::

   >>> from qsarkit.validation import BootstrapValidator
   >>> boot = BootstrapValidator(n_iterations=30, random_state=0).run(model, X, y)
   >>> round(boot["mean_score"], 2)
   0.34
   >>> round(boot["ci_upper"] - boot["ci_lower"], 1)
   1.9

An interval nearly two :math:`R^2` units wide. **Any comparison between two
models on this dataset that turns on less than that is noise** — and the
interval is the only thing that says so.

Scoring is out-of-bag, not in-bag: about 36.8% of the data is left out of
each resample, and scoring there rather than on the fitted rows is what
makes this an estimate of generalization instead of of fit.

Predictivity: external validation
---------------------------------

.. doctest::

   >>> from qsarkit.model_selection import RandomSplitter
   >>> from qsarkit.validation import ExternalValidator
   >>> train, test = next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> cv = CrossValidator(n_splits=5, random_state=0).evaluate(model, X[train], y[train])
   >>> fitted = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> result = ExternalValidator(q2=cv["q2"]).validate(
   ...     fitted, X[test], y[test], y[train])
   >>> round(result["r2"], 2), round(result["q2_f1"], 2)
   (0.82, 0.82)

Supply ``y_train`` so Q²F1 is scaled by the *training* set variance, which
is what makes it comparable across differently-centred test sets. Supply
``q2`` so Golbraikh-Tropsha criterion 1 can be evaluated rather than
reporting ``None``:

.. doctest::

   >>> result["golbraikh_tropsha"]["passed"]
   False
   >>> gt = result["golbraikh_tropsha"]
   >>> [k for k in sorted(gt) if k.startswith("criterion") and gt[k] is False]
   ['criterion_1_q2', 'criterion_3_r0']

An :math:`R^2` of 0.82 on the test set looks respectable and would have
been reported as a success. Criterion 1 fails because the cross-validated
:math:`Q^2` of 0.24 is below the 0.5 threshold; criterion 3 concerns
regression through the origin — the predictions correlate with the truth
but are systematically offset. Running the full check is what turns a
respectable-looking number into an accurate picture.

.. note::

   None of these validators mutates the estimator you hand them: each
   clones it before fitting, so the same configured model can be passed to
   all four.

   .. doctest::

      >>> template = QSARRegressor("ridge")
      >>> _ = YScrambling(n_iterations=5, random_state=0).run(template, X, y)
      >>> hasattr(template, "estimator_")
      False

Choosing the metric
-------------------

All four validators take ``scoring``. It defaults to :math:`R^2`, which is
right for a regression QSAR and wrong for everything else: a toxicity
classifier has to be argued in ROC-AUC or average precision, and a regulator
asking for RMSE is not asking for :math:`R^2` reported next to it.

.. doctest::

   >>> from qsarkit.validation import available_metrics
   >>> len(available_metrics())
   18
   >>> [m for m in available_metrics() if "auc" in m]
   ['pr_auc', 'roc_auc']

Pass several and every score becomes an array **in the order given**, so one
pass reports them all:

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.validation import CrossValidator
   >>> cv = CrossValidator(n_splits=5, random_state=0, scoring=["r2", "rmse", "mae"])
   >>> result = cv.evaluate(QSARRegressor("rf", random_state=0), X, y)
   >>> result["metric"]
   ('r2', 'rmse', 'mae')
   >>> result["score"].round(2)
   array([0.69, 0.6 , 0.44])

One metric returns a float; an iterable returns an array even when it holds a
single entry, so adding a second metric never changes the shape of your code.

Classification metrics see probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``roc_auc``, ``pr_auc`` and ``brier`` rank or calibrate, so they are given
``predict_proba``'s positive-class column, never a thresholded label.
Thresholding first throws away the ranking that ROC-AUC exists to measure.

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.models import QSARClassifier
   >>> labels = (DEMO_Y > np.median(DEMO_Y)).astype(int)
   >>> model = QSARClassifier("rf", random_state=0)
   >>> cv = CrossValidator(n_splits=5, random_state=0,
   ...                     scoring=["roc_auc", "pr_auc", "mcc"])
   >>> report = cv.evaluate(model, X, labels)
   >>> report["score"].round(2)
   array([0.93, 0.94, 0.75])
   >>> report["y_score_cv"].shape          # out-of-fold probabilities
   (24,)

For a metric that is not on the list, or one that needs probabilities:

.. doctest::

   >>> from sklearn.metrics import average_precision_score
   >>> from qsarkit.validation import make_scorer
   >>> scorer = make_scorer(average_precision_score, needs_proba=True, name="ap")
   >>> CrossValidator(n_splits=5, random_state=0, scoring=scorer).evaluate(
   ...     model, X, labels)["metric"]
   'ap'

A bare callable is assumed to take ``(y_true, y_pred)`` and to improve as it
grows; :func:`~qsarkit.validation.make_scorer` is how the other cases are
declared. Declaring a loss matters more than it looks:
``greater_is_better=False`` is what keeps "did the scrambled model do at least
as well" comparing in the right direction, so a y-randomization p-value
computed on RMSE is not reported backwards.

.. doctest::

   >>> from qsarkit.validation import YScrambling
   >>> scramble = YScrambling(n_iterations=20, random_state=0,
   ...                        scoring=["r2", "rmse"]).run(
   ...     QSARRegressor("rf", random_state=0), X, y)
   >>> scramble["p_value"].round(3)        # same verdict from a gain and a loss
   array([0.048, 0.048])

.. warning::

   **Score out of fold before reading anything into a ranking metric.**
   :class:`~qsarkit.validation.YScrambling` scores the apparent, in-sample fit
   by default, which is what earlier releases did and what makes the classic
   over-fitting demonstration work for :math:`R^2`. It cannot work for
   ROC-AUC: a random forest separates *permuted* labels in-sample as perfectly
   as real ones, so both sides read near 1.0 and the test reports nothing.

   .. doctest::

      >>> in_sample = YScrambling(n_iterations=20, random_state=0,
      ...                         scoring="roc_auc").run(model, X, labels)
      >>> round(in_sample["real_score"], 2), round(in_sample["mean_scrambled_score"], 2)
      (1.0, 1.0)

   Pass ``cv`` — and ``stratify=True`` on an imbalanced endpoint — and the
   same test becomes informative:

   .. doctest::

      >>> honest = YScrambling(n_iterations=20, random_state=0, scoring="roc_auc",
      ...                      cv=5, stratify=True).run(model, X, labels)
      >>> round(honest["real_score"], 2), round(honest["mean_scrambled_score"], 2)
      (0.93, 0.43)
      >>> honest["scored_out_of_fold"]
      True

See :doc:`../guide/oecd` for how these fit together into a reportable
validation, and :doc:`metrics` for the statistics they compute.

API
---

.. automodule:: qsarkit.validation
   :members:
   :show-inheritance:

References
----------

- OECD (2007). *Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models*, ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" *J. Mol. Graph.
  Model.*, 20(4), 269-276. :doi:`10.1016/S1093-3263(01)00123-1`
- Rücker, C., Rücker, G. & Meringer, M. (2007). "y-Randomization and Its
  Variants in QSPR/QSAR." *J. Chem. Inf. Model.*, 47(6), 2345-2357.
  :doi:`10.1021/ci700157b`
- Tropsha, A., Gramatica, P. & Gombar, V. K. (2003). "The Importance of
  Being Earnest." *QSAR Comb. Sci.*, 22(1), 69-77.
  :doi:`10.1002/qsar.200390007`
- Efron, B. & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap.*
  Chapman & Hall. :doi:`10.1201/9780429246593`
- Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
  Definition of the Q2 Parameter for QSAR Validation." *J. Chem. Inf.
  Model.*, 49(7), 1669-1678. :doi:`10.1021/ci900115y`
- Varma, S. & Simon, R. (2006). "Bias in Error Estimation When Using
  Cross-Validation for Model Selection." *BMC Bioinformatics*, 7, 91.
  :doi:`10.1186/1471-2105-7-91`
