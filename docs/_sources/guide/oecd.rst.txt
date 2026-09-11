OECD validation
===============

The OECD defines five principles a QSAR model must satisfy to be
acceptable for regulatory use. qsarkit is organized around them.

.. list-table::
   :header-rows: 1
   :widths: 6 34 60

   * - #
     - Principle
     - Where it lives
   * - 1
     - A defined endpoint
     - Dataset metadata; :class:`~qsarkit.reporting.QSARReport`
   * - 2
     - An unambiguous algorithm
     - Every estimator's documented hyperparameters and citation;
       :attr:`~qsarkit.functional.MoleculeSet.history`
   * - 3
     - A defined applicability domain
     - :mod:`qsarkit.applicability`
   * - 4
     - Appropriate goodness-of-fit, robustness and predictivity
     - :mod:`qsarkit.metrics`, :mod:`qsarkit.validation`
   * - 5
     - A mechanistic interpretation, if possible
     - :mod:`qsarkit.explainability`, :mod:`qsarkit.sar`

Principle 4 in practice
-----------------------

Goodness of fit alone proves nothing — a sufficiently flexible model fits
noise perfectly. The OECD asks for three separate things.

**Goodness of fit** is R² on the training set. It is the least
informative of the three, and the one most often reported alone:

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.model_selection import RandomSplitter
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> train, test = next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> round(model.score(X[train], y[train]), 3)
   0.921

**Robustness** is cross-validated Q², plus y-scrambling to show the model
has not simply memorized:

.. doctest::

   >>> from qsarkit.validation import CrossValidator
   >>> cv = CrossValidator(n_splits=5, random_state=0).evaluate(
   ...     QSARRegressor("rf", random_state=0), X[train], y[train])
   >>> round(cv["q2"], 3)
   0.225

Training R² of 0.92 against a cross-validated Q² of 0.23. The first
number describes how well the model memorized 18 compounds; the second
is the one that describes prediction.

y-scrambling refits the model on randomly permuted activities. If the
scrambled models score anywhere near the real one, the apparent
performance came from the model's flexibility relative to the dataset
size, not from a structure-activity relationship:

.. doctest::

   >>> from qsarkit.validation import YScrambling
   >>> scramble = YScrambling(n_iterations=50, random_state=0).run(
   ...     QSARRegressor("rf", random_state=0), X[train], y[train])
   >>> round(scramble["real_score"], 3)
   0.921
   >>> round(scramble["mean_scrambled_score"], 3)
   0.837
   >>> round(scramble["max_scrambled_score"], 3)
   0.907
   >>> scramble["p_value"] < 0.05
   True

.. danger::

   Read those numbers again. The model fits **randomly permuted
   activities** to R² = 0.84 on average, and one permutation reached
   0.907 against the real model's 0.921. The p-value technically clears
   0.05, but the honest reading is that almost all of this model's
   apparent fit is capacity, not chemistry: 18 compounds described by 256
   features will fit essentially anything.

   This is precisely the failure y-scrambling exists to expose, and it is
   invisible in the training R² that would otherwise be reported. It is
   also cheap to run, so there is no excuse for omitting it.

The bootstrap gives the same score an error bar, which is what tells you
whether a difference between two models means anything:

.. doctest::

   >>> from qsarkit.validation import BootstrapValidator
   >>> boot = BootstrapValidator(n_iterations=30, random_state=0).run(
   ...     QSARRegressor("rf", random_state=0), X[train], y[train])
   >>> round(boot["mean_score"], 2), round(boot["ci_upper"] - boot["ci_lower"], 2)
   (0.26, 2.87)

An out-of-bag R² of 0.26 with a 95% interval nearly three R² units wide.
Any comparison between two models on this dataset that turns on less than
that is noise, and the interval is the only thing that says so.

**Predictivity** is external validation on compounds the model never saw,
scored with metrics designed for it:

.. doctest::

   >>> from qsarkit.validation import ExternalValidator
   >>> report = ExternalValidator(q2=cv["q2"]).validate(
   ...     model, X[test], y[test], y[train])
   >>> round(report["r2"], 3), round(report["q2_f1"], 3)
   (0.816, 0.818)
   >>> report["golbraikh_tropsha"]["passed"]
   False

The Golbraikh–Tropsha check fails. Reading which criterion failed is the
point of running it:

.. doctest::

   >>> gt = report["golbraikh_tropsha"]
   >>> [k for k in sorted(gt) if k.startswith("criterion") and gt[k] is False]
   ['criterion_1_q2', 'criterion_3_r0', 'criterion_5_delta_r0']

Three of the five fail. Criterion 1 fails because Q² = 0.23 is below the
0.5 threshold; criteria 3 and 5 concern regression through the origin —
the predictions correlate with the truth but are systematically offset.

An R² of 0.82 on the test set looks respectable and would have been
reported as a success. Running the full check is what turns that into an
accurate picture.

Use Q²F1–F3 rather than plain R² on the test set: they are scaled by the
*training* set variance, so they cannot be inflated by choosing a test set
that happens to span a wide activity range.

Principle 3: the applicability domain
-------------------------------------

.. doctest::

   >>> from qsarkit.applicability import ADAnalyzer, TanimotoSimilarityAD
   >>> analyzer = ADAnalyzer(TanimotoSimilarityAD(threshold=0.6)).fit(X[train])
   >>> ad = analyzer.report(X[test], y[test], model.predict(X[test]))
   >>> ad["n_inside"], ad["n_outside"]
   (3, 3)
   >>> round(ad["rmse_ratio"], 2)
   1.15

``rmse_ratio`` above 1 is the evidence principle 3 asks for: predictions
inside the domain really are more accurate than those outside it.

The threshold is a real choice, not a formality. At 0.35 every test
compound falls inside, the ratio is undefined, and the domain has told
you nothing:

.. doctest::

   >>> import numpy as np
   >>> loose = ADAnalyzer(TanimotoSimilarityAD(threshold=0.35)).fit(X[train])
   >>> loose_report = loose.report(X[test], y[test], model.predict(X[test]))
   >>> loose_report["coverage"], bool(np.isnan(loose_report["rmse_ratio"]))
   (1.0, True)

Reporting
---------

:class:`~qsarkit.reporting.OECDReportBuilder` structures all of the above
into a QMRF-style document — and refuses to be quiet about what you
skipped:

.. doctest::

   >>> from qsarkit.reporting import OECDReportBuilder
   >>> builder = OECDReportBuilder(title="Demo QMRF", endpoint="pIC50")
   >>> builder.unaddressed
   [1, 2, 3, 4, 5]
   >>> _ = builder.add_evidence(1, True, {"endpoint": "pIC50"})
   >>> _ = builder.add_evidence(4, True, {"q2": round(cv["q2"], 3),
   ...                                    "y_scrambling_p": scramble["p_value"]})
   >>> builder.unaddressed
   [2, 3, 5]

A submission fails review over a principle nobody noticed was missing, so
an unaddressed principle is recorded as an explicit gap rather than
omitted silently:

.. doctest::

   >>> report_doc = builder.build()
   >>> "not addressed" in report_doc.to_markdown()
   True

Output formats
--------------

The same report renders five ways, and every one carries the tables:

.. doctest::

   >>> report_doc = builder.build()
   >>> print(report_doc.to_text(width=52).splitlines()[1])
   Demo QMRF
   >>> report_doc.to_markdown().startswith("# Demo QMRF")
   True
   >>> report_doc.to_html().startswith("<!DOCTYPE html>")
   True
   >>> sorted(report_doc.to_dict())
   ['author', 'created', 'endpoint', 'sections', 'title']

PDF needs ``reportlab``, and embedding the plots needs ``kaleido``
(``pip install qsarkit[reporting]``):

.. doctest::

   >>> import os, tempfile
   >>> out = os.path.join(tempfile.mkdtemp(), "qmrf.pdf")
   >>> _ = report_doc.to_pdf(out)
   >>> os.path.getsize(out) > 0
   True

Attach figures to a section and they travel with it into HTML, Markdown
and PDF:

.. doctest::

   >>> from qsarkit.reporting import plot_predicted_vs_observed
   >>> y_pred = model.predict(X[test])
   >>> _ = report_doc.add_section(
   ...     "Predicted vs observed",
   ...     figures=[plot_predicted_vs_observed(y[test], y_pred)])
   >>> "[figure:" in report_doc.to_text()
   True

.. note::

   ``to_pdf`` and ``to_markdown`` raise rather than silently dropping
   plots when ``kaleido`` is missing. A report that quietly lost its
   evidence looks complete and is not; pass ``include_figures=False`` if
   you deliberately want the tables alone.

References
----------

- OECD (2007). *Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models.* OECD Series on Testing
  and Assessment No. 69, ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q²!" *J. Mol. Graph.
  Model.*, 20(4), 269-276. :doi:`10.1016/S1093-3263(01)00123-1`
- Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
  Definition of the Q² Parameter for QSAR Validation." *J. Chem. Inf.
  Model.*, 49(7), 1669-1678. :doi:`10.1021/ci900115y`
- Rücker, C., Rücker, G. & Meringer, M. (2007). "y-Randomization and Its
  Variants in QSPR/QSAR." *J. Chem. Inf. Model.*, 47(6), 2345-2357.
  :doi:`10.1021/ci700157b`
- Tropsha, A., Gramatica, P. & Gombar, V. K. (2003). "The Importance of
  Being Earnest." *QSAR Comb. Sci.*, 22(1), 69-77.
  :doi:`10.1002/qsar.200390007`
- Gramatica, P. (2007). *QSAR Comb. Sci.*, 26(5), 694-701.
  :doi:`10.1002/qsar.200610151`
