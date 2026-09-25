Applicability domain
====================

OECD validation principle 3 requires a QSAR model to declare *the chemical
space in which its predictions are reliable*. A model with no stated domain
is, formally, not a validated QSAR model.

The interface
-------------

Every domain estimator in :mod:`qsarkit.applicability` shares one
interface:

.. doctest::

   >>> from qsarkit.applicability import KNNApplicabilityDomain
   >>> from qsarkit.model_selection import ScaffoldSplitter
   >>> X, y = demo_fingerprints(512), DEMO_Y
   >>> train, test = next(ScaffoldSplitter(test_size=0.25).split_mols(demo_mols, y))
   >>> domain = KNNApplicabilityDomain(n_neighbors=3).fit(X[train])
   >>> domain.predict(X[test])                   # bool array: True = inside
   array([ True, False, False, False, False, False])
   >>> domain.score_samples(X[test]).shape       # larger = further outside
   (6,)
   >>> domain.decision_function(X[test]).shape   # positive = inside
   (6,)
   >>> round(domain.coverage(X[test]), 3)        # fraction inside
   0.167

Five of the six held-out compounds fall outside — which is *correct* here,
because the scaffold split held out an entire chemotype. Swap in a random
split and the same domain on the same data says nearly the opposite:

.. doctest::

   >>> from qsarkit.model_selection import RandomSplitter
   >>> rtrain, rtest = next(
   ...     RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> KNNApplicabilityDomain(n_neighbors=3).fit(X[rtrain]).predict(X[rtest])
   array([ True,  True,  True,  True,  True, False])

17% in-domain against 83%. Nothing changed but which compounds were held
out. **If your applicability domain marks nearly everything in-domain,
suspect the split before congratulating the model.**

Choosing a definition
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Use when
   * - :class:`~qsarkit.applicability.LeverageAD`
     - You need the regulatory standard (the Williams plot's ``h*``).
   * - :class:`~qsarkit.applicability.KNNApplicabilityDomain`
     - Default choice. Handles clustered, multi-series datasets.
   * - :class:`~qsarkit.applicability.TanimotoSimilarityAD`
     - Your model is built on fingerprints. Speaks chemists' language.
   * - :class:`~qsarkit.applicability.RangeAD`
     - You want something transparent and conservative for a regulator.
   * - :class:`~qsarkit.applicability.ConvexHullAD`
     - You need the tightest possible interpolation region.
   * - :class:`~qsarkit.applicability.IsolationForestAD`
     - High-dimensional, non-convex, large training sets.
   * - :class:`~qsarkit.applicability.EnsembleAD`
     - You would rather not bet on any single definition.

Leverage and the bounding box assume the training set is one convex cloud —
which real QSAR datasets, built from several chemical series, usually are
not. That is why the k-NN and similarity definitions tend to win in
comparative studies.

Does the domain actually help?
------------------------------

A domain is only useful if excluding what it rejects improves accuracy on
what it keeps. :class:`~qsarkit.applicability.ADAnalyzer` measures exactly
that:

.. doctest::

   >>> from qsarkit.applicability import ADAnalyzer
   >>> from qsarkit.models import QSARRegressor
   >>> model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> y_pred = model.predict(X[test])
   >>> analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X[train])
   >>> report = analyzer.report(X[test], y[test], y_pred)
   >>> sorted(report)
   ['coverage', 'mae_inside', 'mae_outside', 'n_inside', 'n_outside', 'rmse_inside', 'rmse_outside', 'rmse_ratio']
   >>> report["n_inside"], report["n_outside"]
   (1, 5)
   >>> round(report["rmse_ratio"], 2)
   1.13

``rmse_ratio`` above 1 means out-of-domain predictions really are worse —
the domain is carrying information. A ratio near 1 means it is not, however
statistically principled it looks. 1.13 on one in-domain compound is weak
evidence either way, which is itself worth knowing: this dataset is too
small for the domain to be characterised confidently.

On the random split the same measurement is equally uninformative, for the
opposite reason — almost nothing is outside to compare against:

.. doctest::

   >>> loose = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X[rtrain])
   >>> rmodel = QSARRegressor("rf", random_state=0).fit(X[rtrain], y[rtrain])
   >>> loose_report = loose.report(X[rtest], y[rtest], rmodel.predict(X[rtest]))
   >>> loose_report["n_inside"], loose_report["n_outside"]
   (5, 1)
   >>> round(loose_report["rmse_ratio"], 1)
   1.1

.. warning::

   ``rmse_ratio`` is undefined when *every* test compound lands on one
   side. A domain with 100% coverage has told you nothing, and the report
   returns ``nan`` rather than a misleading number:

   .. doctest::

      >>> import numpy as np
      >>> from qsarkit.applicability import TanimotoSimilarityAD
      >>> permissive = ADAnalyzer(TanimotoSimilarityAD(threshold=0.05)).fit(X[rtrain])
      >>> wide = permissive.report(X[rtest], y[rtest], rmodel.predict(X[rtest]))
      >>> wide["coverage"], bool(np.isnan(wide["rmse_ratio"]))
      (1.0, True)

The accuracy-versus-coverage curve is the more informative view, because it
shows the whole trade-off rather than one threshold's worth of it:

.. doctest::

   >>> curve = analyzer.accuracy_vs_coverage(X[test], y[test], y_pred)
   >>> list(curve.columns)
   ['coverage', 'n_samples', 'rmse', 'mae']
   >>> type(analyzer.plot_accuracy_vs_coverage(X[test], y[test], y_pred)).__name__
   'Figure'
   >>> type(analyzer.williams_plot(X[test], y[test], y_pred)).__name__
   'Figure'

Choosing the threshold
----------------------

The threshold is a real decision, not a formality — it sets how much of
your test set you are declining to predict. On fingerprints,
:class:`~qsarkit.applicability.TanimotoSimilarityAD` is the definition that
matches how a chemist judges novelty:

.. doctest::

   >>> for threshold in (0.2, 0.3, 0.45):
   ...     inside = int(TanimotoSimilarityAD(threshold=threshold).fit(
   ...         X[train]).predict(X[test]).sum())
   ...     print(f"threshold {threshold}: {inside}/6 in domain")
   threshold 0.2: 6/6 in domain
   threshold 0.3: 3/6 in domain
   threshold 0.45: 0/6 in domain

The whole range from "predict everything" to "predict nothing" spans 0.25
Tanimoto units here, so the choice is consequential and the default will
not suit every dataset. Pick it from the accuracy-versus-coverage curve on
validation data, not by whichever value makes the coverage number look
best.

.. note::

   Those thresholds look low for a similarity cutoff. Morgan fingerprints
   of small molecules score far below intuition, because one substituent
   alters every atom environment within the fingerprint radius. A
   threshold calibrated on drug-sized molecules will reject an entire
   fragment dataset.

References
----------

- OECD (2007). *Guidance Document on the Validation of (Q)SAR Models.* OECD
  Series on Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Sahigara, F. et al. (2012). "Comparison of Different Approaches to Define
  the Applicability Domain of QSAR Models." *Molecules*, 17(5), 4791-4810.
  :doi:`10.3390/molecules17054791`
- Gramatica, P. (2007). "Principles of QSAR Models Validation: Internal and
  External." *QSAR Comb. Sci.*, 26(5), 694-701. :doi:`10.1002/qsar.200610151`
- Sheridan, R. P. et al. (2004). "Similarity to Molecules in the Training
  Set Is a Good Discriminator for Prediction Accuracy in QSAR." *J. Chem.
  Inf. Comput. Sci.*, 44(6), 1912-1928. :doi:`10.1021/ci049782w`
