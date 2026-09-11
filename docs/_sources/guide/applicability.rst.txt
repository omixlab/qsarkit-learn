Applicability domain
====================

OECD validation principle 3 requires a QSAR model to declare *the chemical
space in which its predictions are reliable*. A model with no stated domain
is, formally, not a validated QSAR model.

The interface
-------------

Every domain estimator in :mod:`qsarkit.applicability` shares one
interface:

.. code-block:: python

   domain.fit(X_train)             # learn the domain
   domain.predict(X)               # bool array: True = inside
   domain.score_samples(X)         # continuous; larger = further outside
   domain.decision_function(X)     # signed margin; positive = inside
   domain.coverage(X)              # fraction inside

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

.. code-block:: python

   from qsarkit.applicability import ADAnalyzer, KNNApplicabilityDomain

   analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=5)).fit(X_train)

   analyzer.report(X_test, y_test, y_pred)
   # {'coverage': 0.87, 'rmse_inside': 0.42, 'rmse_outside': 1.13,
   #  'rmse_ratio': 2.69, ...}

   analyzer.accuracy_vs_coverage(X_test, y_test, y_pred)   # DataFrame
   analyzer.plot_accuracy_vs_coverage(X_test, y_test, y_pred)
   analyzer.williams_plot(X_test, y_test, y_pred)

``rmse_ratio`` above 1 means out-of-domain predictions really are worse — the
domain is carrying information. A ratio near 1 means it is not, however
statistically principled it looks.

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
