Applicability domain
====================

.. currentmodule:: qsarkit.applicability

OECD validation principle 3: the region of chemical space in which a
model's predictions can be trusted. Eleven domain definitions sharing one
interface, plus coverage and accuracy-versus-coverage analysis.

A prediction outside the domain is not *wrong* — it is unsupported by the
training data, which is a different claim and the one regulators ask
about.

One interface, eleven definitions
---------------------------------

Every domain implements ``fit(X)`` and ``predict(X) -> bool array``:

.. doctest::

   >>> from qsarkit.applicability import KNNApplicabilityDomain
   >>> from qsarkit.model_selection import RandomSplitter
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> train, test = next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> domain = KNNApplicabilityDomain(n_neighbors=3).fit(X[train])
   >>> domain.predict(X[test]).tolist()
   [True, True, True, True, True, True]

What the domain is actually measuring
-------------------------------------

Swap the random split for a scaffold split and the same domain, the same
model and the same data give the opposite answer:

.. doctest::

   >>> from qsarkit.model_selection import ScaffoldSplitter
   >>> train, test = next(ScaffoldSplitter(test_size=0.25).split_mols(demo_mols, y))
   >>> KNNApplicabilityDomain(n_neighbors=3).fit(X[train]).predict(X[test]).tolist()
   [False, False, False, False, False, False]

Nothing changed except which compounds are held out. A random split
leaves every test compound with a close analogue in training, so
everything is in-domain and the AD looks vacuous. A scaffold split holds
out whole chemotypes, and the domain correctly reports that it has never
seen anything like them.

That is the entire point: an applicability domain is only informative
when the evaluation is honest about novelty. If your AD marks everything
in-domain, suspect the split before congratulating the model.

Choosing a definition
---------------------

.. doctest::

   >>> from qsarkit.applicability import (
   ...     LeverageAD, RangeAD, TanimotoSimilarityAD)
   >>> train, test = next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> int(TanimotoSimilarityAD(threshold=0.3).fit(X[train]).predict(X[test]).sum())
   6
   >>> bool(LeverageAD().fit(X[train]).predict(X[test]).all())
   True

The definitions are not interchangeable:

``LeverageAD``
   The classical Williams-plot leverage, ``h*``. Assumes a linear model
   and continuous descriptors; on a 2048-bit fingerprint it is close to
   meaningless, because the hat matrix is degenerate.
``TanimotoSimilarityAD``
   Similarity to the nearest training compound. The right choice for
   fingerprints, and the one that matches how a chemist would judge
   novelty.
``KNNApplicabilityDomain``
   Mean distance to the *k* nearest neighbours. Less sensitive to a
   single close analogue than the Tanimoto rule.
``RangeAD`` / ``BoundingBoxAD`` / ``PCABoundingBoxAD``
   Descriptor-range checks. Cheap and interpretable, but a bounding box
   admits the empty interior of a hollow distribution.
``ConvexHullAD``
   Exact but exponential in dimension; usable only after aggressive
   dimensionality reduction.
``KernelDensityAD`` / ``IsolationForestAD``
   Density-based, making no shape assumption.
``EnsembleAD``
   Combines several, requiring agreement.

Does the domain earn its keep?
------------------------------

A domain is only useful if predictions inside it are actually better than
predictions outside it. :class:`ADAnalyzer` measures that directly:

.. doctest::

   >>> from qsarkit.applicability import ADAnalyzer
   >>> from qsarkit.models import QSARRegressor
   >>> model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X[train])
   >>> report = analyzer.report(X[test], y[test], model.predict(X[test]))
   >>> report["coverage"], report["n_inside"], report["n_outside"]
   (1.0, 6, 0)

``rmse_ratio`` is the number to read: greater than 1 means errors outside
the domain really are larger, so the domain is separating reliable
predictions from unreliable ones. Here every test compound is inside, so
there is nothing to compare against and the ratio is undefined:

.. doctest::

   >>> import numpy as np
   >>> bool(np.isnan(report["rmse_ratio"]))
   True

A domain with 100% coverage is not a good result — it is a domain that
has told you nothing.

API
---

.. automodule:: qsarkit.applicability
   :members:
   :show-inheritance:

References
----------

- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models," ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Sahigara, F. et al. (2012). "Comparison of Different Approaches to
  Define the Applicability Domain of QSAR Models." Molecules, 17(5),
  4791-4810. :doi:`10.3390/molecules17054791`
- Netzeva, T. I. et al. (2005). "Current Status of Methods for Defining
  the Applicability Domain of (Quantitative) Structure-Activity
  Relationships." ATLA, 33(2), 155-173.
  :doi:`10.1177/026119290503300209`
- Sheridan, R. P. et al. (2004). "Similarity to Molecules in the Training
  Set Is a Good Discriminator for Prediction Accuracy in QSAR." J. Chem.
  Inf. Comput. Sci., 44(6), 1912-1928. :doi:`10.1021/ci049782w`
