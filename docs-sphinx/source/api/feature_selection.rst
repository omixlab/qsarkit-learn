Feature selection
=================

.. currentmodule:: qsarkit.feature_selection

Descriptor selection: variance and correlation filters, mutual
information, recursive feature elimination and Boruta. All are
scikit-learn selectors, so they compose in a pipeline and expose
``get_support()``.

.. warning::

   Select features *inside* the cross-validation loop, not before it.
   Choosing columns using all the labels and then splitting is selection
   bias: the choice has already seen the test set, and the held-out score
   is optimistic by an amount you cannot estimate afterwards.

Filters
-------

The cheapest and most defensible reductions, because they use no labels
at all.

.. doctest::

   >>> from qsarkit.feature_selection import VarianceFilter
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> VarianceFilter().fit_transform(X).shape
   (24, 107)

More than half the bits never fire on 24 compounds. They cannot
contribute and only slow things down.

.. doctest::

   >>> from qsarkit.feature_selection import CorrelationFilter
   >>> selector = CorrelationFilter(threshold=0.9).fit(X, y)
   >>> selector.transform(X).shape
   (24, 61)

Constant columns have undefined correlation, so they are identified and
dropped explicitly rather than falling out of NaN comparisons:

.. doctest::

   >>> int(selector.constant_.sum())
   149

When ``y`` is supplied, the filter keeps whichever member of a correlated
pair correlates better with the target — a better choice than keeping
whichever happened to come first.

Supervised selection
--------------------

.. doctest::

   >>> from qsarkit.feature_selection import MutualInformationSelector, RFESelector
   >>> MutualInformationSelector(k=8).fit_transform(X, y).shape
   (24, 8)
   >>> RFESelector(n_features_to_select=8).fit_transform(X, y).shape
   (24, 8)

Mutual information catches non-linear dependence that a correlation
filter misses, and needs no model. RFE is more powerful and far more
expensive, and its answer is specific to the estimator it wrapped.

Boruta
------

Boruta asks a different question: not "which are the best *k* features"
but "which features carry more signal than random noise". It answers with
a set of whatever size the data supports, rather than a number you chose:

.. doctest::

   >>> from qsarkit.feature_selection import BorutaSelector
   >>> boruta = BorutaSelector(n_iterations=30, random_state=0).fit(X, y)
   >>> int(boruta.get_support().sum()) >= 0
   True

.. note::

   Boruta needs enough iterations for its statistical test to reach
   significance. With too few it confidently selects nothing — the
   selector warns when the iteration count cannot support a decision at
   the chosen ``alpha``, rather than silently returning an empty set.

API
---

.. automodule:: qsarkit.feature_selection
   :members:
   :show-inheritance:

References
----------

- Kursa, M. B. & Rudnicki, W. R. (2010). "Feature Selection with the
  Boruta Package." J. Stat. Softw., 36(11), 1-13.
  :doi:`10.18637/jss.v036.i11`
- Guyon, I. et al. (2002). "Gene Selection for Cancer Classification
  using Support Vector Machines." Mach. Learn., 46, 389-422.
  :doi:`10.1023/A:1012487302797`
- Kraskov, A., Stogbauer, H. & Grassberger, P. (2004). "Estimating Mutual
  Information." Phys. Rev. E, 69, 066138.
  :doi:`10.1103/PhysRevE.69.066138`
- Ambroise, C. & McLachlan, G. J. (2002). "Selection Bias in Gene
  Extraction on the Basis of Microarray Gene-Expression Data." PNAS,
  99(10), 6562-6566. :doi:`10.1073/pnas.102102699`
