Data quality
============

.. currentmodule:: qsarkit.data_quality

Dataset curation following Fourches, Muratov and Tropsha: duplicate
detection, activity outliers, structure validation, and a pipeline that
reports what it removed and why.

A QSAR model is bounded by the quality of its activity data. Curation is
not tidying — it is the difference between a model of the chemistry and a
model of the database's accumulated errors.

Duplicates
----------

Duplicates are detected by structural identity, not by string equality:
the same compound registered as a salt, a tautomer or with different
stereo annotation is one record.

.. doctest::

   >>> from qsarkit.data_quality import DuplicateDetector
   >>> groups = DuplicateDetector().find_duplicates(
   ...     [demo_mols[0], demo_mols[0], demo_mols[1]])
   >>> len(groups), groups[0].indices
   (1, [0, 1])

With activities attached, a group also reports whether its measurements
agree. A duplicate pair two log units apart is not a duplicate to be
merged — it is a data problem to be investigated:

.. doctest::

   >>> groups = DuplicateDetector(activity_tolerance=0.5).find_duplicates(
   ...     [demo_mols[0], demo_mols[0]], [5.1, 8.4])
   >>> round(groups[0].spread, 2), groups[0].consistent
   (3.3, False)

Outliers
--------

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.data_quality import ActivityOutlierDetector
   >>> ActivityOutlierDetector().detect(np.array([5.0, 5.1, 5.2, 9.9])).tolist()
   [False, False, False, True]

The default is the modified z-score, which uses the median and MAD rather
than the mean and standard deviation. That matters here: a single extreme
value inflates the standard deviation enough to mask itself, so a plain
z-score is least reliable exactly when you need it.

Structure validation
--------------------

.. doctest::

   >>> from rdkit import Chem
   >>> from qsarkit.data_quality import StructureValidator
   >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "[Na+].[Cl-]", "O")]
   >>> sorted({issue.code for issue in StructureValidator().validate(mols)})
   ['inorganic', 'mixture', 'no_carbon', 'too_small']

The ``charged`` check looks at *net* charge, so a zwitterion is not
mistaken for a record that escaped neutralization:

.. doctest::

   >>> validator = StructureValidator()
   >>> [i.code for i in validator.validate([Chem.MolFromSmiles("[NH3+]CC(=O)[O-]")])]
   []
   >>> [i.code for i in validator.validate([Chem.MolFromSmiles("CC(=O)[O-]")])]
   ['charged']

The whole pipeline
------------------

.. doctest::

   >>> from qsarkit.data_quality import DataCurationPipeline
   >>> mols_in = demo_mols[:6] + [demo_mols[0]]      # one deliberate duplicate
   >>> activities = list(DEMO_Y[:6]) + [5.1]
   >>> mols, y, report = DataCurationPipeline().run(mols_in, activities)
   >>> len(mols), len(y)
   (6, 6)
   >>> print(report.summary())
   Curation: 7 -> 6 records (85.7% retained)
     standardize                 7 -> 7      (0 removed)
     validate                    7 -> 7      (0 removed)
     deduplicate                 7 -> 6      (1 removed)

The report is the audit trail OECD Principle 2 asks for — every removal,
attributed to the stage that made it:

.. doctest::

   >>> report.n_input, report.n_output, round(report.retention, 3)
   (7, 6, 0.857)
   >>> frame = report.to_dataframe()
   >>> len(frame) >= 1
   True

Unit checking
-------------

.. doctest::

   >>> from qsarkit.data_quality import check_activity_units
   >>> report = check_activity_units([5.1, 6.2, 7.3], endpoint="IC50")
   >>> report["looks_logarithmic"], report["warnings"]
   (True, [])

A raw nanomolar column spanning six decades is caught:

.. doctest::

   >>> report = check_activity_units([1.0, 10.0, 1000.0, 1e6], unit="nM")
   >>> report["looks_logarithmic"], round(report["log_range"], 1)
   (False, 6.0)
   >>> print(report["warnings"][0])
   Values span 6.0 orders of magnitude, which suggests a raw concentration scale. Convert to pActivity (-log10 molar) before modeling.

So is the subtler case: molar values spanning little numeric range. They
look narrow, but a pActivity of 1e-9 would mean an IC50 near 1 M:

.. doctest::

   >>> report = check_activity_units([1e-9, 5e-8], endpoint="IC50")
   >>> report["looks_logarithmic"]
   False

Mixing molar and p-scale values in one column produces a model that
learns nothing, and nothing in the numbers themselves announces the
mistake — which is why this check exists.

API
---

.. automodule:: qsarkit.data_quality
   :members:
   :show-inheritance:

References
----------

- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and
  QSAR Modeling Research." J. Chem. Inf. Model., 50(7), 1189-1204.
  :doi:`10.1021/ci100176x`
- Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify II."
  J. Chem. Inf. Model., 56(7), 1243-1252.
  :doi:`10.1021/acs.jcim.6b00129`
- Kalliokoski, T. et al. (2013). "Comparability of Mixed IC50 Data."
  PLoS ONE, 8(4), e61007. :doi:`10.1371/journal.pone.0061007`
