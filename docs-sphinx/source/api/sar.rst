SAR interpretation
==================

.. currentmodule:: qsarkit.sar

Matched molecular pairs, activity cliffs, SALI, SARI, Free-Wilson
analysis and R-group decomposition.

This is the part of the workflow that tells you *why* a model performs as
it does, and — more usefully — predicts before you fit anything whether a
regression model can work on this series at all.

Activity cliffs
---------------

An activity cliff is a pair of near-identical structures with very
different activity. Cliffs are where QSAR fails by construction: any
model built on a smooth similarity assumption must predict them wrong.

.. doctest::

   >>> from qsarkit.sar import activity_cliff_report
   >>> report = activity_cliff_report(demo_mols, DEMO_Y, similarity_threshold=0.5)
   >>> report["n_cliffs"], round(report["cliff_ratio"], 4)
   (4, 0.0145)

The report names the substituent changes responsible, which is the part a
chemist can act on:

.. doctest::

   >>> sorted(report["top_transformations"])
   ['[1*]C>>[1*]Cl', '[1*]Cl>>[1*]Br', '[1*]Cl>>[1*]N']

and the specific pairs:

.. doctest::

   >>> cliff = report["top_cliffs"][0]
   >>> cliff.index_a, cliff.index_b, round(cliff.delta, 2)
   (2, 5, 2.9)

.. note::

   The default similarity threshold is 0.85, the figure usually quoted in
   the cliff literature — calibrated for drug-sized molecules with a
   large shared core. The demo set is small molecules, where a
   single-atom change alters every atom environment within the
   fingerprint radius and similarity scores run far below intuition.
   Threshold to your data, not to the paper.

SALI and the activity landscape
-------------------------------

SALI ranks pairs by how steep the cliff is: activity difference over
structural distance.

.. doctest::

   >>> from qsarkit.sar import SALIAnalyzer
   >>> sali = SALIAnalyzer().sali_matrix(demo_mols, DEMO_Y)
   >>> sali.shape
   (24, 24)
   >>> round(float(sali.max()), 2)
   6.38

SARI condenses the whole landscape into one number, separating the
continuous component (smooth SAR, which a model can learn) from the
discontinuous one (cliffs, which it cannot):

.. doctest::

   >>> from qsarkit.sar import SARIAnalyzer
   >>> scores = SARIAnalyzer().analyze(demo_mols, DEMO_Y)
   >>> round(scores["sari"], 3), round(scores["discontinuity"], 3)
   (0.481, 0.065)

Matched molecular pairs
-----------------------

MMPs are the formalization of "change one thing and see what happens",
which is how medicinal chemistry is actually done.

.. doctest::

   >>> from qsarkit.sar import MatchedMolecularPairs
   >>> pairs = MatchedMolecularPairs().find_pairs(demo_mols[:8])
   >>> len(pairs)
   12
   >>> print(pairs[0])
   MatchedPair([1*]C(=O)O>>[1*]NC(C)=O)

The transformation is recorded as a SMIRKS-like rule, so identical
changes across different cores aggregate — which is what turns a list of
pairs into a transferable design rule.

Free-Wilson
-----------

Free-Wilson analysis fits activity as a sum of substituent contributions:
the oldest QSAR method still in use, and still the most interpretable
when the series shares one core.

.. doctest::

   >>> from qsarkit.sar import FreeWilsonAnalysis
   >>> analysis = FreeWilsonAnalysis()
   >>> hasattr(analysis, "fit")
   True

Its assumption — that substituent effects are additive and independent —
is exactly what an activity cliff violates. A high cliff ratio is a
warning that Free-Wilson will mislead here.

API
---

.. automodule:: qsarkit.sar
   :members:
   :show-inheritance:

References
----------

- Maggiora, G. M. (2006). "On Outliers and Activity Cliffs — Why QSAR
  Often Disappoints." J. Chem. Inf. Model., 46(4), 1535.
  :doi:`10.1021/ci060117s`
- Guha, R. & Van Drie, J. H. (2008). "Structure-Activity Landscape
  Index." J. Chem. Inf. Model., 48(3), 646-658.
  :doi:`10.1021/ci7004093`
- Peltason, L. & Bajorath, J. (2007). "SAR Index: Quantifying the Nature
  of Structure-Activity Relationships." J. Med. Chem., 50(23),
  5571-5578. :doi:`10.1021/jm070562u`
- Hussain, J. & Rea, C. (2010). "Computationally Efficient Algorithm to
  Identify Matched Molecular Pairs (MMPs) in Large Data Sets."
  J. Chem. Inf. Model., 50(3), 339-348. :doi:`10.1021/ci900450m`
- Free, S. M. & Wilson, J. W. (1964). "A Mathematical Contribution to
  Structure-Activity Studies." J. Med. Chem., 7(4), 395-399.
  :doi:`10.1021/jm00334a001`
- van Tilborg, D., Alenicheva, A. & Grisoni, F. (2022). "Exposing the
  Limitations of Molecular Machine Learning with Activity Cliffs."
  J. Chem. Inf. Model., 62(23), 5938-5951.
  :doi:`10.1021/acs.jcim.2c01073`
