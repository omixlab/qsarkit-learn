SAR interpretation
==================

Activity cliffs
---------------

An activity cliff is a pair of structurally similar molecules with very
different potency. Cliffs matter because they break the similarity-property
principle that every regression model relies on: a model that cannot
reproduce them will systematically mispredict the most interesting
compounds in your series.

Run this **before** you tune a model, not after:

.. doctest::

   >>> from qsarkit.sar import activity_cliff_report
   >>> report = activity_cliff_report(demo_mols, DEMO_Y, similarity_threshold=0.5)
   >>> report["n_cliffs"], report["n_pairs"]
   (4, 276)
   >>> round(report["cliff_ratio"], 4)              # cliffs / all pairs
   0.0145
   >>> round(report["cliff_compound_fraction"], 4)  # compounds in >= 1 cliff
   0.2083

The two numbers say different things. 1.5% of *pairs* are cliffs, but they
involve 21% of the *compounds* — so a fifth of this series sits on a
discontinuity a regression model cannot fit.

What a chemist can act on is which change causes them:

.. doctest::

   >>> sorted(report["top_transformations"])
   ['[1*]C>>[1*]Cl', '[1*]Cl>>[1*]Br', '[1*]Cl>>[1*]N']
   >>> report["top_scaffolds"]
   {'c1ccccc1': 8}

Every cliff here is on the benzene scaffold, and every one involves the
4-chloro compound. On a real dataset that is the signal to go back to the
assay records for that compound before fitting anything.

A high cliff ratio predicts that a regression model will underperform on
this series no matter how you tune it — and tells you which substituent
changes are responsible.

.. doctest::

   >>> from qsarkit.sar import ActivityCliffDetector
   >>> detector = ActivityCliffDetector(
   ...     similarity_threshold=0.5,    # ECFP4 Tanimoto
   ...     activity_threshold=2.0,      # 100-fold potency change
   ...     method="fingerprint",        # or "scaffold", or "mmp"
   ... )
   >>> cliffs = detector.detect(demo_mols, DEMO_Y)   # descending SALI
   >>> len(cliffs)
   4
   >>> first = cliffs[0]
   >>> first.index_a, first.index_b, round(first.delta, 2), round(first.sali, 2)
   (2, 5, 2.9, 6.38)

.. note::

   The default ``similarity_threshold`` is 0.85, the figure usually quoted
   in the cliff literature — and calibrated for drug-sized molecules with a
   large shared core. These are small molecules, where a single-atom change
   alters every atom environment within the fingerprint radius, so
   similarity runs far below intuition and 0.85 finds nothing. **Threshold
   to your data, not to the paper.**

SALI
----

The Structure-Activity Landscape Index quantifies how sharply activity
changes with structure:

.. math::

   \mathrm{SALI}(i, j) = \frac{|A_i - A_j|}{1 - \mathrm{sim}(i, j)}

Its real value is scoring *models*, not just datasets. The SALI curve ranks
compound pairs by true SALI and asks how many of them the model orders
correctly — a landscape-aware quality measure that RMSE and R² completely
miss:

.. doctest::

   >>> from qsarkit.sar import SALIAnalyzer
   >>> analyzer = SALIAnalyzer()
   >>> analyzer.sali_matrix(demo_mols, DEMO_Y).shape
   (24, 24)
   >>> network = analyzer.sali_network(demo_mols, DEMO_Y, percentile=95)
   >>> network.number_of_nodes(), network.number_of_edges()
   (24, 14)

``sali_auc`` scores how well a model reproduces the *landscape* rather than
the individual values — 1.0 means it ranks every cliff pair correctly, 0.5
is chance:

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> X = demo_fingerprints(512)
   >>> y_pred = QSARRegressor("rf", random_state=0).fit(X, DEMO_Y).predict(X)
   >>> round(analyzer.sali_auc(demo_mols, DEMO_Y, y_pred), 3)
   0.996

That 0.996 is on the *training* predictions, so it measures memorization
rather than skill. Compute it on held-out or out-of-fold predictions for it
to mean anything.

:class:`~qsarkit.sar.SARIAnalyzer` condenses the whole landscape into one
number, separating the part a model can learn from the part it cannot:

.. doctest::

   >>> from qsarkit.sar import SARIAnalyzer
   >>> scores = SARIAnalyzer().analyze(demo_mols, DEMO_Y)
   >>> round(scores["sari"], 3), round(scores["discontinuity"], 3)
   (0.481, 0.065)

Matched molecular pairs
-----------------------

.. doctest::

   >>> from qsarkit.sar import MMPAnalyzer
   >>> mmp = MMPAnalyzer()
   >>> pairs = mmp.find_pairs(demo_mols, DEMO_Y)
   >>> len(pairs)
   38
   >>> summary = mmp.transformation_summary(pairs)
   >>> list(summary.columns)
   ['transformation', 'count', 'mean_delta', 'median_delta', 'std_delta']
   >>> summary.iloc[0]["transformation"], int(summary.iloc[0]["count"])
   ('[1*]C>>[1*]Cl', 4)

Because the transformation is recorded as a rule rather than a pair of
structures, the same change across different cores aggregates — which is
what turns a list of pairs into a transferable design rule.

Read ``std_delta`` beside ``mean_delta``. A transformation with a mean of
+0.77 and a standard deviation of 1.12 is not a design rule; it is a change
whose effect depends entirely on context:

.. doctest::

   >>> row = summary.iloc[0]
   >>> round(float(row["mean_delta"]), 2), round(float(row["std_delta"]), 2)
   (0.77, 1.12)

Each pair records the shared core, the transformation as
``fragment_a>>fragment_b``, and the activity change. Aggregating by
transformation answers the medicinal chemist's question directly: *what
does this substituent change usually do to potency?*

The activity landscape
----------------------

A SAS map plots every pair as (structure similarity, activity similarity)
and reads the quadrants as SAR regimes:

.. doctest::

   >>> from qsarkit.sar import ActivityLandscapePlotter
   >>> plotter = ActivityLandscapePlotter()
   >>> sas = plotter.sas_data(demo_mols, DEMO_Y)
   >>> list(sas.columns)
   ['index_a', 'index_b', 'structure_similarity', 'activity_similarity', 'delta_activity', 'quadrant']
   >>> sas["quadrant"].value_counts().to_dict()
   {'scaffold hop': 158, 'nondescript': 107, 'smooth SAR': 11}
   >>> type(plotter.plot(demo_mols, DEMO_Y)).__name__
   'Figure'

No pair lands in the "activity cliff" quadrant here, because the quadrant
assignment uses a structural-similarity threshold these small molecules do
not clear — the same caveat as above.

=================  =================  ==========================
Structure sim.     Activity sim.      Interpretation
=================  =================  ==========================
high               high               smooth / continuous SAR
high               low                **activity cliff**
low                high               scaffold hop
low                low                nondescript
=================  =================  ==========================

Free-Wilson analysis
--------------------

The original QSAR method, and still the most interpretable: activity as a
baseline plus an additive contribution per substituent per position. Its
residuals are themselves informative — large ones mark exactly the
non-additive SAR that additive models cannot represent.

.. doctest::

   >>> from qsarkit.sar import FreeWilsonAnalysis
   >>> benzoic_acids = demo_mols[:6]               # one congeneric series
   >>> fw = FreeWilsonAnalysis(core="O=C(O)c1ccccc1", alpha=0.1).fit(
   ...     benzoic_acids, DEMO_Y[:6])
   >>> contributions = fw.to_dataframe()
   >>> len(contributions) > 0
   True

The residuals are the informative part. The largest one marks the compound
the additive model cannot account for — which is the same 4-chloro
compound the cliff analysis flagged:

.. doctest::

   >>> residuals = fw.residuals()
   >>> worst = residuals.iloc[0]
   >>> int(worst["molecule_index"]), round(float(worst["residual"]), 2)
   (2, 0.2)

Free-Wilson assumes substituent effects are additive and independent —
exactly what an activity cliff violates. A high cliff ratio is a warning
that this analysis will mislead on that series.

References
----------

- Maggiora, G. M. (2006). "On Outliers and Activity Cliffs — Why QSAR Often
  Disappoints." *J. Chem. Inf. Model.*, 46(4), 1535. :doi:`10.1021/ci060117s`
- Guha, R. & Van Drie, J. H. (2008). "Structure-Activity Landscape Index."
  *J. Chem. Inf. Model.*, 48(3), 646-658. :doi:`10.1021/ci7004093`
- Hussain, J. & Rea, C. (2010). "Computationally Efficient Algorithm to
  Identify Matched Molecular Pairs." *J. Chem. Inf. Model.*, 50(3), 339-348.
  :doi:`10.1021/ci900450m`
- Stumpfe, D. & Bajorath, J. (2012). "Exploring Activity Cliffs in Medicinal
  Chemistry." *J. Med. Chem.*, 55(7), 2932-2942. :doi:`10.1021/jm300288g`
- Free, S. M. & Wilson, J. W. (1964). "A Mathematical Contribution to
  Structure-Activity Studies." *J. Med. Chem.*, 7(4), 395-399.
  :doi:`10.1021/jm00334a001`
- van Tilborg, D., Alenicheva, A. & Grisoni, F. (2022). "Exposing the
  Limitations of Molecular Machine Learning with Activity Cliffs."
  *J. Chem. Inf. Model.*, 62(23), 5938-5951.
  :doi:`10.1021/acs.jcim.2c01073`
