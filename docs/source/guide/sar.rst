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

.. code-block:: python

   from qsarkit.sar import activity_cliff_report

   report = activity_cliff_report(mols, pIC50)
   report["n_cliffs"]
   report["cliff_ratio"]              # cliffs / all pairs
   report["cliff_compound_fraction"]  # compounds in at least one cliff
   report["top_transformations"]      # which R-group swaps cause them
   report["top_scaffolds"]

A high cliff ratio predicts that a regression model will underperform on
this series no matter how you tune it — and tells you which substituent
changes are responsible.

.. code-block:: python

   from qsarkit.sar import ActivityCliffDetector

   detector = ActivityCliffDetector(
       similarity_threshold=0.85,   # ECFP4 Tanimoto
       activity_threshold=2.0,      # 100-fold potency change
       method="fingerprint",        # or "scaffold", or "mmp"
   )
   cliffs = detector.detect(mols, pIC50)   # sorted by descending SALI

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

.. code-block:: python

   from qsarkit.sar import SALIAnalyzer

   analyzer = SALIAnalyzer()
   analyzer.sali_matrix(mols, y_true)
   analyzer.sali_network(mols, y_true, percentile=95)   # networkx graph
   analyzer.sali_auc(mols, y_true, y_pred)             # 1.0 = perfect, 0.5 = chance

Matched molecular pairs
-----------------------

.. code-block:: python

   from qsarkit.sar import MMPAnalyzer

   analyzer = MMPAnalyzer()
   pairs = analyzer.find_pairs(mols, pIC50)
   analyzer.transformation_summary(pairs)
   # transformation      count  mean_delta  median_delta  std_delta
   # [1*]C>>[1*]N            1         3.5           3.5        0.0

Each pair records the shared core, the transformation as
``fragment_a>>fragment_b``, and the activity change. Aggregating by
transformation answers the medicinal chemist's question directly: *what
does this substituent change usually do to potency?*

The activity landscape
----------------------

A SAS map plots every pair as (structure similarity, activity similarity)
and reads the quadrants as SAR regimes:

.. code-block:: python

   from qsarkit.sar import ActivityLandscapePlotter

   plotter = ActivityLandscapePlotter()
   plotter.sas_data(mols, y)   # DataFrame with a `quadrant` column
   plotter.plot(mols, y)       # plotly Figure

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

.. code-block:: python

   from qsarkit.sar import FreeWilsonAnalysis

   fw = FreeWilsonAnalysis(core="O=C(O)c1ccccc1", alpha=0.1).fit(mols, y)
   fw.to_dataframe()   # contribution of each substituent, ranked
   fw.residuals()      # largest residual = the most non-additive compound

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
