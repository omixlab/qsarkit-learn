Changelog
=========

.. _changelog-0-8-0:

0.8.0 (unreleased)
------------------

Narrowed the package back to the QSAR workflow proper, and filled the gaps
that narrowing exposed.

**Scope**

qsarkit is now a focused QSAR library rather than a general
cheminformatics platform. Literature mining, database retrieval, docking,
de-novo design, ADMET filtering, active learning, experiment design and
benchmarking were removed: they are separate disciplines with separate
failure modes, and covering them made the package broad rather than
trustworthy. 29 subpackages became 21.

**New modules**

- ``persistence`` — pickle-free model saving. A directory bundle holding
  JSON metadata beside a `skops <https://skops.readthedocs.io>`_
  representation of the estimator, so a saved model survives a
  scikit-learn upgrade and can be inspected before it is loaded. Carries
  the provenance OECD principle 2 asks for.

**Fixed**

- ``QSARRegressor`` and ``QSARClassifier`` now expose their fitted backend's
  attributes. The facades delegate ``fit`` and ``predict`` but hid everything
  a *consumer* introspects, so ``BorutaSelector`` and scikit-learn's ``RFE``
  rejected qsarkit's own default estimator for lacking
  ``feature_importances_`` or ``coef_``, and ``EnsembleUncertainty`` missed
  ``estimators_`` and silently refitted a bagging ensemble of forests instead
  of reusing the forest's trees --- slower, and markedly worse calibrated
  (ENCE 2.9 against 0.3 on the example in :doc:`api/uncertainty`). Names
  beginning with an underscore are not forwarded, so cloning, copying and
  pickling are unaffected.
- ``ChemicalSpaceAnalyzer.fit`` consumed entries from ``self.kwargs`` with
  ``pop()``, mutating a constructor argument. A user's ``perplexity`` was
  honoured by the first ``fit()`` and silently forgotten by the second, so two
  identical calls on one object returned different embeddings. The options are
  now copied before use.

- ``KNNApplicabilityDomain`` with the default ``metric="euclidean"`` computed
  distances by materialising an ``(n_query, n_train, n_features)`` array. On
  5000 training compounds described by 2048-bit fingerprints that is over
  100 TB, so the domain was killed by the operating system on any realistic
  dataset while working fine on the two dozen molecules in the documentation.
  It now uses ``scipy.spatial.distance.cdist``: the same distances in
  :math:`O(n \times m)` memory, 19 s and 0.6 GB for that case.
- ``SHAPExplainer`` could not explain qsarkit's own estimators.
  ``shap.TreeExplainer`` rejects a ``QSARClassifier`` outright, and
  ``explainer_type="auto"`` classified the facade by its own class name and
  fell back to the kernel explainer, which is far slower and needs a
  background set. The facade is now unwrapped to its fitted backend --- by
  type, since scikit-learn's ensembles also carry an ``estimator_`` attribute
  holding an *unfitted* base-estimator template.
- ``SHAPExplainer`` now uses interventional perturbation when a ``background``
  sample is supplied. It is the better-defined estimator and avoids the
  additivity-check failure that tree-path-dependent perturbation hits on wide
  fingerprint matrices.
- ``QSARRegressor`` and ``QSARClassifier`` raised
  ``TypeError: got multiple values for keyword argument`` when
  ``model_params`` set a keyword the facade also sets, so
  ``QSARClassifier("rf", model_params={"n_estimators": 100})`` --- the
  package's default estimator --- failed outright. The caller's value now
  wins.

**New functionality**

- ``chemspace`` gained ``projection_trustworthiness`` and
  ``ChemicalSpaceAnalyzer.trustworthiness``, which measure how much of a
  2D projection's local structure survived the embedding. A t-SNE or UMAP
  figure that destroyed the neighbourhood structure looks exactly like one
  that preserved it, so the score belongs with the figure. One neighbourhood
  size returns a float and an iterable returns an array in the order given,
  matching the convention the validators use; ``subsample`` makes the
  :math:`O(n^2)` distance matrix tractable on a screening library.
- Every validator in ``validation`` -- ``CrossValidator``, ``YScrambling``,
  ``ExternalValidator`` and ``BootstrapValidator`` -- now takes a
  ``scoring`` argument instead of being fixed to :math:`R^2`. Name a metric
  from ``available_metrics()``, pass a ``(y_true, y_pred)`` callable, or
  wrap one with ``make_scorer`` when it needs probabilities or is a loss.
  Pass an iterable of metrics and every score in the result becomes an
  array in the order given, computed from a single pass over the folds.

  ``roc_auc``, ``pr_auc`` and ``brier`` are given ``predict_proba``'s
  positive-class column rather than a thresholded label. A loss declares
  ``greater_is_better=False``, which is what keeps a y-randomization
  p-value from being reported backwards when it is computed on RMSE.

  ``YScrambling`` also gained ``cv`` and ``stratify``, which score out of
  fold rather than in sample. This is necessary rather than cosmetic for a
  ranking metric: a random forest separates permuted labels in-sample as
  perfectly as real ones, so an in-sample ROC-AUC comparison reads 1.0
  against 1.0 and detects nothing. The default stays in-sample, so existing
  results are unchanged.
- ``validation`` gained ``YScrambling`` (y-randomization, required
  evidence under OECD principle 4), ``BootstrapValidator`` (out-of-bag
  score with a confidence interval) and ``ExternalValidator``.
- ``metrics`` gained probability calibration (``calibration_curve``,
  ``expected_calibration_error``, ``maximum_calibration_error``,
  ``calibration_report``), residual diagnostics (``qq_data``,
  ``residual_normality``) and decision-threshold selection
  (``threshold_sweep``, ``optimal_threshold``, ``threshold_report``).
- ``explainability`` gained ``AttributionAtomMapper``, which projects
  SHAP or LIME per-bit attributions back onto atoms through the
  fingerprint's bit-provenance map, and ``draw_atom_weights``, which
  renders them as an RDKit similarity map.
- ``QSARRegressor`` and ``QSARClassifier`` now accept any
  scikit-learn-compatible estimator — XGBoost, LightGBM, CatBoost or your
  own — as a class or an instance, with ``model_args``, ``model_params``,
  ``fit_params``, ``predict_params`` and ``predict_proba_params``.
- ``functional.molecules`` accepts InChI as well as SMILES and RDKit
  molecules, auto-detected per entry.
- ``functional.balance`` accepts an imbalanced-learn sampler. Samplers
  that synthesize feature vectors (SMOTE and relatives) are rejected at
  the molecule stage with an explanation, since no molecule corresponds
  to an interpolated vector.
- The functional pipe reaches the whole workflow: ``featurize``,
  ``fingerprint``, ``describe``, ``scale``, ``impute``, ``drop_constant``,
  ``drop_correlated``, ``select_features``, ``split``, ``fit``,
  ``cross_validate``, ``applicability_domain``, ``collect``, plus
  flowchart rendering to PNG, PDF and SVG.
- ``QSARReport`` renders to plain text and PDF in addition to Markdown,
  HTML and JSON, with tables and plots carried into each.
- New plots: ``plot_calibration_curve``, ``plot_qq``,
  ``plot_threshold_sweep``, ``plot_precision_recall``,
  ``plot_atom_contributions``.

**Fixes**

- ``StructureValidator`` now checks *net* formal charge. It previously
  flagged every zwitterion — glycine, ciprofloxacin, any betaine — as a
  record that had escaped neutralization.
- ``check_activity_units`` no longer reports raw molar concentrations as
  logarithmic when they happen to span a narrow numeric range. A
  pActivity of 1e-9 would mean an IC50 near 1 M.
- ``NearestNeighborAnalyzer.nearest_similarity(exclude_self=True)`` masks
  each query's own row rather than every perfect match, so ``redundancy``
  detects exact duplicates instead of hiding them.
- ``DiversityAnalyzer`` and ``ScaffoldAnalyzer`` now agree on
  ``n_scaffolds``: both exclude acyclic molecules, which have no
  Bemis-Murcko framework. Previously a library of straight chains looked
  scaffold-diverse.
- ``CorrelationFilter`` drops zero-variance columns deliberately instead
  of as a side effect of NaN comparisons, and no longer emits a
  divide-by-zero warning per column on fingerprint input.
- ``GlycanDescriptors`` and ``FragmentRemover`` no longer substitute
  defaults in ``__init__``, which broke ``clone()`` and made
  ``get_params()`` report values the caller never passed.
- ``QSARReport.to_markdown`` no longer requires the undeclared
  ``tabulate`` package for reports containing a table.
- ``functional.molecules`` and the SMILES readers restore the caller's
  RDKit logging state instead of switching it back on.
- Per-molecule error handling no longer uses bare ``except Exception``,
  which swallowed ``MemoryError``. A documented
  ``RDKIT_MOLECULE_ERRORS`` tuple replaces it.
- ``DuplicateDetector`` computes activity spread with vectorized NumPy.

**Documentation**

- Every example in the docstrings, the guide, the API reference and the
  notebooks is executed by the test suite, so none can go stale silently.
- Four example notebooks covering every public subpackage, committed with
  their output.
- ``package.md`` describes the whole package organization in one file.
- LaTeX formulas render in HTML and PDF; the Read the Docs build treats
  warnings as errors.
- Added a ``LICENSE`` file, which ``pyproject.toml`` had referenced
  without shipping.

.. _changelog-0-2-0:

0.2.0
-----

Expanded qsarkit from a QSAR modeling library into a broad cheminformatics
platform. Much of this was removed again in 0.4.0; see above.

**New modules**

- ``chemistry`` — standardization, glycan detection/removal/descriptors,
  fragment and protecting-group removal, scaffold extraction, molecular
  graphs.
- ``sar`` — matched molecular pairs, activity cliffs, SALI, SARI,
  activity-landscape (SAS) maps, R-group tables, Free-Wilson analysis.
- ``applicability`` — eleven applicability-domain definitions behind one
  interface, plus coverage and accuracy-vs-coverage analysis.
- ``neighbors`` and ``cluster`` — Tanimoto/Jaccard similarity search and
  k-NN estimators; Taylor-Butina, sphere exclusion, MaxMin and
  hierarchical clustering with a scikit-learn API.
- ``data_quality``, ``representation``, ``uncertainty``,
  ``explainability``, ``chemspace``, ``reporting``, ``functional``.

**Infrastructure**

- ``mypy --strict`` clean; ships a ``py.typed`` marker.
- All plotting standardized on Plotly.
- Heavy dependencies made lazy and grouped into pip extras.
- Sphinx documentation with per-class scientific references.

.. _changelog-0-1-0:

0.1.0
-----

Initial QSAR modeling library.
