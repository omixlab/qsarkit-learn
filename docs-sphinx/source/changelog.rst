Changelog
=========

.. _changelog-0-4-1:

0.4.1 (unreleased)
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

**New functionality**

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
