Changelog
=========

0.2.0
-----

Expanded qsarkit from a QSAR modeling library into a full cheminformatics
platform covering the discovery workflow end to end.

**New modules**

- ``chemistry`` — standardization, glycan detection/removal/descriptors,
  fragment and protecting-group removal, scaffold extraction, reaction
  templates and named metabolic transformations, molecular graphs.
- ``sar`` — matched molecular pairs, activity cliffs, SALI, SARI,
  activity-landscape (SAS) maps, R-group tables, Free-Wilson analysis.
- ``applicability`` — eleven applicability-domain definitions behind one
  interface, plus coverage and accuracy-vs-coverage analysis.
- ``neighbors`` and ``cluster`` — Tanimoto/Jaccard similarity search and
  k-NN estimators; Taylor-Butina, sphere exclusion, MaxMin and
  hierarchical clustering with a scikit-learn API.
- ``docking`` — docking output parsing, binding-pocket descriptors,
  protein-ligand interaction fingerprints, rescoring and pose validation.
- ``text_mining``, ``databases``, ``data_quality``, ``representation``,
  ``design``, ``admet``, ``active_learning``, ``experiment_design``,
  ``reporting``, ``benchmarking``.

**Infrastructure**

- ``mypy --strict`` clean; ships a ``py.typed`` marker.
- All plotting standardized on Plotly.
- Heavy dependencies made lazy and grouped into pip extras.
- Sphinx documentation with per-class scientific references.

0.1.0
-----

Initial QSAR modeling library.
