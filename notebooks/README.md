# Example notebooks

Four notebooks covering the whole QSAR workflow, in order. Each is
self-contained — it defines its own dataset and imports — so you can start
anywhere, but they build on each other and are best read in sequence.

| Notebook | Covers |
|---|---|
| [01 — Curation and the functional API](01_curation_and_the_functional_api.ipynb) | `chemistry`, `data_quality`, `functional`, `utils` |
| [02 — Representation and chemical space](02_representation_and_chemical_space.ipynb) | `representation`, `transform`, `chemspace`, `neighbors`, `cluster` |
| [03 — Modelling, validation and applicability](03_modeling_validation_and_applicability.ipynb) | `models`, `model_selection`, `validation`, `metrics`, `feature_selection`, `applicability`, `uncertainty` |
| [04 — SAR, explainability and reporting](04_sar_explainability_and_reporting.ipynb) | `sar`, `explainability`, `reporting` |
| [05 — Classification, calibration and deployment](05_classification_calibration_and_deployment.ipynb) | `metrics` (calibration, thresholds), `functional` (`resample`), `explainability` (atom-level SHAP/LIME), `persistence` |

Together they touch every public subpackage.

## Running them

```bash
pip install -e ".[dev]"      # includes ipykernel, nbformat, nbclient
jupyter lab notebooks/
```

The notebooks are committed **with their output**, so you can read them on
GitHub without running anything.

## They are tested

`tests/docs/test_notebooks.py` checks that each notebook is well-formed and
was committed with executed output containing no errors, and — under the
`slow` marker — executes every one of them top to bottom:

```bash
pytest tests/docs/test_notebooks.py      # structural checks (fast)
pytest -m slow                           # actually execute them
```

Documentation that is never run stops being true, so a change that breaks an
example fails CI like any other regression.

## A note on the dataset

All four use the same 24-compound synthetic series: four substituent
families (benzoic acids, anilides, pyridine carboxamides, benzimidazoles)
with made-up pIC50 values, resolving to three Bemis-Murcko scaffolds.

It is deliberately tiny, so every cell runs instantly and every number is
reproducible. It is also deliberately *hard*: one compound is a planted
activity-cliff outlier, and the scaffold families are distinct enough that a
scaffold split is genuinely difficult. Several notebooks show models scoring
badly on it — a scaffold-split Q²F1 of −1.0, residuals that fail every
normality check, an applicability domain that rejects the whole test set.
That is the point. The numbers are honest, and a worked example where
everything succeeds teaches nothing about the failure modes these tools
exist to detect.

Notebook 5 additionally resamples the set into a 200-compound, 18%-active
screening deck with 12% label noise, because a separable problem cannot
demonstrate a threshold trade-off.
