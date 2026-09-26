# qsarkit-learn

A focused, open-source Python library for QSAR (Quantitative Structure-Activity Relationship) modeling.

**[Documentation](https://omixlab.github.io/qsarkit-learn/)** | **[Package reference](package.md)** | **[Notebooks](notebooks/)** | **[Source Code](https://github.com/omixlab/qsarkit-learn)**

`qsarkit-learn` covers the QSAR workflow proper — curating structures, turning them into features, fitting and validating a model, defining where it applies, and interpreting what it learned. It deliberately stops there: it is not a literature-mining, database-retrieval, docking or de-novo design toolkit, and does not pretend to be.

## The Workflow

The package is organized around the core stages of a QSAR project:

1. **Curation** (`qsarkit.chemistry`, `qsarkit.data_quality`): Curate structures and resolve activity data conflicts.
2. **Representation** (`qsarkit.representation`): Convert molecules into machine-learning ready features (descriptors, fingerprints, learned embeddings).
3. **Modeling** (`qsarkit.models`, `qsarkit.model_selection`, `qsarkit.feature_selection`): Select features and fit models.
4. **Validation** (`qsarkit.validation`, `qsarkit.metrics`): Evaluate models aligned with OECD principles.
5. **Applicability Domain** (`qsarkit.applicability`): Define the chemical space where your model's predictions are trustworthy.
6. **Uncertainty** (`qsarkit.uncertainty`): Estimate prediction confidence intervals and error bars.
7. **Interpretation** (`qsarkit.sar`, `qsarkit.explainability`): Explain model decisions and identify activity cliffs.
8. **Reporting** (`qsarkit.reporting`, `qsarkit.persistence`): Generate model reports and save models reproducibly.

## Design Principles

- **Everything is a scikit-learn estimator**: Transformers accept `Iterable[rdkit.Chem.Mol]` and implement `fit` / `transform`. Models implement `fit` / `predict`. Everything composes seamlessly in `sklearn.pipeline.Pipeline`, works with `GridSearchCV`, and supports `clone()`.
- **Every algorithm cites its source**: Each class documents its original scientific publication with a DOI.
- **Typed and checked**: The codebase is strictly typed (`mypy --strict`) and ships with a `py.typed` marker.
- **Every example is executed**: Docstrings, guide pages, API reference and notebooks all run in the test suite, so none of them can go stale silently.
- **Narrow on purpose**: Everything here earns its place in the core QSAR workflow. Data acquisition and generative modeling are deliberately kept out of scope to maintain a highly trustworthy, specialized tool.

## Installation

Install from PyPI with pip. Python 3.9+ is required.

```bash
pip install qsarkit-learn
```

> The distribution is **`qsarkit-learn`**; the package you import is **`qsarkit`** — the same split as `scikit-learn` and `sklearn`. The bare name `qsarkit` on PyPI belongs to an unrelated project.

The package uses optional dependencies to avoid bloating your environment. You can install specific extras depending on your use case:

```bash
# For embeddings and NLP-based representations
pip install qsarkit-learn[embeddings,nlp]

# For tree-based models and explainability tools
pip install qsarkit-learn[boosting,explainability]

# For pickle-free model saving and PDF reports
pip install qsarkit-learn[persistence,reporting]

# To install everything
pip install qsarkit-learn[all]
```

A missing optional dependency raises an error naming the extra that provides it, rather than an `ImportError` you have to interpret.

| Extra | Enables |
|---|---|
| `embeddings` | Mol2Vec embeddings (`gensim`) |
| `nlp` | ChemBERTa embeddings, MC-dropout (`transformers`, `torch`) |
| `explainability` | SHAP and LIME attribution |
| `boosting` | XGBoost and LightGBM estimators |
| `embedding_viz` | UMAP chemical-space projections (`umap-learn`) |
| `reporting` | PDF and static image export (`reportlab`, `kaleido`) |
| `persistence` | Pickle-free model saving (`skops`) |
| `balancing` | imbalanced-learn samplers |

## A First Example

Here is a simple example showing how to curate a molecule and analyze the Structure-Activity Relationship (SAR) of a dataset:

```python
from rdkit import Chem
from qsarkit.chemistry import MolecularStandardizer
from qsarkit.sar import activity_cliff_report

# Curate: strip the salt, neutralize the charge
standardizer = MolecularStandardizer()
mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
curated_mols = standardizer.transform([mol])
print(Chem.MolToSmiles(curated_mols[0]))
# Output: 'CC(=O)Oc1ccccc1C(=O)O'

# Diagnose a dataset before modeling it to find activity cliffs
# (Assuming `mols` is a list of RDKit molecules and `pIC50_values` is an array of activities)
report = activity_cliff_report(mols, pIC50_values)
print(report["cliff_ratio"])           # Proportion of the SAR that is discontinuous
print(report["top_transformations"])   # R-group swaps that cause the cliffs
```

## The Functional Pipe API

A second way to write the same workflow, reading in the order the work happens — in the spirit of R's `%>%`. It is not a separate implementation: `featurize` takes any transformer and `fit` any estimator, so the pipe reaches the whole package.

`molecules()` accepts RDKit molecules, SMILES and InChI in any mixture, and every step keeps `y` index-aligned with the molecules — dropping a molecule drops its label with it.

```python
from qsarkit.functional import *

train, test = (
    molecules(smiles_or_inchi, activities)
    >> standardize() >> drop_invalid()       # MoleculeSet
    >> remove_duplicates(agg="mean")
    >> fingerprint("morgan", n_bits=2048)    # -> FeatureSet
    >> split("scaffold", test_size=0.2)
)
model = train >> select_features(k=200) >> fit("rf")
```

Steps compose, so a curation protocol can be defined once and applied to train and test alike:

```python
curate = standardize() >> drop_invalid() >> remove_duplicates(max_spread=1.0)
train = molecules(train_smiles, train_y) >> curate
test  = molecules(test_smiles,  test_y)  >> curate
```

> **Use `>>`, not `>`.** Python parses `a > b > c` as the chained comparison `(a > b) and (b > c)`, so a `>`-based pipe silently discards everything but the last two stages. Piping with `>` raises a `TypeError` explaining why.

A pipeline is a graph, and drawing it is the quickest way to confirm the stages are in the order you meant:

```python
pipe = standardize() >> drop_invalid() >> fingerprint() >> scale() >> fit("rf")

pipe.plot()                    # a Plotly figure, no extra dependency
pipe.render("workflow.pdf")    # PNG / PDF / SVG
print(pipe.to_dot())           # Graphviz DOT source
```

## Any Estimator You Like

`QSARRegressor` and `QSARClassifier` dispatch on a name (`"rf"`, `"svm"`, `"gbm"`, `"xgboost"`, `"lightgbm"`, `"knn"`, `"pls"`, `"gp"`, …), but they also accept anything following the scikit-learn `fit`/`predict` protocol — XGBoost, LightGBM, CatBoost, or your own wrapper — as a class or an instance:

```python
from catboost import CatBoostRegressor

model = QSARRegressor(CatBoostRegressor,
                      model_params={"depth": 6},
                      fit_params={"verbose": False}).fit(X, y)
```

`model_args` / `model_params` go to the constructor; `fit_params`, `predict_params` and `predict_proba_params` reach arguments that belong to the call rather than the constructor. An instance you pass is cloned, never mutated, and the facade remains a real scikit-learn estimator.

## Honest Validation

The split is the experiment. A random split of a QSAR dataset measures *interpolation*, because public sets are dense with near-duplicate analogues. On the demo dataset that difference is Q²F1 = 0.82 random against −1.0 by scaffold — same data, same model.

OECD principle 4 asks for three separate things, and a single R² addresses only the first:

```python
from qsarkit.validation import BootstrapValidator, CrossValidator, YScrambling

CrossValidator(n_splits=5).evaluate(model, X_train, y_train)              # predictivity
YScrambling(n_iterations=100).run(model, X_train, y_train)["p_value"]     # robustness
BootstrapValidator(n_iterations=100).run(model, X_train, y_train)         # precision
```

y-randomization is the check that catches the classic QSAR failure — a few dozen compounds described by thousands of descriptors, where something will always correlate. It is cheap, so there is no excuse for omitting it.

All four validators take `scoring`, so none of them is tied to R². Pass a list and every score comes back as an array in that order, from one pass over the folds:

```python
from qsarkit.validation import CrossValidator, available_metrics, make_scorer

CrossValidator(scoring=["roc_auc", "pr_auc", "mcc", "brier"]).evaluate(model, X, y)["score"]
available_metrics()                                  # the 18 named ones

# Anything else, including a metric that needs probabilities:
from sklearn.metrics import average_precision_score
make_scorer(average_precision_score, needs_proba=True)
```

`roc_auc`, `pr_auc` and `brier` are handed `predict_proba`'s positive column, never a thresholded label — scoring a hard label with ROC-AUC throws away the ranking the metric exists to measure. A loss declares `greater_is_better=False`, which is what keeps a y-randomization p-value on RMSE from coming out backwards.

> **Score y-randomization out of fold.** The default scores the apparent, in-sample fit. A random forest separates *permuted* labels in-sample as perfectly as real ones, so an in-sample ROC-AUC comparison reads 1.000 against 0.999 and detects nothing. Pass `cv` — and `stratify=True` on an imbalanced endpoint:
>
> ```python
> YScrambling(n_iterations=100, scoring="roc_auc", cv=5, stratify=True).run(model, X, y)
> ```

For classification, three traps worth knowing about:

```python
from qsarkit.metrics import calibration_report, optimal_threshold, threshold_report

# `predict()` cuts at 0.5, which is almost never right on an imbalanced set.
optimal_threshold(y_val, scores, criterion="mcc")
optimal_threshold(y_val, scores, criterion="cost", cost_fn=5.0, cost_fp=1.0)
threshold_report(y_val, scores)          # every criterion, against the 0.5 default

# ROC-AUC depends only on the *ranking* of scores, so a model can have
# excellent AUC and useless probabilities. Check before you threshold them.
calibration_report(y_val, scores)["brier_skill_score"]
```

And for regression, every metric here assumes roughly normal, homoscedastic errors — when that fails, the numbers still compute and quietly mean something else:

```python
from qsarkit.metrics import qq_data, residual_normality

residual_normality(y_test, y_pred)       # skew, kurtosis, heteroscedasticity
qq_data(y_test - y_pred)                 # the data behind a normal Q-Q plot
```

## Looking at the Chemical Space First

Before modelling: does this library cover one region or several, does the test set sit inside the training set's cloud, is this hit an outlier.

```python
from qsarkit.chemspace import ChemicalSpaceAnalyzer, projection_trustworthiness

space = ChemicalSpaceAnalyzer(method="umap", metric="jaccard", random_state=0).fit(X)
space.embedding_                        # (n_molecules, 2)

# How much of the picture can be believed:
space.trustworthiness(X, n_neighbors=[5, 15, 30])
```

t-SNE and UMAP produce convincing islands whose *between*-cluster distances mean nothing, and a projection that destroyed the neighbourhood structure looks exactly like one that preserved it. Trustworthiness is the difference, and it is worth reporting with the figure. UMAP needs the `embedding_viz` extra; PCA, t-SNE and MDS need nothing extra.

## Applicability Domain

A prediction outside the domain is not *wrong* — it is unsupported by the training data, which is a different claim and the one regulators ask about.

```python
from qsarkit.applicability import ADAnalyzer, TanimotoSimilarityAD

domain = TanimotoSimilarityAD(threshold=0.35).fit(X_train)
report = ADAnalyzer(domain).fit(X_train).report(X_test, y_test, y_pred)
report["rmse_ratio"]     # > 1 means the domain is doing its job
```

A domain with 100% coverage has told you nothing — and usually indicates a random split rather than a good model.

## Explaining a Model on the Molecule

SHAP and LIME attribute a prediction to *features*. For a fingerprint model those are hash buckets, and "bit 1743 contributed +0.21" is not an explanation a chemist can act on. `AttributionAtomMapper` projects it back onto atoms through the fingerprint's bit-provenance map, and RDKit draws the result:

```python
from qsarkit.explainability import AttributionAtomMapper, draw_atom_weights

mapper = AttributionAtomMapper(fingerprint)
weights = mapper.from_shap(mol, explainer, X, index=0)

mapper.collision_rate(mol)               # how much to trust the picture
svg = draw_atom_weights(mol, weights)    # RDKit similarity map
```

## Saving a Model That Still Works Next Year

**Not pickle.** A pickled model embeds the exact class layout of every object, so a file written under one scikit-learn release can fail to load — or load into a subtly different object — under the next; and loading one executes arbitrary code.

`qsarkit.persistence` writes a directory bundle instead: plain JSON metadata beside a [skops](https://skops.readthedocs.io) representation of the estimator, which stores parameters as data and refuses to reconstruct untrusted types.

```python
from qsarkit.persistence import ModelMetadata, inspect_bundle, load_model, save_model

path = save_model(
    model,
    "egfr_pIC50",
    pipeline=fingerprint,                # so the bundle can take molecules
    metadata=ModelMetadata(name="EGFR", endpoint="pIC50 (-log10 M)"),
)

inspect_bundle(path)["untrusted"]        # safe to run on a stranger's bundle
bundle = load_model(path)
bundle.predict_mols(new_mols)            # straight from structures
```

A feature-width mismatch is refused rather than producing confident nonsense, and a model loaded under different package versions says so.

## Reports

`QSARReport` renders to plain text, Markdown, HTML, JSON and PDF, with tables and plots carried into each. `OECDReportBuilder` structures the same material around the five validation principles and tracks which are **unaddressed** — a submission fails review over a principle nobody noticed was missing.

```python
from qsarkit.reporting import OECDReportBuilder, QSARReport

builder = OECDReportBuilder(title="QMRF for EGFR model", endpoint="pIC50")
builder.add_evidence(1, True, {"endpoint": "pIC50, CHEMBL203"})
builder.unaddressed                      # [2, 3, 4, 5] — explicit gaps

builder.build().to_pdf("qmrf.pdf")
```

All plotting returns `plotly.graph_objects.Figure` objects. Functions never call `.show()` and never write files, so the same figure composes into a notebook, a dashboard and a report.

## Learning the Package

- **[`package.md`](package.md)** — the whole package organization in one file: every subpackage, its main classes and functions, and the literature behind them.
- **[Notebooks](notebooks/)** — five worked walkthroughs covering every public subpackage, committed with their output:
  1. [Curation and the functional API](notebooks/01_curation_and_the_functional_api.ipynb)
  2. [Representation and chemical space](notebooks/02_representation_and_chemical_space.ipynb)
  3. [Modelling, validation and applicability](notebooks/03_modeling_validation_and_applicability.ipynb)
  4. [SAR, explainability and reporting](notebooks/04_sar_explainability_and_reporting.ipynb)
  5. [Classification, calibration and deployment](notebooks/05_classification_calibration_and_deployment.ipynb)
- **[Documentation](https://omixlab.github.io/qsarkit-learn/)** — the API reference, with a worked example and scientific references for every class.

The notebooks use a deliberately *hard* 24-compound dataset: one planted activity-cliff outlier, distinct scaffold families, non-normal residuals. Several of them show models scoring badly on it. That is the point — a worked example where everything succeeds teaches nothing about the failure modes these tools exist to detect.

## Development

```bash
git clone https://github.com/omixlab/qsarkit-learn
cd qsarkit-learn
pip install -e ".[dev]"

pytest                              # the suite
pytest -m slow                      # plus executing the notebooks
pytest --cov=qsarkit --cov-branch   # with coverage
mypy qsarkit                        # strict type check
ruff check qsarkit                  # lint
```

Building the documentation. `docs-sphinx/` is the source; `docs/` is the
built site served by GitHub Pages, and it is committed, so rebuilding it is
part of preparing a commit that touches the docs:

```bash
pip install -e ".[docs]"

make docs        # rebuild and copy the site into docs/ — the one to run
make preview     # build into docs-sphinx/build/html, leaving docs/ alone
make doctest     # execute every example in the documentation
make linkcheck   # verify external links resolve
```

`make docs` builds from scratch with warnings treated as errors, so a broken
cross-reference or a page deleted from the source cannot reach the published
site. The same targets exist inside `docs-sphinx/` if you prefer to work
there (`make publish` is the one that writes to `../docs`).

> `typings/` holds a local stub shadow for RDKit. The `rdkit-stubs` bundled with the RDKit wheel contain an auto-generation bug — a C++ enum member named `None`, which is an illegal annotation target — that otherwise aborts every mypy run and silently hides all real type errors. Don't delete it.

## OECD Compliance

The package is organized around the five OECD validation principles:

| # | Principle | Where it lives |
|---|---|---|
| 1 | A defined endpoint | `ModelMetadata`, `QSARReport` |
| 2 | An unambiguous algorithm | Documented hyperparameters; `MoleculeSet.history`; `persistence` provenance |
| 3 | A defined applicability domain | `qsarkit.applicability` |
| 4 | Goodness-of-fit, robustness, predictivity | `qsarkit.metrics`, `qsarkit.validation` |
| 5 | A mechanistic interpretation, if possible | `qsarkit.sar`, `qsarkit.explainability` |

> OECD (2007). *Guidance Document on the Validation of (Quantitative) Structure-Activity Relationship [(Q)SAR] Models.* OECD Series on Testing and Assessment No. 69, ENV/JM/MONO(2007)2. [doi:10.1787/9789264085442-en](https://doi.org/10.1787/9789264085442-en)

## Contributing

We welcome contributions! Please see the [`CONTRIBUTING_STYLE.md`](CONTRIBUTING_STYLE.md) guide for our conventions on class structures, documentation, error handling, static typing (`mypy`), and unit testing.

## Citing

If `qsarkit-learn` contributes to work you publish, please cite the package along with the primary reference for whichever algorithm you used — each class docstring names it.

## License

MIT — see [`LICENSE`](LICENSE).
