# qsarkit — package reference

A map of the whole package: what each subpackage is for, the classes and
functions it exposes, and the literature each rests on.

This is the orientation document. For executable examples see the five
[notebooks](notebooks/), and for full parameter-level documentation the
[API reference](https://omixlab.github.io/qsarkit-learn/).

Current at version 0.5.0: 21 subpackages, 289 public classes and functions,
2129 tests at 92% branch coverage.

**Contents**

- [Scope](#scope)
- [Architecture](#architecture)
- [Two ways to write a workflow](#two-ways-to-write-a-workflow)
- Stages: [Curation](#1-curation) · [Representation](#2-representation) ·
  [Modelling](#3-modelling) · [Validation](#4-validation) ·
  [Applicability](#5-applicability-domain) · [Uncertainty](#6-uncertainty) ·
  [Interpretation](#7-interpretation) · [Reporting](#8-reporting-and-persistence)
- [Supporting modules](#supporting-modules)
- [Optional dependencies](#optional-dependencies)
- [Conventions](#conventions)
- [References](#references)

---

## Scope

qsarkit covers the QSAR workflow proper: curating structures, turning them
into features, fitting and validating a model, defining where it applies,
and interpreting what it learned.

It deliberately stops there. Literature mining, database retrieval,
docking and de-novo design are separate disciplines with separate failure
modes; a library that covered all of them would be broad rather than
trustworthy. Where a boundary exists, the documentation names the tool that
does the job properly — for graph neural networks, for instance,
[Chemprop](https://github.com/chemprop/chemprop) — and qsarkit accepts the
resulting features as a plain array.

## Architecture

21 subpackages, organized by workflow stage rather than by technique:

```
     structures + measured activities
                  │
    ┌─────────────▼──────────────┐
    │ 1. CURATION                │  chemistry, data_quality
    │ 2. REPRESENTATION          │  representation, transform
    └─────────────┬──────────────┘
                  │  feature matrix
    ┌─────────────▼──────────────┐
    │ 3. MODELLING               │  models, model_selection,
    │                            │  feature_selection, neighbors, cluster
    │ 4. VALIDATION              │  validation, metrics
    │ 5. APPLICABILITY DOMAIN    │  applicability
    │ 6. UNCERTAINTY             │  uncertainty
    └─────────────┬──────────────┘
                  │  a model you can defend
    ┌─────────────▼──────────────┐
    │ 7. INTERPRETATION          │  sar, explainability, chemspace
    │ 8. REPORTING               │  reporting, persistence
    └────────────────────────────┘

    cross-cutting: base (estimator protocol), functional (pipe API),
                   utils (I/O, units)
```

Every component is a scikit-learn estimator. Transformers take
`Iterable[rdkit.Chem.Mol]` and implement `fit`/`transform`; models
implement `fit`/`predict`. They compose in `sklearn.pipeline.Pipeline`,
work with `GridSearchCV`, and survive `clone()`.

## Two ways to write a workflow

The estimator API, for composing with the rest of scikit-learn:

```python
from qsarkit.chemistry import MolecularStandardizer
from qsarkit.representation import MorganFingerprint
from qsarkit.models import QSARRegressor

clean = MolecularStandardizer().transform(mols)
X = MorganFingerprint(radius=2, n_bits=2048).transform(clean)
model = QSARRegressor("rf").fit(X, y)
```

Or the functional pipe, which reads in the order the work happens and
keeps `y` aligned with the molecules at every step:

```python
from qsarkit.functional import *

train, test = (
    molecules(smiles, y)                  # Mol, SMILES or InChI
    >> standardize() >> drop_invalid()    # MoleculeSet
    >> fingerprint("morgan", n_bits=2048) # → FeatureSet
    >> split("scaffold", test_size=0.2)
)
model = train >> select_features(k=200) >> fit("rf")
```

Both call the same code. `featurize` accepts any transformer and `fit` any
estimator, so the pipe reaches the whole package rather than
reimplementing a slice of it.

---

## 1. Curation

A QSAR model is bounded by the quality of the data behind it. Curation is
not tidying — it is the difference between a model of the chemistry and a
model of the database's accumulated errors.

### `qsarkit.chemistry` — structure normalization

| Class | Purpose |
|---|---|
| `MolecularStandardizer` | Sanitize, strip salts and solvates, neutralize charges, canonicalize tautomers, handle stereochemistry, normalize hydrogens |
| `GlycanDetector` | Find carbohydrate rings by the Sugar Removal Utility strategy |
| `GlycanRemover` | Cleave glycosidic bonds, returning the aglycone |
| `GlycanDescriptors` | Sugar count, glycan mass fraction, glycosylation pattern |
| `FragmentRemover` | Strip protecting groups, linkers, tags and click handles |
| `CoreExtractor` | Bemis–Murcko scaffold, generic skeleton, MCS, med-chem core |
| `MolecularGraph` | NetworkX conversion plus topological indices (Wiener, Balaban J, κ shape) |

Two records of the same compound differing only in salt form, protonation
or tautomer are the same compound; a model that sees them as different is
learning the registration system. Failures become `None` **in place**, so a
parallel array of activities stays aligned.

### `qsarkit.data_quality` — dataset-level checks

| Name | Purpose |
|---|---|
| `DuplicateDetector` | Structural duplicates by InChIKey, SMILES, connectivity or scaffold, with activity-agreement checking |
| `ActivityOutlierDetector` | z-score, modified z-score (median/MAD), IQR, neighbour-based |
| `StructureValidator` | Mixtures, inorganics, isotopes, size limits, net charge |
| `DataCurationPipeline` | The stages above in order, with a `CurationReport` naming every removal |
| `check_activity_units` | Catch a column that mixes molar and p-scale values |
| `merge_replicates` | Aggregate duplicate measurements, refusing when they disagree |

The default outlier method is the modified z-score, which uses the median
and MAD: a single extreme value inflates the standard deviation enough to
mask itself, so a plain z-score is least reliable exactly when needed.

`StructureValidator` checks **net** charge, so a zwitterion — glycine,
ciprofloxacin, any betaine — is not mistaken for a record that escaped
neutralization.

> Fourches, Muratov & Tropsha (2010), *Trust, But Verify*,
> [10.1021/ci100176x](https://doi.org/10.1021/ci100176x) ·
> Fischer et al. (2020), *Sugar Removal Utility*,
> [10.3390/molecules25081988](https://doi.org/10.3390/molecules25081988)

---

## 2. Representation

Choosing a representation matters more than choosing a model. A random
forest on good features beats a tuned network on bad ones, and no amount of
hyperparameter search recovers information the representation discarded.

### `qsarkit.representation`

**Fingerprints** — `MorganFingerprint` (ECFP), `FeatureMorganFingerprint`
(FCFP), `RDKitFingerprint`, `PatternFingerprint`, `LayeredFingerprint`,
`AtomPairFingerprint`, `TopologicalTorsionFingerprint`,
`MACCSKeysFingerprint`, `AvalonFingerprint`, `PharmacophoreFingerprint`,
`MHFPFingerprint`, `MAP4Fingerprint`, `FingerprintCombiner`.

**Descriptors** — `RDKitDescriptors` (all 217),
`PhysicochemicalDescriptors`, `ConstitutionalDescriptors`,
`LipinskiDescriptors`, `FragmentDescriptors`, `Descriptors3D`,
`DescriptorCalculator`.

**Learned embeddings** — `Mol2VecTransformer` (needs `embeddings`),
`ChemBERTaTransformer` (needs `nlp`).

Atom pairs and topological torsions are *count* vectors by design. MACCS
keys are 166 curated substructure questions: interpretable and
fixed-length, but far less expressive than a hashed fingerprint.

Descriptors are continuous and on wildly different scales, so any
distance-based or regularized model needs them scaled. Fingerprints, being
binary, do not.

### `qsarkit.transform` — scikit-learn glue

`SmilesToMol`, `MolToSmiles`, `MoleculeFeatureUnion`, `NaNHandler`,
`DescriptorScaler`, `VarianceThresholdMol`, `make_qsar_pipeline`.

`make_qsar_pipeline(..., from_smiles=True)` puts structure parsing *inside*
the cross-validation fold, so no preprocessing happens where it could leak.

> Rogers & Hahn (2010), *Extended-Connectivity Fingerprints*,
> [10.1021/ci100050t](https://doi.org/10.1021/ci100050t) ·
> Todeschini & Consonni (2009), *Molecular Descriptors for Chemoinformatics*,
> [10.1002/9783527628766](https://doi.org/10.1002/9783527628766)

---

## 3. Modelling

### `qsarkit.models`

| Class | Purpose |
|---|---|
| `QSARRegressor`, `QSARClassifier` | One facade over `"rf"`, `"svm"`, `"gbm"`, `"xgboost"`, `"lightgbm"`, `"knn"`, `"pls"`, `"ridge"`, `"lasso"`, `"elasticnet"`, `"mlp"`, `"gp"` |
| `PLSRegressor` | Partial least squares with VIP scores |
| `GaussianProcessQSAR` | Tanimoto-kernel GP with predictive uncertainty |
| `RandomForestQSAR`, `SVMQSAR`, `NeuralNetworkQSAR` | Direct wrappers |
| `ConsensusModel` | Weighted or averaged ensemble across models |
| `BaselineModel` | Mean/median/stratified baseline, for the comparison every model owes |
| `TanimotoKernel` | A scikit-learn kernel for fingerprints |

The facades also accept **any** estimator following the scikit-learn
protocol — XGBoost, LightGBM, CatBoost, or your own — as a class or an
instance:

```python
QSARRegressor(CatBoostRegressor, model_params={"depth": 6},
              fit_params={"verbose": False})
```

`model_args`/`model_params` go to the constructor; `fit_params`,
`predict_params` and `predict_proba_params` reach arguments that belong to
the call rather than the constructor (XGBoost's `eval_set`, LightGBM's
`callbacks`, a `sample_weight` array). An instance you pass is **cloned,
never mutated**, and the facade remains a real scikit-learn estimator.

PLS is the chemometric workhorse: it handles more descriptors than
compounds, which is the normal QSAR situation and where ordinary
regression fails. `GaussianProcessQSAR` chooses its kernel from the data —
a Tanimoto kernel is not positive semi-definite on signed descriptors, so
hardcoding it would silently fail to converge.

### `qsarkit.model_selection` — how you split *is* the experiment

| Splitter | What it measures |
|---|---|
| `ScaffoldSplitter` | Generalization to new chemistry. The default hard split; deterministic |
| `StratifiedScaffoldSplitter` | The same, preserving class balance |
| `ButinaClusterSplitter`, `SphereExclusionSplitter` | Similarity clusters, catching series that share no framework |
| `KennardStoneSplitter`, `PerimeterSplitter` | Deterministic coverage-driven selection |
| `MaxMinSplitter` | Diversity-driven |
| `TimeSplitter` | Prospective use. Reliably the most pessimistic |
| `RandomSplitter` | A baseline, and the comparison that shows what the others cost |

Also `NestedCV` and `hyperparameter_search` (grid, random, Optuna).

A random split of a QSAR dataset measures *interpolation*: public sets are
dense with near-duplicate analogues, so random assignment scatters a
congeneric series across both sides and scores the model on compounds whose
close relatives it has memorized. On the demo dataset the gap is
Q²F1 = 0.82 random against −1.0 by scaffold.

`NestedCV` separates model selection from assessment. Tuning on the folds
you report from leaks the test set into the hyperparameter choice, and the
reported figure is optimistic by an amount nobody can estimate afterwards.

### `qsarkit.feature_selection`

`VarianceFilter`, `CorrelationFilter`, `MutualInformationSelector`,
`RFESelector`, `BorutaSelector`.

Select **inside** the cross-validation loop. Choosing columns using all the
labels and then splitting is selection bias. Boruta asks a different
question from the rest — "which features carry more signal than noise" —
and answers with a set of whatever size the data supports.

### `qsarkit.neighbors` and `qsarkit.cluster`

`JaccardNeighborSearch`, `JaccardKNeighborsRegressor`,
`JaccardKNeighborsClassifier`, `tanimoto_similarity_matrix`,
`jaccard_distance`, `jaccard_distance_matrix`, `is_binary`.

`ButinaClustering`, `SphereExclusionClustering`, `HierarchicalClustering`,
`MaxMinPicker`.

Euclidean distance is wrong for sparse binary fingerprints: it reduces to
the count of differing bits and never looks at how many bits two molecules
share relative to how many they set, so it confuses "two elaborate
molecules with most features in common" with "two sparse molecules with
almost nothing in common". Everything here uses Tanimoto/Jaccard, verified
against RDKit's `BulkTanimotoSimilarity`.

> Bemis & Murcko (1996),
> [10.1021/jm9602928](https://doi.org/10.1021/jm9602928) ·
> Sheridan (2013), *Time-Split Cross-Validation*,
> [10.1021/ci400084k](https://doi.org/10.1021/ci400084k) ·
> Butina (1999), [10.1021/ci9803381](https://doi.org/10.1021/ci9803381) ·
> Bajusz et al. (2015), *Why Is Tanimoto Index Appropriate*,
> [10.1186/s13321-015-0069-3](https://doi.org/10.1186/s13321-015-0069-3)

---

## 4. Validation

OECD principle 4 asks for three distinct things — goodness of fit,
robustness and predictivity — and a single R² addresses only the first.

### `qsarkit.validation`

| Class | Addresses |
|---|---|
| `CrossValidator` | k-fold, repeated k-fold, leave-one-out, leave-group-out. Reports Q² and out-of-fold predictions |
| `YScrambling` | **Robustness.** Refit on permuted labels; if scrambled models score near the real one, the fit is capacity rather than chemistry |
| `BootstrapValidator` | Out-of-bag score with a confidence interval, so a difference between two models can be judged |
| `ExternalValidator` | **Predictivity.** Held-out scoring with the full QSAR metric set and the Golbraikh–Tropsha criteria |

`n_splits` is reported from the splitter, not the constructor argument:
leave-one-out on 24 compounds is 24 splits.

### `qsarkit.metrics`

**Regression** — `q2_f1`, `q2_f2`, `q2_f3`, `ccc`, `r2m`, `r2m_prime`,
`average_r2m`, `delta_r2m`, `r0_squared`, `r0_prime_squared`, `k_slope`,
`k_prime_slope`, `golbraikh_tropsha_criteria`, `rmse`, `rmsep`, `mae`,
`median_ae`, `mse`, `see`, `press`, `bias`, `adjusted_r2_score`,
`r2_score`, `qsar_regression_report`.

**Classification and screening** — `roc_auc`, `pr_auc`, `bedroc`,
`enrichment_factor`, `robust_initial_enhancement`, `matthews_corrcoef`,
`cohen_kappa`, `balanced_accuracy`, `f1_score`, `precision`, `recall`,
`sensitivity`, `specificity`, `accuracy`, `brier_score`,
`confusion_counts`, `qsar_classification_report`.

**Calibration** — `calibration_curve`, `expected_calibration_error`,
`maximum_calibration_error`, `calibration_report`.

**Residual diagnostics** — `qq_data`, `residual_normality`.

**Threshold selection** — `threshold_sweep`, `optimal_threshold`,
`threshold_report`.

Three points worth stating plainly:

- **ROC-AUC does not notice miscalibration.** It depends only on the
  *ranking* of scores, so a model can have perfect AUC and useless
  probabilities. `calibration_report` also gives `brier_skill_score`,
  which compares against predicting the base rate — on an imbalanced
  screening set a raw Brier score near 0.05 looks excellent and is often
  worse than the constant prediction.
- **Every regression metric assumes roughly normal, homoscedastic
  errors.** When that fails the numbers still compute and quietly mean
  something else. `qq_data` and `residual_normality` check it. Read the
  skew, kurtosis and heteroscedasticity terms rather than the Shapiro–Wilk
  p-value, which is sample-size sensitive in both directions.
- **`predict()` cuts at 0.5, which is almost never right.** On a 3%-active
  set the Youden-optimal threshold found 100% of actives where 0.5 found
  43%. `optimal_threshold` maximizes Youden's J, MCC, F1 or balanced
  accuracy, minimizes an explicit `cost_fn`/`cost_fp`, or maximizes one of
  precision/recall subject to a floor on the other. Select it on
  validation data, never on the test set.

> Golbraikh & Tropsha (2002), *Beware of q2!*,
> [10.1016/S1093-3263(01)00123-1](https://doi.org/10.1016/S1093-3263(01)00123-1) ·
> Consonni, Ballabio & Todeschini (2009),
> [10.1021/ci900115y](https://doi.org/10.1021/ci900115y) ·
> Rücker, Rücker & Meringer (2007), *y-Randomization*,
> [10.1021/ci700157b](https://doi.org/10.1021/ci700157b) ·
> Truchon & Bayly (2007), *BEDROC*,
> [10.1021/ci600426e](https://doi.org/10.1021/ci600426e) ·
> Chicco & Jurman (2020), *Advantages of MCC*,
> [10.1186/s12864-019-6413-7](https://doi.org/10.1186/s12864-019-6413-7) ·
> Saito & Rehmsmeier (2015), *Precision-Recall vs ROC*,
> [10.1371/journal.pone.0118432](https://doi.org/10.1371/journal.pone.0118432)

---

## 5. Applicability domain

OECD principle 3. A prediction outside the domain is not *wrong* — it is
unsupported by the training data, which is a different claim and the one
regulators ask about.

### `qsarkit.applicability`

Eleven definitions behind one interface (`fit(X)`, `predict(X) -> bool`,
`score_samples`, `decision_function`):

| Class | Notes |
|---|---|
| `TanimotoSimilarityAD` | Similarity to the nearest training compound. The right choice for fingerprints |
| `KNNApplicabilityDomain` | Mean distance to *k* neighbours; less sensitive to one close analogue |
| `LeverageAD` | Classical Williams-plot `h*`. Assumes a linear model; near-meaningless on a 2048-bit fingerprint, where the hat matrix is degenerate |
| `RangeAD`, `BoundingBoxAD`, `PCABoundingBoxAD` | Descriptor-range checks. Cheap, but a box admits the hollow interior of a real distribution |
| `ConvexHullAD` | Exact, exponential in dimension; needs aggressive reduction first |
| `KernelDensityAD`, `IsolationForestAD` | Density-based, no shape assumption |
| `DistanceToModelAD` | Mahalanobis or Euclidean distance to the training centroid |
| `EnsembleAD` | Requires agreement among several |
| `ADAnalyzer` | Coverage, and accuracy inside versus outside |

`ADAnalyzer`'s `rmse_ratio` is the number to read: above 1 means errors
outside the domain really are larger, so the domain is separating reliable
predictions from unreliable ones. **A domain with 100% coverage has told
you nothing** — and usually indicates a random split rather than a good
model.

> Sahigara et al. (2012),
> [10.3390/molecules17054791](https://doi.org/10.3390/molecules17054791) ·
> Netzeva et al. (2005),
> [10.1177/026119290503300209](https://doi.org/10.1177/026119290503300209)

---

## 6. Uncertainty

An applicability domain answers "should I trust this at all"; uncertainty
answers "how wrong is it likely to be".

### `qsarkit.uncertainty`

| Class | Notes |
|---|---|
| `ConformalRegressor`, `ConformalClassifier` | Distribution-free coverage guarantee: set `alpha=0.2` and 80% of intervals contain the truth, whatever the model |
| `EnsembleUncertainty` | Spread across a bootstrap or native ensemble |
| `GaussianProcessUncertainty` | Posterior variance |
| `QuantileRegressionUncertainty` | Quantile regression intervals |
| `MCDropoutUncertainty` | Monte-Carlo dropout (needs `nlp`/torch) |
| `UncertaintyCalibration` | ENCE, miscalibration area, coverage curve, Spearman(σ, |error|) |

Only conformal prediction has a guarantee. Ensemble spread is a *proxy*,
which is why the calibration diagnostics matter: a negative correlation
between predicted σ and actual error means the model is most confident
exactly where it is most wrong, which is worse than useless.

> Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World*,
> [10.1007/b106715](https://doi.org/10.1007/b106715) ·
> Norinder et al. (2014),
> [10.1021/ci5001168](https://doi.org/10.1021/ci5001168)

---

## 7. Interpretation

### `qsarkit.sar` — why the model behaves as it does

| Name | Purpose |
|---|---|
| `ActivityCliffDetector`, `ActivityCliff` | Near-identical structures with very different activity |
| `activity_cliff_report` | Cliff count and ratio, the steepest pairs, and the R-group swaps responsible |
| `SALIAnalyzer` | Structure-Activity Landscape Index: Δactivity over structural distance |
| `SARIAnalyzer` | Splits the landscape into continuous (learnable) and discontinuous (not) parts |
| `ActivityLandscapePlotter` | SAS-map quadrants |
| `MatchedMolecularPairs`, `MatchedPair`, `MMPAnalyzer` | Hussain–Rea fragment indexing; transformations as reusable rules |
| `FreeWilsonAnalysis` | Additive substituent contributions |
| `RGroupAnalyzer`, `SARTable` | R-group decomposition into an SAR table |

Cliffs are where QSAR fails *by construction*: any model resting on a
smooth similarity assumption must predict them wrong. Running
`activity_cliff_report` **before** modelling tells you whether a
regression model can work on this series at all.

The default similarity threshold is 0.85, the figure usually quoted — but
it is calibrated for drug-sized molecules with a large shared core. On
small molecules a single-atom change alters every atom environment within
the fingerprint radius, and similarity runs far below intuition. Threshold
to your data.

### `qsarkit.explainability` — which part of *this molecule* mattered

| Name | Purpose |
|---|---|
| `PermutationImportance` | Model-agnostic; reflects predictive reliance rather than internal structure |
| `SHAPExplainer`, `LIMEExplainer` | Per-feature attribution (needs `explainability`) |
| `AtomicContributionMap`, `AtomicContribution` | Mask each atom, re-predict, attribute the change |
| `AttributionAtomMapper` | **Project SHAP/LIME per-bit values back onto atoms** via the fingerprint's bit-provenance map |
| `bit_atom_environments`, `bit_weights_to_atom_weights` | The mapping primitives |
| `draw_atom_weights` | RDKit similarity map, as SVG or PNG |
| `FragmentContributionAnalyzer` | Aggregate atom attributions over BRICS fragments |
| `PartialDependence`, `CounterfactualExplainer` | Feature response curves; minimal edits that flip a prediction |

"Bit 1743 contributed +0.21" is not an explanation a chemist can act on.
`AttributionAtomMapper` turns it into one, with two caveats the arithmetic
cannot remove — hashed bits collide (`collision_rate` reports how much),
and an atom in several environments accumulates credit, so weights do not
sum to the prediction.

Flat weights across a ring are not a defect: masking one atom leaves the
others setting the same bits, so no single one is *necessary*. Read it as
"this ring matters as a unit". Symmetry-equivalent atoms receiving equal
weight is a correctness check.

### `qsarkit.chemspace` — the questions to ask before modelling

`ChemicalSpaceAnalyzer` (PCA/t-SNE/MDS/UMAP), `DiversityAnalyzer`,
`ClusterAnalyzer`, `NearestNeighborAnalyzer`, `ScaffoldAnalyzer`,
`ChemicalSpaceCoverage`, plus `bemis_murcko_smiles`, `tanimoto_matrix`,
`compute_fingerprints`, `fingerprints_to_array`, `morgan_generator`.

t-SNE and UMAP preserve local neighbourhoods and give the familiar island
plots, but **between-cluster distances in those plots are not
meaningful** — reading them as chemical distance is the commonest misuse.
PCA's axes are interpretable and its distances mean something.

Acyclic molecules have no Bemis–Murcko framework; they are reported
separately rather than counted as a scaffold, or a library of straight
chains would look scaffold-diverse.

> Maggiora (2006), *On Outliers and Activity Cliffs*,
> [10.1021/ci060117s](https://doi.org/10.1021/ci060117s) ·
> Guha & Van Drie (2008), *SALI*,
> [10.1021/ci7004093](https://doi.org/10.1021/ci7004093) ·
> Hussain & Rea (2010), *MMP algorithm*,
> [10.1021/ci900450m](https://doi.org/10.1021/ci900450m) ·
> Riniker & Landrum (2013), *Similarity Maps*,
> [10.1186/1758-2946-5-43](https://doi.org/10.1186/1758-2946-5-43) ·
> Lundberg & Lee (2017), *SHAP*,
> [NeurIPS](https://papers.nips.cc/paper/7062) ·
> Wattenberg, Viégas & Johnson (2016), *How to Use t-SNE Effectively*,
> [10.23915/distill.00002](https://doi.org/10.23915/distill.00002)

---

## 8. Reporting and persistence

### `qsarkit.reporting`

`QSARReport` renders to **plain text, Markdown, HTML, JSON and PDF**, with
tables and plots carried into each. `OECDReportBuilder` structures the same
material around the five validation principles and tracks which are
**unaddressed** — a submission fails review over a principle nobody noticed
was missing, so an omission is recorded as an explicit gap.

Plots: `plot_predicted_vs_observed`, `plot_residuals`, `plot_williams`,
`plot_roc_curve`, `plot_precision_recall`, `plot_calibration_curve`,
`plot_qq`, `plot_threshold_sweep`, `plot_learning_curve`,
`plot_feature_importance`, `plot_atom_contributions`, `figure_to_html`.

All plotting returns `plotly.graph_objects.Figure`. Nothing calls `.show()`
or writes a file, so the same figure composes into a notebook, a dashboard
and a report. The exception is `plot_atom_contributions`, which returns SVG
because a chemical depiction is RDKit's job.

### `qsarkit.persistence` — saving a model that still works next year

`save_model`, `load_model`, `inspect_bundle`, `ModelBundle`,
`ModelMetadata`, `environment_summary`.

**Not pickle.** Pickle embeds the exact class layout of every object, so a
file written under one scikit-learn release can fail to load — or load into
a subtly different object — under the next; and loading one executes
arbitrary code.

A qsarkit bundle is a directory:

```
model.qsar/
  manifest.json      what is in here, and what wrote it
  metadata.json      endpoint, task, feature names, provenance
  estimator.skops    the estimator, in skops' inspectable format
  pipeline.skops     the preprocessing pipeline, when there is one
```

Everything but the estimator is plain JSON, readable without importing
qsarkit. The estimator uses [skops](https://skops.readthedocs.io), which
stores parameters as data and refuses to reconstruct untrusted types.
`inspect_bundle` reports what a stranger's bundle contains **without
executing it**.

Guardrails that matter in practice: a feature-width mismatch is refused
rather than producing confident nonsense; `predict_mols` works only when
the transformer was saved too; and loading under different package
versions warns, because a model is not guaranteed to reproduce its
original predictions across them.

> OECD (2007), *Guidance Document No. 69*, ENV/JM/MONO(2007)2,
> [10.1787/9789264085442-en](https://doi.org/10.1787/9789264085442-en) ·
> skops, [Secure persistence](https://skops.readthedocs.io/en/stable/persistence.html)

---

## Supporting modules

### `qsarkit.base` — the estimator protocol

`MoleculeTransformer` (molecules → features),
`MoleculeToMoleculeTransformer` (molecules → molecules, so curation steps
chain), `FittableMoleculeTransformer` (adds an `is_fitted` guard),
`ensure_mol_list`, `require`, and the exception hierarchy (`QsarkitError`,
`InvalidMoleculeError`, `ModelNotFittedError`, `OptionalDependencyError`).

Implement `_transform` on a list of molecules and inherit input validation,
`fit`, `fit_transform` and scikit-learn compatibility. `None` passes
through `ensure_mol_list` deliberately: a molecule that failed earlier must
keep its position, since dropping it would shift every downstream label.

### `qsarkit.functional` — the pipe API

Values: `MoleculeSet` (molecules + labels + provenance), `FeatureSet`
(matrix + labels + originating molecules). Steps: `PipeStep`, `Step`,
`FeatureStep`, built by the `step`/`feature_step` decorators.

- **Entry** — `molecules` (Mol, SMILES or InChI, auto-detected)
- **Curation** — `standardize`, `desalt`, `neutralize`,
  `canonicalize_tautomers`, `deglycate`, `remove_protecting_groups`,
  `drop_invalid`, `remove_duplicates`, `balance`, `keep_if`, `drop_if`,
  `filter_by_property`, `to_pactivity`, `sample`, `shuffle`, `apply`
- **Representation** — `featurize`, `fingerprint`, `describe`
- **Features** — `scale`, `impute`, `drop_constant`, `drop_correlated`,
  `select_features`, `resample`
- **Terminals** — `split`, `fit`, `cross_validate`,
  `applicability_domain`, `collect`
- **Flowchart** — `plot_pipeline`, `to_dot`, `render_pipeline` (PNG/PDF/SVG),
  `pipeline_nodes`, `PipelineNode`

**Class balancing** happens at either stage, and the distinction matters.
`balance` works on molecules and accepts `"undersample"`, `"oversample"`,
or any [imbalanced-learn](https://imbalanced-learn.org) sampler that
*selects* existing samples. SMOTE and its relatives are rejected there with
an explanation: they interpolate new feature vectors, and no molecule
corresponds to an interpolated vector.

`resample` works on features, where synthesis is meaningful, and takes any
sampler. When one synthesizes rows it drops the molecules and says so,
because a mismatched molecule list is worse than none — and any later step
needing them (a scaffold split, a Tanimoto domain, an atom-level
explanation) would then be working on the wrong structures.

Resample the **training set only**. Rebalancing the test set changes the
class prior you are measuring against, so a balanced test score does not
describe the population the model will meet.

**Use `>>`, not `>`.** Python parses `a > b > c` as the chained comparison
`(a > b) and (b > c)`, so a `>`-based pipe silently discards everything but
the last two stages. Piping with `>` raises a `TypeError` explaining why.

### `qsarkit.utils`

I/O (`read_smiles`, `write_smiles`, `read_sdf`, `write_sdf`,
`read_csv_mols`, `mols_to_dataframe`, `dataframe_to_mols`), units
(`to_pactivity`, `from_pactivity`, `nm_to_molar`, `convert_concentration`,
`pactivity_to_delta_g`), validation (`check_mols`, `check_X_y_mols`),
logging, and constants (`CONCENTRATION_TO_MOLAR`, `PACTIVITY_ENDPOINTS`,
`LIPINSKI_THRESHOLDS`, gas constants, `DEFAULT_RANDOM_STATE`).

pIC50 = −log10(IC50 in molar). Working on the p-scale makes the unit
explicit and the errors approximately normal, which is what every
regression metric here assumes.

---

## Optional dependencies

`import qsarkit` never pulls in PyTorch, transformers or gensim. Those load
lazily, and a missing one raises `OptionalDependencyError` naming the extra
to install.

| Extra | Enables |
|---|---|
| `embeddings` | Mol2Vec (`gensim`) |
| `nlp` | ChemBERTa, MC-dropout (`transformers`, `torch`) |
| `explainability` | SHAP and LIME |
| `boosting` | XGBoost and LightGBM |
| `embedding_viz` | UMAP projections (`umap-learn`) |
| `reporting` | PDF and static image export (`reportlab`, `kaleido`) |
| `persistence` | Pickle-free model saving (`skops`) |
| `balancing` | imbalanced-learn samplers |

## Conventions

- **Typed.** `mypy --strict` clean, ships `py.typed`.
- **Cited.** Every algorithm names its original publication with a DOI, and
  the official implementation docs where it wraps one.
- **Tested.** Every example in the docstrings, the documentation and the
  notebooks is executed by the test suite, so none can go stale silently.
- **Plotly for plots.** Figures are returned, never shown or written.
- **Alignment is preserved.** Anything that drops a molecule drops its
  label with it.
- **Failures are explicit.** A record that cannot be parsed becomes `None`
  in place rather than vanishing; an unreachable constraint is reported
  with the best actually available; an optional dependency is named.

## References

The five OECD validation principles underpin the package's organization:

> OECD (2007). *Guidance Document on the Validation of (Quantitative)
> Structure-Activity Relationship [(Q)SAR] Models.* OECD Series on Testing
> and Assessment No. 69, ENV/JM/MONO(2007)2.
> [10.1787/9789264085442-en](https://doi.org/10.1787/9789264085442-en)

| # | Principle | Where it lives |
|---|---|---|
| 1 | A defined endpoint | `ModelMetadata`, `QSARReport` |
| 2 | An unambiguous algorithm | Documented hyperparameters; `MoleculeSet.history`; `persistence` provenance |
| 3 | A defined applicability domain | `qsarkit.applicability` |
| 4 | Goodness-of-fit, robustness, predictivity | `qsarkit.metrics`, `qsarkit.validation` |
| 5 | A mechanistic interpretation, if possible | `qsarkit.sar`, `qsarkit.explainability` |

Foundational works cited throughout:

- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify."
  *J. Chem. Inf. Model.*, 50(7), 1189–1204.
  [10.1021/ci100176x](https://doi.org/10.1021/ci100176x)
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" *J. Mol. Graph.
  Model.*, 20(4), 269–276.
  [10.1016/S1093-3263(01)00123-1](https://doi.org/10.1016/S1093-3263(01)00123-1)
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  *J. Chem. Inf. Model.*, 50(5), 742–754.
  [10.1021/ci100050t](https://doi.org/10.1021/ci100050t)
- Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs. 1.
  Molecular Frameworks." *J. Med. Chem.*, 39(15), 2887–2893.
  [10.1021/jm9602928](https://doi.org/10.1021/jm9602928)
- Maggiora, G. M. (2006). "On Outliers and Activity Cliffs — Why QSAR Often
  Disappoints." *J. Chem. Inf. Model.*, 46(4), 1535.
  [10.1021/ci060117s](https://doi.org/10.1021/ci060117s)
- Sahigara, F. et al. (2012). "Comparison of Different Approaches to Define
  the Applicability Domain of QSAR Models." *Molecules*, 17(5), 4791–4810.
  [10.3390/molecules17054791](https://doi.org/10.3390/molecules17054791)
- van Tilborg, D., Alenicheva, A. & Grisoni, F. (2022). "Exposing the
  Limitations of Molecular Machine Learning with Activity Cliffs."
  *J. Chem. Inf. Model.*, 62(23), 5938–5951.
  [10.1021/acs.jcim.2c01073](https://doi.org/10.1021/acs.jcim.2c01073)
- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in Python."
  *JMLR*, 12, 2825–2830.
  [jmlr.org](https://jmlr.org/papers/v12/pedregosa11a.html)
- RDKit: Open-source cheminformatics. [rdkit.org](https://www.rdkit.org)

Every class and function carries its own primary reference in its
docstring; this list is the shared foundation, not the whole bibliography.
