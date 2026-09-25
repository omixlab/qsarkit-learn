Modeling
========

Estimators
----------

:class:`~qsarkit.models.QSARRegressor` and
:class:`~qsarkit.models.QSARClassifier` put a broad menu of algorithms
behind one name, so comparing backends is a one-word change:

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> X, y = demo_fingerprints(512), DEMO_Y
   >>> round(QSARRegressor("rf", random_state=0).fit(X, y).score(X, y), 3)
   0.953

Backend-specific arguments go through ``model_params``, not as keywords —
the facade's own signature stays the same whichever algorithm you pick:

.. doctest::

   >>> pls = QSARRegressor("pls", model_params={"n_components": 3}).fit(X, y)
   >>> pls.predict(X).shape
   (24,)
   >>> gp = QSARRegressor("gp", random_state=0).fit(X, y)   # Tanimoto-kernel GP
   >>> gp.predict(X).shape
   (24,)

:class:`~qsarkit.models.ConsensusModel` takes ``(name, estimator)`` pairs,
like scikit-learn's own ensembles:

.. doctest::

   >>> from qsarkit.models import ConsensusModel
   >>> ensemble = ConsensusModel([
   ...     ("rf", QSARRegressor("rf", random_state=0)),
   ...     ("svm", QSARRegressor("svm")),
   ... ]).fit(X, y)
   >>> ensemble.predict(X).shape
   (24,)

All follow the scikit-learn API, so they drop into
:class:`~sklearn.pipeline.Pipeline`,
:class:`~sklearn.model_selection.GridSearchCV` and
:func:`~sklearn.base.clone`.

Any estimator, not just the menu
--------------------------------

``name`` also accepts anything following the scikit-learn ``fit``/
``predict`` protocol — XGBoost, LightGBM, CatBoost, or your own wrapper —
as a class or an instance:

.. doctest::

   >>> from sklearn.linear_model import Ridge
   >>> model = QSARRegressor(Ridge(alpha=2.0)).fit(X, y)
   >>> type(model.estimator_).__name__, model.estimator_.alpha
   ('Ridge', 2.0)

The instance you pass is cloned, never mutated, so one configured template
can seed several models:

.. doctest::

   >>> template = Ridge(alpha=1.0)
   >>> _ = QSARRegressor(template).fit(X, y)
   >>> hasattr(template, "coef_")
   False

``model_args``/``model_params`` go to the constructor; ``fit_params``,
``predict_params`` and ``predict_proba_params`` reach arguments that belong
to the call rather than the constructor — XGBoost's ``eval_set``,
LightGBM's ``callbacks``, a ``sample_weight`` array:

.. doctest::

   >>> import numpy as np
   >>> weights = np.linspace(0.5, 1.5, len(y))
   >>> model = QSARRegressor(Ridge, fit_params={"sample_weight": weights}).fit(X, y)
   >>> model.predict(X).shape
   (24,)

Splitting
---------

.. doctest::

   >>> from qsarkit.model_selection import ScaffoldSplitter
   >>> train, test = next(ScaffoldSplitter(test_size=0.2).split_mols(demo_mols, y))
   >>> len(train), len(test)
   (18, 6)

The guarantee is that no scaffold appears on both sides:

.. doctest::

   >>> from qsarkit.chemspace import bemis_murcko_smiles
   >>> train_scaffolds = {bemis_murcko_smiles(demo_mols[i]) for i in train}
   >>> test_scaffolds = {bemis_murcko_smiles(demo_mols[i]) for i in test}
   >>> train_scaffolds & test_scaffolds
   set()

.. important::

   **A random split flatters a QSAR model.** Public datasets are full of
   near-duplicate analogues, so a random split puts close relatives on both
   sides and measures interpolation, not generalization.

   On this dataset the difference is not subtle — same data, same model,
   same seed:

   .. doctest::

      >>> from qsarkit.metrics import q2_f1
      >>> from qsarkit.model_selection import RandomSplitter
      >>> def score(train, test):
      ...     model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
      ...     return round(q2_f1(y[test], model.predict(X[test]), y[train]), 2)
      >>> score(*next(RandomSplitter(test_size=0.2, random_state=0).split(X, y)))
      0.81
      >>> score(train, test)
      -0.89

   Both numbers are honestly computed. Only the second answers "will this
   work on chemistry I have not seen". Report a scaffold or cluster split
   unless you have a specific reason not to.

:class:`~qsarkit.model_selection.ButinaClusterSplitter` splits by
similarity cluster rather than scaffold, which catches analogue series that
share no Bemis-Murcko framework:

.. doctest::

   >>> from qsarkit.model_selection import ButinaClusterSplitter
   >>> btrain, btest = next(
   ...     ButinaClusterSplitter(test_size=0.25).split_mols(demo_mols, y))
   >>> len(btrain), len(btest)
   (18, 6)

:class:`~qsarkit.model_selection.TimeSplitter` is the most honest of all
where you have dates: it reproduces the real prospective task, and is
reliably the most pessimistic.

Feature selection
-----------------

.. doctest::

   >>> from qsarkit.feature_selection import CorrelationFilter, VarianceFilter
   >>> VarianceFilter().fit_transform(X).shape
   (24, 121)
   >>> CorrelationFilter(threshold=0.95).fit_transform(X, y).shape
   (24, 59)

More than three-quarters of a 512-bit fingerprint never fires on 24
compounds. The unsupervised filters are safe anywhere because they never
look at ``y``.

:class:`~qsarkit.feature_selection.BorutaSelector` asks a different
question — which features carry more signal than random noise — and answers
with a set of whatever size the data supports:

.. doctest::

   >>> from qsarkit.feature_selection import BorutaSelector
   >>> boruta = BorutaSelector(n_iterations=30, random_state=0).fit(X, y)
   >>> int(boruta.get_support().sum())
   2

.. warning::

   **Select features inside the cross-validation loop.** Choosing columns
   using all the labels and then splitting is selection bias: the choice
   has already seen the test labels, and the held-out score is optimistic
   by an amount you cannot estimate afterwards.

   .. doctest::

      >>> from sklearn.model_selection import cross_val_score
      >>> from sklearn.pipeline import Pipeline
      >>> from qsarkit.feature_selection import MutualInformationSelector
      >>> leaky = MutualInformationSelector(k=20).fit_transform(X, y)
      >>> honest = Pipeline([
      ...     ("select", MutualInformationSelector(k=20)),
      ...     ("model", QSARRegressor("rf", random_state=0)),
      ... ])
      >>> outside = cross_val_score(
      ...     QSARRegressor("rf", random_state=0), leaky, y, cv=5).mean()
      >>> inside = cross_val_score(honest, X, y, cv=5).mean()
      >>> bool(outside > inside)      # the optimism, measured
      True

Hyperparameters
---------------

Tuning on the same folds you report from leaks the test set into the
hyperparameter choice. :class:`~qsarkit.model_selection.NestedCV` keeps
selection and assessment apart:

.. doctest::

   >>> from qsarkit.model_selection import NestedCV
   >>> nested = NestedCV(
   ...     QSARRegressor("ridge"),
   ...     {"model_params": [{"alpha": 0.1}, {"alpha": 1.0}, {"alpha": 10.0}]},
   ...     inner_cv=3, outer_cv=5,
   ... )
   >>> result = nested.run(X, y)
   >>> sorted(result)
   ['best_params', 'mean_score', 'scores', 'std_score']
   >>> len(result["scores"])
   5

Disagreement between the outer folds about the best value is itself the
finding: it says the choice is not well determined by this much data, so a
single "best" hyperparameter reported from one grid search would be noise.

Metrics
-------

.. doctest::

   >>> from qsarkit.metrics import qsar_regression_report
   >>> model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> report = qsar_regression_report(y[test], model.predict(X[test]))
   >>> sorted(report)
   ['average_r2m', 'ccc', 'delta_r2m', 'golbraikh_tropsha', 'mae', 'q2_f2', 'r2', 'rmse']

Prefer Q²F1–F3 to R² on external sets: they are scaled by the *training*
set variance, so they cannot be inflated by a test set that happens to span
a wide activity range. For virtual screening prefer BEDROC to ROC-AUC,
since only the top of the ranked list will ever be tested.

Every regression metric here assumes roughly normal, homoscedastic errors.
When that fails they still compute and quietly mean something else — see
:doc:`oecd` for the diagnostics, and for the full validation picture.

The same thing as a pipe
------------------------

.. doctest::

   >>> from qsarkit.functional import fingerprint, fit, molecules, split
   >>> train_set, test_set = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> fingerprint("morgan", n_bits=512)
   ...     >> split("scaffold", test_size=0.2)
   ... )
   >>> model = train_set >> fit("rf", random_state=0)
   >>> model.predict(test_set.X).shape
   (6,)

References
----------

- Wold, S., Sjöström, M. & Eriksson, L. (2001). "PLS-Regression."
  *Chemom. Intell. Lab. Syst.*, 58(2), 109-130.
  :doi:`10.1016/S0169-7439(01)00155-1`
- Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
  *Neural Netw.*, 18(8), 1093-1110. :doi:`10.1016/j.neunet.2005.07.009`
- Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine
  Learning." *Chem. Sci.*, 9, 513-530. :doi:`10.1039/C7SC02664A`
- Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method for
  Estimating the Goodness of Prospective Prediction." *J. Chem. Inf.
  Model.*, 53(4), 783-790. :doi:`10.1021/ci400084k`
- Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
  Methods: Good and Bad Metrics for the 'Early Recognition' Problem."
  *J. Chem. Inf. Model.*, 47(2), 488-508. :doi:`10.1021/ci600426e`
- Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model
  Selection and Subsequent Selection Bias in Performance Evaluation."
  *J. Mach. Learn. Res.*, 11, 2079-2107.
  https://jmlr.org/papers/v11/cawley10a.html
- Kursa, M. B. & Rudnicki, W. R. (2010). "Feature Selection with the Boruta
  Package." *J. Stat. Softw.*, 36(11), 1-13. :doi:`10.18637/jss.v036.i11`
