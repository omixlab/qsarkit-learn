Model selection
===============

.. currentmodule:: qsarkit.model_selection

QSAR-aware data splitting, nested cross-validation and hyperparameter
search.

How you split is a bigger decision than which model you fit. A random
split of a QSAR dataset measures interpolation: public sets are dense
with near-duplicate analogues, so random assignment scatters a congeneric
series across both sides and scores the model on compounds whose close
relatives it has memorized.

The splitters
-------------

Every splitter yields ``(train_idx, test_idx)`` and follows the
scikit-learn protocol, so they drop into ``cross_val_score`` unchanged:

.. doctest::

   >>> from qsarkit.model_selection import ScaffoldSplitter
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> train, test = next(ScaffoldSplitter(test_size=0.25).split_mols(demo_mols, y))
   >>> len(train), len(test)
   (18, 6)

The guarantee is that no scaffold appears on both sides:

.. doctest::

   >>> from qsarkit.chemspace import bemis_murcko_smiles
   >>> train_scaffolds = {bemis_murcko_smiles(demo_mols[i]) for i in train}
   >>> test_scaffolds = {bemis_murcko_smiles(demo_mols[i]) for i in test}
   >>> train_scaffolds & test_scaffolds
   set()

``split_mols`` takes molecules; ``split`` takes the matrix and accepts
molecules as ``groups`` for the chemistry-aware splitters:

.. doctest::

   >>> from qsarkit.model_selection import (
   ...     ButinaClusterSplitter, KennardStoneSplitter, RandomSplitter)
   >>> len(next(RandomSplitter(test_size=0.25, random_state=0).split(X, y))[1])
   6
   >>> len(next(ButinaClusterSplitter(test_size=0.25).split_mols(demo_mols, y))[1])
   6
   >>> len(next(KennardStoneSplitter(test_size=0.25).split(X, y))[1])
   6

Which to use:

``ScaffoldSplitter``
   The default hard split. Keeps whole chemotypes together, so the test
   set is chemistry the model has never seen. Deterministic — no seed.
``StratifiedScaffoldSplitter``
   The same, preserving class balance. Use for classification when the
   actives are scarce.
``ButinaClusterSplitter`` / ``SphereExclusionSplitter``
   Split by similarity cluster rather than scaffold. Catches analogue
   series that share no Bemis-Murcko framework.
``KennardStoneSplitter`` / ``PerimeterSplitter``
   Deterministic coverage-driven selection. Kennard-Stone puts the most
   spread-out compounds in *training*, which is what you want when
   training data is precious and the test set need only be representative.
``TimeSplitter``
   Train on early compounds, test on later ones. The only split that
   simulates prospective use, and reliably the most pessimistic.
``RandomSplitter``
   Included for baselines and for the comparison that shows how much the
   others cost you.

Cross-validation and search
---------------------------

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.validation import CrossValidator
   >>> report = CrossValidator(n_splits=3, random_state=0).evaluate(
   ...     QSARRegressor("rf", random_state=0), X, y)
   >>> round(report["q2"], 2)
   0.61

Compare that with the training R² of 0.952 from :doc:`models`: the gap is
the size of the illusion, and it is the honest number.

.. doctest::

   >>> from qsarkit.model_selection import hyperparameter_search
   >>> search = hyperparameter_search(
   ...     QSARRegressor("rf", random_state=0),
   ...     {"model_params": [{"n_estimators": 10}, {"n_estimators": 50}]},
   ...     X, y, cv=3,
   ... )
   >>> sorted(search.best_params_["model_params"])
   ['n_estimators']

:class:`NestedCV` separates model selection from model assessment. Tuning
on the same folds you report scores from leaks the test set into the
choice of hyperparameters, and the reported figure is optimistic by an
amount nobody can estimate after the fact:

.. doctest::

   >>> from qsarkit.model_selection import NestedCV
   >>> nested = NestedCV(
   ...     QSARRegressor("ridge"),
   ...     {"model_params": [{"alpha": 0.1}, {"alpha": 1.0}]},
   ...     inner_cv=2, outer_cv=3,
   ... )
   >>> result = nested.run(X, y)
   >>> sorted(result)
   ['best_params', 'mean_score', 'scores', 'std_score']
   >>> len(result["scores"]), len(result["best_params"])
   (3, 3)

Each outer fold reports the hyperparameters chosen on its *own* inner
folds, so disagreement between them is itself informative — it says the
choice is not well determined by this much data:

.. doctest::

   >>> len({str(p) for p in result["best_params"]}) > 1
   True

API
---

.. automodule:: qsarkit.model_selection
   :members:
   :show-inheritance:

References
----------

- Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs. 1.
  Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
  :doi:`10.1021/jm9602928`
- Sheridan, R. P. (2013). "Time-Split Cross-Validation as a Method for
  Estimating the Goodness of Prospective Prediction." J. Chem. Inf.
  Model., 53(4), 783-790. :doi:`10.1021/ci400084k`
- Kennard, R. W. & Stone, L. A. (1969). "Computer Aided Design of
  Experiments." Technometrics, 11(1), 137-148.
  :doi:`10.1080/00401706.1969.10490666`
- Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine
  Learning." Chem. Sci., 9, 513-530. :doi:`10.1039/C7SC02664A`
- Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model
  Selection and Subsequent Selection Bias in Performance Evaluation."
  J. Mach. Learn. Res., 11, 2079-2107.
  https://jmlr.org/papers/v11/cawley10a.html
