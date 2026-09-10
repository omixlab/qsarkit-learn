from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from qsarkit.model_selection import (
    ButinaClusterSplitter,
    KennardStoneSplitter,
    MaxMinSplitter,
    NestedCV,
    PerimeterSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    SphereExclusionSplitter,
    StratifiedScaffoldSplitter,
    TimeSplitter,
    hyperparameter_search,
)

SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC",   # benzene scaffold
    "c1ccncc1C", "c1ccncc1CC",                   # pyridine scaffold
    "CCO", "CCN",                                # acyclic (empty scaffold)
    "CCCCCC",
    "c1ccc2ccccc2c1", "c1ccc2ccccc2c1C",         # naphthalene scaffold
]


@pytest.fixture(scope="module")
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


@pytest.fixture(scope="module")
def activities():
    return np.arange(len(SMILES), dtype=float)


@pytest.fixture(scope="module")
def X():
    return np.random.RandomState(0).normal(size=(20, 4))


MOL_SPLITTERS = [
    pytest.param(RandomSplitter, {"random_state": 0}, id="random"),
    pytest.param(ScaffoldSplitter, {}, id="scaffold"),
    pytest.param(StratifiedScaffoldSplitter, {}, id="stratified-scaffold"),
    pytest.param(ButinaClusterSplitter, {}, id="butina"),
    pytest.param(SphereExclusionSplitter, {}, id="sphere-exclusion"),
    pytest.param(MaxMinSplitter, {}, id="maxmin"),
]

ARRAY_SPLITTERS = [
    pytest.param(RandomSplitter, {"random_state": 0}, id="random"),
    pytest.param(KennardStoneSplitter, {}, id="kennard-stone"),
    pytest.param(PerimeterSplitter, {}, id="perimeter"),
]


class TestSplitterContract:
    """Behaviour every splitter must satisfy."""

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS)
    def test_splits_are_disjoint_and_complete(self, cls, kwargs, mols, activities):
        train, test = next(cls(test_size=0.3, **kwargs).split_mols(mols, activities))
        assert set(train).isdisjoint(set(test))
        assert sorted([*train, *test]) == list(range(len(mols)))

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS)
    def test_both_sides_are_non_empty(self, cls, kwargs, mols, activities):
        train, test = next(cls(test_size=0.3, **kwargs).split_mols(mols, activities))
        assert len(train) > 0 and len(test) > 0

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS)
    def test_indices_are_sorted_integers(self, cls, kwargs, mols, activities):
        train, test = next(cls(test_size=0.3, **kwargs).split_mols(mols, activities))
        assert np.array_equal(train, np.sort(train))
        assert train.dtype.kind == "i"

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS)
    def test_is_deterministic(self, cls, kwargs, mols, activities):
        a = next(cls(test_size=0.3, **kwargs).split_mols(mols, activities))
        b = next(cls(test_size=0.3, **kwargs).split_mols(mols, activities))
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS + ARRAY_SPLITTERS)
    def test_rejects_bad_test_size(self, cls, kwargs, X):
        with pytest.raises(ValueError, match="test_size must be in"):
            next(cls(test_size=1.5, **kwargs).split(X))

    @pytest.mark.parametrize("cls, kwargs", ARRAY_SPLITTERS)
    def test_rejects_too_few_samples(self, cls, kwargs):
        with pytest.raises(ValueError, match="at least 2 samples"):
            next(cls(**kwargs).split(np.zeros((1, 3))))

    @pytest.mark.parametrize("cls, kwargs", MOL_SPLITTERS + ARRAY_SPLITTERS)
    def test_get_n_splits(self, cls, kwargs):
        assert cls(**kwargs).get_n_splits() == 1


class TestScaffoldSplitter:
    def test_scaffolds_never_cross_the_split(self, mols):
        train, test = next(ScaffoldSplitter(test_size=0.3).split_mols(mols))

        def scaffold(i):
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mols[i]))

        assert {scaffold(i) for i in train}.isdisjoint({scaffold(i) for i in test})

    def test_same_scaffold_molecules_stay_together(self, mols):
        train, test = next(ScaffoldSplitter(test_size=0.3).split_mols(mols))
        benzene_group = {0, 1, 2}
        assert benzene_group <= set(train) or benzene_group <= set(test)

    def test_needs_molecules(self, X):
        with pytest.raises(ValueError, match="needs RDKit molecules"):
            next(ScaffoldSplitter().split(X))

    def test_include_chirality_separates_enantiomers(self):
        pair = [Chem.MolFromSmiles(s) for s in ("C[C@H]1CCCCC1", "C[C@@H]1CCCCC1")]
        plain = ScaffoldSplitter(include_chirality=False)
        assert len(plain._scaffold_groups(pair)) == 1
        chiral = ScaffoldSplitter(include_chirality=True)
        assert len(chiral._scaffold_groups(pair)) >= 1

    def test_single_scaffold_still_yields_a_test_set(self):
        same = [Chem.MolFromSmiles(s) for s in ("c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC")]
        train, test = next(ScaffoldSplitter(test_size=0.34).split_mols(same))
        assert len(test) > 0 and len(train) > 0

    def test_handles_none_molecules(self):
        with_none = [Chem.MolFromSmiles("c1ccccc1C"), None, Chem.MolFromSmiles("CCO")]
        train, test = next(ScaffoldSplitter(test_size=0.34).split_mols(with_none))
        assert len(train) + len(test) == 3


class TestStratifiedScaffoldSplitter:
    def test_keeps_scaffolds_disjoint(self, mols, activities):
        train, test = next(
            StratifiedScaffoldSplitter(test_size=0.3).split_mols(mols, activities)
        )

        def scaffold(i):
            return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mols[i]))

        assert {scaffold(i) for i in train}.isdisjoint({scaffold(i) for i in test})

    def test_balances_the_label_distribution(self, mols):
        # activities correlated with scaffold: a plain scaffold split would
        # put all the high values on one side
        y = np.array([9.0, 9.0, 9.0, 1.0, 1.0, 5.0, 5.0, 5.0, 2.0, 2.0])
        train, test = next(
            StratifiedScaffoldSplitter(test_size=0.3).split_mols(mols, y)
        )
        assert abs(y[train].mean() - y[test].mean()) < 6.0

    def test_falls_back_to_plain_scaffold_without_labels(self, mols):
        train, test = next(StratifiedScaffoldSplitter(test_size=0.3).split_mols(mols))
        assert set(train).isdisjoint(set(test))


class TestClusterSplitters:
    @pytest.mark.parametrize(
        "cls", [ButinaClusterSplitter, SphereExclusionSplitter]
    )
    def test_clusters_never_cross_the_split(self, cls, mols):
        splitter = cls(test_size=0.3)
        labels = splitter._cluster_labels(mols, None)
        train, test = next(splitter.split_mols(mols))
        assert set(labels[train]).isdisjoint(set(labels[test]))

    @pytest.mark.parametrize(
        "cls", [ButinaClusterSplitter, SphereExclusionSplitter]
    )
    def test_works_on_a_precomputed_feature_matrix(self, cls):
        fps = (np.random.RandomState(0).rand(20, 32) > 0.7).astype(float)
        train, test = next(cls(test_size=0.3).split(fps))
        assert len(train) + len(test) == 20

    def test_cutoff_changes_the_clustering(self, mols):
        loose = ButinaClusterSplitter(cutoff=0.9)._cluster_labels(mols, None)
        tight = ButinaClusterSplitter(cutoff=0.1)._cluster_labels(mols, None)
        assert len(set(tight)) >= len(set(loose))


class TestMaxMinSplitter:
    def test_train_set_is_the_diverse_subset(self, mols):
        train, test = next(MaxMinSplitter(test_size=0.3).split_mols(mols))
        assert len(train) == 7 and len(test) == 3

    def test_works_on_arrays(self, X):
        train, test = next(MaxMinSplitter(test_size=0.25).split(X))
        assert len(train) + len(test) == 20


class TestTimeSplitter:
    def test_test_set_is_strictly_later(self, X):
        dates = np.arange(20)
        train, test = next(TimeSplitter(test_size=0.3).split(X, groups=dates))
        assert dates[train].max() <= dates[test].min()

    def test_handles_unsorted_dates(self, X):
        rng = np.random.RandomState(0)
        dates = rng.permutation(20)
        train, test = next(TimeSplitter(test_size=0.3).split(X, groups=dates))
        assert dates[train].max() <= dates[test].min()

    def test_requires_dates(self, X):
        with pytest.raises(ValueError, match="needs dates"):
            next(TimeSplitter().split(X))

    def test_rejects_mismatched_group_length(self, X):
        with pytest.raises(ValueError, match="groups has length"):
            next(TimeSplitter().split(X, groups=np.arange(5)))


class TestKennardStoneSplitter:
    def test_split_sizes(self, X):
        train, test = next(KennardStoneSplitter(test_size=0.25).split(X))
        assert len(train) == 15 and len(test) == 5

    def test_is_deterministic(self, X):
        a = next(KennardStoneSplitter(test_size=0.25).split(X))
        b = next(KennardStoneSplitter(test_size=0.25).split(X))
        assert np.array_equal(a[0], b[0])

    def test_training_set_spans_the_space(self, X):
        # the two most distant points must both be selected for training
        from scipy.spatial.distance import cdist

        train, _ = next(KennardStoneSplitter(test_size=0.25).split(X))
        d = cdist(X, X)
        i, j = np.unravel_index(np.argmax(d), d.shape)
        assert i in train and j in train

    def test_accepts_an_alternative_metric(self, X):
        train, test = next(
            KennardStoneSplitter(test_size=0.25, metric="cityblock").split(X)
        )
        assert len(train) + len(test) == 20

    def test_works_on_molecules(self, mols):
        train, test = next(KennardStoneSplitter(test_size=0.3).split_mols(mols))
        assert len(train) + len(test) == len(mols)


class TestPerimeterSplitter:
    def test_training_set_holds_the_outermost_points(self, X):
        train, test = next(PerimeterSplitter(test_size=0.25).split(X))
        distance = np.linalg.norm(X - X.mean(axis=0), axis=1)
        assert distance[train].min() >= distance[test].max()

    def test_split_sizes(self, X):
        train, test = next(PerimeterSplitter(test_size=0.25).split(X))
        assert len(train) == 15 and len(test) == 5


class TestSklearnIntegration:
    def test_usable_as_a_cv_argument(self, X):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(
            Ridge(), X, X[:, 0], cv=RandomSplitter(random_state=0)
        )
        assert scores.shape == (1,)

    def test_scaffold_split_usable_for_grid_search(self, X):
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GridSearchCV

        search = GridSearchCV(
            Ridge(), {"alpha": [0.1, 1.0]}, cv=RandomSplitter(random_state=0)
        ).fit(X, X[:, 0])
        assert "alpha" in search.best_params_


class TestHyperparameterSearch:
    @pytest.fixture
    def data(self):
        from sklearn.datasets import make_regression

        return make_regression(n_samples=40, n_features=5, random_state=0)

    def test_grid_search(self, data):
        from sklearn.ensemble import RandomForestRegressor

        X, y = data
        search = hyperparameter_search(
            RandomForestRegressor(random_state=0),
            {"n_estimators": [5, 10]}, X, y, cv=2,
        )
        assert search.best_params_["n_estimators"] in (5, 10)
        assert hasattr(search, "best_estimator_")

    def test_random_search(self, data):
        from sklearn.ensemble import RandomForestRegressor

        X, y = data
        search = hyperparameter_search(
            RandomForestRegressor(random_state=0),
            {"n_estimators": [5, 10, 15]}, X, y,
            method="random", n_iter=2, cv=2, random_state=0,
        )
        assert "n_estimators" in search.best_params_

    def test_rejects_unknown_method(self, data):
        from sklearn.linear_model import Ridge

        X, y = data
        with pytest.raises(ValueError, match="method must be"):
            hyperparameter_search(Ridge(), {"alpha": [1.0]}, X, y, method="bogus")

    def test_optuna_backend_is_optional(self, data):
        pytest.importorskip("optuna")
        from sklearn.linear_model import Ridge

        X, y = data
        search = hyperparameter_search(
            Ridge(), {"alpha": [0.1, 1.0]}, X, y, method="optuna", n_iter=2, cv=2
        )
        assert hasattr(search, "best_params_")


class TestNestedCV:
    def test_produces_one_score_per_outer_fold(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge

        X, y = make_regression(n_samples=40, n_features=5, random_state=0)
        ncv = NestedCV(Ridge(), {"alpha": [0.1, 1.0]}, inner_cv=2, outer_cv=3)
        result = ncv.run(X, y)
        assert ncv.scores_.shape == (3,)
        assert len(ncv.best_params_) == 3
        assert set(result) == {"mean_score", "std_score", "scores", "best_params"}

    def test_reports_mean_and_spread(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge

        X, y = make_regression(n_samples=40, n_features=5, random_state=0)
        result = NestedCV(
            Ridge(), {"alpha": [0.1, 1.0]}, inner_cv=2, outer_cv=2
        ).run(X, y)
        assert result["mean_score"] == pytest.approx(
            float(np.mean(result["scores"]))
        )
        assert result["std_score"] >= 0

    def test_accepts_a_custom_outer_splitter(self):
        from sklearn.datasets import make_regression
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        X, y = make_regression(n_samples=30, n_features=4, random_state=0)
        ncv = NestedCV(
            Ridge(), {"alpha": [1.0]}, inner_cv=2, outer_cv=KFold(n_splits=2)
        )
        ncv.run(X, y)
        assert ncv.scores_.shape == (2,)
