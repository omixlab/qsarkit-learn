from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from qsarkit.feature_selection import (
    BorutaSelector,
    CorrelationFilter,
    MutualInformationSelector,
    RFESelector,
    VarianceFilter,
)


@pytest.fixture(scope="module")
def regression():
    # shuffle=False puts the informative features first, so a selector
    # that works must prefer the leading columns
    return make_regression(
        n_samples=120, n_features=10, n_informative=3,
        random_state=0, shuffle=False,
    )


@pytest.fixture(scope="module")
def classification():
    return make_classification(
        n_samples=120, n_features=10, n_informative=3, n_redundant=0,
        random_state=0, shuffle=False,
    )


ALL_SELECTORS = [
    pytest.param(VarianceFilter, {}, id="variance"),
    pytest.param(CorrelationFilter, {}, id="correlation"),
    pytest.param(MutualInformationSelector, {"k": 3}, id="mutual-info"),
    pytest.param(RFESelector, {"n_features_to_select": 3}, id="rfe"),
    pytest.param(
        BorutaSelector, {"n_iterations": 12, "random_state": 0}, id="boruta"
    ),
]


class TestSelectorContract:
    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_support_mask_has_one_entry_per_feature(self, cls, kwargs, regression):
        X, y = regression
        support = cls(**kwargs).fit(X, y).get_support()
        assert support.shape == (X.shape[1],)
        assert support.dtype == bool

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_transform_width_matches_the_mask(self, cls, kwargs, regression):
        X, y = regression
        selector = cls(**kwargs).fit(X, y)
        assert selector.transform(X).shape[1] == selector.get_support().sum()

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_transform_preserves_row_count(self, cls, kwargs, regression):
        X, y = regression
        assert cls(**kwargs).fit(X, y).transform(X).shape[0] == X.shape[0]

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_keeps_at_least_one_feature(self, cls, kwargs, regression):
        X, y = regression
        assert cls(**kwargs).fit(X, y).get_support().sum() >= 1

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_support_indices(self, cls, kwargs, regression):
        X, y = regression
        selector = cls(**kwargs).fit(X, y)
        indices = selector.get_support(indices=True)
        assert set(indices.tolist()) == set(
            np.flatnonzero(selector.get_support()).tolist()
        )

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_clone_roundtrip(self, cls, kwargs):
        selector = cls(**kwargs)
        assert clone(selector).get_params() == selector.get_params()

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_works_inside_a_pipeline(self, cls, kwargs, regression):
        X, y = regression
        pipe = Pipeline([("select", cls(**kwargs)), ("model", Ridge())]).fit(X, y)
        assert pipe.predict(X).shape == y.shape

    @pytest.mark.parametrize("cls, kwargs", ALL_SELECTORS)
    def test_is_deterministic(self, cls, kwargs, regression):
        X, y = regression
        a = cls(**kwargs).fit(X, y).get_support()
        b = cls(**kwargs).fit(X, y).get_support()
        assert np.array_equal(a, b)


class TestInformativeFeatureRecovery:
    """Each selector must prefer the genuinely informative columns."""

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            pytest.param(MutualInformationSelector, {"k": 3}, id="mutual-info"),
            pytest.param(RFESelector, {"n_features_to_select": 3}, id="rfe"),
            pytest.param(
                BorutaSelector, {"n_iterations": 25, "random_state": 0}, id="boruta"
            ),
        ],
    )
    def test_selects_mostly_informative_features(self, cls, kwargs, regression):
        X, y = regression
        selected = set(cls(**kwargs).fit(X, y).get_support(indices=True).tolist())
        informative = {0, 1, 2}
        # at least two of the three real signals must survive
        assert len(selected & informative) >= 2


class TestVarianceFilter:
    def test_drops_constant_columns(self):
        X = np.column_stack([np.arange(20.0), np.full(20, 5.0)])
        y = np.arange(20.0)
        assert VarianceFilter().fit(X, y).get_support().tolist() == [True, False]

    def test_threshold_controls_strictness(self):
        X = np.column_stack([np.arange(20.0), np.arange(20.0) * 0.01])
        y = np.arange(20.0)
        assert VarianceFilter(threshold=1.0).fit(X, y).get_support().sum() == 1

    def test_keeps_everything_by_default_on_varied_data(self, regression):
        X, y = regression
        assert VarianceFilter().fit(X, y).get_support().all()


class TestCorrelationFilter:
    def test_drops_a_duplicated_column(self):
        base = np.random.RandomState(0).normal(size=(60, 3))
        X = np.column_stack([base, base[:, 0]])  # column 3 duplicates column 0
        y = base[:, 0] * 2
        support = CorrelationFilter(threshold=0.95).fit(X, y).get_support()
        assert support.sum() == 3
        # exactly one of the correlated pair survives
        assert support[0] != support[3]

    def test_keeps_uncorrelated_columns(self):
        X = np.random.RandomState(0).normal(size=(80, 4))
        y = X[:, 0]
        assert CorrelationFilter(threshold=0.95).fit(X, y).get_support().all()

    def test_spearman_method(self):
        base = np.random.RandomState(0).normal(size=(60, 3))
        X = np.column_stack([base, base[:, 0] ** 3])  # monotonic, not linear
        y = base[:, 0]
        support = CorrelationFilter(
            threshold=0.9, method="spearman"
        ).fit(X, y).get_support()
        assert support.sum() == 3

    def test_lower_threshold_drops_more(self, regression):
        X, y = regression
        strict = CorrelationFilter(threshold=0.1).fit(X, y).get_support().sum()
        loose = CorrelationFilter(threshold=0.99).fit(X, y).get_support().sum()
        assert strict <= loose

    def test_correlation_matrix_is_square_and_symmetric(self, regression):
        X, y = regression
        selector = CorrelationFilter().fit(X, y)
        matrix = selector.correlation_matrix_
        assert matrix.shape == (X.shape[1], X.shape[1])
        assert np.allclose(matrix, matrix.T)


class TestMutualInformationSelector:
    def test_k_controls_the_count(self, regression):
        X, y = regression
        assert MutualInformationSelector(k=4).fit(X, y).get_support().sum() == 4

    def test_percentile_mode(self, regression):
        X, y = regression
        selector = MutualInformationSelector(percentile=50.0, random_state=0)
        assert 1 <= selector.fit(X, y).get_support().sum() <= X.shape[1]

    def test_classification_task(self, classification):
        X, y = classification
        selector = MutualInformationSelector(
            task="classification", k=3, random_state=0
        ).fit(X, y)
        assert selector.get_support().sum() == 3

    def test_k_larger_than_the_feature_count_is_clamped(self, regression):
        X, y = regression
        selector = MutualInformationSelector(k=99).fit(X, y)
        assert selector.get_support().sum() == X.shape[1]

    def test_scores_are_non_negative(self, regression):
        X, y = regression
        assert (MutualInformationSelector(k=3).fit(X, y).scores_ >= 0).all()


class TestRFESelector:
    def test_selects_the_requested_count(self, regression):
        X, y = regression
        assert RFESelector(n_features_to_select=4).fit(X, y).get_support().sum() == 4

    def test_fraction_of_features(self, regression):
        X, y = regression
        selector = RFESelector(n_features_to_select=0.3).fit(X, y)
        assert selector.get_support().sum() == 3

    def test_custom_estimator(self, regression):
        X, y = regression
        selector = RFESelector(
            estimator=Ridge(), n_features_to_select=3
        ).fit(X, y)
        assert selector.get_support().sum() == 3

    def test_classification_task(self, classification):
        X, y = classification
        selector = RFESelector(
            task="classification", n_features_to_select=3, random_state=0
        ).fit(X, y)
        assert selector.get_support().sum() == 3

    def test_step_larger_than_one(self, regression):
        X, y = regression
        selector = RFESelector(n_features_to_select=3, step=2).fit(X, y)
        assert selector.get_support().sum() == 3


class TestBorutaSelector:
    def test_rejects_pure_noise_features(self):
        # only the first two columns carry signal; Boruta should drop the rest
        rng = np.random.RandomState(0)
        signal = rng.normal(size=(150, 2))
        noise = rng.normal(size=(150, 8))
        X = np.column_stack([signal, noise])
        y = signal[:, 0] * 3 + signal[:, 1] * 2
        selected = set(
            BorutaSelector(n_iterations=40, random_state=0)
            .fit(X, y)
            .get_support(indices=True)
            .tolist()
        )
        assert {0, 1} <= selected
        # most of the eight noise columns must be rejected
        assert len(selected & set(range(2, 10))) <= 3

    def test_classification_task(self, classification):
        X, y = classification
        selector = BorutaSelector(
            task="classification", n_iterations=12, random_state=0
        ).fit(X, y)
        assert selector.get_support().shape == (X.shape[1],)

    def test_include_tentative_keeps_at_least_as_many(self, regression):
        X, y = regression
        strict = BorutaSelector(
            n_iterations=10, random_state=0, include_tentative=False
        ).fit(X, y).get_support().sum()
        loose = BorutaSelector(
            n_iterations=10, random_state=0, include_tentative=True
        ).fit(X, y).get_support().sum()
        assert loose >= strict

    def test_custom_estimator(self, regression):
        from sklearn.ensemble import RandomForestRegressor

        X, y = regression
        selector = BorutaSelector(
            estimator=RandomForestRegressor(n_estimators=20, random_state=0),
            n_iterations=12, random_state=0,
        ).fit(X, y)
        assert selector.get_support().shape == (X.shape[1],)

    def test_warns_when_too_few_iterations_to_confirm_anything(self, regression):
        # The Bonferroni-corrected binomial test cannot reach significance
        # with fewer trials than log2(2 * n_features / alpha); returning an
        # empty selection silently would break any downstream pipeline.
        X, y = regression
        with pytest.warns(UserWarning, match="confirmed no features"):
            selector = BorutaSelector(n_iterations=4, random_state=0).fit(X, y)
        assert selector.get_support().sum() == 0

    def test_minimum_iterations_matches_the_formula(self):
        from math import ceil, log2

        selector = BorutaSelector(alpha=0.05)
        assert selector._minimum_iterations(10) == int(ceil(log2(2 * 10 / 0.05)))

    def test_more_iterations_do_not_break_it(self, regression):
        X, y = regression
        selector = BorutaSelector(n_iterations=30, random_state=0).fit(X, y)
        assert selector.get_support().sum() >= 1
