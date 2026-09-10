from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from qsarkit.validation import CrossValidator


@pytest.fixture(scope="module")
def data():
    return make_regression(
        n_samples=60, n_features=5, n_informative=3, noise=5.0, random_state=0
    )


class TestCrossValidator:
    def test_report_keys(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(
            Ridge(), X, y
        )
        assert set(result) == {
            "q2", "rmse_cv", "mae_cv", "y_pred_cv", "method", "n_splits"
        }

    def test_out_of_fold_predictions_align_with_input(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(
            Ridge(), X, y
        )
        assert result["y_pred_cv"].shape == y.shape

    def test_q2_is_high_for_a_learnable_problem(self, data):
        X, y = data
        assert CrossValidator(n_splits=5, random_state=0).evaluate(
            Ridge(), X, y
        )["q2"] > 0.8

    def test_q2_is_poor_for_pure_noise(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(60, 5))
        y = rng.normal(size=60)
        # cross-validated Q2 on noise must not look like a real fit; a
        # naive in-sample R2 would happily report a positive number here
        assert CrossValidator(n_splits=5, random_state=0).evaluate(
            Ridge(), X, y
        )["q2"] < 0.3

    def test_q2_matches_a_hand_computation(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(Ridge(), X, y)
        pred = result["y_pred_cv"]
        press = float(np.sum((y - pred) ** 2))
        total = float(np.sum((y - y.mean()) ** 2))
        assert result["q2"] == pytest.approx(1.0 - press / total)

    def test_rmse_matches_a_hand_computation(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(Ridge(), X, y)
        expected = float(np.sqrt(np.mean((y - result["y_pred_cv"]) ** 2)))
        assert result["rmse_cv"] == pytest.approx(expected)

    def test_mae_matches_a_hand_computation(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(Ridge(), X, y)
        expected = float(np.mean(np.abs(y - result["y_pred_cv"])))
        assert result["mae_cv"] == pytest.approx(expected)

    def test_leave_one_out(self, data):
        X, y = data
        result = CrossValidator(method="loo").evaluate(Ridge(), X[:20], y[:20])
        assert result["method"] == "loo"
        assert result["n_splits"] == 20

    def test_repeated_kfold(self, data):
        X, y = data
        result = CrossValidator(
            method="repeated_kfold", n_splits=3, n_repeats=2, random_state=0
        ).evaluate(Ridge(), X, y)
        assert result["method"] == "repeated_kfold"
        assert result["y_pred_cv"].shape == y.shape

    def test_leave_group_out(self, data):
        X, y = data
        groups = np.repeat(np.arange(6), 10)
        result = CrossValidator(method="leave_group_out").evaluate(
            Ridge(), X, y, groups=groups
        )
        assert result["n_splits"] == 6

    def test_leave_group_out_requires_groups(self, data):
        X, y = data
        with pytest.raises(ValueError):
            CrossValidator(method="leave_group_out").evaluate(Ridge(), X, y)

    def test_does_not_mutate_the_estimator(self, data):
        X, y = data
        model = LinearRegression()
        CrossValidator(n_splits=3, random_state=0).evaluate(model, X, y)
        # a fresh clone is fitted per fold, so the original stays unfitted
        assert not hasattr(model, "coef_")

    def test_is_deterministic_under_a_seed(self, data):
        X, y = data
        a = CrossValidator(n_splits=3, random_state=42).evaluate(Ridge(), X, y)
        b = CrossValidator(n_splits=3, random_state=42).evaluate(Ridge(), X, y)
        assert a["q2"] == pytest.approx(b["q2"])

    def test_works_with_a_forest(self, data):
        X, y = data
        result = CrossValidator(n_splits=3, random_state=0).evaluate(
            RandomForestRegressor(n_estimators=10, random_state=0), X, y
        )
        assert np.isfinite(result["q2"])

    def test_rejects_an_unknown_method(self, data):
        X, y = data
        with pytest.raises(ValueError):
            CrossValidator(method="bogus").evaluate(Ridge(), X, y)
