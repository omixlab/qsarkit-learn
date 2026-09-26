"""Tests for the OECD principle-4 robustness and predictivity checks."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

from qsarkit.base import ModelNotFittedError
from qsarkit.validation import BootstrapValidator, ExternalValidator, YScrambling


@pytest.fixture(scope="module")
def signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.2, size=60)
    return X, y


@pytest.fixture(scope="module")
def noise():
    rng = np.random.default_rng(1)
    return rng.normal(size=(60, 4)), rng.normal(size=60)


class TestYScrambling:
    def test_real_signal_beats_scrambled_labels(self, signal):
        X, y = signal
        result = YScrambling(n_iterations=50, random_state=0).run(Ridge(), X, y)
        assert result["p_value"] < 0.05
        assert result["real_score"] > result["max_scrambled_score"]

    def test_noise_does_not(self, noise):
        X, y = noise
        result = YScrambling(n_iterations=50, random_state=0).run(Ridge(), X, y)
        assert result["p_value"] > 0.05

    def test_an_overfitting_model_on_noise_is_still_caught(self, noise):
        # The point of the test: a deep tree fits noise perfectly, so its
        # real score is high — but so are its scrambled scores, and the
        # p-value is what exposes it.
        X, y = noise
        result = YScrambling(n_iterations=30, random_state=0).run(
            DecisionTreeRegressor(random_state=0), X, y)
        assert result["real_score"] > 0.9        # looks excellent
        assert result["p_value"] > 0.05          # but is meaningless

    def test_reports_every_documented_key(self, signal):
        X, y = signal
        result = YScrambling(n_iterations=10, random_state=0).run(Ridge(), X, y)
        assert set(result) == {
            "real_score", "mean_scrambled_score", "std_scrambled_score",
            "max_scrambled_score", "best_scrambled_score", "p_value",
            "n_iterations", "metric", "scored_out_of_fold",
        }

    def test_p_value_is_never_zero(self, signal):
        # A permutation test cannot distinguish "very unlikely" from
        # "impossible"; reporting 0 would claim more than was measured.
        X, y = signal
        result = YScrambling(n_iterations=20, random_state=0).run(Ridge(), X, y)
        assert result["p_value"] == pytest.approx(1 / 21)
        assert result["p_value"] > 0

    def test_is_reproducible(self, signal):
        X, y = signal
        a = YScrambling(n_iterations=10, random_state=42).run(Ridge(), X, y)
        b = YScrambling(n_iterations=10, random_state=42).run(Ridge(), X, y)
        assert a == b

    def test_does_not_mutate_the_estimator(self, signal):
        X, y = signal
        model = Ridge()
        YScrambling(n_iterations=5, random_state=0).run(model, X, y)
        assert not hasattr(model, "coef_")

    def test_rejects_zero_iterations(self, signal):
        X, y = signal
        with pytest.raises(ValueError, match="n_iterations"):
            YScrambling(n_iterations=0).run(Ridge(), X, y)

    def test_stores_the_scores(self, signal):
        X, y = signal
        scrambler = YScrambling(n_iterations=12, random_state=0)
        scrambler.run(Ridge(), X, y)
        assert scrambler.scrambled_scores_.shape == (12,)

    def test_plot_returns_a_figure(self, signal):
        X, y = signal
        scrambler = YScrambling(n_iterations=10, random_state=0)
        scrambler.run(Ridge(), X, y)
        assert scrambler.plot().__class__.__name__ == "Figure"

    def test_plot_before_run_is_an_error(self):
        with pytest.raises(ModelNotFittedError, match="run"):
            YScrambling().plot()


class TestExternalValidator:
    def test_reports_the_standard_statistics(self, signal):
        X, y = signal
        model = Ridge().fit(X[:45], y[:45])
        report = ExternalValidator().validate(model, X[45:], y[45:])
        assert {"r2", "rmse", "mae", "ccc", "golbraikh_tropsha", "n_test"} <= set(report)
        assert report["n_test"] == 15

    def test_q2_f1_needs_the_training_activities(self, signal):
        X, y = signal
        model = Ridge().fit(X[:45], y[:45])
        without = ExternalValidator().validate(model, X[45:], y[45:])
        with_train = ExternalValidator().validate(model, X[45:], y[45:], y[:45])
        assert "q2_f1" not in without
        assert "q2_f1" in with_train

    def test_golbraikh_tropsha_criterion_1_needs_q2(self, signal):
        X, y = signal
        model = Ridge().fit(X[:45], y[:45])
        without = ExternalValidator().validate(model, X[45:], y[45:])
        assert without["golbraikh_tropsha"]["q2_available"] is False

        supplied = ExternalValidator(q2=0.9).validate(model, X[45:], y[45:])
        assert supplied["golbraikh_tropsha"]["q2_available"] is True

    def test_a_good_model_passes(self, signal):
        X, y = signal
        model = Ridge().fit(X[:45], y[:45])
        report = ExternalValidator(q2=0.9).validate(model, X[45:], y[45:])
        assert report["golbraikh_tropsha"]["passed"]

    def test_a_noise_model_does_not(self, noise):
        X, y = noise
        model = Ridge().fit(X[:45], y[:45])
        report = ExternalValidator(q2=0.0).validate(model, X[45:], y[45:])
        assert not report["golbraikh_tropsha"]["passed"]


class TestBootstrapValidator:
    def test_estimates_a_score_with_an_interval(self, signal):
        X, y = signal
        result = BootstrapValidator(n_iterations=25, random_state=0).run(Ridge(), X, y)
        assert result["mean_score"] > 0.9
        assert result["ci_lower"] <= result["mean_score"] <= result["ci_upper"]

    def test_noise_scores_near_zero_or_below(self, noise):
        X, y = noise
        result = BootstrapValidator(n_iterations=25, random_state=0).run(Ridge(), X, y)
        assert result["mean_score"] < 0.3

    def test_reports_every_documented_key(self, signal):
        X, y = signal
        result = BootstrapValidator(n_iterations=10, random_state=0).run(Ridge(), X, y)
        assert set(result) == {
            "mean_score", "std_score", "ci_lower", "ci_upper",
            "confidence", "n_iterations", "n_effective", "metric",
        }

    def test_confidence_widens_the_interval(self, signal):
        X, y = signal
        validator = BootstrapValidator(n_iterations=40, random_state=0)
        narrow = validator.run(Ridge(), X, y, confidence=0.5)
        wide = validator.run(Ridge(), X, y, confidence=0.99)
        assert (wide["ci_upper"] - wide["ci_lower"]) >= (
            narrow["ci_upper"] - narrow["ci_lower"])

    def test_scores_out_of_bag_not_in_bag(self, signal):
        # Scoring in-bag would report fit, not generalization, and would sit
        # far above the honest figure.
        X, y = signal
        result = BootstrapValidator(n_iterations=25, random_state=0).run(Ridge(), X, y)
        in_bag = Ridge().fit(X, y).score(X, y)
        assert result["mean_score"] <= in_bag

    def test_is_reproducible(self, signal):
        X, y = signal
        a = BootstrapValidator(n_iterations=10, random_state=7).run(Ridge(), X, y)
        b = BootstrapValidator(n_iterations=10, random_state=7).run(Ridge(), X, y)
        assert a == b

    def test_does_not_mutate_the_estimator(self, signal):
        X, y = signal
        model = Ridge()
        BootstrapValidator(n_iterations=5, random_state=0).run(model, X, y)
        assert not hasattr(model, "coef_")

    def test_rejects_bad_arguments(self, signal):
        X, y = signal
        with pytest.raises(ValueError, match="n_iterations"):
            BootstrapValidator(n_iterations=0).run(Ridge(), X, y)
        with pytest.raises(ValueError, match="confidence"):
            BootstrapValidator(n_iterations=5).run(Ridge(), X, y, confidence=1.5)

    def test_too_small_a_dataset_is_reported(self):
        # With 2 samples a resample almost always covers both, leaving no
        # out-of-bag set; that must be reported, not silently averaged away.
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="too small"):
            BootstrapValidator(n_iterations=5, random_state=0).run(Ridge(), X, y)

    def test_stores_the_scores(self, signal):
        X, y = signal
        validator = BootstrapValidator(n_iterations=12, random_state=0)
        validator.run(Ridge(), X, y)
        assert validator.scores_.shape[0] == 12
