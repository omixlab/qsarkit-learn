"""Metric selection, shared by every validator in :mod:`qsarkit.validation`.

The behaviour under test is the contract stated in the module docstring: one
metric gives a float, several give an array in the order requested, a loss is
compared the right way round, and a probability metric receives probabilities
rather than labels.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from qsarkit.metrics import rmse, roc_auc
from qsarkit.models import QSARClassifier, QSARRegressor
from qsarkit.validation import (
    BootstrapValidator,
    CrossValidator,
    ExternalValidator,
    YScrambling,
    available_metrics,
    make_scorer,
)


@pytest.fixture(scope="module")
def regression():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5))
    y = X[:, 0] * 3.0 + rng.normal(scale=0.3, size=80)
    return X, y


@pytest.fixture(scope="module")
def classification():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=150) > 0.7).astype(int)
    return X, y


class TestMetricResolution:
    def test_available_metrics_covers_both_problem_types(self):
        names = available_metrics()
        assert {"r2", "rmse", "mae"} <= set(names)
        assert {"roc_auc", "pr_auc", "mcc", "balanced_accuracy"} <= set(names)

    def test_an_unknown_name_says_what_is_available(self, regression):
        X, y = regression
        with pytest.raises(ValueError, match="Unknown metric 'nope'"):
            CrossValidator(scoring="nope").evaluate(Ridge(), X, y)

    def test_an_empty_iterable_is_refused(self, regression):
        X, y = regression
        with pytest.raises(ValueError, match="scoring is empty"):
            CrossValidator(scoring=[]).evaluate(Ridge(), X, y)

    def test_a_callable_is_accepted(self, regression):
        X, y = regression
        result = CrossValidator(scoring=rmse).evaluate(Ridge(), X, y)
        assert result["metric"] == "rmse"
        assert isinstance(result["score"], float)

    def test_make_scorer_rejects_a_non_callable(self):
        with pytest.raises(TypeError, match="must be callable"):
            make_scorer("rmse")  # type: ignore[arg-type]


class TestSingleVersusMultiple:
    """One metric is a float; an iterable is an array, even of length one."""

    def test_one_metric_gives_a_float(self, regression):
        X, y = regression
        assert isinstance(
            YScrambling(n_iterations=5, random_state=0, scoring="r2").run(
                Ridge(), X, y
            )["real_score"],
            float,
        )

    def test_several_metrics_give_an_array_in_order(self, regression):
        X, y = regression
        result = YScrambling(
            n_iterations=5, random_state=0, scoring=["rmse", "r2", "mae"]
        ).run(Ridge(), X, y)

        assert result["metric"] == ("rmse", "r2", "mae")
        for key in (
            "real_score",
            "mean_scrambled_score",
            "std_scrambled_score",
            "p_value",
        ):
            assert isinstance(result[key], np.ndarray)
            assert result[key].shape == (3,)

        # The order is the requested one, not a sorted or canonical one.
        single = [
            YScrambling(n_iterations=5, random_state=0, scoring=name).run(
                Ridge(), X, y
            )["real_score"]
            for name in ("rmse", "r2", "mae")
        ]
        assert np.allclose(result["real_score"], single)

    def test_a_one_element_iterable_still_gives_an_array(self, regression):
        """So adding a second metric never changes the caller's result shape."""
        X, y = regression
        result = CrossValidator(scoring=["r2"]).evaluate(Ridge(), X, y)
        assert isinstance(result["score"], np.ndarray)
        assert result["score"].shape == (1,)
        assert result["metric"] == ("r2",)


class TestLossDirection:
    """"At least as good" means smaller for a loss."""

    def test_a_loss_gives_the_same_verdict_as_the_matching_gain(self, regression):
        X, y = regression
        result = YScrambling(
            n_iterations=20, random_state=0, scoring=["r2", "rmse"]
        ).run(Ridge(), X, y)
        # R^2 and RMSE are monotonically related on one dataset, so a correct
        # implementation reaches the same p-value from both. Comparing RMSE
        # with >= would invert it and report ~1.0.
        assert result["p_value"][0] == pytest.approx(result["p_value"][1])
        assert result["p_value"][1] < 0.2

    def test_best_scrambled_score_follows_the_direction(self, regression):
        X, y = regression
        result = YScrambling(
            n_iterations=10, random_state=0, scoring=["r2", "rmse"]
        ).run(Ridge(), X, y)
        assert result["best_scrambled_score"][0] == result["max_scrambled_score"][0]
        # For RMSE the best scrambled model is the one with the lowest error,
        # which the maximum does not give.
        assert result["best_scrambled_score"][1] <= result["max_scrambled_score"][1]

    def test_make_scorer_declares_a_loss(self):
        assert make_scorer(rmse, greater_is_better=False).greater_is_better is False


class TestProbabilityMetrics:
    """A ranking metric must see probabilities, never hard labels."""

    def test_roc_auc_uses_probabilities(self, classification):
        X, y = classification
        model = QSARClassifier("rf", random_state=0)
        result = CrossValidator(n_splits=5, random_state=0, scoring="roc_auc").evaluate(
            model, X, y
        )

        # Out-of-fold probabilities are reported, and they are not 0/1.
        scores = result["y_score_cv"]
        assert scores.min() >= 0.0 and scores.max() <= 1.0
        assert not np.all(np.isin(scores, (0.0, 1.0)))

        # Scoring the hard labels instead would give a strictly worse AUC,
        # because thresholding throws the ranking away.
        from qsarkit.metrics import roc_auc as auc

        thresholded = auc(y, (scores >= 0.5).astype(float))
        assert result["score"] > thresholded

    def test_a_probability_metric_needs_a_probabilistic_estimator(self, regression):
        X, y = regression
        with pytest.raises(ValueError, match="neither predict_proba nor"):
            CrossValidator(scoring="roc_auc").evaluate(QSARRegressor("pls"), X, y)

    def test_make_scorer_with_needs_proba(self, classification):
        X, y = classification
        scorer = make_scorer(roc_auc, needs_proba=True, name="auc")
        model = QSARClassifier("rf", random_state=0).fit(X[:100], y[:100])
        result = ExternalValidator(scoring=scorer).validate(model, X[100:], y[100:])
        assert result["metric"] == "auc"
        assert 0.0 <= result["score"] <= 1.0


class TestOutOfFoldScrambling:
    """Why ``cv`` exists: in-sample scoring cannot see a ranking gap."""

    def test_in_sample_roc_auc_hides_the_signal(self, classification):
        X, y = classification
        model = QSARClassifier("rf", random_state=0)
        result = YScrambling(n_iterations=5, random_state=0, scoring="roc_auc").run(
            model, X, y
        )
        # A forest separates permuted labels in-sample just as well as real
        # ones, so the test reports no gap even though the signal is real.
        assert result["real_score"] > 0.99
        assert result["mean_scrambled_score"] > 0.9
        assert result["scored_out_of_fold"] is False

    def test_out_of_fold_recovers_it(self, classification):
        X, y = classification
        model = QSARClassifier("rf", random_state=0)
        result = YScrambling(
            n_iterations=5, random_state=0, scoring="roc_auc", cv=5, stratify=True
        ).run(model, X, y)

        assert result["scored_out_of_fold"] is True
        assert result["real_score"] > 0.75
        # Permuted labels score around chance once scoring is honest.
        assert result["mean_scrambled_score"] < 0.65
        assert result["real_score"] - result["mean_scrambled_score"] > 0.15

    def test_cv_below_two_is_refused(self, regression):
        X, y = regression
        with pytest.raises(ValueError, match="cv must be at least 2"):
            YScrambling(n_iterations=2, cv=1).run(Ridge(), X, y)


class TestBackwardsCompatibility:
    """The default must be exactly what earlier releases computed."""

    def test_cross_validator_default_keys_are_unchanged(self, regression):
        X, y = regression
        result = CrossValidator(n_splits=5, random_state=0).evaluate(Ridge(), X, y)
        assert {"q2", "rmse_cv", "mae_cv", "y_pred_cv"} <= set(result)
        assert "score" not in result

    def test_external_validator_default_is_the_regression_report(self, regression):
        X, y = regression
        model = Ridge().fit(X[:60], y[:60])
        report = ExternalValidator().validate(model, X[60:], y[60:])
        assert "golbraikh_tropsha" in report
        assert "score" not in report

    def test_explicit_r2_matches_the_default(self, regression):
        X, y = regression
        default = CrossValidator(n_splits=5, random_state=0).evaluate(Ridge(), X, y)
        explicit = CrossValidator(n_splits=5, random_state=0, scoring="r2").evaluate(
            Ridge(), X, y
        )
        assert explicit["score"] == pytest.approx(default["q2"])

    def test_bootstrap_default_matches_explicit_r2(self, regression):
        X, y = regression
        default = BootstrapValidator(n_iterations=10, random_state=0).run(Ridge(), X, y)
        explicit = BootstrapValidator(
            n_iterations=10, random_state=0, scoring="r2"
        ).run(Ridge(), X, y)
        assert explicit["mean_score"] == pytest.approx(default["mean_score"])
