"""Tests for decision-threshold selection."""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.metrics import optimal_threshold, threshold_report, threshold_sweep


@pytest.fixture(scope="module")
def separable():
    """A perfectly separable toy problem."""
    return np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])


@pytest.fixture(scope="module")
def imbalanced():
    """3% actives, with signal — the realistic screening case."""
    rng = np.random.default_rng(0)
    y = np.zeros(1000, dtype=int)
    y[:30] = 1
    scores = rng.beta(2, 8, size=1000) + y * 0.25
    return y, scores


class TestSweep:
    def test_finds_the_perfect_split(self, separable):
        y, scores = separable
        sweep = threshold_sweep(y, scores)
        best = int(np.argmax(sweep["youden_j"]))
        assert sweep["thresholds"][best] == pytest.approx(0.6)
        assert sweep["youden_j"][best] == pytest.approx(1.0)

    def test_sensitivity_falls_and_specificity_rises(self, imbalanced):
        y, scores = imbalanced
        sweep = threshold_sweep(y, scores)
        assert np.all(np.diff(sweep["sensitivity"]) <= 1e-12)
        assert np.all(np.diff(sweep["specificity"]) >= -1e-12)

    def test_confusion_counts_are_consistent(self, imbalanced):
        y, scores = imbalanced
        sweep = threshold_sweep(y, scores)
        total = sweep["tp"] + sweep["fp"] + sweep["tn"] + sweep["fn"]
        assert np.all(total == len(y))
        assert np.all(sweep["tp"] + sweep["fn"] == y.sum())

    def test_includes_a_predict_nothing_endpoint(self, imbalanced):
        """The degenerate end of the curve should be explicit, not absent."""
        y, scores = imbalanced
        sweep = threshold_sweep(y, scores)
        assert sweep["tp"][-1] == 0 and sweep["fp"][-1] == 0
        assert sweep["sensitivity"][-1] == 0.0

    def test_lowest_threshold_predicts_everything_positive(self, imbalanced):
        y, scores = imbalanced
        sweep = threshold_sweep(y, scores)
        assert sweep["sensitivity"][0] == pytest.approx(1.0)
        assert sweep["specificity"][0] == pytest.approx(0.0)

    def test_n_thresholds_reduces_the_grid(self, imbalanced):
        y, scores = imbalanced
        full = threshold_sweep(y, scores)["thresholds"].size
        coarse = threshold_sweep(y, scores, n_thresholds=20)["thresholds"].size
        assert coarse < full

    def test_derived_rates_are_in_range(self, imbalanced):
        y, scores = imbalanced
        sweep = threshold_sweep(y, scores)
        for key in ("sensitivity", "specificity", "precision", "f1", "accuracy"):
            assert np.all((sweep[key] >= 0.0) & (sweep[key] <= 1.0)), key
        assert np.all((sweep["mcc"] >= -1.0) & (sweep["mcc"] <= 1.0))

    def test_boolean_labels_work_without_pos_label(self):
        sweep = threshold_sweep(
            np.array([False, False, True, True]), np.array([0.1, 0.2, 0.8, 0.9])
        )
        assert sweep["sensitivity"][0] == pytest.approx(1.0)

    def test_string_labels_require_an_explicit_positive_class(self):
        """Choosing by sort order would make 'inactive' positive."""
        with pytest.raises(ValueError, match="not numeric"):
            threshold_sweep(
                np.array(["active", "inactive", "active", "inactive"]),
                np.array([0.9, 0.1, 0.8, 0.2]),
            )

    def test_string_labels_work_when_told_which_is_positive(self):
        sweep = threshold_sweep(
            np.array(["active", "inactive", "active", "inactive"]),
            np.array([0.9, 0.1, 0.8, 0.2]),
            pos_label="active",
        )
        assert sweep["tp"][0] == 2
        best = int(np.argmax(sweep["youden_j"]))
        assert sweep["youden_j"][best] == pytest.approx(1.0)

    def test_an_absent_pos_label_is_reported(self):
        with pytest.raises(ValueError, match="not one of the labels"):
            threshold_sweep([0, 1], [0.1, 0.9], pos_label=7)

    def test_non_finite_scores_are_dropped(self):
        y = np.array([0, 0, 1, 1])
        scores = np.array([0.1, np.nan, 0.8, 0.9])
        sweep = threshold_sweep(y, scores)
        assert sweep["tp"][0] + sweep["fp"][0] == 3

    def test_rejects_a_single_class(self):
        with pytest.raises(ValueError, match="two classes"):
            threshold_sweep(np.zeros(5, dtype=int), np.linspace(0, 1, 5))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="entries"):
            threshold_sweep([0, 1], [0.5])

    def test_rejects_all_non_finite_scores(self):
        with pytest.raises(ValueError, match="No finite scores"):
            threshold_sweep([0, 1], [np.nan, np.nan])

    def test_rejects_too_few_thresholds(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="n_thresholds"):
            threshold_sweep(y, scores, n_thresholds=1)


class TestOptimalThreshold:
    @pytest.mark.parametrize(
        "criterion", ["youden", "mcc", "f1", "balanced_accuracy"]
    )
    def test_each_criterion_returns_a_usable_threshold(self, imbalanced, criterion):
        y, scores = imbalanced
        result = optimal_threshold(y, scores, criterion=criterion)
        assert result["criterion"] == criterion
        assert scores.min() <= result["threshold"] <= np.nextafter(scores.max(), np.inf)

    def test_the_reported_score_matches_the_reported_confusion_matrix(
        self, imbalanced
    ):
        y, scores = imbalanced
        result = optimal_threshold(y, scores, criterion="youden")
        expected = (
            result["tp"] / (result["tp"] + result["fn"])
            + result["tn"] / (result["tn"] + result["fp"])
            - 1.0
        )
        assert result["score"] == pytest.approx(expected)

    def test_on_imbalanced_data_the_optimum_is_well_below_one_half(
        self, imbalanced
    ):
        """The whole reason not to use predict()'s 0.5 default."""
        y, scores = imbalanced
        assert optimal_threshold(y, scores, criterion="mcc")["threshold"] < 0.5

    def test_costly_false_negatives_lower_the_threshold(self, imbalanced):
        y, scores = imbalanced
        symmetric = optimal_threshold(
            y, scores, criterion="cost", cost_fn=1.0, cost_fp=1.0
        )["threshold"]
        asymmetric = optimal_threshold(
            y, scores, criterion="cost", cost_fn=10.0, cost_fp=1.0
        )["threshold"]
        assert asymmetric <= symmetric

    def test_costly_false_positives_raise_it(self, imbalanced):
        y, scores = imbalanced
        symmetric = optimal_threshold(
            y, scores, criterion="cost", cost_fn=1.0, cost_fp=1.0
        )["threshold"]
        strict = optimal_threshold(
            y, scores, criterion="cost", cost_fn=1.0, cost_fp=10.0
        )["threshold"]
        assert strict >= symmetric

    def test_cost_rejects_negative_costs(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="non-negative"):
            optimal_threshold(y, scores, criterion="cost", cost_fn=-1.0)

    def test_precision_constraint_is_respected(self, imbalanced):
        y, scores = imbalanced
        result = optimal_threshold(
            y, scores, criterion="precision", min_precision=0.3
        )
        assert result["precision"] >= 0.3

    def test_recall_constraint_is_respected(self, imbalanced):
        y, scores = imbalanced
        result = optimal_threshold(y, scores, criterion="recall", min_recall=0.8)
        assert result["sensitivity"] >= 0.8

    def test_an_unreachable_precision_is_reported_with_the_best_available(
        self, imbalanced
    ):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="best available"):
            optimal_threshold(y, scores, criterion="precision", min_precision=0.99)

    def test_an_unreachable_recall_is_reported(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="best available"):
            optimal_threshold(y, scores, criterion="recall", min_recall=1.01)

    def test_precision_criterion_requires_its_constraint(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="needs min_precision"):
            optimal_threshold(y, scores, criterion="precision")

    def test_recall_criterion_requires_its_constraint(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="needs min_recall"):
            optimal_threshold(y, scores, criterion="recall")

    def test_rejects_an_unknown_criterion(self, imbalanced):
        y, scores = imbalanced
        with pytest.raises(ValueError, match="Unknown criterion"):
            optimal_threshold(y, scores, criterion="vibes")

    def test_reports_the_documented_keys(self, imbalanced):
        y, scores = imbalanced
        result = optimal_threshold(y, scores)
        assert {
            "threshold", "criterion", "score", "tp", "fp", "tn", "fn",
            "sensitivity", "specificity", "precision", "f1", "mcc",
            "balanced_accuracy", "accuracy",
        } <= set(result)

    def test_precision_ignores_thresholds_predicting_nothing(self, imbalanced):
        """Precision is undefined when nothing is called positive."""
        y, scores = imbalanced
        result = optimal_threshold(
            y, scores, criterion="precision", min_precision=0.2
        )
        assert result["tp"] + result["fp"] > 0


class TestThresholdReport:
    def test_compares_every_criterion_against_the_default(self, imbalanced):
        y, scores = imbalanced
        report = threshold_report(y, scores)
        assert sorted(k for k in report if isinstance(report[k], dict)) == [
            "balanced_accuracy", "default_0.5", "f1", "mcc", "youden",
        ]

    def test_the_tuned_threshold_beats_the_default_on_its_own_criterion(
        self, imbalanced
    ):
        y, scores = imbalanced
        report = threshold_report(y, scores)
        assert report["mcc"]["mcc"] >= report["default_0.5"]["mcc"]
        assert report["f1"]["f1"] >= report["default_0.5"]["f1"]

    def test_the_default_misses_actives_on_imbalanced_data(self, imbalanced):
        y, scores = imbalanced
        report = threshold_report(y, scores)
        assert report["default_0.5"]["sensitivity"] < report["youden"]["sensitivity"]

    def test_includes_the_threshold_free_summary(self, imbalanced):
        y, scores = imbalanced
        report = threshold_report(y, scores)
        assert 0.0 <= report["roc_auc"] <= 1.0
        assert 0.0 <= report["pr_auc"] <= 1.0
        assert report["base_rate"] == pytest.approx(y.mean())

    def test_n_thresholds_is_honoured(self, imbalanced):
        y, scores = imbalanced
        report = threshold_report(y, scores, n_thresholds=25)
        assert "youden" in report
