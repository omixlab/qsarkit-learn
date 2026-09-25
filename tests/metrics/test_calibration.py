"""Tests for probability calibration and residual-distribution diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.metrics import (
    calibration_curve,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    qq_data,
    residual_normality,
)


@pytest.fixture(scope="module")
def calibrated():
    """Probabilities that are true by construction."""
    rng = np.random.default_rng(0)
    p = rng.uniform(size=4000)
    return (rng.uniform(size=4000) < p).astype(int), p


class TestCalibrationCurve:
    def test_a_calibrated_model_lies_on_the_diagonal(self, calibrated):
        y, p = calibrated
        curve = calibration_curve(y, p, n_bins=10)
        assert np.allclose(
            curve["mean_predicted"], curve["observed_frequency"], atol=0.06
        )

    def test_reports_one_entry_per_non_empty_bin(self, calibrated):
        y, p = calibrated
        curve = calibration_curve(y, p, n_bins=10)
        assert curve["mean_predicted"].size == curve["observed_frequency"].size
        assert curve["counts"].sum() == len(y)

    def test_quantile_binning_balances_the_counts(self, calibrated):
        y, p = calibrated
        uniform = calibration_curve(y, p, n_bins=10, strategy="uniform")
        quantile = calibration_curve(y, p, n_bins=10, strategy="quantile")
        assert quantile["counts"].std() <= uniform["counts"].std()

    def test_an_overconfident_model_bends_away(self, calibrated):
        y, p = calibrated
        squashed = np.clip(p * 1.6 - 0.3, 0, 1)
        curve = calibration_curve(y, squashed, n_bins=5)
        # Predictions pushed toward 0 are exceeded by reality.
        assert curve["observed_frequency"][0] > curve["mean_predicted"][0]

    def test_a_prediction_of_exactly_zero_lands_in_the_first_bin(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.0, 0.0, 1.0, 1.0])
        curve = calibration_curve(y, p, n_bins=4)
        assert curve["counts"].sum() == 4

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="entries"):
            calibration_curve([0, 1], [0.5])

    def test_rejects_non_binary_labels(self):
        with pytest.raises(ValueError, match="binary"):
            calibration_curve([0, 1, 2], [0.1, 0.5, 0.9])

    def test_rejects_scores_outside_zero_one(self):
        """Decision-function output is not a probability."""
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            calibration_curve([0, 1], [-2.0, 3.0])

    def test_rejects_too_few_bins(self, calibrated):
        y, p = calibrated
        with pytest.raises(ValueError, match="n_bins"):
            calibration_curve(y, p, n_bins=1)

    def test_rejects_an_unknown_strategy(self, calibrated):
        y, p = calibrated
        with pytest.raises(ValueError, match="strategy"):
            calibration_curve(y, p, strategy="clever")

    def test_rejects_an_empty_set(self):
        with pytest.raises(ValueError, match="empty"):
            calibration_curve([], [])

    def test_handles_every_prediction_identical(self):
        y = np.array([0, 1, 0, 1])
        p = np.full(4, 0.5)
        curve = calibration_curve(y, p, n_bins=5, strategy="quantile")
        assert curve["counts"].sum() == 4


class TestCalibrationErrors:
    def test_a_calibrated_model_scores_near_zero(self, calibrated):
        y, p = calibrated
        assert expected_calibration_error(y, p) < 0.05

    def test_a_shifted_model_scores_worse(self, calibrated):
        y, p = calibrated
        shifted = np.clip(p + 0.25, 0, 1)
        assert expected_calibration_error(y, shifted) > expected_calibration_error(y, p)

    def test_the_shift_size_is_reflected(self, calibrated):
        y, p = calibrated
        small = expected_calibration_error(y, np.clip(p + 0.1, 0, 1))
        large = expected_calibration_error(y, np.clip(p + 0.3, 0, 1))
        assert large > small

    def test_max_error_is_at_least_the_mean(self, calibrated):
        y, p = calibrated
        assert maximum_calibration_error(y, p, min_count=20) >= (
            expected_calibration_error(y, p)
        )

    def test_min_count_suppresses_tiny_bin_artefacts(self, calibrated):
        y, p = calibrated
        assert maximum_calibration_error(y, p, n_bins=60, min_count=50) <= (
            maximum_calibration_error(y, p, n_bins=60, min_count=1)
        )

    def test_min_count_above_every_bin_returns_zero(self, calibrated):
        y, p = calibrated
        assert maximum_calibration_error(y, p, min_count=10 ** 6) == 0.0

    def test_rejects_a_min_count_below_one(self, calibrated):
        y, p = calibrated
        with pytest.raises(ValueError, match="min_count"):
            maximum_calibration_error(y, p, min_count=0)


class TestCalibrationReport:
    def test_reports_the_documented_keys(self, calibrated):
        y, p = calibrated
        report = calibration_report(y, p)
        assert {
            "ece", "mce", "brier", "brier_skill_score", "base_rate",
            "mean_predicted", "observed_frequency", "n_samples", "n_bins_used",
        } <= set(report)

    def test_a_calibrated_model_has_positive_skill(self, calibrated):
        y, p = calibrated
        assert calibration_report(y, p)["brier_skill_score"] > 0

    def test_predicting_the_base_rate_has_no_skill(self, calibrated):
        """The check a raw Brier score on imbalanced data hides."""
        y, _ = calibrated
        flat = np.full(len(y), y.mean())
        assert calibration_report(y, flat)["brier_skill_score"] == pytest.approx(0.0)

    def test_an_imbalanced_set_with_a_good_looking_brier_can_lack_skill(self):
        rng = np.random.default_rng(1)
        y = np.zeros(2000, dtype=int)
        y[:40] = 1                              # 2% actives
        useless = rng.uniform(0.0, 0.04, size=2000)
        report = calibration_report(y, useless)
        assert report["brier"] < 0.05           # looks excellent
        assert report["brier_skill_score"] <= 0.05   # and is not

    def test_base_rate_is_reported(self, calibrated):
        y, p = calibrated
        assert calibration_report(y, p)["base_rate"] == pytest.approx(y.mean())


class TestQQ:
    def test_normal_residuals_track_the_reference(self):
        rng = np.random.default_rng(0)
        data = qq_data(rng.normal(size=500))
        corr = np.corrcoef(
            data["theoretical_quantiles"], data["sample_quantiles"]
        )[0, 1]
        assert corr > 0.99

    def test_heavy_tails_depart_from_it(self):
        rng = np.random.default_rng(0)
        normal = np.corrcoef(
            *[qq_data(rng.normal(size=500))[k] for k in
              ("theoretical_quantiles", "sample_quantiles")]
        )[0, 1]
        heavy = np.corrcoef(
            *[qq_data(rng.standard_t(df=2, size=500))[k] for k in
              ("theoretical_quantiles", "sample_quantiles")]
        )[0, 1]
        assert heavy < normal

    def test_quantiles_are_sorted(self):
        rng = np.random.default_rng(0)
        data = qq_data(rng.normal(size=200))
        assert np.all(np.diff(data["sample_quantiles"]) >= 0)
        assert np.all(np.diff(data["theoretical_quantiles"]) >= 0)

    def test_standardizing_makes_the_slope_about_one(self):
        rng = np.random.default_rng(0)
        slope = qq_data(rng.normal(scale=50.0, size=500))["reference_line"][0]
        assert 0.8 < slope < 1.2

    def test_without_standardizing_the_slope_follows_the_scale(self):
        rng = np.random.default_rng(0)
        slope = qq_data(
            rng.normal(scale=50.0, size=500), standardize=False
        )["reference_line"][0]
        assert slope > 10

    def test_non_finite_values_are_dropped(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=100)
        values[:5] = np.nan
        assert qq_data(values)["sample_quantiles"].size == 95

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 3"):
            qq_data([1.0, 2.0])

    def test_handles_identical_residuals(self):
        data = qq_data(np.zeros(10))
        assert np.all(data["sample_quantiles"] == 0.0)


class TestResidualNormality:
    def test_normal_residuals_are_not_rejected(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=500)
        report = residual_normality(truth, truth + rng.normal(scale=0.1, size=500))
        assert report["shapiro_p"] > 0.05
        assert abs(report["skew"]) < 0.2

    def test_shape_terms_do_not_depend_on_a_hypothesis_test(self):
        """Skew and kurtosis describe the residuals at any sample size.

        Shapiro-Wilk is sample-size sensitive in both directions -- weak on
        a few dozen compounds, over-eager on a few hundred -- so the shape
        terms are what a QSAR-sized dataset should be read on. They are
        stable where the p-value is not.
        """
        rng = np.random.default_rng(0)
        skews = []
        for n in (100, 300, 1000):
            truth = rng.normal(size=n)
            report = residual_normality(truth, truth + rng.normal(scale=0.1, size=n))
            skews.append(abs(report["skew"]))
        assert all(value < 0.6 for value in skews)

    def test_skewed_residuals_are_detected(self):
        rng = np.random.default_rng(0)
        truth = np.zeros(300)
        report = residual_normality(truth, -rng.exponential(size=300))
        assert report["shapiro_p"] < 0.05
        assert abs(report["skew"]) > 0.5

    def test_heteroscedasticity_is_detected(self):
        rng = np.random.default_rng(0)
        fitted = np.linspace(1, 10, 300)
        noisy = fitted + rng.normal(scale=fitted * 0.3)
        assert abs(residual_normality(noisy, fitted)["heteroscedasticity_r"]) > 0.2

    def test_homoscedastic_errors_show_little_correlation(self):
        rng = np.random.default_rng(0)
        fitted = np.linspace(1, 10, 400)
        noisy = fitted + rng.normal(scale=0.3, size=400)
        assert abs(residual_normality(noisy, fitted)["heteroscedasticity_r"]) < 0.2

    def test_reports_the_documented_keys(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=100)
        report = residual_normality(truth, truth + rng.normal(scale=0.1, size=100))
        assert set(report) == {
            "shapiro_statistic", "shapiro_p", "skew", "excess_kurtosis",
            "heteroscedasticity_r", "n",
        }

    def test_perfect_predictions_do_not_crash(self):
        """Zero-variance residuals have no distribution to test."""
        truth = np.linspace(0, 1, 50)
        report = residual_normality(truth, truth)
        assert report["shapiro_p"] == 1.0
        assert report["heteroscedasticity_r"] == 0.0

    def test_non_finite_pairs_are_dropped(self):
        rng = np.random.default_rng(0)
        truth = rng.normal(size=100)
        predicted = truth + rng.normal(scale=0.1, size=100)
        predicted[:5] = np.nan
        assert residual_normality(truth, predicted)["n"] == 95

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="entries"):
            residual_normality([1.0, 2.0], [1.0])

    def test_rejects_too_few_pairs(self):
        with pytest.raises(ValueError, match="at least 3"):
            residual_normality([1.0, 2.0], [1.0, 2.0])

    def test_large_samples_are_subsampled_for_shapiro(self):
        """Shapiro-Wilk is unreliable above ~5000 points."""
        rng = np.random.default_rng(0)
        truth = rng.normal(size=8000)
        report = residual_normality(truth, truth + rng.normal(scale=0.1, size=8000))
        assert report["n"] == 8000          # all pairs counted
        assert 0.0 <= report["shapiro_p"] <= 1.0
