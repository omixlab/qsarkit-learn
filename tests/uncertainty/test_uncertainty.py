from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from qsarkit.base import ModelNotFittedError
from qsarkit.uncertainty import (
    ConformalClassifier,
    ConformalPredictor,
    ConformalRegressor,
    EnsembleUncertainty,
    GaussianProcessUncertainty,
    MCDropoutUncertainty,
    QuantileRegressionUncertainty,
    UncertaintyCalibration,
)


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 4))
    y = X[:, 0] * 2 + rng.normal(scale=0.3, size=400)
    return X[:300], X[300:], y[:300], y[300:]


@pytest.fixture(scope="module")
def classification_data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(400, 4))
    y = (X[:, 0] > 0).astype(int)
    return X[:300], X[300:], y[:300], y[300:]


def rf_reg():
    return RandomForestRegressor(n_estimators=20, random_state=0)


def rf_clf():
    return RandomForestClassifier(n_estimators=20, random_state=0)


class TestConformalRegressor:
    def test_achieves_nominal_coverage(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        cp = ConformalRegressor(rf_reg(), alpha=0.1, random_state=0).fit(X_tr, y_tr)
        # Conformal prediction guarantees coverage >= 1 - alpha; allow a
        # small finite-sample slack below nominal on one draw.
        assert cp.evaluate(X_te, y_te)["coverage"] >= 0.85

    def test_tighter_alpha_gives_wider_intervals(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(rf_reg(), alpha=0.1, random_state=0).fit(X_tr, y_tr)
        assert np.mean(cp.interval_width(X_te, alpha=0.01)) >= np.mean(
            cp.interval_width(X_te, alpha=0.2)
        )

    def test_intervals_bracket_the_point_prediction(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(rf_reg(), random_state=0).fit(X_tr, y_tr)
        lower, upper = cp.predict_interval(X_te)
        point = cp.predict(X_te)
        assert np.all(lower <= point) and np.all(point <= upper)

    def test_plain_intervals_are_constant_width(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(rf_reg(), random_state=0).fit(X_tr, y_tr)
        assert np.std(cp.interval_width(X_te)) == pytest.approx(0.0, abs=1e-9)

    def test_normalized_intervals_vary_per_sample(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(
            rf_reg(), normalized=True, random_state=0
        ).fit(X_tr, y_tr)
        assert np.std(cp.interval_width(X_te)) > 0.0

    def test_normalized_still_covers(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        cp = ConformalRegressor(
            rf_reg(), alpha=0.1, normalized=True, random_state=0
        ).fit(X_tr, y_tr)
        assert cp.evaluate(X_te, y_te)["coverage"] >= 0.85

    def test_custom_difficulty_estimator(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(
            rf_reg(),
            normalized=True,
            difficulty_estimator=LinearRegression(),
            random_state=0,
        ).fit(X_tr, y_tr)
        assert cp.interval_width(X_te).shape == (len(X_te),)

    def test_evaluate_reports_expected_coverage(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        cp = ConformalRegressor(rf_reg(), alpha=0.2, random_state=0).fit(X_tr, y_tr)
        report = cp.evaluate(X_te, y_te)
        assert report["expected_coverage"] == pytest.approx(0.8)
        assert set(report) == {
            "coverage", "expected_coverage", "mean_width", "median_width"
        }

    def test_alpha_can_be_changed_without_refitting(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        cp = ConformalRegressor(rf_reg(), alpha=0.1, random_state=0).fit(X_tr, y_tr)
        assert cp.evaluate(X_te, y_te, alpha=0.5)["expected_coverage"] == 0.5

    def test_requires_fitting(self, regression_data):
        _, X_te, _, _ = regression_data
        with pytest.raises(ModelNotFittedError):
            ConformalRegressor(rf_reg()).predict(X_te)

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_invalid_alpha(self, regression_data, alpha):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalRegressor(rf_reg(), alpha=alpha).fit(X_tr, y_tr)

    def test_rejects_invalid_alpha_at_predict_time(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        cp = ConformalRegressor(rf_reg(), random_state=0).fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="alpha must be in"):
            cp.predict_interval(X_te, alpha=1.5)

    def test_rejects_invalid_calibration_size(self, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError, match="calibration_size must be in"):
            ConformalRegressor(rf_reg(), calibration_size=1.5).fit(X_tr, y_tr)

    def test_tiny_calibration_set_raises(self):
        X = np.random.RandomState(0).normal(size=(4, 2))
        with pytest.raises(ValueError, match="fewer than 2 samples"):
            ConformalRegressor(
                LinearRegression(), calibration_size=0.25, random_state=0
            ).fit(X, X[:, 0])


class TestConformalClassifier:
    def test_achieves_nominal_coverage(self, classification_data):
        X_tr, X_te, y_tr, y_te = classification_data
        cp = ConformalClassifier(rf_clf(), alpha=0.1, random_state=0).fit(X_tr, y_tr)
        assert cp.evaluate(X_te, y_te)["coverage"] >= 0.85

    def test_p_values_are_probabilities(self, classification_data):
        X_tr, X_te, y_tr, _ = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        p = cp.p_values(X_te)
        assert p.shape == (len(X_te), 2)
        assert np.all((p >= 0) & (p <= 1))

    def test_prediction_sets_are_label_lists(self, classification_data):
        X_tr, X_te, y_tr, _ = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        sets = cp.predict_set(X_te[:5])
        assert all(isinstance(s, list) for s in sets)
        assert all(set(s) <= {0, 1} for s in sets)

    def test_smaller_alpha_gives_larger_sets(self, classification_data):
        X_tr, X_te, y_tr, _ = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        strict = np.mean([len(s) for s in cp.predict_set(X_te, alpha=0.01)])
        loose = np.mean([len(s) for s in cp.predict_set(X_te, alpha=0.3)])
        assert strict >= loose

    def test_mondrian_calibration_runs(self, classification_data):
        X_tr, X_te, y_tr, y_te = classification_data
        cp = ConformalClassifier(
            rf_clf(), alpha=0.1, mondrian=True, random_state=0
        ).fit(X_tr, y_tr)
        assert cp.evaluate(X_te, y_te)["coverage"] >= 0.85

    def test_evaluate_keys(self, classification_data):
        X_tr, X_te, y_tr, y_te = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        assert set(cp.evaluate(X_te, y_te)) == {
            "coverage", "expected_coverage", "mean_set_size",
            "singleton_fraction", "empty_fraction",
        }

    def test_point_predict_delegates(self, classification_data):
        X_tr, X_te, y_tr, _ = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        assert cp.predict(X_te).shape == (len(X_te),)

    def test_requires_fitting(self, classification_data):
        _, X_te, _, _ = classification_data
        with pytest.raises(ModelNotFittedError):
            ConformalClassifier(rf_clf()).p_values(X_te)

    def test_rejects_invalid_alpha(self, classification_data):
        X_tr, _, y_tr, _ = classification_data
        with pytest.raises(ValueError, match="alpha must be in"):
            ConformalClassifier(rf_clf(), alpha=0.0).fit(X_tr, y_tr)

    def test_rejects_invalid_alpha_at_predict_time(self, classification_data):
        X_tr, X_te, y_tr, _ = classification_data
        cp = ConformalClassifier(rf_clf(), random_state=0).fit(X_tr, y_tr)
        with pytest.raises(ValueError, match="alpha must be in"):
            cp.predict_set(X_te, alpha=2.0)


class TestConformalPredictorFactory:
    def test_dispatches_on_estimator_type(self):
        assert isinstance(
            ConformalPredictor(RandomForestRegressor()), ConformalRegressor
        )
        assert isinstance(
            ConformalPredictor(RandomForestClassifier()), ConformalClassifier
        )

    def test_explicit_task(self):
        assert isinstance(
            ConformalPredictor(RandomForestRegressor(), task="regression"),
            ConformalRegressor,
        )

    def test_rejects_unknown_task(self):
        with pytest.raises(ValueError, match="task must be"):
            ConformalPredictor(RandomForestRegressor(), task="ranking")

    def test_forwards_kwargs(self):
        assert ConformalPredictor(RandomForestRegressor(), alpha=0.2).alpha == 0.2


class TestEnsembleUncertainty:
    def test_uses_a_forests_own_trees(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(rf_reg(), random_state=0).fit(X_tr, y_tr)
        assert len(est.estimators_) == 20

    def test_bagging_a_non_ensemble_learner(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(
            DecisionTreeRegressor(), n_estimators=6, random_state=0
        ).fit(X_tr, y_tr)
        assert len(est.estimators_) == 6
        assert est.predict_uncertainty(X_te)[1].mean() > 0

    def test_without_bootstrap(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(
            DecisionTreeRegressor(), n_estimators=4, bootstrap=False,
            use_native_ensemble=False, random_state=0,
        ).fit(X_tr, y_tr)
        assert est.predict_uncertainty(X_te)[1].shape == (len(X_te),)

    def test_uncertainty_is_non_negative(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(rf_reg(), random_state=0).fit(X_tr, y_tr)
        assert np.all(est.predict_uncertainty(X_te)[1] >= 0)

    def test_predict_matches_the_mean(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(rf_reg(), random_state=0).fit(X_tr, y_tr)
        assert np.allclose(est.predict(X_te), est.predict_uncertainty(X_te)[0])

    def test_interval_widens_with_n_std(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        est = EnsembleUncertainty(rf_reg(), random_state=0).fit(X_tr, y_tr)
        narrow = np.ptp(np.array(est.predict_interval(X_te, n_std=1.0)), axis=0)
        wide = np.ptp(np.array(est.predict_interval(X_te, n_std=3.0)), axis=0)
        assert np.all(wide >= narrow)

    def test_single_member_rejected(self, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleUncertainty(DecisionTreeRegressor(), n_estimators=1).fit(
                X_tr, y_tr
            )

    def test_requires_fitting(self, regression_data):
        _, X_te, _, _ = regression_data
        with pytest.raises(ModelNotFittedError):
            EnsembleUncertainty(rf_reg()).predict_uncertainty(X_te)


class TestGaussianProcessUncertainty:
    def test_fit_and_predict(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        gp = GaussianProcessUncertainty(random_state=0).fit(X_tr[:60], y_tr[:60])
        mean, std = gp.predict_uncertainty(X_te)
        assert mean.shape == std.shape == (len(X_te),)
        assert np.all(std >= 0)

    def test_uncertainty_is_lower_on_training_points(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(40, 2))
        y = X[:, 0]
        gp = GaussianProcessUncertainty(random_state=0).fit(X, y)
        near = gp.predict_uncertainty(X)[1].mean()
        far = gp.predict_uncertainty(np.full((5, 2), 20.0))[1].mean()
        assert far > near

    def test_accepts_a_custom_kernel(self):
        from sklearn.gaussian_process.kernels import RBF

        rng = np.random.RandomState(0)
        X = rng.normal(size=(30, 2))
        gp = GaussianProcessUncertainty(kernel=RBF(), random_state=0).fit(X, X[:, 0])
        assert gp.predict_uncertainty(X)[1].shape == (30,)

    def test_requires_fitting(self, regression_data):
        _, X_te, _, _ = regression_data
        with pytest.raises(ModelNotFittedError):
            GaussianProcessUncertainty().predict_uncertainty(X_te)


class TestQuantileRegressionUncertainty:
    def test_intervals_are_ordered(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        q = QuantileRegressionUncertainty(random_state=0).fit(X_tr, y_tr)
        lower, upper = q.predict_interval(X_te)
        assert np.all(upper >= lower)

    def test_intervals_cover_most_points(self, regression_data):
        X_tr, X_te, y_tr, y_te = regression_data
        q = QuantileRegressionUncertainty(random_state=0).fit(X_tr, y_tr)
        lower, upper = q.predict_interval(X_te)
        assert np.mean((y_te >= lower) & (y_te <= upper)) > 0.7

    def test_captures_heteroscedasticity(self):
        # noise grows with x, so intervals should widen with x
        rng = np.random.RandomState(0)
        x = rng.uniform(0, 10, size=600)
        y = x + rng.normal(scale=0.05 + x / 10, size=600)
        X = x.reshape(-1, 1)
        q = QuantileRegressionUncertainty(random_state=0).fit(X, y)
        lower, upper = q.predict_interval(X)
        width = upper - lower
        low_x, high_x = x < 2, x > 8
        assert width[high_x].mean() > width[low_x].mean()

    def test_uncertainty_is_non_negative(self, regression_data):
        X_tr, X_te, y_tr, _ = regression_data
        q = QuantileRegressionUncertainty(random_state=0).fit(X_tr, y_tr)
        assert np.all(q.predict_uncertainty(X_te)[1] >= 0)

    def test_rejects_bad_quantiles(self, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError, match="0 < low < high < 1"):
            QuantileRegressionUncertainty(quantiles=(0.9, 0.1)).fit(X_tr, y_tr)

    def test_rejects_estimator_without_a_quantile_parameter(self, regression_data):
        X_tr, _, y_tr, _ = regression_data
        with pytest.raises(ValueError, match="no 'quantile' or 'alpha'"):
            QuantileRegressionUncertainty(estimator=LinearRegression()).fit(
                X_tr, y_tr
            )

    def test_requires_fitting(self, regression_data):
        _, X_te, _, _ = regression_data
        with pytest.raises(ModelNotFittedError):
            QuantileRegressionUncertainty().predict_interval(X_te)
        with pytest.raises(ModelNotFittedError):
            QuantileRegressionUncertainty().predict_uncertainty(X_te)


class TestMCDropoutUncertainty:
    def test_samples_a_dropout_network(self):
        torch = pytest.importorskip("torch")

        net = torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.ReLU(),
            torch.nn.Dropout(0.5), torch.nn.Linear(8, 1),
        )
        est = MCDropoutUncertainty(net, n_samples=10).fit(None, None)
        mean, std = est.predict_uncertainty(np.random.RandomState(0).normal(size=(6, 4)))
        assert mean.shape == std.shape == (6,)
        # dropout must actually produce variation
        assert std.mean() > 0

    def test_network_without_dropout_is_rejected(self):
        torch = pytest.importorskip("torch")

        net = torch.nn.Sequential(torch.nn.Linear(4, 1))
        est = MCDropoutUncertainty(net, n_samples=5)
        with pytest.raises(ValueError, match="no dropout layers"):
            est.predict_uncertainty(np.zeros((3, 4)))

    def test_too_few_samples_rejected(self):
        pytest.importorskip("torch")
        import torch

        net = torch.nn.Sequential(torch.nn.Dropout(0.5), torch.nn.Linear(4, 1))
        with pytest.raises(ValueError, match="n_samples must be at least 2"):
            MCDropoutUncertainty(net, n_samples=1).predict_uncertainty(
                np.zeros((2, 4))
            )


class TestUncertaintyCalibration:
    @pytest.fixture
    def well_calibrated(self):
        rng = np.random.RandomState(0)
        y_true = rng.normal(size=800)
        sigma = np.full(800, 1.0)
        y_pred = y_true + rng.normal(scale=1.0, size=800)
        return y_true, y_pred, sigma

    def test_report_keys(self, well_calibrated):
        report = UncertaintyCalibration().report(*well_calibrated)
        assert set(report) == {
            "ence", "miscalibration_area", "spearman_error_correlation",
            "coverage_68", "coverage_95", "mean_sigma", "rmse",
        }

    def test_well_calibrated_model_scores_well(self, well_calibrated):
        report = UncertaintyCalibration().report(*well_calibrated)
        assert report["coverage_68"] == pytest.approx(0.68, abs=0.08)
        assert report["coverage_95"] == pytest.approx(0.95, abs=0.06)
        assert report["miscalibration_area"] < 0.1

    def test_overconfident_model_is_detected(self, well_calibrated):
        y_true, y_pred, sigma = well_calibrated
        # claim ten times more precision than the model actually has
        report = UncertaintyCalibration().report(y_true, y_pred, sigma / 10)
        assert report["coverage_95"] < 0.5
        assert report["miscalibration_area"] > 0.2

    def test_ence_is_lower_for_a_calibrated_model(self, well_calibrated):
        y_true, y_pred, sigma = well_calibrated
        cal = UncertaintyCalibration()
        assert cal.ence(y_true, y_pred, sigma) < cal.ence(y_true, y_pred, sigma / 10)

    def test_spearman_detects_informative_uncertainty(self):
        rng = np.random.RandomState(0)
        sigma = rng.uniform(0.1, 2.0, size=500)
        y_true = np.zeros(500)
        y_pred = rng.normal(scale=sigma)  # error genuinely scales with sigma
        rho = UncertaintyCalibration().spearman_error_correlation(
            y_true, y_pred, sigma
        )
        assert rho > 0.3

    def test_constant_uncertainty_has_no_ranking_power(self, well_calibrated):
        y_true, y_pred, sigma = well_calibrated
        assert UncertaintyCalibration().spearman_error_correlation(
            y_true, y_pred, sigma
        ) == 0.0

    def test_coverage_curve_is_monotonic(self, well_calibrated):
        nominal, observed = UncertaintyCalibration().coverage_curve(*well_calibrated)
        assert np.all(np.diff(nominal) > 0)
        assert np.all(np.diff(observed) >= -1e-9)

    def test_plot_returns_a_plotly_figure(self, well_calibrated):
        import plotly.graph_objects as go

        fig = UncertaintyCalibration().plot_calibration(*well_calibrated)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            UncertaintyCalibration().report([1.0, 2.0], [1.0], [1.0])

    def test_rejects_negative_sigma(self):
        with pytest.raises(ValueError, match="sigma must be non-negative"):
            UncertaintyCalibration().report([1.0], [1.0], [-1.0])

    def test_rejects_bad_bin_count(self):
        with pytest.raises(ValueError, match="n_bins must be positive"):
            UncertaintyCalibration(n_bins=0).report([1.0], [1.0], [1.0])

    def test_zero_sigma_does_not_divide_by_zero(self):
        report = UncertaintyCalibration().report([1.0, 2.0], [1.0, 2.0], [0.0, 0.0])
        assert np.isfinite(report["coverage_95"])
