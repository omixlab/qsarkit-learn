from __future__ import annotations

import numpy as np
import pytest

from qsarkit.applicability import (
    ADAnalyzer,
    BoundingBoxAD,
    ConvexHullAD,
    DistanceToModelAD,
    EnsembleAD,
    IsolationForestAD,
    KernelDensityAD,
    KNNApplicabilityDomain,
    LeverageAD,
    PCABoundingBoxAD,
    RangeAD,
    TanimotoSimilarityAD,
)
from qsarkit.base import ModelNotFittedError


@pytest.fixture(scope="module")
def X():
    return np.random.RandomState(0).normal(size=(80, 4))


@pytest.fixture(scope="module")
def far_point():
    return np.full((1, 4), 50.0)


# Every AD flavour, constructed so each is valid for a 4-feature, 80-row set.
ALL_DOMAINS = [
    pytest.param(LeverageAD, {}, id="leverage"),
    pytest.param(DistanceToModelAD, {}, id="distance-euclidean"),
    pytest.param(DistanceToModelAD, {"metric": "mahalanobis"}, id="distance-mahalanobis"),
    pytest.param(DistanceToModelAD, {"metric": "cityblock"}, id="distance-cityblock"),
    pytest.param(KNNApplicabilityDomain, {"n_neighbors": 5}, id="knn"),
    pytest.param(RangeAD, {}, id="range"),
    pytest.param(PCABoundingBoxAD, {"n_components": 2}, id="pca-box"),
    pytest.param(ConvexHullAD, {"n_components": 2}, id="convex-hull"),
    pytest.param(KernelDensityAD, {}, id="kde"),
    pytest.param(IsolationForestAD, {"random_state": 0}, id="isolation-forest"),
    pytest.param(EnsembleAD, {}, id="ensemble"),
]


class TestDomainContract:
    """Behaviour every applicability domain must satisfy."""

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_far_point_is_out_of_domain(self, cls, kwargs, X, far_point):
        ad = cls(**kwargs).fit(X)
        assert not ad.predict(far_point)[0]

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_training_set_is_mostly_in_domain(self, cls, kwargs, X):
        ad = cls(**kwargs).fit(X)
        assert ad.coverage(X) >= 0.9

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_far_point_scores_worse_than_training(self, cls, kwargs, X, far_point):
        ad = cls(**kwargs).fit(X)
        assert ad.score_samples(far_point)[0] >= np.median(ad.score_samples(X))

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_decision_function_sign_matches_predict(self, cls, kwargs, X, far_point):
        ad = cls(**kwargs).fit(X)
        both = np.vstack([X[:5], far_point])
        assert np.array_equal(ad.decision_function(both) >= 0, ad.predict(both))

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_methods_require_fitting(self, cls, kwargs, X):
        with pytest.raises(ModelNotFittedError):
            cls(**kwargs).score_samples(X)

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_rejects_non_2d_input(self, cls, kwargs):
        with pytest.raises(ValueError, match="2-dimensional"):
            cls(**kwargs).fit(np.zeros(5))

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_rejects_feature_count_mismatch(self, cls, kwargs, X):
        ad = cls(**kwargs).fit(X)
        with pytest.raises(ValueError, match="expected 4"):
            ad.score_samples(np.zeros((2, 7)))

    @pytest.mark.parametrize("cls, kwargs", ALL_DOMAINS)
    def test_sklearn_clone_roundtrip(self, cls, kwargs):
        from sklearn.base import clone

        ad = cls(**kwargs)
        assert clone(ad).get_params() == ad.get_params()


class TestLeverageAD:
    def test_threshold_matches_the_published_formula(self, X):
        ad = LeverageAD().fit(X)
        n, p = X.shape
        assert ad.threshold_ == pytest.approx(3 * (p + 1) / n)

    def test_threshold_factor_is_honoured(self, X):
        assert LeverageAD(threshold_factor=2.0).fit(X).threshold_ == pytest.approx(
            2 * (X.shape[1] + 1) / X.shape[0]
        )

    def test_leverage_is_non_negative(self, X):
        assert (LeverageAD().fit(X).score_samples(X) >= 0).all()

    def test_handles_rank_deficient_descriptors(self):
        # a duplicated column makes X'X singular; pinv must cope
        base = np.random.RandomState(1).normal(size=(30, 2))
        X = np.hstack([base, base[:, :1]])
        assert np.isfinite(LeverageAD().fit(X).score_samples(X)).all()

    def test_empty_training_set_raises(self):
        with pytest.raises(ValueError, match="empty training set"):
            LeverageAD().fit(np.zeros((0, 3)))


class TestDistanceToModelAD:
    def test_centroid_is_the_closest_point(self, X):
        ad = DistanceToModelAD().fit(X)
        assert ad.score_samples(X.mean(axis=0, keepdims=True))[0] == pytest.approx(
            0.0, abs=1e-9
        )

    def test_percentile_controls_coverage(self, X):
        loose = DistanceToModelAD(percentile=99).fit(X)
        tight = DistanceToModelAD(percentile=50).fit(X)
        assert loose.coverage(X) > tight.coverage(X)

    def test_invalid_metric_raises(self, X):
        with pytest.raises(ValueError, match="metric must be"):
            DistanceToModelAD(metric="cosine").fit(X)

    def test_invalid_percentile_raises(self, X):
        with pytest.raises(ValueError, match="percentile must be"):
            DistanceToModelAD(percentile=101).fit(X)


class TestKNNApplicabilityDomain:
    def test_too_many_neighbors_raises(self, X):
        with pytest.raises(ValueError, match="must be smaller than"):
            KNNApplicabilityDomain(n_neighbors=len(X)).fit(X)

    def test_invalid_metric_raises(self, X):
        with pytest.raises(ValueError, match="metric must be"):
            KNNApplicabilityDomain(metric="manhattan").fit(X)

    def test_invalid_percentile_raises(self, X):
        with pytest.raises(ValueError, match="percentile must be"):
            KNNApplicabilityDomain(percentile=-1).fit(X)

    def test_tanimoto_metric_on_fingerprints(self):
        rng = np.random.RandomState(3)
        fps = (rng.rand(40, 64) > 0.7).astype(float)
        ad = KNNApplicabilityDomain(n_neighbors=3, metric="tanimoto").fit(fps)
        assert ad.coverage(fps) >= 0.9
        assert not ad.predict(np.zeros((1, 64)))[0]

    def test_empty_training_set_raises(self):
        with pytest.raises(ValueError, match="empty training set"):
            KNNApplicabilityDomain().fit(np.zeros((0, 3)))


class TestRangeAD:
    def test_inside_and_outside_the_box(self):
        ad = RangeAD().fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert ad.predict(np.array([[0.5, 0.5]]))[0]
        assert not ad.predict(np.array([[9.0, 0.5]]))[0]

    def test_score_counts_violated_descriptors(self):
        ad = RangeAD().fit(np.array([[0.0, 0.0], [1.0, 1.0]]))
        assert ad.score_samples(np.array([[9.0, 9.0]]))[0] == 2.0
        assert ad.score_samples(np.array([[9.0, 0.5]]))[0] == 1.0

    def test_tolerance_widens_the_box(self):
        X = np.array([[0.0], [1.0]])
        assert not RangeAD().fit(X).predict(np.array([[1.05]]))[0]
        assert RangeAD(tolerance=0.1).fit(X).predict(np.array([[1.05]]))[0]

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            RangeAD(tolerance=-0.1).fit(np.array([[0.0], [1.0]]))

    def test_bounding_box_is_an_alias(self):
        assert BoundingBoxAD is RangeAD


class TestConvexHullAD:
    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="needs more than"):
            ConvexHullAD(n_components=2).fit(np.zeros((2, 2)))

    def test_interior_point_is_inside(self, X):
        ad = ConvexHullAD(n_components=2).fit(X)
        assert ad.score_samples(X.mean(axis=0, keepdims=True))[0] == 0.0


class TestTanimotoSimilarityAD:
    @pytest.fixture
    def fps(self):
        return np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=float)

    def test_identical_fingerprint_is_inside(self, fps):
        ad = TanimotoSimilarityAD(threshold=0.5).fit(fps)
        assert ad.predict(np.array([[1, 1, 0, 0]], dtype=float))[0]

    def test_disjoint_fingerprint_is_outside(self, fps):
        ad = TanimotoSimilarityAD(threshold=0.5).fit(fps)
        assert not ad.predict(np.array([[0, 0, 0, 1]], dtype=float))[0]

    def test_similarity_is_reported_directly(self, fps):
        ad = TanimotoSimilarityAD().fit(fps)
        sim = ad.similarity_to_training(np.array([[1, 1, 0, 0]], dtype=float))
        assert sim[0] == pytest.approx(1.0)

    def test_score_complements_similarity(self, fps):
        ad = TanimotoSimilarityAD().fit(fps)
        q = np.array([[1, 0, 0, 0]], dtype=float)
        assert ad.score_samples(q)[0] == pytest.approx(
            1 - ad.similarity_to_training(q)[0]
        )

    def test_invalid_threshold_raises(self, fps):
        with pytest.raises(ValueError, match="threshold must be"):
            TanimotoSimilarityAD(threshold=1.5).fit(fps)

    def test_too_many_neighbors_raises(self, fps):
        with pytest.raises(ValueError, match="exceeds the training set size"):
            TanimotoSimilarityAD(n_neighbors=99).fit(fps)


class TestEnsembleAD:
    def test_voting_modes_change_strictness(self, X, far_point):
        both = np.vstack([X, far_point])
        strict = EnsembleAD(voting="any").fit(X)
        loose = EnsembleAD(voting="unanimous").fit(X)
        assert strict.coverage(both) <= loose.coverage(both)

    def test_invalid_voting_raises(self, X):
        with pytest.raises(ValueError, match="voting must be"):
            EnsembleAD(voting="plurality").fit(X)

    def test_empty_member_list_raises(self, X):
        with pytest.raises(ValueError, match="at least one member"):
            EnsembleAD(estimators=[]).fit(X)

    def test_custom_members_are_used(self, X, far_point):
        ad = EnsembleAD(estimators=[RangeAD(), LeverageAD()]).fit(X)
        members = ad.member_predictions(far_point)
        assert list(members.columns) == ["RangeAD", "LeverageAD"]
        assert not members.iloc[0].any()

    def test_score_is_the_fraction_outside(self, X, far_point):
        ad = EnsembleAD(estimators=[RangeAD(), LeverageAD()]).fit(X)
        assert ad.score_samples(far_point)[0] == pytest.approx(1.0)


class TestADAnalyzer:
    @pytest.fixture
    def setup(self, X):
        rng = np.random.RandomState(7)
        y = X[:, 0] * 2.0
        y_pred = y + rng.normal(scale=0.1, size=len(y))
        return X, y, y_pred

    def test_report_keys_and_coverage(self, setup):
        X, y, y_pred = setup
        analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X)
        report = analyzer.report(X, y, y_pred)
        assert set(report) == {
            "coverage", "n_inside", "n_outside", "rmse_inside",
            "rmse_outside", "mae_inside", "mae_outside", "rmse_ratio",
        }
        assert report["n_inside"] + report["n_outside"] == len(X)

    def test_domain_separates_good_from_bad_predictions(self, X):
        # deliberately wreck the predictions of the out-of-domain points
        rng = np.random.RandomState(11)
        X_all = np.vstack([X, np.full((5, 4), 20.0)])
        y = X_all[:, 0] * 2.0
        y_pred = y + rng.normal(scale=0.05, size=len(y))
        y_pred[-5:] += 50.0

        analyzer = ADAnalyzer(RangeAD()).fit(X)
        report = analyzer.report(X_all, y, y_pred)
        assert report["rmse_outside"] > report["rmse_inside"]
        assert report["rmse_ratio"] > 1.0

    def test_empty_subset_gives_nan(self, setup):
        X, y, y_pred = setup
        analyzer = ADAnalyzer(RangeAD()).fit(X)
        report = analyzer.report(X, y, y_pred)
        assert report["n_outside"] == 0
        assert np.isnan(report["rmse_outside"])

    def test_report_rejects_mismatched_shapes(self, setup):
        X, y, y_pred = setup
        analyzer = ADAnalyzer(RangeAD()).fit(X)
        with pytest.raises(ValueError, match="y_pred has"):
            analyzer.report(X, y, y_pred[:5])
        with pytest.raises(ValueError, match="but y_true has"):
            analyzer.report(X, y[:5], y_pred[:5])

    def test_accuracy_vs_coverage_curve(self, setup):
        X, y, y_pred = setup
        analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X)
        curve = analyzer.accuracy_vs_coverage(X, y, y_pred, n_points=10)
        assert list(curve.columns) == ["coverage", "n_samples", "rmse", "mae"]
        assert curve["coverage"].is_monotonic_increasing
        assert curve["coverage"].iloc[-1] == pytest.approx(1.0)

    def test_curve_rejects_bad_arguments(self, setup):
        X, y, y_pred = setup
        analyzer = ADAnalyzer(RangeAD()).fit(X)
        with pytest.raises(ValueError, match="same length"):
            analyzer.accuracy_vs_coverage(X, y[:5], y_pred[:5])
        with pytest.raises(ValueError, match="n_points must be positive"):
            analyzer.accuracy_vs_coverage(X, y, y_pred, n_points=0)

    def test_plots_return_plotly_figures(self, setup):
        import plotly.graph_objects as go

        X, y, y_pred = setup
        analyzer = ADAnalyzer(LeverageAD()).fit(X)
        assert isinstance(analyzer.plot_accuracy_vs_coverage(X, y, y_pred), go.Figure)
        assert isinstance(analyzer.williams_plot(X, y, y_pred), go.Figure)

    def test_williams_plot_handles_zero_variance_residuals(self, X):
        analyzer = ADAnalyzer(LeverageAD()).fit(X)
        y = np.ones(len(X))
        fig = analyzer.williams_plot(X, y, y)
        assert len(fig.data) == 1
