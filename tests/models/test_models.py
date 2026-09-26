from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score

from qsarkit.models import (
    BaselineModel,
    ConsensusModel,
    GaussianProcessQSAR,
    NeuralNetworkQSAR,
    PLSRegressor,
    QSARClassifier,
    QSARRegressor,
    RandomForestQSAR,
    SVMQSAR,
    TanimotoKernel,
)

REGRESSOR_NAMES = [
    "rf", "svm", "gbm", "knn", "pls", "ridge", "lasso", "elasticnet", "mlp", "gp",
]
CLASSIFIER_NAMES = ["rf", "svm", "gbm", "knn", "ridge", "mlp", "gp", "pls"]


@pytest.fixture(scope="module")
def regression():
    return make_regression(
        n_samples=80, n_features=8, n_informative=4, noise=5.0, random_state=0
    )


@pytest.fixture(scope="module")
def classification():
    return make_classification(
        n_samples=80, n_features=8, n_informative=4, random_state=0
    )


@pytest.fixture(scope="module")
def fingerprints():
    """Binary fingerprints, where the Tanimoto kernel is valid."""
    rng = np.random.RandomState(0)
    X = (rng.rand(60, 32) > 0.7).astype(float)
    y = X[:, :5].sum(axis=1).astype(float)
    return X, y


class TestQSARRegressor:
    @pytest.mark.parametrize("name", REGRESSOR_NAMES)
    def test_every_backend_fits_and_predicts(self, name, regression):
        X, y = regression
        model = QSARRegressor(name, random_state=0).fit(X, y)
        assert model.predict(X).shape == y.shape

    @pytest.mark.parametrize("name", REGRESSOR_NAMES)
    def test_every_backend_scores(self, name, regression):
        X, y = regression
        assert np.isfinite(QSARRegressor(name, random_state=0).fit(X, y).score(X, y))

    def test_is_a_sklearn_regressor(self):
        assert is_regressor(QSARRegressor("rf"))

    def test_learns_a_learnable_problem(self, regression):
        X, y = regression
        assert QSARRegressor("rf", random_state=0).fit(X, y).score(X, y) > 0.7

    def test_model_params_are_forwarded(self, regression):
        X, y = regression
        model = QSARRegressor(
            "rf", random_state=0, model_params={"n_estimators": 7}
        ).fit(X, y)
        assert model.estimator_.n_estimators == 7

    def test_knn_backend_uses_the_tanimoto_estimator(self, fingerprints):
        from qsarkit.neighbors import JaccardKNeighborsRegressor

        X, y = fingerprints
        model = QSARRegressor("knn").fit(X, y)
        assert isinstance(model.estimator_, JaccardKNeighborsRegressor)

    def test_clone_roundtrip(self):
        model = QSARRegressor("svm", random_state=3)
        assert clone(model).get_params() == model.get_params()

    def test_works_in_cross_validation(self, regression):
        X, y = regression
        assert cross_val_score(
            QSARRegressor("ridge"), X, y, cv=3
        ).shape == (3,)

    def test_rejects_an_unknown_name(self, regression):
        X, y = regression
        with pytest.raises(ValueError):
            QSARRegressor("nonexistent").fit(X, y)


class TestQSARClassifier:
    @pytest.mark.parametrize("name", CLASSIFIER_NAMES)
    def test_every_backend_fits_and_predicts(self, name, classification):
        X, y = classification
        model = QSARClassifier(name, random_state=0).fit(X, y)
        assert model.predict(X).shape == y.shape

    def test_is_a_sklearn_classifier(self):
        assert is_classifier(QSARClassifier("rf"))

    def test_learns_a_learnable_problem(self, classification):
        X, y = classification
        assert QSARClassifier("rf", random_state=0).fit(X, y).score(X, y) > 0.8

    def test_predict_proba_sums_to_one(self, classification):
        X, y = classification
        proba = QSARClassifier("rf", random_state=0).fit(X, y).predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_is_refused_where_unavailable(self, classification):
        X, y = classification
        model = QSARClassifier("ridge", random_state=0).fit(X, y)
        with pytest.raises(AttributeError, match="predict_proba"):
            model.predict_proba(X)

    def test_classes_are_exposed(self, classification):
        X, y = classification
        assert QSARClassifier("rf", random_state=0).fit(X, y).classes_.tolist() == [
            0, 1
        ]

    def test_works_in_cross_validation(self, classification):
        X, y = classification
        assert cross_val_score(
            QSARClassifier("rf", random_state=0), X, y, cv=3
        ).shape == (3,)

    def test_rejects_an_unknown_name(self, classification):
        X, y = classification
        with pytest.raises(ValueError):
            QSARClassifier("nonexistent").fit(X, y)


class TestPLSRegressor:
    def test_fits_and_predicts(self, regression):
        X, y = regression
        model = PLSRegressor(n_components=3).fit(X, y)
        assert model.predict(X).shape == y.shape

    def test_predict_returns_a_flat_array(self, regression):
        # sklearn's PLSRegression returns (n, 1); a QSAR API should not
        X, y = regression
        assert PLSRegressor(n_components=2).fit(X, y).predict(X).ndim == 1

    def test_vip_scores_have_one_entry_per_feature(self, regression):
        X, y = regression
        assert PLSRegressor(n_components=3).fit(X, y).vip_scores_.shape == (
            X.shape[1],
        )

    def test_vip_scores_are_non_negative(self, regression):
        X, y = regression
        assert (PLSRegressor(n_components=3).fit(X, y).vip_scores_ >= 0).all()

    def test_vip_identifies_informative_features(self):
        # only the first two columns drive y; their VIP must lead
        rng = np.random.RandomState(0)
        X = rng.normal(size=(120, 6))
        y = X[:, 0] * 3 + X[:, 1] * 2 + rng.normal(scale=0.1, size=120)
        vip = PLSRegressor(n_components=3).fit(X, y).vip_scores_
        assert set(np.argsort(-vip)[:2].tolist()) == {0, 1}

    def test_vip_mean_square_is_about_one(self, regression):
        # the VIP normalization makes the mean of squared VIPs equal 1,
        # which is what justifies the conventional "VIP > 1" cutoff
        X, y = regression
        vip = PLSRegressor(n_components=3).fit(X, y).vip_scores_
        assert float(np.mean(vip**2)) == pytest.approx(1.0, rel=1e-6)

    def test_is_a_sklearn_regressor(self):
        assert is_regressor(PLSRegressor())

    def test_clone_roundtrip(self):
        model = PLSRegressor(n_components=4)
        assert clone(model).get_params() == model.get_params()

    def test_works_in_cross_validation(self, regression):
        X, y = regression
        assert cross_val_score(PLSRegressor(n_components=2), X, y, cv=3).shape == (3,)


class TestTanimotoKernel:
    def test_matches_the_similarity_matrix(self, fingerprints):
        from qsarkit.neighbors import tanimoto_similarity_matrix

        X, _ = fingerprints
        assert np.allclose(TanimotoKernel()(X), tanimoto_similarity_matrix(X))

    def test_diagonal_is_sigma_squared(self, fingerprints):
        X, _ = fingerprints
        assert np.allclose(TanimotoKernel(sigma_0=2.0).diag(X), 4.0)

    def test_is_symmetric_and_psd(self, fingerprints):
        X, _ = fingerprints
        K = TanimotoKernel()(X)
        assert np.allclose(K, K.T)
        # a valid kernel matrix has no meaningfully negative eigenvalues
        assert np.linalg.eigvalsh(K).min() > -1e-8

    def test_two_argument_form(self, fingerprints):
        X, _ = fingerprints
        assert TanimotoKernel()(X[:5], X).shape == (5, len(X))

    def test_rejects_negative_input(self):
        with pytest.raises(ValueError, match="non-negative"):
            TanimotoKernel()(np.array([[-1.0, 2.0]]))

    def test_rejects_negative_second_argument(self, fingerprints):
        X, _ = fingerprints
        with pytest.raises(ValueError, match="negative values"):
            TanimotoKernel()(X, np.array([[-1.0] * X.shape[1]]))

    def test_gradient_shape(self, fingerprints):
        X, _ = fingerprints
        K, grad = TanimotoKernel()(X[:5], eval_gradient=True)
        assert K.shape == (5, 5)
        assert grad.shape[:2] == (5, 5)

    def test_gradient_requires_a_single_argument(self, fingerprints):
        X, _ = fingerprints
        with pytest.raises(ValueError, match="Gradient can only be evaluated"):
            TanimotoKernel()(X, X, eval_gradient=True)


class TestGaussianProcessQSAR:
    def test_uses_tanimoto_on_fingerprints(self, fingerprints):
        X, y = fingerprints
        model = GaussianProcessQSAR(random_state=0).fit(X, y)
        assert isinstance(model.kernel_, TanimotoKernel) or "Tanimoto" in str(
            model.kernel_
        )

    def test_uses_rbf_on_signed_descriptors(self, regression):
        # Tanimoto is not positive semi-definite on signed data, so the
        # default must fall back rather than fail inside Cholesky
        X, y = regression
        model = GaussianProcessQSAR(random_state=0).fit(X[:40], y[:40])
        assert "Tanimoto" not in str(model.kernel_)

    def test_kernel_argument_is_preserved_for_cloning(self, fingerprints):
        X, y = fingerprints
        model = GaussianProcessQSAR(random_state=0).fit(X, y)
        # the fitted estimator must still report kernel=None so clone works
        assert model.get_params()["kernel"] is None
        assert clone(model).get_params()["kernel"] is None

    def test_predicts_with_uncertainty(self, fingerprints):
        X, y = fingerprints
        model = GaussianProcessQSAR(random_state=0).fit(X, y)
        mean, std = model.predict(X, return_std=True)
        assert mean.shape == std.shape == y.shape
        assert (std >= 0).all()

    def test_explicit_kernel_is_used(self, fingerprints):
        X, y = fingerprints
        model = GaussianProcessQSAR(
            kernel=TanimotoKernel(sigma_0=2.0), random_state=0
        ).fit(X, y)
        assert model.get_params()["kernel"] is not None


class TestOtherEstimators:
    def test_random_forest_qsar(self, regression):
        X, y = regression
        model = RandomForestQSAR(n_estimators=20, random_state=0).fit(X, y)
        assert model.score(X, y) > 0.7
        assert model.feature_importances_.shape == (X.shape[1],)

    def test_random_forest_defaults_to_many_trees(self):
        assert RandomForestQSAR().n_estimators == 500

    def test_svm_qsar(self, regression):
        X, y = regression
        model = SVMQSAR().fit(X, y)
        assert model.predict(X).shape == y.shape

    def test_neural_network_qsar(self, regression):
        X, y = regression
        model = NeuralNetworkQSAR(max_iter=50, random_state=0).fit(X, y)
        assert model.predict(X).shape == y.shape

    @pytest.mark.parametrize("cls", [RandomForestQSAR, SVMQSAR, NeuralNetworkQSAR])
    def test_are_sklearn_regressors(self, cls):
        assert is_regressor(cls())

    @pytest.mark.parametrize("cls", [RandomForestQSAR, SVMQSAR, NeuralNetworkQSAR])
    def test_clone_roundtrip(self, cls):
        model = cls()
        assert clone(model).get_params() == model.get_params()


class TestBaselineModel:
    def test_regression_baseline_scores_zero(self, regression):
        # predicting the training mean gives R2 exactly 0 by construction
        X, y = regression
        assert BaselineModel().fit(X, y).score(X, y) == pytest.approx(0.0)

    def test_regression_baseline_predicts_the_mean(self, regression):
        X, y = regression
        assert BaselineModel().fit(X, y).predict(X) == pytest.approx(y.mean())

    def test_classification_baseline(self, classification):
        X, y = classification
        model = BaselineModel(task="classification").fit(X, y)
        assert model.predict(X).shape == y.shape
        assert 0.0 <= model.score(X, y) <= 1.0

    def test_classification_baseline_probabilities(self, classification):
        X, y = classification
        proba = BaselineModel(task="classification").fit(X, y).predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_estimator_type_follows_the_task(self):
        assert is_regressor(BaselineModel(task="regression"))
        assert is_classifier(BaselineModel(task="classification"))

    def test_a_real_model_beats_the_baseline(self, regression):
        # this is the OECD null-model comparison in one line
        X, y = regression
        assert QSARRegressor("rf", random_state=0).fit(X, y).score(X, y) > (
            BaselineModel().fit(X, y).score(X, y)
        )

    def test_requires_fitting(self, regression):
        from qsarkit.base import ModelNotFittedError

        X, _ = regression
        with pytest.raises(ModelNotFittedError):
            BaselineModel().predict(X)

    def test_custom_strategy(self, regression):
        X, y = regression
        model = BaselineModel(strategy="median").fit(X, y)
        assert model.predict(X) == pytest.approx(np.median(y))


class TestConsensusModel:
    def test_averaging_regression(self, regression):
        X, y = regression
        model = ConsensusModel(
            estimators=[("lr", LinearRegression()), ("ridge", Ridge())],
            task="regression", method="averaging",
        ).fit(X, y)
        assert model.predict(X).shape == y.shape
        assert model.score(X, y) > 0.5

    def test_consensus_beats_its_weakest_member(self, regression):
        X, y = regression
        weak = QSARRegressor("knn", random_state=0).fit(X, y).score(X, y)
        consensus = ConsensusModel(
            estimators=[
                ("rf", QSARRegressor("rf", random_state=0)),
                ("ridge", Ridge()),
            ],
            task="regression",
        ).fit(X, y)
        assert consensus.score(X, y) > weak

    def test_stacking_method(self, regression):
        X, y = regression
        model = ConsensusModel(
            estimators=[("lr", LinearRegression()), ("ridge", Ridge())],
            task="regression", method="stacking", cv=3,
        ).fit(X, y)
        assert np.isfinite(model.predict(X)).all()

    def test_weights_are_honoured(self, regression):
        X, y = regression
        model = ConsensusModel(
            estimators=[("lr", LinearRegression()), ("ridge", Ridge())],
            task="regression", weights=[0.9, 0.1],
        ).fit(X, y)
        assert model.predict(X).shape == y.shape

    def test_classification_task(self, classification):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        X, y = classification
        model = ConsensusModel(
            estimators=[
                ("rf", RandomForestClassifier(n_estimators=10, random_state=0)),
                ("lr", LogisticRegression(max_iter=500)),
            ],
            task="classification",
        ).fit(X, y)
        assert model.score(X, y) > 0.7
        assert np.allclose(model.predict_proba(X).sum(axis=1), 1.0)

    def test_estimator_type_follows_the_task(self):
        assert is_regressor(
            ConsensusModel(estimators=[("lr", LinearRegression())])
        )
        assert is_classifier(
            ConsensusModel(
                estimators=[("lr", LinearRegression())], task="classification"
            )
        )

    def test_rejects_an_unknown_task(self, regression):
        X, y = regression
        with pytest.raises(ValueError, match="task must be"):
            ConsensusModel(
                estimators=[("lr", LinearRegression())], task="bogus"
            ).fit(X, y)

    def test_requires_fitting(self, regression):
        from qsarkit.base import ModelNotFittedError

        X, _ = regression
        with pytest.raises(ModelNotFittedError):
            ConsensusModel(estimators=[("lr", LinearRegression())]).predict(X)


class TestModelParamsOverrideFacadeDefaults:
    """``model_params`` must win over whatever the facade sets for a backend.

    ``model_params`` is the documented way to configure a backend, so passing
    a keyword the facade also sets has to work. Splatting both into the
    constructor raised ``TypeError: got multiple values for keyword argument``
    instead -- ``QSARClassifier("rf", model_params={"n_estimators": 100})``
    failed outright, and that is the package's default estimator.
    """

    def test_classifier_n_estimators_is_overridable(self):
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 100, "n_jobs": -1}
        )
        built = model._build()
        assert built.n_estimators == 100
        assert built.n_jobs == -1
        # The facade's own settings survive where the caller did not override.
        assert built.random_state == 0

    def test_regressor_n_estimators_is_overridable(self):
        built = QSARRegressor(
            "rf", random_state=0, model_params={"n_estimators": 50}
        )._build()
        assert built.n_estimators == 50

    def test_overriding_one_default_keeps_the_others(self):
        built = QSARClassifier("svm", model_params={"kernel": "linear"})._build()
        assert built.kernel == "linear"
        # probability=True is what makes predict_proba available, so the facade
        # must keep setting it.
        assert built.probability is True

    def test_it_fits_end_to_end(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6))
        y = (X[:, 0] > 0).astype(int)
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 10}
        ).fit(X, y)
        assert model.predict_proba(X).shape == (40, 2)
