"""QSARRegressor / QSARClassifier with user-supplied estimators.

The facades accept any object following the scikit-learn ``fit``/``predict``
protocol -- XGBoost, LightGBM, CatBoost or a hand-written wrapper -- as a
class or an instance, plus per-method keyword arguments. These tests use
scikit-learn estimators and a deliberately non-sklearn stub so they run
without the optional boosting dependencies installed.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from qsarkit.models import QSARClassifier, QSARRegressor


@pytest.fixture(scope="module")
def regression_data():
    return make_regression(n_samples=40, n_features=5, random_state=0)


@pytest.fixture(scope="module")
def classification_data():
    return make_classification(n_samples=40, n_features=5, random_state=0)


class _BareRegressor:
    """An estimator following the protocol without inheriting from sklearn.

    Stands in for XGBoost/CatBoost, which do the same: they expose
    ``fit``/``predict`` but are not ``BaseEstimator`` subclasses, so an
    ``isinstance`` check would wrongly reject them.
    """

    def __init__(self, constant=0.0, note=None):
        self.constant = constant
        self.note = note
        self.fit_kwargs = None

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        self.mean_ = float(np.mean(y)) + self.constant
        return self

    def predict(self, X, **kwargs):
        self.predict_kwargs = kwargs
        return np.full(len(X), self.mean_)


class TestCustomRegressor:
    def test_accepts_an_instance(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(Ridge(alpha=2.0)).fit(X, y)
        assert type(model.estimator_).__name__ == "Ridge"
        assert model.estimator_.alpha == 2.0

    def test_accepts_a_class_with_model_params(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(Ridge, model_params={"alpha": 5.0}).fit(X, y)
        assert model.estimator_.alpha == 5.0

    def test_accepts_a_class_with_positional_model_args(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(_BareRegressor, model_args=[3.0]).fit(X, y)
        assert model.estimator_.constant == 3.0

    def test_accepts_a_non_sklearn_estimator(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(_BareRegressor()).fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_does_not_mutate_the_instance_it_was_given(self, regression_data):
        X, y = regression_data
        template = Ridge(alpha=1.0)
        QSARRegressor(template).fit(X, y)
        assert not hasattr(template, "coef_")

    def test_model_args_alongside_an_instance_is_an_error(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="model_args applies only"):
            QSARRegressor(Ridge(), model_args=[1.0]).fit(X, y)

    def test_fit_params_reach_the_estimator(self, regression_data):
        X, y = regression_data
        weights = np.linspace(0.5, 1.5, len(y))
        model = QSARRegressor(
            DecisionTreeRegressor, fit_params={"sample_weight": weights}
        ).fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_predict_params_reach_the_estimator(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(
            _BareRegressor, predict_params={"verbose": True}
        ).fit(X, y)
        model.predict(X)
        assert model.estimator_.predict_kwargs == {"verbose": True}

    def test_fit_params_are_seen_by_fit(self, regression_data):
        X, y = regression_data
        model = QSARRegressor(_BareRegressor, fit_params={"early_stop": 3}).fit(X, y)
        assert model.estimator_.fit_kwargs == {"early_stop": 3}

    def test_remains_a_scikit_learn_regressor(self):
        model = QSARRegressor(Ridge())
        assert is_regressor(model)
        assert type(clone(model)).__name__ == "QSARRegressor"

    def test_works_in_a_pipeline_and_grid_search(self, regression_data):
        from sklearn.model_selection import GridSearchCV

        X, y = regression_data
        search = GridSearchCV(
            QSARRegressor(Ridge),
            {"model_params": [{"alpha": 0.1}, {"alpha": 10.0}]},
            cv=3,
        ).fit(X, y)
        assert search.best_params_["model_params"]["alpha"] in (0.1, 10.0)

    def test_rejects_an_unknown_name(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="name must be one of"):
            QSARRegressor("nonesuch").fit(X, y)

    def test_rejects_an_object_that_is_not_an_estimator(self, regression_data):
        X, y = regression_data
        with pytest.raises(ValueError, match="scikit-learn API"):
            QSARRegressor(object()).fit(X, y)

    def test_builtin_names_still_work(self, regression_data):
        X, y = regression_data
        assert QSARRegressor("rf", random_state=0).fit(X, y).predict(X).shape == (40,)


class TestCustomClassifier:
    def test_accepts_a_class_with_model_params(self, classification_data):
        X, y = classification_data
        model = QSARClassifier(
            DecisionTreeClassifier, model_params={"max_depth": 3}
        ).fit(X, y)
        assert model.estimator_.max_depth == 3

    def test_predict_proba_is_forwarded(self, classification_data):
        X, y = classification_data
        model = QSARClassifier(DecisionTreeClassifier).fit(X, y)
        assert model.predict_proba(X).shape == (len(X), 2)

    def test_classes_come_from_the_estimator(self, classification_data):
        X, y = classification_data
        model = QSARClassifier(DecisionTreeClassifier).fit(X, y)
        assert model.classes_.tolist() == [0, 1]

    def test_classes_fall_back_to_the_labels_seen(self, classification_data):
        X, y = classification_data

        class _NoClassesAttr:
            def fit(self, X, y, **kwargs):
                self.seen_ = np.unique(y)
                return self

            def predict(self, X, **kwargs):
                return np.zeros(len(X), dtype=int)

        model = QSARClassifier(_NoClassesAttr()).fit(X, y)
        assert model.classes_.tolist() == [0, 1]

    def test_predict_proba_reports_backends_that_lack_it(self, classification_data):
        X, y = classification_data

        class _NoProba:
            def fit(self, X, y, **kwargs):
                return self

            def predict(self, X, **kwargs):
                return np.zeros(len(X), dtype=int)

        model = QSARClassifier(_NoProba()).fit(X, y)
        with pytest.raises(AttributeError, match="does not support predict_proba"):
            model.predict_proba(X)

    def test_fit_params_reach_the_estimator(self, classification_data):
        X, y = classification_data
        weights = np.ones(len(y))
        model = QSARClassifier(
            DecisionTreeClassifier, fit_params={"sample_weight": weights}
        ).fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_remains_a_scikit_learn_classifier(self):
        model = QSARClassifier(DecisionTreeClassifier())
        assert is_classifier(model)
        assert type(clone(model)).__name__ == "QSARClassifier"

    def test_builtin_names_still_work(self, classification_data):
        X, y = classification_data
        model = QSARClassifier("rf", random_state=0).fit(X, y)
        assert model.predict_proba(X).shape == (40, 2)
