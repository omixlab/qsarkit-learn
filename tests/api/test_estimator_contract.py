"""The scikit-learn contract, checked across the whole public API.

The package's central promise is that everything composes with
scikit-learn. That rests on rules which are easy to break in one class and
hard to notice: ``__init__`` storing parameters unmodified, ``get_params``
round-tripping, ``clone`` producing an equivalent estimator, and tags
resolving so ``is_regressor``/``is_classifier`` answer correctly.

These are collected per class so a failure names the offender.
"""

from __future__ import annotations

import importlib
import inspect
import warnings
from typing import Any, List, Tuple

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor

import qsarkit


def _public_estimators() -> List[Tuple[str, str, Any]]:
    """Every exported class that is a scikit-learn estimator."""
    found: List[Tuple[str, str, Any]] = []
    for sub in qsarkit._SUBPACKAGES:
        module = importlib.import_module(f"qsarkit.{sub}")
        for name in getattr(module, "__all__", []):
            obj = getattr(module, name)
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseEstimator)
                and not inspect.isabstract(obj)
            ):
                found.append((sub, name, obj))
    return sorted(found, key=lambda t: (t[0], t[1]))


def _constructible() -> List[Tuple[str, str, Any]]:
    """Those constructible with no arguments, which most are."""
    out = []
    for sub, name, cls in _public_estimators():
        try:
            cls()
        except TypeError:
            continue
        out.append((sub, name, cls))
    return out


ESTIMATORS = _constructible()
IDS = [f"{sub}.{name}" for sub, name, _ in ESTIMATORS]


def test_the_scan_found_estimators() -> None:
    """Guard against the parametrization silently becoming empty."""
    assert len(ESTIMATORS) > 40


@pytest.mark.parametrize(("sub", "name", "cls"), ESTIMATORS, ids=IDS)
def test_get_params_round_trips(sub: str, name: str, cls: Any) -> None:
    instance = cls()
    params = instance.get_params()
    instance.set_params(**params)
    assert instance.get_params() == params


@pytest.mark.parametrize(("sub", "name", "cls"), ESTIMATORS, ids=IDS)
def test_clone_produces_an_equal_estimator(sub: str, name: str, cls: Any) -> None:
    """``clone`` is what ``GridSearchCV`` and ``cross_val_score`` rely on."""
    instance = cls()
    assert clone(instance).get_params() == instance.get_params()


@pytest.mark.parametrize(("sub", "name", "cls"), ESTIMATORS, ids=IDS)
def test_init_stores_parameters_unmodified(sub: str, name: str, cls: Any) -> None:
    """``__init__`` must not substitute defaults or coerce values.

    An ``__init__`` that writes ``self.x = x or Default()`` makes
    ``get_params`` report something the caller never passed, and ``clone``
    then yields a non-identical estimator. Resolve defaults at fit time
    instead.
    """
    signature = inspect.signature(cls.__init__)
    defaults = {
        parameter.name: parameter.default
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.default is not inspect.Parameter.empty
    }
    if not defaults:
        pytest.skip("no defaulted parameters")

    stored = cls().get_params()
    mismatched = {
        key: (defaults[key], stored[key])
        for key in defaults
        if key in stored and stored[key] is not defaults[key]
        and stored[key] != defaults[key]
    }
    assert not mismatched, (
        f"{sub}.{name}.__init__ altered these parameters: {mismatched}. "
        "Store them as given and resolve the default where it is used."
    )


@pytest.mark.parametrize(("sub", "name", "cls"), ESTIMATORS, ids=IDS)
def test_repr_does_not_raise(sub: str, name: str, cls: Any) -> None:
    assert repr(cls())


class TestTaskTags:
    """Mixin order matters: a mixin after ``BaseEstimator`` breaks tags."""

    def test_regressors_are_recognized(self) -> None:
        from qsarkit.models import (
            GaussianProcessQSAR,
            PLSRegressor,
            QSARRegressor,
            RandomForestQSAR,
            SVMQSAR,
        )
        from qsarkit.neighbors import JaccardKNeighborsRegressor

        for cls in (
            QSARRegressor,
            RandomForestQSAR,
            SVMQSAR,
            PLSRegressor,
            GaussianProcessQSAR,
            JaccardKNeighborsRegressor,
        ):
            instance = cls("rf") if cls is QSARRegressor else cls()
            assert is_regressor(instance), cls.__name__
            assert not is_classifier(instance), cls.__name__

    def test_classifiers_are_recognized(self) -> None:
        from qsarkit.models import QSARClassifier
        from qsarkit.neighbors import JaccardKNeighborsClassifier

        for instance in (QSARClassifier("rf"), JaccardKNeighborsClassifier()):
            assert is_classifier(instance), type(instance).__name__
            assert not is_regressor(instance), type(instance).__name__

    def test_regressors_expose_score(self) -> None:
        """``RegressorMixin.score`` is what ``cross_val_score`` calls."""
        from qsarkit.models import QSARRegressor

        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(30, 4)), rng.normal(size=30)
        model = QSARRegressor("ridge").fit(X, y)
        assert isinstance(model.score(X, y), float)


class TestPipelineComposition:
    def test_transformers_compose_in_a_pipeline(self) -> None:
        from rdkit import Chem
        from sklearn.pipeline import Pipeline

        from qsarkit.models import QSARRegressor
        from qsarkit.representation import MorganFingerprint

        mols = [Chem.MolFromSmiles(s) for s in
                ("CCO", "CCN", "CCC", "c1ccccc1", "c1ccncc1", "CCCl")]
        y = np.arange(len(mols), dtype=float)
        pipeline = Pipeline([
            ("features", MorganFingerprint(n_bits=64)),
            ("model", QSARRegressor("ridge")),
        ])
        assert pipeline.fit(mols, y).predict(mols).shape == (len(mols),)

    def test_a_pipeline_survives_grid_search(self) -> None:
        from rdkit import Chem
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        from qsarkit.models import QSARRegressor
        from qsarkit.representation import MorganFingerprint

        mols = [Chem.MolFromSmiles(s) for s in
                ("CCO", "CCN", "CCC", "c1ccccc1", "c1ccncc1", "CCCl",
                 "CCBr", "CCI", "c1ccoc1")]
        y = np.arange(len(mols), dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = GridSearchCV(
                Pipeline([
                    ("features", MorganFingerprint(n_bits=64)),
                    ("model", QSARRegressor("ridge")),
                ]),
                {"features__radius": [1, 2]},
                cv=3,
            ).fit(mols, y)
        assert search.best_params_["features__radius"] in (1, 2)

    def test_transformers_expose_feature_names_where_meaningful(self) -> None:
        from rdkit import Chem

        from qsarkit.representation import PhysicochemicalDescriptors

        block = PhysicochemicalDescriptors()
        names = block.get_feature_names_out()
        X = block.transform([Chem.MolFromSmiles("CCO")])
        assert len(names) == X.shape[1]
