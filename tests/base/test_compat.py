"""Forwarding parameters only to the scikit-learn versions that have them.

``requires-python`` is ``>=3.9`` and the dependency floor is
``scikit-learn>=1.2``. ``MLPRegressor.loss`` and ``MDS.init`` arrived in
releases requiring Python 3.10, so passing them unconditionally raised
``TypeError`` on the oldest interpreter the package supports — which CI caught
on Python 3.9 while a 3.13 development environment was perfectly happy.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor

from qsarkit.base import accepts_parameter, forward_supported


class _Old:
    """Stands in for an older estimator that lacks the parameter."""

    def __init__(self, hidden_layer_sizes=(100,), random_state=None):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.random_state = random_state


class _Flexible:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestAcceptsParameter:
    def test_a_declared_parameter(self):
        assert accepts_parameter(RandomForestRegressor, "n_estimators") is True

    def test_an_absent_parameter(self):
        assert accepts_parameter(RandomForestRegressor, "not_a_parameter") is False

    def test_kwargs_accepts_anything(self):
        assert accepts_parameter(_Flexible, "whatever") is True

    def test_the_answer_is_cached(self):
        first = accepts_parameter(_Old, "hidden_layer_sizes")
        assert first is accepts_parameter(_Old, "hidden_layer_sizes")


class TestForwardSupported:
    def test_a_supported_parameter_is_passed_through(self):
        assert forward_supported(
            _Old, {"hidden_layer_sizes": (50,)}, {"hidden_layer_sizes": (100,)}
        ) == {"hidden_layer_sizes": (50,)}

    def test_an_unsupported_default_is_dropped_quietly(self):
        """The caller did not ask for it, so dropping it changes nothing."""
        assert forward_supported(
            _Old, {"loss": "squared_error"}, {"loss": "squared_error"}
        ) == {}

    def test_an_unsupported_non_default_is_refused(self):
        """Silently ignoring a requested value would give a different model."""
        with pytest.raises(TypeError, match="has no 'loss' parameter"):
            forward_supported(_Old, {"loss": "poisson"}, {"loss": "squared_error"})

    def test_the_error_names_the_version(self):
        import sklearn

        with pytest.raises(TypeError, match=sklearn.__version__):
            forward_supported(_Old, {"loss": "poisson"}, {"loss": "squared_error"})


class TestTheEstimatorsThatNeededIt:
    """End-to-end: both classes construct on whatever scikit-learn is here."""

    @pytest.fixture(scope="class")
    def data(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6))
        return X, X[:, 0] * 2.0

    def test_neural_network_constructs_and_fits(self, data):
        from qsarkit.models import NeuralNetworkQSAR

        X, y = data
        model = NeuralNetworkQSAR(max_iter=30, random_state=0).fit(X, y)
        assert model.predict(X).shape == (40,)

    def test_neural_network_still_exposes_loss_for_cloning(self):
        """`loss` stays in the signature, so get_params and clone are intact."""
        from sklearn.base import clone

        from qsarkit.models import NeuralNetworkQSAR

        assert "loss" in inspect.signature(NeuralNetworkQSAR.__init__).parameters
        assert isinstance(clone(NeuralNetworkQSAR()), NeuralNetworkQSAR)

    def test_mds_projects(self, data):
        from qsarkit.chemspace import ChemicalSpaceAnalyzer

        X, _ = data
        analyzer = ChemicalSpaceAnalyzer(
            method="mds", metric="euclidean", random_state=0
        )
        assert analyzer.fit_transform(X).shape == (40, 2)

    def test_mds_honours_n_init(self, data):
        from qsarkit.chemspace import ChemicalSpaceAnalyzer

        X, _ = data
        analyzer = ChemicalSpaceAnalyzer(
            method="mds", metric="euclidean", random_state=0, n_init=2
        )
        assert analyzer.fit_transform(X).shape == (40, 2)
