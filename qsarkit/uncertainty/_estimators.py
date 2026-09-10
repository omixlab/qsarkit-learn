"""Uncertainty estimators: ensemble spread, MC dropout, GP variance, quantiles."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone

from qsarkit.base.exceptions import ModelNotFittedError

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

__all__ = [
    "BaseUncertaintyEstimator",
    "EnsembleUncertainty",
    "MCDropoutUncertainty",
    "GaussianProcessUncertainty",
    "QuantileRegressionUncertainty",
]


class BaseUncertaintyEstimator(BaseEstimator, ABC):
    """Common interface: ``fit``, ``predict``, and ``predict_uncertainty``.

    Every estimator here returns a point prediction and a per-sample
    standard deviation, so they are interchangeable wherever a model's
    confidence is needed — active-learning acquisition functions,
    applicability-domain scoring, or simply reporting error bars.

    References
    ----------
    - Hirschfeld, L. et al. (2020). "Uncertainty Quantification Using
      Neural Networks for Molecular Property Prediction." J. Chem. Inf.
      Model., 60(8), 3770-3780. https://doi.org/10.1021/acs.jcim.0c00502
    - Scalia, G. et al. (2020). "Evaluating Scalable Uncertainty
      Estimation Methods for Deep Learning-Based Molecular Property
      Prediction." J. Chem. Inf. Model., 60(6), 2697-2717.
      https://doi.org/10.1021/acs.jcim.9b00975
    """

    @abstractmethod
    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "BaseUncertaintyEstimator":
        """Fit the underlying model(s)."""

    @abstractmethod
    def predict_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Return ``(mean, std)`` per sample."""

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Point predictions (the mean of :meth:`predict_uncertainty`)."""
        return self.predict_uncertainty(X)[0]

    def predict_interval(
        self, X: npt.ArrayLike, n_std: float = 1.96
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Gaussian prediction interval at ``n_std`` standard deviations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        n_std : float, default 1.96
            Multiplier; 1.96 gives a nominal 95% interval *if* the errors
            are Gaussian. When that assumption is doubtful — which for
            QSAR it usually is — prefer
            :class:`~qsarkit.uncertainty.ConformalRegressor`, whose
            coverage guarantee is distribution-free.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        mean, std = self.predict_uncertainty(X)
        return mean - n_std * std, mean + n_std * std


class EnsembleUncertainty(BaseUncertaintyEstimator):
    """Uncertainty from the disagreement among an ensemble.

    Trains several models — on bootstrap resamples, or with different
    seeds — and reports the spread of their predictions. Where the
    members agree the prediction is well determined by the data; where
    they diverge it is not. For a random forest the trees already form
    an ensemble, so their per-tree predictions are used directly rather
    than refitting.

    Parameters
    ----------
    estimator : sklearn regressor
        Base model. Cloned, not modified.
    n_estimators : int, default 10
        Number of ensemble members (ignored when reusing a forest's own
        trees).
    bootstrap : bool, default True
        Resample the training data for each member. Without it, members
        differ only through their own randomness, which understates
        uncertainty for deterministic learners.
    use_native_ensemble : bool, default True
        For estimators exposing ``estimators_`` (forests, bagging), use
        the existing members instead of training new ones.
    random_state : int, optional
        Seed.

    Attributes
    ----------
    estimators_ : list
        The fitted members.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(60, 3)); y = X[:, 0] * 2
    >>> est = EnsembleUncertainty(DecisionTreeRegressor(), n_estimators=5,
    ...                           random_state=0).fit(X, y)
    >>> mean, std = est.predict_uncertainty(X)
    >>> bool(np.all(std >= 0))
    True

    References
    ----------
    - Breiman, L. (1996). "Bagging Predictors." Mach. Learn., 24, 123-140.
      https://doi.org/10.1007/BF00058655
    - Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). "Simple and
      Scalable Predictive Uncertainty Estimation Using Deep Ensembles."
      NeurIPS 2017. https://arxiv.org/abs/1612.01474
    - Sheridan, R. P. (2013). "Using Random Forest to Model the Domain
      Applicability of Another Random Forest Model." J. Chem. Inf.
      Model., 53(11), 2837-2850. https://doi.org/10.1021/ci400482e
    """

    estimators_: List[Any]

    def __init__(
        self,
        estimator: Any,
        n_estimators: int = 10,
        bootstrap: bool = True,
        use_native_ensemble: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.use_native_ensemble = use_native_ensemble
        self.random_state = random_state

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "EnsembleUncertainty":
        """Fit the ensemble members.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        EnsembleUncertainty
        """
        if self.n_estimators < 2:
            raise ValueError(
                f"n_estimators must be at least 2 to have a spread, got "
                f"{self.n_estimators}."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)

        fitted = clone(self.estimator).fit(X_arr, y_arr)
        if self.use_native_ensemble and hasattr(fitted, "estimators_"):
            self.estimators_ = list(fitted.estimators_)
            self._native = True
            return self

        self._native = False
        members: List[Any] = []
        for i in range(self.n_estimators):
            member = clone(self.estimator)
            if hasattr(member, "random_state"):
                member.set_params(random_state=int(rng.randint(0, 2**31 - 1)))
            if self.bootstrap:
                idx = rng.choice(len(X_arr), size=len(X_arr), replace=True)
                members.append(member.fit(X_arr[idx], y_arr[idx]))
            else:
                members.append(member.fit(X_arr, y_arr))
        self.estimators_ = members
        return self

    def predict_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Mean and standard deviation across the ensemble members."""
        if not hasattr(self, "estimators_"):
            raise ModelNotFittedError(
                "EnsembleUncertainty must be fitted before predicting."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        preds = np.array(
            [np.asarray(m.predict(X_arr), dtype=np.float64) for m in self.estimators_]
        )
        return (
            np.asarray(preds.mean(axis=0), dtype=np.float64),
            np.asarray(preds.std(axis=0, ddof=1), dtype=np.float64),
        )


class MCDropoutUncertainty(BaseUncertaintyEstimator):
    """Monte-Carlo dropout uncertainty for a PyTorch network.

    Keeps dropout active at prediction time and samples the network
    several times. Gal and Ghahramani showed this approximates
    variational inference in a deep Gaussian process, so the sample
    spread is a principled posterior estimate rather than just noise —
    at the cost of one forward pass per sample.

    Parameters
    ----------
    model : torch.nn.Module
        A network containing at least one dropout layer. Without one,
        every pass is identical and the reported uncertainty is zero.
    n_samples : int, default 50
        Forward passes per prediction.
    device : str, optional
        Torch device; defaults to the model's own.

    Examples
    --------
    >>> import pytest
    >>> torch = pytest.importorskip("torch")  # doctest: +SKIP

    References
    ----------
    - Gal, Y. & Ghahramani, Z. (2016). "Dropout as a Bayesian
      Approximation: Representing Model Uncertainty in Deep Learning."
      ICML 2016. https://arxiv.org/abs/1506.02142
    - Scalia, G. et al. (2020). J. Chem. Inf. Model., 60(6), 2697-2717.
      https://doi.org/10.1021/acs.jcim.9b00975
    """

    def __init__(
        self,
        model: Any,
        n_samples: int = 50,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.n_samples = n_samples
        self.device = device

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "MCDropoutUncertainty":
        """No-op: the wrapped network is expected to be trained already.

        Parameters
        ----------
        X, y : array-like
            Ignored; present for API compatibility.

        Returns
        -------
        MCDropoutUncertainty
        """
        return self

    def _enable_dropout(self) -> int:
        """Put dropout layers (and only those) into training mode."""
        torch = __import__("torch")

        self.model.eval()
        n_dropout = 0
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.train()
                n_dropout += 1
        return n_dropout

    def predict_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Mean and standard deviation over ``n_samples`` stochastic passes."""
        from qsarkit.base import require

        torch = require("torch")

        if self.n_samples < 2:
            raise ValueError(
                f"n_samples must be at least 2, got {self.n_samples}."
            )
        n_dropout = self._enable_dropout()
        if n_dropout == 0:
            raise ValueError(
                "The model contains no dropout layers, so MC dropout would "
                "report zero uncertainty. Add dropout, or use "
                "EnsembleUncertainty instead."
            )

        tensor = torch.as_tensor(
            np.asarray(X, dtype=np.float32),
            device=self.device or next(self.model.parameters()).device,
        )
        with torch.no_grad():
            samples = torch.stack(
                [self.model(tensor).squeeze(-1) for _ in range(self.n_samples)]
            )
        arr = samples.cpu().numpy().astype(np.float64)
        return arr.mean(axis=0), arr.std(axis=0, ddof=1)


class GaussianProcessUncertainty(BaseUncertaintyEstimator):
    """Posterior standard deviation of a Gaussian process.

    The only method here whose uncertainty is exact rather than
    approximate: a GP returns a full posterior, so the standard
    deviation is the model's own belief, not a sample statistic. The
    cost is cubic in training-set size, which caps it at a few thousand
    compounds.

    Parameters
    ----------
    kernel : sklearn kernel, optional
        Defaults to an RBF with a white-noise term. For fingerprints,
        pass :class:`~qsarkit.models.TanimotoKernel`.
    alpha : float, default 1e-10
        Value added to the diagonal for numerical stability.
    normalize_y : bool, default True
        Standardize the target before fitting.
    random_state : int, optional
        Seed.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(40, 2)); y = X[:, 0]
    >>> gp = GaussianProcessUncertainty(random_state=0).fit(X, y)
    >>> mean, std = gp.predict_uncertainty(X)
    >>> bool(np.all(std >= 0))
    True

    References
    ----------
    - Rasmussen, C. E. & Williams, C. K. I. (2006). "Gaussian Processes
      for Machine Learning." MIT Press.
      https://gaussianprocess.org/gpml/
    - Obrezanova, O. et al. (2007). "Gaussian Processes: A Method for
      Automatic QSAR Modeling of ADME Properties." J. Chem. Inf. Model.,
      47(5), 1847-1857. https://doi.org/10.1021/ci7000633
    """

    def __init__(
        self,
        kernel: Optional[Any] = None,
        alpha: float = 1e-10,
        normalize_y: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        self.kernel = kernel
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.random_state = random_state

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "GaussianProcessUncertainty":
        """Fit the Gaussian process.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        GaussianProcessUncertainty
        """
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        kernel = self.kernel if self.kernel is not None else RBF() + WhiteKernel()
        self._model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=self.alpha,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        ).fit(np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.float64))
        return self

    def predict_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Posterior mean and standard deviation."""
        if not hasattr(self, "_model"):
            raise ModelNotFittedError(
                "GaussianProcessUncertainty must be fitted before predicting."
            )
        mean, std = self._model.predict(
            np.asarray(X, dtype=np.float64), return_std=True
        )
        return (
            np.asarray(mean, dtype=np.float64),
            np.asarray(std, dtype=np.float64),
        )


class QuantileRegressionUncertainty(BaseUncertaintyEstimator):
    """Uncertainty from quantile regression.

    Fits separate models for a lower quantile, the median and an upper
    quantile. Unlike every other estimator here it does not assume the
    error is symmetric or constant, so it captures heteroscedasticity —
    the common situation where potent compounds are measured more
    precisely than weak ones.

    Parameters
    ----------
    estimator : sklearn regressor, optional
        Must accept a ``quantile``/``alpha`` parameter. Defaults to
        ``GradientBoostingRegressor(loss="quantile")``.
    quantiles : (float, float), default (0.05, 0.95)
        Lower and upper quantiles to fit.
    random_state : int, optional
        Seed.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(120, 3)); y = X[:, 0] * 2 + rng.normal(size=120)
    >>> q = QuantileRegressionUncertainty(random_state=0).fit(X, y)
    >>> lower, upper = q.predict_interval(X)
    >>> bool(np.all(upper >= lower))
    True

    References
    ----------
    - Koenker, R. & Bassett, G. (1978). "Regression Quantiles."
      Econometrica, 46(1), 33-50. https://doi.org/10.2307/1913643
    - Meinshausen, N. (2006). "Quantile Regression Forests." J. Mach.
      Learn. Res., 7, 983-999.
      https://jmlr.org/papers/v7/meinshausen06a.html
    """

    def __init__(
        self,
        estimator: Optional[Any] = None,
        quantiles: Tuple[float, float] = (0.05, 0.95),
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.quantiles = quantiles
        self.random_state = random_state

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "QuantileRegressionUncertainty":
        """Fit lower, median and upper quantile models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        QuantileRegressionUncertainty
        """
        from sklearn.ensemble import GradientBoostingRegressor

        low, high = self.quantiles
        if not 0.0 < low < high < 1.0:
            raise ValueError(
                f"quantiles must satisfy 0 < low < high < 1, got {self.quantiles}."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        def _make(q: float) -> Any:
            if self.estimator is None:
                return GradientBoostingRegressor(
                    loss="quantile", alpha=q, random_state=self.random_state
                )
            model = clone(self.estimator)
            params = model.get_params()
            if "quantile" in params:
                model.set_params(quantile=q)
            elif "alpha" in params:
                model.set_params(alpha=q)
            else:
                raise ValueError(
                    f"{type(model).__name__} has no 'quantile' or 'alpha' "
                    "parameter, so it cannot do quantile regression."
                )
            return model

        self._lower = _make(low).fit(X_arr, y_arr)
        self._median = _make(0.5).fit(X_arr, y_arr)
        self._upper = _make(high).fit(X_arr, y_arr)
        self._z = float(high - low)
        return self

    def predict_uncertainty(
        self, X: npt.ArrayLike
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Median prediction and a standard deviation implied by the quantile span."""
        if not hasattr(self, "_median"):
            raise ModelNotFittedError(
                "QuantileRegressionUncertainty must be fitted before predicting."
            )
        from scipy.stats import norm

        X_arr = np.asarray(X, dtype=np.float64)
        median = np.asarray(self._median.predict(X_arr), dtype=np.float64)
        lower = np.asarray(self._lower.predict(X_arr), dtype=np.float64)
        upper = np.asarray(self._upper.predict(X_arr), dtype=np.float64)
        # Convert the quantile span to an equivalent Gaussian sigma so the
        # value is comparable with the other estimators here.
        low, high = self.quantiles
        span = norm.ppf(high) - norm.ppf(low)
        return median, np.asarray(np.abs(upper - lower) / span, dtype=np.float64)

    def predict_interval(
        self, X: npt.ArrayLike, n_std: float = 1.96
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The fitted quantiles directly — no Gaussian assumption needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        n_std : float
            Ignored; the interval is the fitted quantile pair.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        if not hasattr(self, "_median"):
            raise ModelNotFittedError(
                "QuantileRegressionUncertainty must be fitted before predicting."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        return (
            np.asarray(self._lower.predict(X_arr), dtype=np.float64),
            np.asarray(self._upper.predict(X_arr), dtype=np.float64),
        )
