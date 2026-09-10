"""Naive baseline model for OECD-principle-4 goodness-of-fit comparisons."""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor

from qsarkit.base.exceptions import ModelNotFittedError

__all__ = ["BaselineModel"]

_Task = Literal["regression", "classification"]


class BaselineModel(BaseEstimator):
    """Naive baseline model for the OECD "beat the null model" check.

    OECD (2007) Guidance Document No. 69, validation principle 4,
    requires that a QSAR model be reported together with "appropriate
    measures of goodness-of-fit" — and implicit in that requirement,
    echoed throughout the QSAR validation literature, is that a model
    which cannot outperform a trivial, feature-blind predictor (predict
    the training mean, predict the majority class, ...) has told you
    nothing about the structure-activity relationship. ``BaselineModel``
    is exactly that trivial predictor, packaged as a scikit-learn
    estimator so it can sit in the same cross-validation harness as the
    real model and produce a directly comparable score. Any QSAR model
    reported without first clearing this bar is not informative.

    Parameters
    ----------
    task : {"regression", "classification"}, default "regression"
        Which naive strategy family to use.
    strategy : str, optional
        Strategy forwarded to the underlying scikit-learn dummy
        estimator. For ``task="regression"`` one of ``"mean"``,
        ``"median"``, ``"quantile"``, ``"constant"`` (default ``"mean"``).
        For ``task="classification"`` one of ``"most_frequent"``,
        ``"stratified"``, ``"uniform"``, ``"prior"`` (default
        ``"most_frequent"``).

    Attributes
    ----------
    dummy_ : DummyRegressor or DummyClassifier
        The fitted scikit-learn dummy estimator doing the actual work.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.zeros((10, 3))
    >>> y = np.arange(10.0)
    >>> model = BaselineModel(task="regression").fit(X, y)
    >>> float(model.predict(X)[0]) == float(np.mean(y))
    True

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of
      (Quantitative) Structure-Activity Relationship [(Q)SAR] Models."
      OECD Series on Testing and Assessment No. 69, ENV/JM/MONO(2007)2,
      Principle 4 ("a model should be associated with... appropriate
      measures of goodness-of-fit"). https://doi.org/10.1787/9789264085442-en
    - scikit-learn ``DummyRegressor``/``DummyClassifier`` documentation:
      https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators
    """

    dummy_: Union[DummyRegressor, DummyClassifier]

    def __init__(
        self,
        task: _Task = "regression",
        strategy: Optional[str] = None,
    ) -> None:
        self.task = task
        self.strategy = strategy

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "BaselineModel":
        """Fit the naive baseline for the configured task.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ignored except for shape/length checks (the whole point of a
            baseline is that it does not use the features).
        y : array-like of shape (n_samples,)
            Target values or class labels.

        Returns
        -------
        BaselineModel
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``task`` is not "regression" or "classification".
        """
        if self.task == "regression":
            self.dummy_ = DummyRegressor(strategy=self.strategy or "mean")
        elif self.task == "classification":
            self.dummy_ = DummyClassifier(strategy=self.strategy or "most_frequent")
        else:
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )
        self.dummy_.fit(X, y)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "dummy_"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Predict using the naive strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Constant (regression) or majority/sampled (classification)
            predictions, ignoring the actual feature values.
        """
        self._check_fitted()
        return np.asarray(self.dummy_.predict(X))

    def score(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: Optional[npt.ArrayLike] = None,
    ) -> float:
        """Score the baseline: R^2 for regression, accuracy for classification.

        Implemented explicitly rather than inherited, because this
        estimator switches task at construction time and so cannot carry
        either ``RegressorMixin`` or ``ClassifierMixin``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            True values.
        sample_weight : array-like of shape (n_samples,), optional

        Returns
        -------
        float
            R^2 when ``task="regression"``, accuracy when
            ``task="classification"``. A real model that cannot beat this
            number has learned nothing from the descriptors.
        """
        self._check_fitted()
        return float(self.dummy_.score(X, y, sample_weight=sample_weight))

    def __sklearn_tags__(self) -> object:
        """Report the estimator type sklearn should assume for this task."""
        tags = super().__sklearn_tags__()
        from sklearn.utils import Tags  # noqa: F401  (import guards the API)

        tags.estimator_type = (
            "classifier" if self.task == "classification" else "regressor"
        )
        return tags

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities for the naive classification baseline.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        ModelNotFittedError
            If called before :meth:`fit`.
        AttributeError
            If ``task="regression"``, which has no notion of class
            probabilities.
        """
        self._check_fitted()
        if not isinstance(self.dummy_, DummyClassifier):
            raise AttributeError(
                "predict_proba is only available for task='classification'."
            )
        return np.asarray(self.dummy_.predict_proba(X), dtype=np.float64)
