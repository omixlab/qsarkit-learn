"""Consensus (ensemble) QSAR modeling, built on scikit-learn's voting/stacking."""

from __future__ import annotations

from typing import Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from qsarkit.base.exceptions import ModelNotFittedError

__all__ = ["ConsensusModel"]

_Task = Literal["regression", "classification"]
_Method = Literal["averaging", "stacking"]


class ConsensusModel(BaseEstimator):
    """Consensus QSAR model built from several member estimators.

    Consensus (ensemble) modeling is one of the most robust, widely
    reproduced findings in QSAR practice: averaging or stacking several
    structurally different models (a random forest, an SVM, a PLS model,
    ...) routinely outperforms any single member, because the members'
    errors are only partially correlated and combining them cancels out
    some of each model's idiosyncratic mistakes. Rather than
    reimplementing ensembling machinery, this class is a thin,
    task-aware dispatcher onto scikit-learn's own
    :class:`~sklearn.ensemble.VotingRegressor` /
    :class:`~sklearn.ensemble.VotingClassifier` (``method="averaging"``)
    and :class:`~sklearn.ensemble.StackingRegressor` /
    :class:`~sklearn.ensemble.StackingClassifier`
    (``method="stacking"``).

    Parameters
    ----------
    estimators : sequence of (str, estimator) tuples
        Named member estimators, in the format scikit-learn's own
        voting/stacking ensembles expect.
    task : {"regression", "classification"}, default "regression"
        Prediction task, selecting which family of scikit-learn ensemble
        is built.
    method : {"averaging", "stacking"}, default "averaging"
        ``"averaging"`` combines member predictions by a (possibly
        weighted) vote/mean; ``"stacking"`` trains a meta-estimator on
        out-of-fold member predictions.
    weights : sequence of float, optional
        Per-member weights used only when ``method="averaging"``.
    final_estimator : estimator, optional
        Meta-estimator used only when ``method="stacking"``. Defaults to
        the stacking ensemble's own default (ridge/logistic regression)
        when None.
    cv : int, default 5
        Number of cross-validation folds used to generate the out-of-fold
        predictions that train the meta-estimator, when
        ``method="stacking"``.

    Attributes
    ----------
    estimator_ : estimator
        The fitted internal scikit-learn voting/stacking ensemble.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression, Ridge
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    >>> model = ConsensusModel(
    ...     estimators=[("lr", LinearRegression()), ("ridge", Ridge())],
    ...     task="regression",
    ...     method="averaging",
    ... ).fit(X, y)
    >>> model.predict(X).shape
    (40,)

    References
    ----------
    - Wolpert, D. H. (1992). "Stacked Generalization." Neural Networks,
      5(2), 241-259. https://doi.org/10.1016/S0893-6080(05)80023-1
    - Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning."
      In: Multiple Classifier Systems (MCS 2000), LNCS 1857, 1-15.
      https://doi.org/10.1007/3-540-45014-9_1
    """

    estimator_: BaseEstimator

    def __init__(
        self,
        estimators: Sequence[Tuple[str, BaseEstimator]],
        task: _Task = "regression",
        method: _Method = "averaging",
        weights: Optional[Sequence[float]] = None,
        final_estimator: Optional[BaseEstimator] = None,
        cv: int = 5,
    ) -> None:
        self.estimators = estimators
        self.task = task
        self.method = method
        self.weights = weights
        self.final_estimator = final_estimator
        self.cv = cv

    def _build(self) -> BaseEstimator:
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )
        if self.method not in ("averaging", "stacking"):
            raise ValueError(
                f"method must be 'averaging' or 'stacking', got {self.method!r}."
            )
        if not self.estimators:
            raise ValueError("ConsensusModel needs at least one member estimator.")

        estimators = list(self.estimators)
        if self.method == "averaging":
            if self.task == "regression":
                from sklearn.ensemble import VotingRegressor

                return VotingRegressor(estimators=estimators, weights=self.weights)
            from sklearn.ensemble import VotingClassifier

            return VotingClassifier(
                estimators=estimators, voting="soft", weights=self.weights
            )

        if self.task == "regression":
            from sklearn.ensemble import StackingRegressor

            return StackingRegressor(
                estimators=estimators,
                final_estimator=self.final_estimator,
                cv=self.cv,
            )
        from sklearn.ensemble import StackingClassifier

        return StackingClassifier(
            estimators=estimators,
            final_estimator=self.final_estimator,
            cv=self.cv,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "ConsensusModel":
        """Build and fit the internal voting/stacking ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        ConsensusModel
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``task``/``method`` are not among the allowed literals, or
            ``estimators`` is empty.
        """
        self.estimator_ = self._build()
        self.estimator_.fit(X, y)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Predict by delegating to the fitted internal ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return np.asarray(self.estimator_.predict(X))

    def score(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: Optional[npt.ArrayLike] = None,
    ) -> float:
        """Score the consensus: R^2 for regression, accuracy for classification.

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
        """
        self._check_fitted()
        return float(self.estimator_.score(X, y, sample_weight=sample_weight))

    def __sklearn_tags__(self) -> object:
        """Report the estimator type sklearn should assume for this task."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = (
            "classifier" if self.task == "classification" else "regressor"
        )
        return tags

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities by delegating to the fitted internal ensemble.

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
            If the internal ensemble does not support ``predict_proba``
            (e.g. stacking classification with a ``final_estimator`` that
            itself has no ``predict_proba``).
        """
        self._check_fitted()
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"The fitted {type(self.estimator_).__name__} does not support "
                "predict_proba (task must be 'classification', and for "
                "method='stacking' the final_estimator must support "
                "predict_proba)."
            )
        return np.asarray(self.estimator_.predict_proba(X), dtype=np.float64)
