"""Recursive feature elimination."""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFE, SelectorMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["RFESelector"]


class RFESelector(SelectorMixin, BaseEstimator):
    """Recursive feature elimination, with a QSAR-sane default estimator.

    Thin wrapper around ``sklearn.feature_selection.RFE``: repeatedly
    fits ``estimator``, ranks descriptors by its feature-importance (or
    coefficient) attribute, and prunes the weakest until
    ``n_features_to_select`` remain. When no ``estimator`` is given, a
    random-forest of 200 trees is used — a robust, hyperparameter-light
    default that handles the non-linear structure typical of molecular
    descriptors.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Estimator exposing ``coef_`` or ``feature_importances_`` after
        fitting. Defaults to a ``RandomForestRegressor``/
        ``RandomForestClassifier`` with 200 trees, chosen by ``task``.
    task : {"regression", "classification"}, default "regression"
        Selects the default estimator when ``estimator`` is ``None``.
        Ignored if ``estimator`` is given.
    n_features_to_select : int or float, optional
        Number (or, if a float in (0, 1), fraction) of descriptors to
        keep. Defaults to sklearn's ``RFE`` default (half the features).
    step : int or float, default 1
        Number (or fraction) of descriptors pruned at each iteration.
    random_state : int, optional
        Forwarded to the default random-forest estimator.

    Attributes
    ----------
    support_ : ndarray of bool of shape (n_features,)
        True for descriptors kept.
    ranking_ : ndarray of int of shape (n_features,)
        Selection ranking; selected descriptors are ranked 1.
    n_features_in_ : int
        Number of descriptors seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.feature_selection import RFESelector
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(100, 4))
    >>> y = X[:, 0] * 3.0
    >>> sel = RFESelector(n_features_to_select=1, random_state=0).fit(X, y)
    >>> int(sel.ranking_[0])
    1

    References
    ----------
    - Guyon, I., Weston, J., Barnhill, S. & Vapnik, V. (2002). "Gene
      Selection for Cancer Classification Using Support Vector
      Machines." Machine Learning, 46(1-3), 389-422.
      https://doi.org/10.1023/A:1012487302797
    """

    support_: npt.NDArray[np.bool_]
    ranking_: npt.NDArray[np.intp]
    n_features_in_: int

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task: Literal["regression", "classification"] = "regression",
        n_features_to_select: Optional[Union[int, float]] = None,
        step: Union[int, float] = 1,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.task = task
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.random_state = random_state

    def _default_estimator(self) -> BaseEstimator:
        if self.task == "classification":
            return RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        return RandomForestRegressor(n_estimators=200, random_state=self.random_state)

    def fit(self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None) -> "RFESelector":
        """Run recursive feature elimination.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        RFESelector
            The fitted selector.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )

        self.n_features_in_ = arr.shape[1]
        estimator = self.estimator if self.estimator is not None else self._default_estimator()
        rfe = RFE(
            estimator=estimator,
            n_features_to_select=self.n_features_to_select,
            step=self.step,
        )
        rfe.fit(arr, y)
        self.support_ = np.asarray(rfe.support_, dtype=np.bool_)
        self.ranking_ = np.asarray(rfe.ranking_, dtype=np.intp)
        return self

    def _get_support_mask(self) -> npt.NDArray[np.bool_]:
        check_is_fitted(self)
        return self.support_
