"""Mutual-information-based feature selection."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.feature_selection import (
    SelectorMixin,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.utils.validation import check_is_fitted

__all__ = ["MutualInformationSelector"]


class MutualInformationSelector(SelectorMixin, BaseEstimator):
    """Keep the descriptors most informative about the target, by mutual information.

    Thin wrapper around ``sklearn.feature_selection.mutual_info_regression``
    / ``mutual_info_classif``, which estimate mutual information between
    each descriptor and the target using a k-nearest-neighbour entropy
    estimator. Unlike Pearson correlation, mutual information captures
    non-linear dependencies, which matters for QSAR descriptors that
    often relate to activity non-monotonically.

    Parameters
    ----------
    task : {"regression", "classification"}, default "regression"
        Whether ``y`` is continuous or categorical.
    k : int, optional
        Keep the top-``k`` scoring descriptors. Mutually exclusive with
        ``percentile``.
    percentile : float, optional
        Keep descriptors scoring at or above this percentile (0-100) of
        the score distribution. Mutually exclusive with ``k``. If
        neither ``k`` nor ``percentile`` is given, defaults to 50 (keep
        the top half).
    random_state : int, optional
        Seed forwarded to the mutual-information estimator's internal
        noise injection (used to break ties in the nearest-neighbour
        distances).

    Attributes
    ----------
    scores_ : ndarray of shape (n_features,)
        Raw mutual-information score per descriptor.
    support_ : ndarray of bool of shape (n_features,)
        True for descriptors kept.
    n_features_in_ : int
        Number of descriptors seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.feature_selection import MutualInformationSelector
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(300, 3))
    >>> y = X[:, 0] ** 2
    >>> sel = MutualInformationSelector(k=1, random_state=0).fit(X, y)
    >>> sel.support_.tolist()
    [True, False, False]

    References
    ----------
    - Kraskov, A., Stogbauer, H. & Grassberger, P. (2004). "Estimating
      Mutual Information." Phys. Rev. E, 69(6), 066138.
      https://doi.org/10.1103/PhysRevE.69.066138
    """

    scores_: npt.NDArray[np.float64]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        k: Optional[int] = None,
        percentile: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.task = task
        self.k = k
        self.percentile = percentile
        self.random_state = random_state

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "MutualInformationSelector":
        """Score every descriptor by mutual information with ``y``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        MutualInformationSelector
            The fitted selector.

        Raises
        ------
        ValueError
            If both ``k`` and ``percentile`` are given, or ``task`` is
            invalid.
        """
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )
        if self.k is not None and self.percentile is not None:
            raise ValueError("Only one of `k` and `percentile` may be given, not both.")

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")

        n_features = arr.shape[1]
        self.n_features_in_ = n_features
        y_arr = np.asarray(y)

        score_func = (
            mutual_info_regression if self.task == "regression" else mutual_info_classif
        )
        self.scores_ = np.asarray(
            score_func(arr, y_arr, random_state=self.random_state), dtype=np.float64
        )

        k = self.k
        percentile = self.percentile
        if k is None and percentile is None:
            percentile = 50.0

        if k is not None:
            k_eff = min(k, n_features)
            top_idx = np.argsort(-self.scores_)[:k_eff]
            support = np.zeros(n_features, dtype=bool)
            support[top_idx] = True
        else:
            assert percentile is not None  # narrowed by the branch above
            cutoff = np.percentile(self.scores_, percentile)
            support = self.scores_ >= cutoff

        self.support_ = np.asarray(support, dtype=np.bool_)
        return self

    def _get_support_mask(self) -> npt.NDArray[np.bool_]:
        check_is_fitted(self)
        return self.support_
