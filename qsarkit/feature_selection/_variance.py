"""Near-zero-variance descriptor filtering."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["VarianceFilter"]


class VarianceFilter(SelectorMixin, BaseEstimator):
    """Drop near-constant descriptors.

    Constant or near-constant descriptor columns carry no discriminating
    information and can break downstream scalers (division by a
    near-zero standard deviation) or destabilize linear-model fits. This
    is the standard first curation step applied to any QSAR descriptor
    matrix before feature selection or modelling proper begins.

    The algorithm is the two-line computation
    ``variances_ = X.var(axis=0)``, ``support_ = variances_ > threshold``
    — exactly what ``sklearn.feature_selection.VarianceThreshold``
    implements, reproduced directly here so this estimator shares the
    ``qsarkit`` selector contract (``support_``, ``get_support()``,
    ``transform()`` via :class:`sklearn.feature_selection.SelectorMixin`).

    Parameters
    ----------
    threshold : float, default 0.0
        Descriptors with a training-set variance at or below this value
        are dropped. The default removes only exactly-constant columns.

    Attributes
    ----------
    variances_ : ndarray of shape (n_features,)
        Per-descriptor variance computed on the training data.
    support_ : ndarray of bool of shape (n_features,)
        True for descriptors kept (``variances_ > threshold``).
    n_features_in_ : int
        Number of descriptors seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.feature_selection import VarianceFilter
    >>> X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    >>> vf = VarianceFilter().fit(X)
    >>> vf.support_.tolist()
    [True, False]

    References
    ----------
    - scikit-learn documentation, ``VarianceThreshold``.
      https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.VarianceThreshold.html
    """

    variances_: npt.NDArray[np.float64]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "VarianceFilter":
        """Learn per-descriptor variances and the resulting support mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like, optional
            Ignored. Present for API consistency.

        Returns
        -------
        VarianceFilter
            The fitted selector.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")

        self.n_features_in_ = arr.shape[1]
        self.variances_ = np.asarray(arr.var(axis=0), dtype=np.float64)
        self.support_ = np.asarray(self.variances_ > self.threshold, dtype=np.bool_)
        return self

    def _get_support_mask(self) -> npt.NDArray[np.bool_]:
        check_is_fitted(self)
        return self.support_
