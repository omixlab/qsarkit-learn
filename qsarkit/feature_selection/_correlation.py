"""Pairwise-correlation redundancy filtering."""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["CorrelationFilter"]


class CorrelationFilter(SelectorMixin, BaseEstimator):
    """Drop one descriptor from every pair whose correlation exceeds a threshold.

    Collinear descriptors add noise to linear models, inflate coefficient
    variance and make coefficient interpretation unreliable — a standard
    QSAR descriptor-curation concern. This selector removes redundant
    descriptors while trying to keep the more informative member of each
    correlated pair.

    Notes
    -----
    The algorithm is:

    1. Compute the descriptor-descriptor correlation matrix (Pearson or
       Spearman).
    2. Determine a processing order: if ``y`` is supplied, features are
       visited in *descending* order of ``|corr(feature, y)|`` so the
       more target-relevant member of a correlated pair is considered
       first; otherwise features are visited in their natural column
       order.
    3. Walk the ordered features, keeping a running set of accepted
       indices. A candidate feature is kept if its absolute correlation
       with *every* already-kept feature is at or below ``threshold``;
       otherwise it is redundant with something already kept, and is
       dropped.

    Because the target-relevant feature is visited first when ``y`` is
    given, a duplicated/near-duplicated copy of an informative descriptor
    is reliably the one dropped, not the informative descriptor itself.

    Parameters
    ----------
    threshold : float, default 0.95
        Absolute correlation above which a later feature is considered
        redundant with an earlier, already-kept one.
    method : {"pearson", "spearman"}, default "pearson"
        Correlation measure. Spearman uses rank correlation and is
        robust to monotonic non-linear relationships between descriptors.

    Attributes
    ----------
    correlation_matrix_ : ndarray of shape (n_features, n_features)
        The fitted descriptor-descriptor correlation matrix.
    support_ : ndarray of bool of shape (n_features,)
        True for descriptors kept.
    n_features_in_ : int
        Number of descriptors seen during ``fit``.
    constant_ : ndarray of bool of shape (n_features,)
        True for descriptors with zero variance, whose correlation is
        undefined. These are dropped before the walk rather than as a
        side effect of comparing against NaN.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.feature_selection import CorrelationFilter
    >>> rng = np.random.RandomState(0)
    >>> x0 = rng.normal(size=200)
    >>> X = np.column_stack([x0, x0 + rng.normal(scale=1e-3, size=200), rng.normal(size=200)])
    >>> y = x0 * 2.0
    >>> cf = CorrelationFilter(threshold=0.95).fit(X, y)
    >>> cf.support_.tolist()
    [True, False, True]

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics," 2nd ed. Wiley-VCH.
      https://doi.org/10.1002/9783527628766
    """

    correlation_matrix_: npt.NDArray[np.float64]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(
        self,
        threshold: float = 0.95,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> None:
        self.threshold = threshold
        self.method = method

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "CorrelationFilter":
        """Learn the correlation matrix and resulting support mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like of shape (n_samples,), optional
            Target values. When given, features that correlate more
            strongly with ``y`` are preferred as the "kept" member of a
            redundant pair.

        Returns
        -------
        CorrelationFilter
            The fitted selector.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.method not in ("pearson", "spearman"):
            raise ValueError(
                f"method must be 'pearson' or 'spearman', got {self.method!r}."
            )

        n_features = arr.shape[1]
        self.n_features_in_ = n_features

        # Correlation is undefined for a column with zero variance, and a
        # fingerprint matrix is mostly such columns. Identifying them up
        # front does three things: it suppresses a stream of divide-by-zero
        # warnings, it drops them deliberately rather than as a side effect
        # of NaN comparisons, and it stops the first constant column
        # encountered from being kept by accident (`all([])` is True).
        self.constant_ = np.std(arr, axis=0) == 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            self.correlation_matrix_ = self._correlation_matrix(arr)
            if y is not None:
                y_arr = np.asarray(y, dtype=np.float64).ravel()
                target_corr = self._target_correlation(arr, y_arr)
            else:
                target_corr = None
        abs_corr = np.abs(self.correlation_matrix_)

        candidates = np.flatnonzero(~self.constant_)
        if target_corr is not None:
            scores = np.where(np.isfinite(target_corr), np.abs(target_corr), -np.inf)
            order = candidates[np.argsort(-scores[candidates], kind="stable")]
        else:
            order = candidates

        kept: List[int] = []
        support = np.zeros(n_features, dtype=bool)
        for idx in order:
            idx = int(idx)
            if all(abs_corr[idx, k] <= self.threshold for k in kept):
                kept.append(idx)
                support[idx] = True
        self.support_ = support
        return self

    def _correlation_matrix(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n_features = arr.shape[1]
        if n_features < 2:
            return np.ones((n_features, n_features), dtype=np.float64)
        if self.method == "pearson":
            corr = np.corrcoef(arr, rowvar=False)
        else:
            from scipy.stats import spearmanr

            corr = spearmanr(arr).statistic
        corr = np.atleast_2d(np.asarray(corr, dtype=np.float64))
        if corr.shape != (n_features, n_features):
            # scipy.stats.spearmanr collapses to a scalar for exactly two
            # columns; rebuild the 2x2 matrix explicitly in that case.
            r = float(corr.flat[0])
            corr = np.array([[1.0, r], [r, 1.0]], dtype=np.float64)
        return np.asarray(corr, dtype=np.float64)

    def _target_correlation(
        self, arr: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        n_features = arr.shape[1]
        if self.method == "pearson":
            corr = [np.corrcoef(arr[:, j], y)[0, 1] for j in range(n_features)]
        else:
            from scipy.stats import spearmanr

            corr = [float(spearmanr(arr[:, j], y).statistic) for j in range(n_features)]
        return np.asarray(corr, dtype=np.float64)

    def _get_support_mask(self) -> npt.NDArray[np.bool_]:
        check_is_fitted(self)
        return self.support_
