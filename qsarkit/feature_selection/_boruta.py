"""Boruta all-relevant feature selection."""

from __future__ import annotations

import warnings

from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ["BorutaSelector"]


class BorutaSelector(SelectorMixin, BaseEstimator):
    """Boruta all-relevant feature selection.

    Boruta answers a different question than most feature selectors: not
    "which subset gives the best predictive score" but "which descriptors
    carry *any* signal at all." It does so by comparing every real
    descriptor against "shadow" copies of every descriptor — the same
    values, independently permuted across samples, which by construction
    carry no relationship to ``y``. A tree ensemble is fit on the
    concatenation of real and shadow descriptors; a real descriptor that
    beats the *best* shadow descriptor's importance is a "hit" for that
    iteration. Repeated over many iterations, the hit count of a
    genuinely important descriptor should exceed that of an irrelevant
    one (which is statistically indistinguishable from a shadow feature)
    — formalized with a two-sided binomial test against the null
    ``hits ~ Binomial(n_iterations, 0.5)``, Bonferroni-corrected across
    descriptors.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Estimator exposing ``feature_importances_`` after fitting (e.g.
        a tree ensemble). Defaults to a ``RandomForestRegressor``/
        ``RandomForestClassifier`` with 200 trees, chosen by ``task``.
    task : {"regression", "classification"}, default "regression"
        Selects the default estimator when ``estimator`` is ``None``.
        Ignored if ``estimator`` is given.
    n_iterations : int, default 100
        Number of shadow-permutation iterations.
    alpha : float, default 0.05
        Family-wise significance level; Bonferroni-corrected to
        ``alpha / n_features`` per descriptor.
    include_tentative : bool, default False
        If True, descriptors that could not be statistically resolved
        within the iteration budget ("Tentative") are also selected.
    random_state : int, optional
        Seed for the shadow-feature permutations.

    Attributes
    ----------
    hits_ : ndarray of int of shape (n_features,)
        Number of iterations in which each real descriptor beat the best
        shadow descriptor.
    decision_ : ndarray of str of shape (n_features,)
        Per-descriptor verdict: ``"Confirmed"``, ``"Tentative"`` or
        ``"Rejected"``.
    support_ : ndarray of bool of shape (n_features,)
        True for ``"Confirmed"`` descriptors (plus ``"Tentative"`` ones
        too when ``include_tentative=True``).
    n_features_in_ : int
        Number of descriptors seen during ``fit``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.feature_selection import BorutaSelector
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = X[:, 0] * 5.0 + rng.normal(scale=0.1, size=200)
    >>> sel = BorutaSelector(n_iterations=20, random_state=0).fit(X, y)
    >>> bool(sel.support_[0])
    True

    References
    ----------
    - Kursa, M. B. & Rudnicki, W. R. (2010). "Feature Selection with the
      Boruta Package." Journal of Statistical Software, 36(11), 1-13.
      https://doi.org/10.18637/jss.v036.i11
    """

    hits_: npt.NDArray[np.intp]
    decision_: npt.NDArray[np.str_]
    support_: npt.NDArray[np.bool_]
    n_features_in_: int

    def __init__(
        self,
        estimator: Optional[BaseEstimator] = None,
        task: Literal["regression", "classification"] = "regression",
        n_iterations: int = 100,
        alpha: float = 0.05,
        include_tentative: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.task = task
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.include_tentative = include_tentative
        self.random_state = random_state

    def _default_estimator(self) -> BaseEstimator:
        if self.task == "classification":
            return RandomForestClassifier(n_estimators=200, random_state=self.random_state)
        return RandomForestRegressor(n_estimators=200, random_state=self.random_state)

    def fit(self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None) -> "BorutaSelector":
        """Run the Boruta shadow-permutation procedure.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        BorutaSelector
            The fitted selector.

        Raises
        ------
        ValueError
            If ``task`` is invalid, or if ``estimator`` does not expose
            ``feature_importances_`` after fitting.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.task not in ("regression", "classification"):
            raise ValueError(
                f"task must be 'regression' or 'classification', got {self.task!r}."
            )

        n_samples, n_features = arr.shape
        self.n_features_in_ = n_features
        y_arr = np.asarray(y)
        base_estimator = (
            self.estimator if self.estimator is not None else self._default_estimator()
        )

        rng = np.random.default_rng(self.random_state)
        hits = np.zeros(n_features, dtype=np.intp)

        for iteration in range(self.n_iterations):
            X_shadow = arr.copy()
            for j in range(n_features):
                X_shadow[:, j] = rng.permutation(arr[:, j])
            X_combined = np.hstack([arr, X_shadow])

            fitted = clone(base_estimator).fit(X_combined, y_arr)
            if iteration == 0 and not hasattr(fitted, "feature_importances_"):
                raise ValueError(
                    "BorutaSelector requires an estimator exposing "
                    "feature_importances_ after fit (e.g. a tree ensemble); "
                    f"got {type(fitted).__name__}."
                )

            importances = np.asarray(fitted.feature_importances_, dtype=np.float64)
            real_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            threshold = shadow_imp.max()
            hits += (real_imp > threshold).astype(np.intp)

        self.hits_ = hits
        bonferroni_alpha = self.alpha / n_features
        decision = np.full(n_features, "Tentative", dtype="<U10")
        for j in range(n_features):
            pvalue = binomtest(
                int(hits[j]), self.n_iterations, 0.5, alternative="two-sided"
            ).pvalue
            if hits[j] > self.n_iterations / 2 and pvalue < bonferroni_alpha:
                decision[j] = "Confirmed"
            elif hits[j] < self.n_iterations / 2 and pvalue < bonferroni_alpha:
                decision[j] = "Rejected"
        self.decision_ = decision

        support = decision == "Confirmed"
        if self.include_tentative:
            support = support | (decision == "Tentative")
        self.support_ = np.asarray(support, dtype=np.bool_)

        # Confirming a feature needs enough trials for the Bonferroni-
        # corrected binomial test to reach significance; with too few
        # iterations nothing can pass and the selector silently returns an
        # empty set, which breaks any downstream pipeline. Say so.
        if not self.support_.any():
            warnings.warn(
                f"Boruta confirmed no features after {self.n_iterations} "
                "iterations. With a Bonferroni-corrected binomial test, "
                f"{n_features} features need roughly "
                f"{self._minimum_iterations(n_features)} iterations before any "
                "can reach significance. Increase n_iterations, or pass "
                "include_tentative=True to keep undecided features.",
                UserWarning,
                stacklevel=2,
            )
        return self

    def _minimum_iterations(self, n_features: int) -> int:
        """Iterations needed before a perfect feature could be confirmed.

        A feature that beats its shadow on every one of ``n`` trials has
        binomial p-value ``2 * 0.5**n``; confirmation needs that below
        ``alpha / n_features``.
        """
        from math import ceil, log2

        if n_features < 1:
            return 1
        return int(ceil(log2(2.0 * n_features / self.alpha)))

    def _get_support_mask(self) -> npt.NDArray[np.bool_]:
        check_is_fitted(self)
        return self.support_
