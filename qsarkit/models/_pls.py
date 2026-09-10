"""Partial least squares regressor with VIP variable-importance scores."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.cross_decomposition import PLSRegression

__all__ = ["PLSRegressor"]


class PLSRegressor(PLSRegression):
    """PLS regressor that additionally reports VIP variable-importance scores.

    A thin subclass of :class:`sklearn.cross_decomposition.PLSRegression`
    that, on top of the usual PLS fit, computes the Variable Importance
    in Projection (VIP) score for every input feature. PLS is a workhorse
    QSAR technique — it handles the many-correlated-descriptors,
    few-compounds regime that ordinary least squares cannot, by
    projecting onto a small number of latent variables that jointly
    explain X and y — but the latent-variable loadings it produces are
    not directly interpretable per descriptor. VIP scores solve that by
    aggregating each descriptor's contribution across all retained
    components, weighted by how much y-variance each component explains,
    giving a single per-descriptor importance number comparable to a
    feature-importance ranking. A VIP score around 1 or above is the
    conventional threshold for "important" in chemometrics, since the
    average of all squared VIP scores is exactly 1 by construction.

    Parameters
    ----------
    n_components : int, default 2
        Number of PLS components (latent variables) to fit.
    scale : bool, default True
        Whether to standardize (z-score) ``X`` and ``y`` before fitting.
    max_iter : int, default 500
        Maximum number of iterations of the NIPALS inner loop.
    tol : float, default 1e-6
        Convergence tolerance for the NIPALS inner loop.
    copy : bool, default True
        Whether to copy ``X`` and ``y`` in :meth:`fit` before scaling.

    Attributes
    ----------
    vip_scores_ : ndarray of shape (n_features,)
        Variable Importance in Projection score for each input feature,
        computed after :meth:`fit`. Non-negative; the mean of their
        squares is 1.0 by construction.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=60, n_features=5, n_informative=2, random_state=0)
    >>> model = PLSRegressor(n_components=2).fit(X, y)
    >>> model.vip_scores_.shape
    (5,)
    >>> bool((model.vip_scores_ >= 0).all())
    True

    References
    ----------
    - Wold, S., Sjostrom, M. & Eriksson, L. (2001). "PLS-regression: a
      basic tool of chemometrics." Chemometrics and Intelligent
      Laboratory Systems, 58(2), 109-130.
      https://doi.org/10.1016/S0169-7439(01)00155-1
    - Wold, S., Ruhe, A., Wold, H. & Dunn, W. J. (1984). "The
      Collinearity Problem in Linear Regression. The Partial Least
      Squares (PLS) Approach to Generalized Inverses." SIAM J. Sci.
      Stat. Comput., 5(3), 735-743. https://doi.org/10.1137/0905052
    """

    vip_scores_: npt.NDArray[np.float64]

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        copy: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            scale=scale,
            max_iter=max_iter,
            tol=tol,
            copy=copy,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "PLSRegressor":
        """Fit the PLS model and compute VIP scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training descriptors/fingerprints.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Training target(s).

        Returns
        -------
        PLSRegressor
            The fitted estimator, with :attr:`vip_scores_` populated.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        super().fit(X_arr, y)
        self.vip_scores_ = self._compute_vip(X_arr)
        return self

    def _compute_vip(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Variable Importance in Projection, per Wold et al. (2001).

        ``VIP_j = sqrt(p * sum_h(w_jh_normalized**2 * SSY_h) / sum_h(SSY_h))``
        where ``SSY_h`` is the y-variance explained by component ``h``.
        """
        t = self.x_scores_  # (n_samples, n_components)
        w = self.x_weights_  # (n_features, n_components)
        q = self.y_loadings_  # (n_targets, n_components)
        p = w.shape[0]

        w_norm_sq = (w / np.linalg.norm(w, axis=0, keepdims=True)) ** 2  # (p, h)
        ssy = np.sum(q**2, axis=0) * np.sum(t**2, axis=0)  # (h,)
        total_ssy = ssy.sum()
        if total_ssy == 0.0:
            return np.zeros(p, dtype=np.float64)
        vip = np.sqrt(p * (w_norm_sq @ ssy) / total_ssy)
        return np.asarray(vip, dtype=np.float64)
