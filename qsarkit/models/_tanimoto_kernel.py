"""Tanimoto kernel for Gaussian-process QSAR models."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel

from qsarkit.neighbors import tanimoto_similarity_matrix

__all__ = ["TanimotoKernel"]


class TanimotoKernel(Kernel):
    r"""Tanimoto (Jaccard) kernel for fingerprint-valued Gaussian processes.

    Implements ``k(x, y) = sigma_0^2 * T(x, y)``, where ``T`` is the
    Tanimoto coefficient ``|x AND y| / |x OR y|`` computed by
    :func:`qsarkit.neighbors.tanimoto_similarity_matrix`. Standard
    Gaussian-process kernels (RBF, dot-product, Matern, ...) measure
    similarity through a Euclidean or dot-product geometry, which is the
    wrong geometry for sparse binary fingerprints: two fingerprints that
    share no bits but both have many zeros still look "close" in
    Euclidean space, even though they encode structurally unrelated
    molecules. The Tanimoto kernel instead puts the GP prior directly on
    the similarity measure chemists already use for virtual screening and
    read-across, so the resulting posterior mean and (crucially for QSAR)
    posterior standard deviation are calibrated to fingerprint chemistry
    rather than to an arbitrary embedding geometry.

    ``k(x, y) = sigma_0^2 * T(x, y)`` is a valid (positive semi-definite)
    kernel because the Tanimoto coefficient itself is known to be
    conditionally positive definite over binary vectors (Ralaivola et al.
    2005, building on Gower's 1971 result for similarity coefficients),
    and scaling a PSD kernel by a positive constant preserves PSD-ness.

    Parameters
    ----------
    sigma_0 : float, default 1.0
        Output-scale hyperparameter. Squared to give the kernel's
        amplitude, i.e. ``k(x, x) = sigma_0**2`` for a binary fingerprint
        ``x`` (self-similarity is always 1 under Tanimoto).
    sigma_0_bounds : tuple of float, default (1e-5, 1e5)
        Lower and upper bound on ``sigma_0`` used by the GP's
        hyperparameter optimizer, or the string ``"fixed"`` to keep
        ``sigma_0`` constant during fitting (as with any scikit-learn
        kernel hyperparameter).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=float)
    >>> kernel = TanimotoKernel(sigma_0=2.0)
    >>> K = kernel(X)
    >>> K.shape
    (3, 3)
    >>> bool(np.allclose(np.diag(K), 4.0))
    True

    References
    ----------
    - Ralaivola, L., Swamidass, S. J., Saigo, H. & Baldi, P. (2005).
      "Graph Kernels for Chemical Informatics." Neural Networks, 18(8),
      1093-1110. https://doi.org/10.1016/j.neunet.2005.07.009
    - Rasmussen, C. E. & Williams, C. K. I. (2006). "Gaussian Processes
      for Machine Learning." MIT Press. ISBN 0-262-18253-X. Freely
      available at http://gaussianprocess.org/gpml/
    - scikit-learn custom kernel documentation:
      https://scikit-learn.org/stable/modules/gaussian_process.html#kernels-for-gaussian-processes
    """

    def __init__(
        self,
        sigma_0: float = 1.0,
        sigma_0_bounds: Union[Tuple[float, float], str] = (1e-5, 1e5),
    ) -> None:
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    @property
    def hyperparameter_sigma_0(self) -> Hyperparameter:
        """The ``sigma_0`` hyperparameter descriptor scikit-learn introspects."""
        return Hyperparameter("sigma_0", "numeric", self.sigma_0_bounds)

    def __call__(
        self,
        X: npt.ArrayLike,
        Y: Optional[npt.ArrayLike] = None,
        eval_gradient: bool = False,
    ) -> Union[
        npt.NDArray[np.float64],
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
    ]:
        """Evaluate the kernel matrix (and optionally its gradient).

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Fingerprint matrix.
        Y : array-like of shape (n_samples_Y, n_features), optional
            Second fingerprint matrix. Defaults to ``X``.
        eval_gradient : bool, default False
            If True, also return the gradient of the kernel with respect
            to the (log-transformed) hyperparameters. Only valid when
            ``Y`` is None.

        Returns
        -------
        ndarray of shape (n_samples_X, n_samples_Y)
            The kernel matrix ``k(X, Y)``.
        tuple of (ndarray, ndarray), only if ``eval_gradient`` is True
            The kernel matrix and its gradient of shape
            ``(n_samples_X, n_samples_X, n_dims)``.

        Raises
        ------
        ValueError
            If ``eval_gradient`` is True and ``Y`` is not None.
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        # The MinMax/Tanimoto kernel is only positive semi-definite on
        # non-negative inputs (Ralaivola et al. 2005). Fed general
        # descriptors, which are routinely negative after scaling, the
        # Gram matrix is indefinite and Cholesky fails deep inside
        # sklearn with an opaque "not positive definite" LinAlgError.
        # Catch it here, where the cause can actually be explained.
        if np.any(X_arr < 0):
            raise ValueError(
                "TanimotoKernel requires non-negative inputs (fingerprints or "
                "counts); got negative values. For general real-valued "
                "descriptors use an RBF kernel instead."
            )
        if Y is None:
            T = tanimoto_similarity_matrix(X_arr)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            Y_arr = np.atleast_2d(np.asarray(Y, dtype=np.float64))
            if np.any(Y_arr < 0):
                raise ValueError(
                    "TanimotoKernel requires non-negative inputs; Y contains "
                    "negative values."
                )
            T = tanimoto_similarity_matrix(X_arr, Y_arr)

        K = (self.sigma_0**2) * T
        if eval_gradient:
            if not self.hyperparameter_sigma_0.fixed:
                grad = (2.0 * self.sigma_0**2 * T)[:, :, np.newaxis]
                return K, np.asarray(grad, dtype=np.float64)
            return K, np.empty((X_arr.shape[0], X_arr.shape[0], 0))
        return K

    def diag(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the diagonal of the kernel matrix ``k(X, X)``.

        Cheaper than computing the full matrix and extracting the
        diagonal, since every diagonal entry is simply ``sigma_0**2``
        (Tanimoto self-similarity is always 1).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.full(X_arr.shape[0], self.sigma_0**2, dtype=np.float64)

    def is_stationary(self) -> bool:
        """Whether the kernel is stationary (depends only on ``x - y``).

        Returns
        -------
        bool
            Always False: the Tanimoto coefficient depends on the sets
            of "on" bits themselves, not merely on their difference.
        """
        return False

    def __repr__(self) -> str:
        """Short representation showing the fitted/initial ``sigma_0``."""
        return f"{self.__class__.__name__}(sigma_0={self.sigma_0:.3g})"
