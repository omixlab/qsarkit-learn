"""Gaussian-process regressor with a Tanimoto kernel default, for QSAR."""

from __future__ import annotations

from typing import Any, Optional

import numpy.typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor

from qsarkit.models._tanimoto_kernel import TanimotoKernel

__all__ = ["GaussianProcessQSAR"]


class GaussianProcessQSAR(GaussianProcessRegressor):
    """Gaussian-process regressor defaulting to a Tanimoto fingerprint kernel.

    A thin subclass of
    :class:`sklearn.gaussian_process.GaussianProcessRegressor` whose only
    behavioural change is the default kernel: when ``kernel=None`` (the
    constructor default, exactly as in the parent class), :meth:`fit`
    builds a :class:`~qsarkit.models.TanimotoKernel` rather than
    scikit-learn's own RBF default — mirroring the parent class's own
    convention of resolving ``kernel=None`` to a concrete kernel lazily
    inside ``fit()``, so ``__init__`` keeps storing exactly what was
    passed to it (sklearn's ``get_params``/``clone`` contract).

    The QSAR motivation for a Gaussian process at all is uncertainty
    quantification: unlike a point-estimate regressor, a fitted GP
    returns both a predictive mean and a predictive standard deviation
    (``predict(X, return_std=True)``), and that per-compound standard
    deviation is a natural, model-native applicability-domain signal —
    it grows exactly where the model is extrapolating away from the
    training fingerprints. Using the Tanimoto kernel rather than RBF
    means that "far from training data" is measured in the similarity
    metric fingerprints were designed for, not in a Euclidean geometry
    that is not meaningful for sparse binary vectors.

    Parameters
    ----------
    kernel : Kernel, optional
        Covariance function. If None (default), a
        :class:`~qsarkit.models.TanimotoKernel` is constructed inside
        :meth:`fit`.
    alpha : float or array-like, default 1e-6
        Value added to the diagonal of the kernel matrix during fitting,
        interpreted as observation noise variance. The default is larger
        than scikit-learn's ``1e-10`` to better tolerate the assay noise
        typical of biological QSAR endpoints.
    optimizer : str or callable, default "fmin_l_bfgs_b"
        Optimizer used to find the kernel hyperparameters maximizing the
        log-marginal-likelihood, or None to keep the initial
        hyperparameters fixed.
    n_restarts_optimizer : int, default 0
        Number of restarts of the optimizer from hyperparameters sampled
        log-uniformly from their bounds, in addition to the initial run.
    normalize_y : bool, default True
        Whether to subtract the mean and scale the training targets to
        unit variance before fitting. True by default here (unlike
        scikit-learn's False) since QSAR endpoints (pIC50, logS, ...)
        are rarely already zero-mean.
    copy_X_train : bool, default True
        Whether a copy of the training data is stored.
    n_targets : int, optional
        Number of targets expected when fitting multi-output data.
    random_state : int, optional
        Seed controlling the optimizer's random restarts.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=30, n_features=6, random_state=0)
    >>> X_binary = (X > np.median(X, axis=0)).astype(float)
    >>> model = GaussianProcessQSAR(random_state=0).fit(X_binary, y)
    >>> mean, std = model.predict(X_binary, return_std=True)
    >>> mean.shape
    (30,)
    >>> bool((std >= 0).all())
    True

    References
    ----------
    - Ralaivola, L., Swamidass, S. J., Saigo, H. & Baldi, P. (2005).
      "Graph Kernels for Chemical Informatics." Neural Networks, 18(8),
      1093-1110. https://doi.org/10.1016/j.neunet.2005.07.009
    - Rasmussen, C. E. & Williams, C. K. I. (2006). "Gaussian Processes
      for Machine Learning." MIT Press. ISBN 0-262-18253-X. Freely
      available at http://gaussianprocess.org/gpml/
    """

    # Declared explicitly because it is set by scikit-learn's __init__,
    # which mypy sees as untyped - without this, `self.kernel` has no
    # inferable type and the None-resolution in fit() cannot type-check.
    kernel: Optional[Any]

    def __init__(
        self,
        kernel: Optional[Any] = None,
        *,
        alpha: npt.ArrayLike = 1e-6,
        optimizer: Optional[Any] = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 0,
        normalize_y: bool = True,
        copy_X_train: bool = True,
        n_targets: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> None:
        super().__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            n_targets=n_targets,
            random_state=random_state,
        )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "GaussianProcessQSAR":
        """Fit the Gaussian process, defaulting to a Tanimoto kernel.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training fingerprints or descriptors.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Training target values.

        Returns
        -------
        GaussianProcessQSAR
            The fitted estimator.
        """
        if self.kernel is None:
            # Resolve the QSAR default lazily, exactly where scikit-learn's
            # own GaussianProcessRegressor.fit() resolves kernel=None to an
            # RBF default - only choosing by what the data actually is.
            #
            # Tanimoto is the right kernel for fingerprints, but it is only
            # positive semi-definite on non-negative inputs, so on signed
            # descriptors (the norm after standardization) it produces an
            # indefinite Gram matrix and Cholesky fails. Pick RBF there.
            #
            # self.kernel is restored to None afterwards so __init__'s
            # "store exactly what was passed" contract still holds for
            # get_params()/clone() on the fitted estimator.
            import numpy as np

            from sklearn.gaussian_process.kernels import RBF, WhiteKernel

            X_arr = np.asarray(X, dtype=np.float64)
            self.kernel = (
                TanimotoKernel()
                if X_arr.size and np.all(X_arr >= 0)
                else RBF() + WhiteKernel()
            )
            try:
                super().fit(X, y)
            finally:
                self.kernel = None
            return self
        super().fit(X, y)
        return self
