"""Support-vector regressor with QSAR-sane defaults."""

from __future__ import annotations

from typing import Union

from sklearn.svm import SVR

__all__ = ["SVMQSAR"]


class SVMQSAR(SVR):
    """Support-vector regressor tuned with QSAR-sane defaults.

    A thin subclass of :class:`sklearn.svm.SVR` that keeps the parent's
    full parameter set but defaults to an RBF kernel with ``gamma="scale"``
    — the combination most consistently reported to work well for QSAR
    on molecular descriptors and fingerprints without extensive tuning.
    Support-vector regression is attractive for QSAR because its
    epsilon-insensitive loss is robust to the noisy, assay-variability-
    laden activity values typical of biological data, and the kernel
    trick lets it capture non-linear structure-activity relationships
    without hand-engineered interaction terms.

    Parameters
    ----------
    kernel : {"linear", "poly", "rbf", "sigmoid", "precomputed"} or callable, default "rbf"
        Kernel used by the support-vector regressor.
    C : float, default 1.0
        Regularization strength (inverse); larger values fit the training
        data more closely.
    gamma : {"scale", "auto"} or float, default "scale"
        Kernel coefficient for "rbf", "poly" and "sigmoid".
    epsilon : float, default 0.1
        Width of the epsilon-insensitive tube within which no penalty is
        incurred.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=40, n_features=5, random_state=0)
    >>> model = SVMQSAR().fit(X, y)
    >>> model.predict(X).shape
    (40,)

    References
    ----------
    - Cortes, C. & Vapnik, V. (1995). "Support-Vector Networks." Machine
      Learning, 20(3), 273-297. https://doi.org/10.1007/BF00994018
    - Burbidge, R., Trotter, M., Buxton, B. & Holden, S. (2001). "Drug
      Design by Machine Learning: Support Vector Machines for
      Pharmaceutical Data Analysis." Comput. Chem., 26(1), 5-14.
      https://doi.org/10.1016/S0097-8485(01)00094-8
    """

    def __init__(
        self,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: Union[str, float] = "scale",
        coef0: float = 0.0,
        tol: float = 1e-3,
        C: float = 1.0,
        epsilon: float = 0.1,
        shrinking: bool = True,
        cache_size: float = 200,
        verbose: bool = False,
        max_iter: int = -1,
    ) -> None:
        super().__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            epsilon=epsilon,
            shrinking=shrinking,
            cache_size=cache_size,
            verbose=verbose,
            max_iter=max_iter,
        )
