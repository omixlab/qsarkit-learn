"""Random-forest regressor with QSAR-sane defaults."""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor

__all__ = ["RandomForestQSAR"]


class RandomForestQSAR(RandomForestRegressor):
    """Random-forest regressor tuned with QSAR-sane defaults.

    A thin subclass of :class:`sklearn.ensemble.RandomForestRegressor`
    that only changes the *defaults* (larger forest, ``sqrt``-of-features
    splitting) to values that repeatedly perform well on molecular
    descriptor/fingerprint QSAR benchmarks, while leaving every parameter
    of the parent estimator overridable. Random forests are a strong
    default QSAR model: they are insensitive to feature scaling, handle
    the mixed continuous/binary descriptor matrices typical of
    cheminformatics without preprocessing, and are robust to the
    correlated, high-dimensional descriptor sets QSAR studies routinely
    produce.

    Parameters
    ----------
    n_estimators : int, default 500
        Number of trees. QSAR forests benefit from more trees than the
        scikit-learn default (100) because descriptor sets are often
        high-dimensional and correlated, so more trees are needed to
        stabilize the feature-subsampling variance.
    max_features : {"sqrt", "log2", None} or int or float, default "sqrt"
        Number of features considered at each split. ``"sqrt"`` is the
        classical Breiman recommendation for regression forests and
        decorrelates trees built from correlated molecular descriptors.
    random_state : int, optional
        Seed for reproducibility. None every other parameter is
        forwarded verbatim to :class:`~sklearn.ensemble.RandomForestRegressor`.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=40, n_features=5, random_state=0)
    >>> model = RandomForestQSAR(n_estimators=10, random_state=0).fit(X, y)
    >>> model.predict(X).shape
    (40,)

    References
    ----------
    - Breiman, L. (2001). "Random Forests." Machine Learning, 45(1),
      5-32. https://doi.org/10.1023/A:1010933404324
    - Svetnik, V., Liaw, A., Tong, C., Culberson, J. C., Sheridan, R. P.
      & Feuston, B. P. (2003). "Random Forest: A Classification and
      Regression Tool for Compound Classification and QSAR Modeling."
      J. Chem. Inf. Comput. Sci., 43(6), 1947-1958.
      https://doi.org/10.1021/ci034160g
    """

    def __init__(
        self,
        n_estimators: int = 500,
        *,
        criterion: str = "squared_error",
        max_depth: Optional[int] = None,
        min_samples_split: Union[int, float] = 2,
        min_samples_leaf: Union[int, float] = 1,
        min_weight_fraction_leaf: float = 0.0,
        max_features: Union[str, int, float, None] = "sqrt",
        max_leaf_nodes: Optional[int] = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: Union[bool, Callable[..., float]] = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        warm_start: bool = False,
        ccp_alpha: float = 0.0,
        max_samples: Union[int, float, None] = None,
        monotonic_cst: Optional[npt.ArrayLike] = None,
    ) -> None:
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
