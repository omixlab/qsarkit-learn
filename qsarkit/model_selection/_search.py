"""Hyperparameter search and nested cross-validation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone

__all__ = ["hyperparameter_search", "NestedCV"]


def hyperparameter_search(
    estimator: BaseEstimator,
    param_grid: Union[Dict[str, Sequence[Any]], List[Dict[str, Sequence[Any]]]],
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    method: Literal["grid", "random", "optuna"] = "grid",
    cv: Any = 5,
    scoring: Optional[str] = None,
    n_iter: int = 20,
    n_jobs: Optional[int] = None,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Tune hyperparameters by grid, random or Bayesian search.

    Parameters
    ----------
    estimator : sklearn estimator
        The model to tune. Cloned, not modified.
    param_grid : dict or list of dict
        Parameter name -> values to try, in scikit-learn's format.
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    method : {"grid", "random", "optuna"}, default "grid"
        Exhaustive grid, random sampling, or Optuna's TPE sampler
        (requires the optional ``optuna`` package). Random search beats
        grid search on the same budget whenever only a few parameters
        actually matter, which is the usual case.
    cv : int or cross-validation splitter, default 5
        Pass a :mod:`qsarkit.model_selection` splitter to tune under a
        scaffold or time split rather than a random one.
    scoring : str, optional
        scikit-learn scorer name. Defaults to the estimator's own score.
    n_iter : int, default 20
        Number of parameter settings sampled by the random and Optuna
        methods.
    n_jobs : int, optional
        Parallel jobs.
    random_state : int, optional
        Seed.
    **kwargs
        Forwarded to the underlying search object.

    Returns
    -------
    sklearn search object
        Fitted, exposing ``best_estimator_``, ``best_params_`` and
        ``best_score_``.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X, y = make_regression(n_samples=40, n_features=5, random_state=0)
    >>> search = hyperparameter_search(
    ...     RandomForestRegressor(random_state=0),
    ...     {"n_estimators": [5, 10]}, X, y, cv=2,
    ... )
    >>> "n_estimators" in search.best_params_
    True

    References
    ----------
    - Bergstra, J. & Bengio, Y. (2012). "Random Search for Hyper-Parameter
      Optimization." J. Mach. Learn. Res., 13, 281-305.
      https://jmlr.org/papers/v13/bergstra12a.html
    - Akiba, T. et al. (2019). "Optuna: A Next-generation Hyperparameter
      Optimization Framework." KDD 2019, 2623-2631.
      https://doi.org/10.1145/3292500.3330701
    - scikit-learn model selection documentation:
      https://scikit-learn.org/stable/modules/grid_search.html
    """
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

    if method == "grid":
        search: Any = GridSearchCV(
            clone(estimator), param_grid, cv=cv, scoring=scoring,
            n_jobs=n_jobs, **kwargs,
        )
    elif method == "random":
        search = RandomizedSearchCV(
            clone(estimator), param_grid, n_iter=n_iter, cv=cv,
            scoring=scoring, n_jobs=n_jobs, random_state=random_state, **kwargs,
        )
    elif method == "optuna":
        from qsarkit.base import require

        require("optuna")
        from optuna.integration import OptunaSearchCV  # type: ignore[import-not-found]

        search = OptunaSearchCV(
            clone(estimator), param_grid, cv=cv, scoring=scoring,
            n_trials=n_iter, random_state=random_state, **kwargs,
        )
    else:
        raise ValueError(
            f"method must be 'grid', 'random' or 'optuna', got {method!r}."
        )

    search.fit(np.asarray(X), np.asarray(y))
    return search


class NestedCV:
    """Nested cross-validation: unbiased performance for a *tuned* model.

    Tuning hyperparameters and reporting the best cross-validated score
    from that same search is one of the most common ways QSAR papers
    overstate performance — the score is optimistically biased because
    the test folds were used to choose the model. Nested CV fixes it by
    tuning inside an inner loop and scoring on outer folds the tuning
    never saw.

    Parameters
    ----------
    estimator : sklearn estimator
        Model to tune and evaluate.
    param_grid : dict
        Search space for the inner loop.
    inner_cv : int or splitter, default 3
        Cross-validation used for tuning.
    outer_cv : int or splitter, default 5
        Cross-validation used for scoring.
    scoring : str, optional
        scikit-learn scorer name.
    n_jobs : int, optional
        Parallel jobs.

    Attributes
    ----------
    scores_ : ndarray
        Outer-fold scores.
    best_params_ : list of dict
        The parameters chosen in each outer fold. Disagreement between
        folds is itself informative: it means the tuning is unstable and
        the "best" parameters from a single search are noise.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> X, y = make_regression(n_samples=40, n_features=5, random_state=0)
    >>> ncv = NestedCV(Ridge(), {"alpha": [0.1, 1.0]}, inner_cv=2, outer_cv=2)
    >>> _ = ncv.run(X, y)
    >>> ncv.scores_.shape
    (2,)

    References
    ----------
    - Varma, S. & Simon, R. (2006). "Bias in Error Estimation when Using
      Cross-Validation for Model Selection." BMC Bioinformatics, 7, 91.
      https://doi.org/10.1186/1471-2105-7-91
    - Cawley, G. C. & Talbot, N. L. C. (2010). "On Over-fitting in Model
      Selection and Subsequent Selection Bias in Performance Evaluation."
      J. Mach. Learn. Res., 11, 2079-2107.
      https://jmlr.org/papers/v11/cawley10a.html
    - Baumann, D. & Baumann, K. (2014). "Reliable Estimation of
      Prediction Errors for QSAR Models under Model Uncertainty Using
      Double Cross-Validation." J. Cheminform., 6, 47.
      https://doi.org/10.1186/s13321-014-0047-1
    """

    scores_: npt.NDArray[np.float64]
    best_params_: List[Dict[str, Any]]

    def __init__(
        self,
        estimator: BaseEstimator,
        param_grid: Dict[str, Sequence[Any]],
        inner_cv: Any = 3,
        outer_cv: Any = 5,
        scoring: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.param_grid = param_grid
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.scoring = scoring
        self.n_jobs = n_jobs

    def run(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Dict[str, Any]:
        """Run the nested loop.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        dict
            ``mean_score``, ``std_score``, ``scores`` (per outer fold),
            and ``best_params`` (the winner in each outer fold).
        """
        from sklearn.model_selection import GridSearchCV, KFold, check_cv

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        outer = (
            KFold(n_splits=self.outer_cv, shuffle=True, random_state=0)
            if isinstance(self.outer_cv, int)
            else check_cv(self.outer_cv)
        )

        scores: List[float] = []
        params: List[Dict[str, Any]] = []
        for train_idx, test_idx in outer.split(X_arr, y_arr):
            search = GridSearchCV(
                clone(self.estimator),
                self.param_grid,
                cv=self.inner_cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            ).fit(X_arr[train_idx], y_arr[train_idx])
            scores.append(
                float(search.score(X_arr[test_idx], y_arr[test_idx]))
            )
            params.append(dict(search.best_params_))

        self.scores_ = np.asarray(scores, dtype=np.float64)
        self.best_params_ = params
        return {
            "mean_score": float(self.scores_.mean()),
            "std_score": float(self.scores_.std(ddof=1)) if len(scores) > 1 else 0.0,
            "scores": self.scores_,
            "best_params": params,
        }
