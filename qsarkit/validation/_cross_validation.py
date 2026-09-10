"""Cross-validated Q^2 estimation for QSAR models (OECD validation principle 4).

Internal cross-validation is the classical way to gauge how well a QSAR
model is expected to generalize before an external test set is even
collected. This module wraps scikit-learn's fold-splitting machinery
(``KFold``, ``LeaveOneOut``, ``RepeatedKFold``, ``LeaveOneGroupOut``) and
reduces the resulting out-of-fold predictions to the single number the
QSAR literature calls Q^2 -- numerically the same formula as R^2, applied
to predictions the model never saw during its own fit.

References
----------
- Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
  Chemoinformatics," 2nd ed. Wiley-VCH.
  https://doi.org/10.1002/9783527628766
- Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
  Definition of the Q^2 Parameter for QSAR Validation." J. Chem. Inf.
  Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, LeaveOneGroupOut, LeaveOneOut, RepeatedKFold

from qsarkit.metrics import mae, r2_score, rmse

__all__ = ["CrossValidator"]


class CrossValidator:
    """Cross-validated Q^2, RMSE_CV and MAE_CV for a scikit-learn estimator.

    Supports the four fold-splitting schemes routinely used in QSAR
    validation studies: k-fold, leave-one-out, repeated k-fold (to
    average away the fold-assignment randomness of a single k-fold run)
    and leave-one-group-out (for scaffold- or assay-based splits passed
    via ``groups``). Every fold clones the estimator with
    ``sklearn.base.clone`` before fitting, so the caller's own estimator
    object is never mutated.

    Parameters
    ----------
    method : {"kfold", "loo", "repeated_kfold", "leave_group_out"}, default "kfold"
        Cross-validation scheme.
    n_splits : int, default 5
        Number of folds for ``"kfold"`` and ``"repeated_kfold"``. Ignored
        by ``"loo"`` and ``"leave_group_out"`` (both use one fold per
        sample / per group).
    n_repeats : int, default 10
        Number of repeats for ``"repeated_kfold"``. Ignored otherwise.
    random_state : int, optional
        Seed controlling the fold shuffling of ``"kfold"`` and
        ``"repeated_kfold"``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.validation import CrossValidator
    >>> X, y = make_regression(n_samples=60, n_features=5, noise=1.0, random_state=0)
    >>> result = CrossValidator(method="kfold", n_splits=5, random_state=0).evaluate(
    ...     Ridge(), X, y
    ... )
    >>> result["q2"] > 0.5
    True

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
      Chemoinformatics," 2nd ed. Wiley-VCH.
      https://doi.org/10.1002/9783527628766
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """

    method: Literal["kfold", "loo", "repeated_kfold", "leave_group_out"]
    n_splits: int
    n_repeats: int
    random_state: Optional[int]

    def __init__(
        self,
        method: Literal["kfold", "loo", "repeated_kfold", "leave_group_out"] = "kfold",
        n_splits: int = 5,
        n_repeats: int = 10,
        random_state: Optional[int] = None,
    ) -> None:
        self.method = method
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def _splitter_and_args(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64],
        groups: Optional[npt.NDArray[np.float64]],
    ) -> Tuple[Any, Tuple[Any, ...]]:
        """Build the sklearn splitter and its ``.split()`` positional args."""
        if self.method == "kfold":
            return (
                KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state),
                (X, y),
            )
        if self.method == "loo":
            return LeaveOneOut(), (X, y)
        if self.method == "repeated_kfold":
            return (
                RepeatedKFold(
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state,
                ),
                (X, y),
            )
        if self.method == "leave_group_out":
            if groups is None:
                raise ValueError(
                    "method='leave_group_out' requires `groups` to be provided."
                )
            return LeaveOneGroupOut(), (X, y, groups)
        raise ValueError(
            "method must be one of 'kfold', 'loo', 'repeated_kfold', "
            f"'leave_group_out'; got {self.method!r}."
        )

    def evaluate(
        self,
        estimator: BaseEstimator,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        groups: Optional[npt.ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Cross-validate ``estimator`` and compute Q^2, RMSE_CV and MAE_CV.

        Parameters
        ----------
        estimator : sklearn-compatible estimator
            Any object implementing ``.fit(X, y)`` / ``.predict(X)``. A
            fresh clone is fit on each fold; the passed-in object is
            never mutated.
        X : array-like of shape (n_samples, n_features)
            Descriptor matrix.
        y : array-like of shape (n_samples,)
            Observed responses.
        groups : array-like of shape (n_samples,), optional
            Group labels, required when ``method="leave_group_out"``.

        Returns
        -------
        dict
            ``"q2"`` (float), ``"rmse_cv"`` (float), ``"mae_cv"`` (float),
            ``"y_pred_cv"`` (ndarray of out-of-fold predictions aligned to
            the original row order -- for ``"repeated_kfold"`` this is
            the average out-of-fold prediction over all repeats),
            ``"method"`` (str) and ``"n_splits"`` (int -- the number of
            folds actually run, which for ``"loo"`` is the sample count
            and for ``"leave_group_out"`` the number of groups).

        Raises
        ------
        ValueError
            If ``method="leave_group_out"`` and ``groups`` is ``None``,
            or if ``method`` is not one of the four supported schemes.

        Examples
        --------
        >>> import numpy as np
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.linear_model import Ridge
        >>> from qsarkit.validation import CrossValidator
        >>> X, y = make_regression(n_samples=50, n_features=4, noise=1.0, random_state=0)
        >>> result = CrossValidator(method="loo").evaluate(Ridge(), X, y)
        >>> sorted(result)
        ['mae_cv', 'method', 'n_splits', 'q2', 'rmse_cv', 'y_pred_cv']
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X_arr.shape}.")
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        groups_arr = None if groups is None else np.asarray(groups)

        splitter, split_args = self._splitter_and_args(X_arr, y_arr, groups_arr)

        n = X_arr.shape[0]
        pred_sum = np.zeros(n, dtype=np.float64)
        pred_count = np.zeros(n, dtype=np.int64)
        # Report the number of folds actually run, not the constructor
        # argument: LOO runs one fold per sample and leave-group-out one
        # per group, so echoing self.n_splits would describe a different
        # experiment from the one performed.
        n_splits_run = int(splitter.get_n_splits(*split_args))
        for train_idx, test_idx in splitter.split(*split_args):
            fitted = clone(estimator).fit(X_arr[train_idx], y_arr[train_idx])
            preds = np.asarray(fitted.predict(X_arr[test_idx]), dtype=np.float64)
            pred_sum[test_idx] += preds
            pred_count[test_idx] += 1

        y_pred_cv = pred_sum / pred_count
        return {
            "q2": r2_score(y_arr, y_pred_cv),
            "rmse_cv": rmse(y_arr, y_pred_cv),
            "mae_cv": mae(y_arr, y_pred_cv),
            "y_pred_cv": y_pred_cv,
            "method": self.method,
            "n_splits": n_splits_run,
        }
