"""Conformal prediction: distribution-free prediction intervals and sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone

from qsarkit.base.exceptions import ModelNotFittedError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["ConformalRegressor", "ConformalClassifier", "ConformalPredictor"]


class ConformalRegressor(BaseEstimator):
    """Inductive (split) conformal prediction intervals for regression.

    Conformal prediction turns any point predictor into an interval
    predictor with a **guaranteed** marginal coverage: at significance
    ``alpha``, at least ``1 - alpha`` of future predictions contain the
    true value. The guarantee needs only that the data be exchangeable —
    no distributional assumption, no assumption that the model is correct.

    The split (inductive) variant fits the model on a proper training
    subset, computes nonconformity scores on a held-out calibration
    subset, and takes the empirical ``1 - alpha`` quantile of those
    scores as the interval half-width.

    Normalized conformal prediction scales each nonconformity score by a
    difficulty estimate, so easy molecules get tighter intervals than
    hard ones. Without it every compound receives the same width, which
    satisfies the coverage guarantee but says nothing useful about any
    individual prediction.

    Parameters
    ----------
    estimator : sklearn regressor
        The underlying point predictor. Cloned, not modified.
    alpha : float, default 0.1
        Significance level; intervals target ``1 - alpha`` coverage.
    normalized : bool, default False
        Scale intervals by a per-sample difficulty estimate.
    difficulty_estimator : sklearn regressor, optional
        Model predicting the log absolute residual, used when
        ``normalized=True``. Defaults to a k-NN regressor.
    beta : float, default 0.1
        Stabilizer added to the difficulty estimate, preventing
        near-zero denominators from producing absurdly tight intervals.
    calibration_size : float, default 0.3
        Fraction of ``fit`` data held out for calibration.
    random_state : int, optional
        Seed for the calibration split.

    Attributes
    ----------
    calibration_scores_ : ndarray
        Nonconformity scores on the calibration set.
    quantile_ : float
        The ``1 - alpha`` quantile used as the interval half-width.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(200, 4))
    >>> y = X[:, 0] * 2 + rng.normal(scale=0.3, size=200)
    >>> cp = ConformalRegressor(RandomForestRegressor(n_estimators=20,
    ...                                              random_state=0),
    ...                         alpha=0.1, random_state=0).fit(X, y)
    >>> lower, upper = cp.predict_interval(X)
    >>> bool(np.all(upper >= lower))
    True

    References
    ----------
    - Vovk, V., Gammerman, A. & Shafer, G. (2005). "Algorithmic Learning
      in a Random World." Springer. https://doi.org/10.1007/b106715
    - Papadopoulos, H. et al. (2002). "Inductive Confidence Machines for
      Regression." ECML 2002, 345-356.
      https://doi.org/10.1007/3-540-36755-1_29
    - Norinder, U. et al. (2014). "Introducing Conformal Prediction in
      Predictive Modeling. A Transparent and Flexible Alternative to
      Applicability Domain Determination." J. Chem. Inf. Model., 54(6),
      1596-1603. https://doi.org/10.1021/ci5001168
    - Svensson, F. et al. (2018). "Conformal Regression for Quantitative
      Structure-Activity Relationship Modeling." J. Chem. Inf. Model.,
      58(5), 1132-1140. https://doi.org/10.1021/acs.jcim.8b00054
    """

    calibration_scores_: npt.NDArray[np.float64]
    quantile_: float

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        normalized: bool = False,
        difficulty_estimator: Optional[Any] = None,
        beta: float = 0.1,
        calibration_size: float = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.normalized = normalized
        self.difficulty_estimator = difficulty_estimator
        self.beta = beta
        self.calibration_size = calibration_size
        self.random_state = random_state

    def _check_fitted(self) -> None:
        if not hasattr(self, "quantile_"):
            raise ModelNotFittedError(
                "ConformalRegressor must be fitted before predicting."
            )

    def _difficulty(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Per-sample interval scaling factor (1.0 when not normalized)."""
        if not self.normalized:
            return np.ones(len(X), dtype=np.float64)
        log_sigma = np.asarray(self._difficulty_model.predict(X), dtype=np.float64)
        return np.exp(log_sigma) + self.beta

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "ConformalRegressor":
        """Fit the model and calibrate nonconformity scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        ConformalRegressor
        """
        from sklearn.model_selection import train_test_split

        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")
        if not 0.0 < self.calibration_size < 1.0:
            raise ValueError(
                f"calibration_size must be in (0, 1), got {self.calibration_size}."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_arr, y_arr, test_size=self.calibration_size,
            random_state=self.random_state,
        )
        if len(X_calib) < 2:
            raise ValueError(
                "Calibration set has fewer than 2 samples; increase "
                "calibration_size or supply more data."
            )

        self._model = clone(self.estimator).fit(X_train, y_train)
        residuals = np.abs(y_calib - np.asarray(self._model.predict(X_calib)))

        if self.normalized:
            from sklearn.neighbors import KNeighborsRegressor

            base = (
                clone(self.difficulty_estimator)
                if self.difficulty_estimator is not None
                else KNeighborsRegressor(n_neighbors=min(5, len(X_train)))
            )
            train_residuals = np.abs(
                y_train - np.asarray(self._model.predict(X_train))
            )
            # Fit on log residuals so the difficulty model cannot predict a
            # negative spread, and exponentiate back in `_difficulty`.
            self._difficulty_model = base.fit(
                X_train, np.log(train_residuals + self.beta)
            )
            scores = residuals / self._difficulty(X_calib)
        else:
            scores = residuals

        self.calibration_scores_ = np.sort(scores)
        # The finite-sample-valid quantile index (Vovk et al.): using the
        # plain empirical quantile under-covers on small calibration sets.
        n = len(scores)
        rank = int(np.ceil((n + 1) * (1 - self.alpha))) - 1
        rank = int(np.clip(rank, 0, n - 1))
        self.quantile_ = float(self.calibration_scores_[rank])
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Point predictions from the underlying model."""
        self._check_fitted()
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)))

    def predict_interval(
        self, X: npt.ArrayLike, alpha: Optional[float] = None
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Prediction intervals at significance ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        alpha : float, optional
            Overrides the fitted ``alpha``. Recomputed from the stored
            calibration scores, so changing it needs no refitting.

        Returns
        -------
        lower, upper : ndarray of shape (n_samples,)
        """
        self._check_fitted()
        X_arr = np.asarray(X, dtype=np.float64)
        point = self.predict(X_arr)

        if alpha is None:
            quantile = self.quantile_
        else:
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
            n = len(self.calibration_scores_)
            rank = int(np.clip(int(np.ceil((n + 1) * (1 - alpha))) - 1, 0, n - 1))
            quantile = float(self.calibration_scores_[rank])

        half_width = quantile * self._difficulty(X_arr)
        return point - half_width, point + half_width

    def interval_width(
        self, X: npt.ArrayLike, alpha: Optional[float] = None
    ) -> npt.NDArray[np.float64]:
        """Width of each prediction interval — the uncertainty estimate."""
        lower, upper = self.predict_interval(X, alpha)
        return np.asarray(upper - lower, dtype=np.float64)

    def evaluate(
        self, X: npt.ArrayLike, y: npt.ArrayLike, alpha: Optional[float] = None
    ) -> dict:
        """Empirical coverage and efficiency on a held-out set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        alpha : float, optional

        Returns
        -------
        dict
            ``coverage`` (fraction of true values inside the interval),
            ``expected_coverage`` (``1 - alpha``), ``mean_width`` and
            ``median_width``. A valid conformal predictor has coverage at
            or just above the expected value; among predictors that
            achieve it, narrower is better.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        lower, upper = self.predict_interval(X, alpha)
        covered = (y_arr >= lower) & (y_arr <= upper)
        width = upper - lower
        return {
            "coverage": float(np.mean(covered)),
            "expected_coverage": 1.0 - (self.alpha if alpha is None else alpha),
            "mean_width": float(np.mean(width)),
            "median_width": float(np.median(width)),
        }


class ConformalClassifier(BaseEstimator):
    """Inductive conformal prediction sets for classification.

    Instead of one label, returns the *set* of labels that cannot be
    rejected at significance ``alpha``. The set size is the honest
    expression of uncertainty: a singleton means a confident call, two or
    more labels means the model genuinely cannot distinguish them, and an
    empty set means the compound resembles no training class at all —
    which a plain ``predict_proba`` can never tell you, since it always
    sums to one no matter how unfamiliar the input.

    Parameters
    ----------
    estimator : sklearn classifier
        Must expose ``predict_proba``. Cloned, not modified.
    alpha : float, default 0.1
        Significance level.
    mondrian : bool, default False
        Calibrate per class rather than globally, which gives per-class
        rather than only marginal validity — important on imbalanced
        datasets, where global calibration lets the majority class absorb
        the error budget.
    calibration_size : float, default 0.3
        Fraction of ``fit`` data held out for calibration.
    random_state : int, optional
        Seed for the calibration split.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    calibration_scores_ : ndarray
        Nonconformity scores on the calibration set.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(200, 4))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> cp = ConformalClassifier(RandomForestClassifier(n_estimators=20,
    ...                                                 random_state=0),
    ...                          alpha=0.1, random_state=0).fit(X, y)
    >>> sets = cp.predict_set(X[:5])
    >>> all(isinstance(s, list) for s in sets)
    True

    References
    ----------
    - Vovk, V., Gammerman, A. & Shafer, G. (2005). "Algorithmic Learning
      in a Random World." Springer. https://doi.org/10.1007/b106715
    - Norinder, U. et al. (2014). J. Chem. Inf. Model., 54(6), 1596-1603.
      https://doi.org/10.1021/ci5001168
    - Vovk, V. (2012). "Conditional Validity of Inductive Conformal
      Predictors." Proc. ACML, 25, 475-490.
      https://proceedings.mlr.press/v25/vovk12.html
    """

    classes_: npt.NDArray[Any]
    calibration_scores_: npt.NDArray[np.float64]

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        mondrian: bool = False,
        calibration_size: float = 0.3,
        random_state: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.alpha = alpha
        self.mondrian = mondrian
        self.calibration_size = calibration_size
        self.random_state = random_state

    def _check_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise ModelNotFittedError(
                "ConformalClassifier must be fitted before predicting."
            )

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "ConformalClassifier":
        """Fit the classifier and calibrate nonconformity scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        ConformalClassifier
        """
        from sklearn.model_selection import train_test_split

        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}.")

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_arr, y_arr, test_size=self.calibration_size,
            random_state=self.random_state, stratify=y_arr,
        )
        self._model = clone(self.estimator).fit(X_train, y_train)
        self.classes_ = np.asarray(self._model.classes_)

        proba = np.asarray(self._model.predict_proba(X_calib), dtype=np.float64)
        positions = np.searchsorted(self.classes_, y_calib)
        # Nonconformity = 1 - P(true class): high when the model is confidently wrong.
        scores = 1.0 - proba[np.arange(len(y_calib)), positions]
        self.calibration_scores_ = scores

        if self.mondrian:
            self._class_scores = {
                int(i): np.sort(scores[positions == i])
                for i in range(len(self.classes_))
            }
        return self

    def p_values(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Conformal p-value for each (sample, class) pair.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The fraction of calibration scores at least as nonconforming
            as this sample would be if it belonged to that class.
        """
        self._check_fitted()
        proba = np.asarray(
            self._model.predict_proba(np.asarray(X, dtype=np.float64)),
            dtype=np.float64,
        )
        candidate_scores = 1.0 - proba
        p = np.empty_like(candidate_scores)
        for c in range(candidate_scores.shape[1]):
            reference = (
                self._class_scores[c]
                if self.mondrian
                else self.calibration_scores_
            )
            n = len(reference)
            if n == 0:  # a class absent from the calibration split
                p[:, c] = 1.0
                continue
            counts = np.sum(
                reference[None, :] >= candidate_scores[:, c][:, None], axis=1
            )
            p[:, c] = (counts + 1.0) / (n + 1.0)
        return p

    def predict_set(
        self, X: npt.ArrayLike, alpha: Optional[float] = None
    ) -> List[List[Any]]:
        """Prediction sets: every label not rejected at ``alpha``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        alpha : float, optional
            Overrides the fitted significance level.

        Returns
        -------
        list of list
            One label list per sample. May be empty (nothing conforms) or
            hold several labels (genuinely ambiguous).
        """
        level = self.alpha if alpha is None else alpha
        if not 0.0 < level < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {level}.")
        p = self.p_values(X)
        return [
            [self.classes_[c] for c in range(p.shape[1]) if row[c] > level]
            for row in p
        ]

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Point predictions from the underlying classifier."""
        self._check_fitted()
        return np.asarray(self._model.predict(np.asarray(X, dtype=np.float64)))

    def evaluate(
        self, X: npt.ArrayLike, y: npt.ArrayLike, alpha: Optional[float] = None
    ) -> dict:
        """Coverage and set-size statistics on a held-out set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        alpha : float, optional

        Returns
        -------
        dict
            ``coverage``, ``expected_coverage``, ``mean_set_size``,
            ``singleton_fraction`` (confident calls) and
            ``empty_fraction`` (compounds resembling no training class).
        """
        y_arr = np.asarray(y)
        sets = self.predict_set(X, alpha)
        covered = [true in s for true, s in zip(y_arr, sets)]
        sizes = np.array([len(s) for s in sets], dtype=np.float64)
        return {
            "coverage": float(np.mean(covered)),
            "expected_coverage": 1.0 - (self.alpha if alpha is None else alpha),
            "mean_set_size": float(np.mean(sizes)),
            "singleton_fraction": float(np.mean(sizes == 1)),
            "empty_fraction": float(np.mean(sizes == 0)),
        }


def ConformalPredictor(
    estimator: Any,
    task: Literal["auto", "regression", "classification"] = "auto",
    **kwargs: Any,
) -> Any:
    """Build the right conformal predictor for an estimator.

    Parameters
    ----------
    estimator : sklearn estimator
        The underlying point predictor.
    task : {"auto", "regression", "classification"}, default "auto"
        ``"auto"`` picks by whether the estimator exposes
        ``predict_proba``.
    **kwargs
        Passed to :class:`ConformalRegressor` or
        :class:`ConformalClassifier`.

    Returns
    -------
    ConformalRegressor or ConformalClassifier

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> type(ConformalPredictor(RandomForestRegressor())).__name__
    'ConformalRegressor'

    References
    ----------
    - Vovk, V., Gammerman, A. & Shafer, G. (2005).
      https://doi.org/10.1007/b106715
    """
    if task == "auto":
        from sklearn.base import is_classifier

        task = "classification" if is_classifier(estimator) else "regression"
    if task == "classification":
        return ConformalClassifier(estimator, **kwargs)
    if task == "regression":
        return ConformalRegressor(estimator, **kwargs)
    raise ValueError(
        f"task must be 'auto', 'regression' or 'classification', got {task!r}."
    )
