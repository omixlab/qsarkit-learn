"""Robustness and external-predictivity checks for OECD principle 4.

Principle 4 asks for three distinct things -- goodness of fit, robustness,
and predictivity -- and a single R² addresses only the first.
:class:`YScrambling` addresses robustness by testing whether the model can
fit noise as well as it fits the data; :class:`ExternalValidator` and
:class:`BootstrapValidator` address predictivity and the stability of the
estimate respectively.

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on Testing
  and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
- Rucker, C., Rucker, G. & Meringer, M. (2007). "y-Randomization and Its
  Variants in QSPR/QSAR." J. Chem. Inf. Model., 47(6), 2345-2357.
  https://doi.org/10.1021/ci700157b
- Efron, B. & Tibshirani, R. J. (1993). "An Introduction to the Bootstrap."
  Chapman & Hall. https://doi.org/10.1201/9780429246593
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone

__all__ = ["YScrambling", "ExternalValidator", "BootstrapValidator"]


def _fit_score(
    estimator: BaseEstimator,
    X: "npt.NDArray[np.float64]",
    y: "npt.NDArray[np.float64]",
) -> float:
    """Fit a fresh clone and return its coefficient of determination."""
    from qsarkit.metrics import r2_score

    model = clone(estimator)
    model.fit(X, y)
    return float(r2_score(y, model.predict(X)))


class YScrambling:
    """Test whether a model can fit randomly permuted labels as well as real ones.

    Also called y-randomization. Refit the model many times on shuffled
    activities: if the scrambled models score anywhere near the real one,
    the apparent performance came from the model's flexibility relative to
    the dataset size, not from a structure-activity relationship.

    This is the check that catches the classic QSAR failure -- a few dozen
    compounds described by thousands of descriptors, where something will
    always correlate. It is required evidence under OECD principle 4, and
    it is cheap, so there is no excuse for omitting it.

    Parameters
    ----------
    n_iterations : int, default 100
        Number of permutations. The smallest p-value obtainable is
        ``1 / (n_iterations + 1)``, so 100 iterations cannot report
        anything below 0.0099.
    random_state : int, optional
        Seed for the permutations.

    Attributes
    ----------
    real_score_ : float
        The model's score on the true labels.
    scrambled_scores_ : ndarray of shape (n_iterations,)
        Scores obtained on permuted labels.

    Examples
    --------
    A real relationship survives the test:

    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.validation import YScrambling
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 4))
    >>> y = X[:, 0] * 3.0 + rng.normal(scale=0.2, size=60)
    >>> result = YScrambling(n_iterations=50, random_state=0).run(Ridge(), X, y)
    >>> result["p_value"] < 0.05
    True

    Pure noise does not:

    >>> y_noise = rng.normal(size=60)
    >>> noise = YScrambling(n_iterations=50, random_state=0).run(Ridge(), X, y_noise)
    >>> noise["p_value"] > 0.05
    True

    The diagnostic value is in the gap between the two scores:

    >>> round(result["real_score"] - result["mean_scrambled_score"], 2) > 0.5
    True

    References
    ----------
    - Rucker, C., Rucker, G. & Meringer, M. (2007). "y-Randomization and
      Its Variants in QSPR/QSAR." J. Chem. Inf. Model., 47(6), 2345-2357.
      https://doi.org/10.1021/ci700157b
    - Tropsha, A., Gramatica, P. & Gombar, V. K. (2003). "The Importance of
      Being Earnest." QSAR Comb. Sci., 22(1), 69-77.
      https://doi.org/10.1002/qsar.200390007
    - OECD (2007). ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """

    real_score_: float
    scrambled_scores_: "npt.NDArray[np.float64]"

    def __init__(
        self, n_iterations: int = 100, random_state: Optional[int] = None
    ) -> None:
        self.n_iterations = n_iterations
        self.random_state = random_state

    def run(
        self,
        estimator: BaseEstimator,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
    ) -> Dict[str, Any]:
        """Fit on the real labels and on ``n_iterations`` permutations of them.

        Parameters
        ----------
        estimator : BaseEstimator
            Cloned before each fit, so the object passed is never modified.
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        dict
            ``real_score``, ``mean_scrambled_score``,
            ``std_scrambled_score``, ``max_scrambled_score``, ``p_value``
            (the fraction of permutations scoring at least as well as the
            real fit, with the conventional +1 correction) and
            ``n_iterations``.

        Raises
        ------
        ValueError
            If ``n_iterations`` is not positive.
        """
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1, got {self.n_iterations}."
            )
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()

        self.real_score_ = _fit_score(estimator, X_arr, y_arr)

        rng = np.random.default_rng(self.random_state)
        scores = np.empty(self.n_iterations, dtype=np.float64)
        for i in range(self.n_iterations):
            scores[i] = _fit_score(estimator, X_arr, rng.permutation(y_arr))
        self.scrambled_scores_ = scores

        # The +1 correction keeps the p-value from ever being exactly zero:
        # a permutation test cannot distinguish "very unlikely" from
        # "impossible", and reporting 0 would claim more than was measured.
        n_better = int(np.sum(scores >= self.real_score_))
        p_value = (n_better + 1) / (self.n_iterations + 1)

        return {
            "real_score": self.real_score_,
            "mean_scrambled_score": float(scores.mean()),
            "std_scrambled_score": float(scores.std()),
            "max_scrambled_score": float(scores.max()),
            "p_value": float(p_value),
            "n_iterations": self.n_iterations,
        }

    def plot(self, title: str = "y-scrambling") -> Any:
        """Histogram of scrambled scores with the real score marked.

        Parameters
        ----------
        title : str, default "y-scrambling"

        Returns
        -------
        plotly.graph_objects.Figure

        Raises
        ------
        ModelNotFittedError
            If :meth:`run` has not been called.
        """
        import plotly.graph_objects as go

        from qsarkit.base.exceptions import ModelNotFittedError

        if not hasattr(self, "scrambled_scores_"):
            raise ModelNotFittedError("Call run() before plot().")

        figure = go.Figure()
        figure.add_trace(
            go.Histogram(
                x=self.scrambled_scores_,
                name="scrambled",
                marker_color="#9AA5B1",
                nbinsx=min(30, max(5, self.n_iterations // 3)),
            )
        )
        figure.add_vline(
            x=self.real_score_,
            line={"color": "#B3261E", "width": 2},
            annotation_text=f"real model ({self.real_score_:.3f})",
        )
        figure.update_layout(
            title=title,
            xaxis_title="R² on (possibly permuted) labels",
            yaxis_title="permutations",
            showlegend=False,
        )
        return figure


class ExternalValidator:
    """Score a fitted model on a held-out set with QSAR-appropriate metrics.

    OECD principle 4's predictivity requirement. Wraps
    :func:`~qsarkit.metrics.qsar_regression_report` and the
    Golbraikh-Tropsha criteria so an external evaluation reports the same
    statistics every time, rather than whichever ones happened to look best.

    Parameters
    ----------
    q2 : float, optional
        A cross-validated Q² from the training set. Supply it so that
        Golbraikh-Tropsha criterion 1 can be evaluated; without it that
        criterion reports ``None`` rather than silently passing.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.validation import ExternalValidator
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 4))
    >>> y = X[:, 0] * 3.0 + rng.normal(scale=0.2, size=60)
    >>> model = Ridge().fit(X[:45], y[:45])
    >>> report = ExternalValidator(q2=0.9).validate(model, X[45:], y[45:], y[:45])
    >>> report["r2"] > 0.9
    True
    >>> report["golbraikh_tropsha"]["passed"]
    True

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" J. Mol. Graph.
      Model., 20(4), 269-276.
      https://doi.org/10.1016/S1093-3263(01)00123-1
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
      Definition of the Q2 Parameter for QSAR Validation." J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """

    def __init__(self, q2: Optional[float] = None) -> None:
        self.q2 = q2

    def validate(
        self,
        estimator: BaseEstimator,
        X_test: npt.ArrayLike,
        y_test: npt.ArrayLike,
        y_train: Optional[npt.ArrayLike] = None,
    ) -> Dict[str, Any]:
        """Evaluate a fitted model on the test set.

        Parameters
        ----------
        estimator : BaseEstimator
            An already-fitted model.
        X_test, y_test : array-like
            The held-out set.
        y_train : array-like, optional
            Training activities. Supplied, Q²F1 and Q²F2 are computed
            against the training mean, which is what makes them comparable
            across differently-centred test sets.

        Returns
        -------
        dict
            The regression report, plus ``q2_f1`` when ``y_train`` is given
            and ``golbraikh_tropsha``.
        """
        from qsarkit.metrics import (
            golbraikh_tropsha_criteria,
            q2_f1,
            qsar_regression_report,
        )

        y_true = np.asarray(y_test, dtype=np.float64).ravel()
        y_pred = np.asarray(estimator.predict(X_test), dtype=np.float64).ravel()

        report: Dict[str, Any] = dict(qsar_regression_report(y_true, y_pred))
        if y_train is not None:
            report["q2_f1"] = float(q2_f1(y_true, y_pred, y_train))
        report["golbraikh_tropsha"] = golbraikh_tropsha_criteria(
            y_true, y_pred, q2=self.q2
        )
        report["n_test"] = int(len(y_true))
        return report


class BootstrapValidator:
    """Bootstrap the training set to estimate how stable a score is.

    A single cross-validated Q² is one number with no error bar. Resampling
    the training set with replacement and refitting gives the spread, which
    is what tells you whether a 0.02 difference between two models means
    anything on this much data -- usually it does not.

    Parameters
    ----------
    n_iterations : int, default 100
        Number of bootstrap resamples.
    random_state : int, optional
        Seed.

    Attributes
    ----------
    scores_ : ndarray of shape (n_iterations,)
        Out-of-bag score from each resample.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import Ridge
    >>> from qsarkit.validation import BootstrapValidator
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 4))
    >>> y = X[:, 0] * 3.0 + rng.normal(scale=0.2, size=60)
    >>> result = BootstrapValidator(n_iterations=25, random_state=0).run(Ridge(), X, y)
    >>> result["mean_score"] > 0.9
    True
    >>> result["ci_lower"] <= result["mean_score"] <= result["ci_upper"]
    True

    References
    ----------
    - Efron, B. & Tibshirani, R. J. (1993). "An Introduction to the
      Bootstrap." Chapman & Hall. https://doi.org/10.1201/9780429246593
    - Wehrens, R., Putter, H. & Buydens, L. M. C. (2000). "The Bootstrap:
      A Tutorial." Chemom. Intell. Lab. Syst., 54(1), 35-52.
      https://doi.org/10.1016/S0169-7439(00)00102-7
    """

    scores_: "npt.NDArray[np.float64]"

    def __init__(
        self, n_iterations: int = 100, random_state: Optional[int] = None
    ) -> None:
        self.n_iterations = n_iterations
        self.random_state = random_state

    def run(
        self,
        estimator: BaseEstimator,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """Refit on bootstrap resamples and score on the out-of-bag remainder.

        Parameters
        ----------
        estimator : BaseEstimator
            Cloned before each fit.
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        confidence : float, default 0.95
            Width of the reported percentile interval.

        Returns
        -------
        dict
            ``mean_score``, ``std_score``, ``ci_lower``, ``ci_upper``,
            ``n_iterations`` and ``n_effective`` (resamples that produced a
            usable out-of-bag set).

        Raises
        ------
        ValueError
            If ``n_iterations`` is not positive, ``confidence`` is not in
            (0, 1), or no resample left any out-of-bag samples.
        """
        from qsarkit.metrics import r2_score

        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1, got {self.n_iterations}."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence}."
            )

        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).ravel()
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)

        scores: List[float] = []
        for _ in range(self.n_iterations):
            train_idx = rng.integers(0, n, size=n)
            # Out-of-bag: about 36.8% of the data is left out of any given
            # resample, and scoring there rather than in-bag is what makes
            # this an estimate of generalization instead of of fit.
            oob = np.setdiff1d(np.arange(n), train_idx, assume_unique=False)
            if oob.size < 2:
                continue
            model = clone(estimator)
            model.fit(X_arr[train_idx], y_arr[train_idx])
            scores.append(float(r2_score(y_arr[oob], model.predict(X_arr[oob]))))

        if not scores:
            raise ValueError(
                "No bootstrap resample left usable out-of-bag samples; the "
                "dataset is too small for this validation."
            )

        self.scores_ = np.asarray(scores, dtype=np.float64)
        alpha = (1.0 - confidence) / 2.0
        return {
            "mean_score": float(self.scores_.mean()),
            "std_score": float(self.scores_.std()),
            "ci_lower": float(np.quantile(self.scores_, alpha)),
            "ci_upper": float(np.quantile(self.scores_, 1.0 - alpha)),
            "confidence": confidence,
            "n_iterations": self.n_iterations,
            "n_effective": len(scores),
        }
