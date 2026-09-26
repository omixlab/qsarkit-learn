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

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, clone

from qsarkit.validation._scoring import (
    Scorer,
    Scoring,
    metric_names,
    resolve_scoring,
    score_all,
    score_estimator,
    unwrap,
)

__all__ = ["YScrambling", "ExternalValidator", "BootstrapValidator"]


def _fit_and_score(
    estimator: BaseEstimator,
    X: "npt.NDArray[np.float64]",
    y: "npt.NDArray[Any]",
    scorers: Sequence[Scorer],
) -> "npt.NDArray[np.float64]":
    """Fit a fresh clone and score it on the same data it was fit on."""
    model = clone(estimator)
    model.fit(X, y)
    return score_estimator(scorers, model, X, y)


def _cross_val_score(
    estimator: BaseEstimator,
    X: "npt.NDArray[np.float64]",
    y: "npt.NDArray[Any]",
    scorers: Sequence[Scorer],
    cv: int,
    random_state: Optional[int],
    stratify: bool,
) -> "npt.NDArray[np.float64]":
    """Score out of fold, pooling the held-out predictions before scoring.

    Pooled rather than averaged per fold because a ranking metric is not a
    mean of per-fold rankings: with a 3% positive rate a fold can contain no
    positives at all, where ROC-AUC is undefined. Pooling the out-of-fold
    predictions and scoring once sidesteps that and matches how a
    cross-validated Q^2 is defined.
    """
    from sklearn.model_selection import KFold, StratifiedKFold

    needs_proba = any(s.needs_proba for s in scorers)
    needs_pred = any(not s.needs_proba for s in scorers)

    splitter: Any
    if stratify:
        splitter = StratifiedKFold(
            n_splits=cv, shuffle=True, random_state=random_state
        )
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    n = len(y)
    pooled_pred = np.empty(n, dtype=np.float64) if needs_pred else None
    pooled_score = np.empty(n, dtype=np.float64) if needs_proba else None

    for train_idx, test_idx in splitter.split(X, y):
        model = clone(estimator)
        model.fit(X[train_idx], y[train_idx])
        if pooled_pred is not None:
            pooled_pred[test_idx] = np.asarray(
                model.predict(X[test_idx]), dtype=np.float64
            )
        if pooled_score is not None:
            from qsarkit.validation._scoring import positive_class_scores

            pooled_score[test_idx] = positive_class_scores(model, X[test_idx])

    return score_all(scorers, y, pooled_pred, pooled_score)


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
    scoring : str, callable, Scorer, or iterable of those, optional
        The metric to argue in. Defaults to :math:`R^2`. Name one of
        :func:`~qsarkit.validation.available_metrics`, pass a
        ``(y_true, y_pred)`` callable, or use
        :func:`~qsarkit.validation.make_scorer` for a metric that needs
        probabilities or is a loss. Pass several and every score in the
        result becomes an array in the order given.
    cv : int, optional
        Score out of fold over this many folds instead of on the training
        data. **Strongly recommended for any flexible model, and required
        for a ranking metric to mean anything**: a random forest reaches an
        in-sample ROC-AUC near 1.0 on permuted labels just as it does on real
        ones, so the in-sample comparison shows no gap and the test reports
        nothing. The default is ``None`` -- the apparent, in-sample fit --
        because that is what earlier releases computed.
    stratify : bool, default False
        Use stratified folds when ``cv`` is set. Needed on an imbalanced
        classification endpoint, where an unstratified fold can contain no
        positives at all.

    Attributes
    ----------
    real_score_ : float or ndarray
        The model's score on the true labels: a float for one metric, an
        array in the given order for several.
    scrambled_scores_ : ndarray
        Shape ``(n_iterations,)`` for one metric, ``(n_iterations,
        n_metrics)`` for several.

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

    real_score_: Any
    scrambled_scores_: "npt.NDArray[np.float64]"

    def __init__(
        self,
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        scoring: Scoring = None,
        cv: Optional[int] = None,
        stratify: bool = False,
    ) -> None:
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.scoring = scoring
        self.cv = cv
        self.stratify = stratify

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
            ``std_scrambled_score``, ``max_scrambled_score``,
            ``best_scrambled_score`` (the largest for a metric where more is
            better, the smallest for a loss), ``p_value`` (the fraction of
            permutations scoring at least as well as the real fit, with the
            conventional +1 correction), ``n_iterations``, ``metric`` (the
            name, or the tuple of names) and ``scored_out_of_fold``.

            Every score is a float when one metric was requested and an
            ndarray in the requested order when several were.

        Raises
        ------
        ValueError
            If ``n_iterations`` is not positive.
        """
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1, got {self.n_iterations}."
            )
        if self.cv is not None and self.cv < 2:
            raise ValueError(f"cv must be at least 2, got {self.cv}.")

        scorers, single = resolve_scoring(self.scoring)
        X_arr = np.asarray(X, dtype=np.float64)
        # The label dtype is left alone: coercing to float would turn class
        # labels into floats, and a metric such as MCC then scores something
        # other than what the caller passed.
        y_arr = np.asarray(y).ravel()

        def evaluate(labels: "npt.NDArray[Any]") -> "npt.NDArray[np.float64]":
            if self.cv is None:
                return _fit_and_score(estimator, X_arr, labels, scorers)
            return _cross_val_score(
                estimator,
                X_arr,
                labels,
                scorers,
                self.cv,
                self.random_state,
                self.stratify,
            )

        real = evaluate(y_arr)

        rng = np.random.default_rng(self.random_state)
        scrambled = np.empty((self.n_iterations, len(scorers)), dtype=np.float64)
        for i in range(self.n_iterations):
            scrambled[i] = evaluate(rng.permutation(y_arr))

        # The +1 correction keeps the p-value from ever being exactly zero:
        # a permutation test cannot distinguish "very unlikely" from
        # "impossible", and reporting 0 would claim more than was measured.
        p_values = np.empty(len(scorers), dtype=np.float64)
        best = np.empty(len(scorers), dtype=np.float64)
        for j, scorer in enumerate(scorers):
            column = scrambled[:, j]
            # Direction matters: for RMSE a scrambled model does "at least as
            # well" by scoring *lower*, so comparing with >= would invert the
            # test and report a loss metric's p-value backwards.
            at_least_as_good = sum(
                scorer.is_at_least_as_good_as(value, real[j]) for value in column
            )
            p_values[j] = (at_least_as_good + 1) / (self.n_iterations + 1)
            best[j] = column.max() if scorer.greater_is_better else column.min()

        self.real_score_ = unwrap(real, single)
        self.scrambled_scores_ = scrambled[:, 0] if single else scrambled

        return {
            "real_score": unwrap(real, single),
            "mean_scrambled_score": unwrap(scrambled.mean(axis=0), single),
            "std_scrambled_score": unwrap(scrambled.std(axis=0), single),
            "max_scrambled_score": unwrap(scrambled.max(axis=0), single),
            # For a loss metric the *best* scrambled score is the smallest,
            # which `max_scrambled_score` (kept for compatibility) does not
            # give.
            "best_scrambled_score": unwrap(best, single),
            "p_value": unwrap(p_values, single),
            "n_iterations": self.n_iterations,
            "metric": metric_names(scorers, single),
            "scored_out_of_fold": self.cv is not None,
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
    scoring : str, callable, Scorer, or iterable of those, optional
        Report these metrics instead of the regression report. The default
        (``None``) keeps the QSAR regression report and the
        Golbraikh-Tropsha criteria, which is what a regression submission
        needs; naming metrics is how a classification endpoint is validated,
        and then ``score`` and ``metric`` replace the report.

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

    def __init__(self, q2: Optional[float] = None, scoring: Scoring = None) -> None:
        self.q2 = q2
        self.scoring = scoring

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
        # With an explicit metric the caller has said what to report, and the
        # regression report would be meaningless anyway on a classification
        # endpoint -- R^2 of 0/1 labels answers no question anyone asked.
        if self.scoring is not None:
            scorers, single = resolve_scoring(self.scoring)
            y_true_any = np.asarray(y_test).ravel()
            return {
                "score": unwrap(
                    score_estimator(scorers, estimator, np.asarray(X_test), y_true_any),
                    single,
                ),
                "metric": metric_names(scorers, single),
                "n_test": int(len(y_true_any)),
            }

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
        self,
        n_iterations: int = 100,
        random_state: Optional[int] = None,
        scoring: Scoring = None,
    ) -> None:
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.scoring = scoring

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
            ``confidence``, ``n_iterations``, ``n_effective`` (resamples that
            produced a usable out-of-bag set) and ``metric``.

            Every score is a float when one metric was requested and an
            ndarray in the requested order when several were.

        Raises
        ------
        ValueError
            If ``n_iterations`` is not positive, ``confidence`` is not in
            (0, 1), or no resample left any out-of-bag samples.
        """
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1, got {self.n_iterations}."
            )
        if not 0.0 < confidence < 1.0:
            raise ValueError(
                f"confidence must be in (0, 1), got {confidence}."
            )

        scorers, single = resolve_scoring(self.scoring)
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y).ravel()
        n = len(y_arr)
        rng = np.random.default_rng(self.random_state)

        collected: List["npt.NDArray[np.float64]"] = []
        for _ in range(self.n_iterations):
            train_idx = rng.integers(0, n, size=n)
            # Out-of-bag: about 36.8% of the data is left out of any given
            # resample, and scoring there rather than in-bag is what makes
            # this an estimate of generalization instead of of fit.
            oob = np.setdiff1d(np.arange(n), train_idx, assume_unique=False)
            if oob.size < 2:
                continue
            # A ranking metric needs both classes present out of bag, and a
            # resample of an imbalanced endpoint can leave only one. Such a
            # resample is skipped rather than scored as if it were valid;
            # `n_effective` reports how many actually counted.
            if any(s.needs_proba for s in scorers) and len(np.unique(y_arr[oob])) < 2:
                continue
            model = clone(estimator)
            model.fit(X_arr[train_idx], y_arr[train_idx])
            collected.append(score_estimator(scorers, model, X_arr[oob], y_arr[oob]))

        if not collected:
            raise ValueError(
                "No bootstrap resample left usable out-of-bag samples; the "
                "dataset is too small for this validation."
            )

        stacked = np.vstack(collected)
        self.scores_ = stacked[:, 0] if single else stacked
        alpha = (1.0 - confidence) / 2.0
        return {
            "mean_score": unwrap(stacked.mean(axis=0), single),
            "std_score": unwrap(stacked.std(axis=0), single),
            "ci_lower": unwrap(np.quantile(stacked, alpha, axis=0), single),
            "ci_upper": unwrap(np.quantile(stacked, 1.0 - alpha, axis=0), single),
            "confidence": confidence,
            "n_iterations": self.n_iterations,
            "n_effective": len(collected),
            "metric": metric_names(scorers, single),
        }
