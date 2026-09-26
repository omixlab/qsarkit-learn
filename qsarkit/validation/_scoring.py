"""Metric selection for the validation methods.

Every class in :mod:`qsarkit.validation` answers "how well does this model
do", and until now each answered it with :math:`R^2`. That is the right
default for a regression QSAR, and wrong for everything else: a
y-randomization test on a toxicity classifier has to be argued in ROC-AUC or
average precision, and a regulator asking for RMSE is not asking for
:math:`R^2` reported alongside.

This module supplies the vocabulary. A metric may be named, passed as a
callable, or wrapped with :func:`make_scorer` when it needs probabilities
rather than hard predictions, and every validation method accepts either one
metric or several. With one it returns a float, as before; with several it
returns an array in the order the metrics were given.

Two details the implementation has to get right:

* **Some metrics need probabilities.** ROC-AUC and average precision rank
  samples, so they need ``predict_proba``; MCC and F1 need a decision. A
  scorer declares which through ``needs_proba``, and a validation method
  calls ``predict`` and ``predict_proba`` at most once each per fit, no
  matter how many metrics are requested.
* **Some metrics are losses.** "Did the scrambled model do at least as well
  as the real one" means *greater* for :math:`R^2` and *smaller* for RMSE, so
  a scorer also declares ``greater_is_better``. Without it a y-randomization
  p-value computed on RMSE would be exactly backwards.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

__all__ = ["Scorer", "make_scorer", "available_metrics"]


@dataclass(frozen=True)
class Scorer:
    """One metric, with what a validation method needs to know about it.

    Attributes
    ----------
    name : str
        How the metric is reported.
    func : callable
        ``func(y_true, y_pred_or_score) -> float``.
    needs_proba : bool
        Whether ``func`` wants ``predict_proba(X)[:, 1]`` rather than
        ``predict(X)``.
    greater_is_better : bool
        Whether a larger value is a better model. ``False`` for losses such
        as RMSE, which is what keeps a permutation p-value the right way
        round.
    """

    name: str
    func: Callable[..., float]
    needs_proba: bool = False
    greater_is_better: bool = True

    def __call__(
        self, y_true: "npt.NDArray[Any]", prediction: "npt.NDArray[Any]"
    ) -> float:
        return float(self.func(y_true, prediction))

    def is_at_least_as_good_as(self, candidate: float, reference: float) -> bool:
        """Whether ``candidate`` matches or beats ``reference`` for this metric."""
        if self.greater_is_better:
            return bool(candidate >= reference)
        return bool(candidate <= reference)


def make_scorer(
    func: Callable[..., float],
    *,
    needs_proba: bool = False,
    greater_is_better: bool = True,
    name: Optional[str] = None,
) -> Scorer:
    """Wrap a metric for use as ``scoring=``.

    Parameters
    ----------
    func : callable
        ``func(y_true, y_pred) -> float``, or ``func(y_true, y_score)`` when
        ``needs_proba`` is set.
    needs_proba : bool, default False
        Pass the positive-class probability instead of the predicted label.
    greater_is_better : bool, default True
        Set ``False`` for a loss, so that "at least as good" compares the
        right way round.
    name : str, optional
        Defaults to the function's ``__name__``.

    Returns
    -------
    Scorer

    Examples
    --------
    A metric a validation method can use directly:

    >>> from sklearn.metrics import average_precision_score
    >>> from qsarkit.validation import make_scorer
    >>> scorer = make_scorer(average_precision_score, needs_proba=True)
    >>> scorer.name
    'average_precision_score'

    A loss, declared as one:

    >>> from qsarkit.metrics import rmse
    >>> make_scorer(rmse, greater_is_better=False).greater_is_better
    False
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func).__name__}.")
    return Scorer(
        name=name if name is not None else str(getattr(func, "__name__", "custom")),
        func=func,
        needs_proba=needs_proba,
        greater_is_better=greater_is_better,
    )


def _registry() -> Dict[str, Scorer]:
    """Named metrics, resolved lazily so importing this module stays cheap."""
    from qsarkit import metrics as m

    def reg(
        name: str,
        func: Callable[..., float],
        needs_proba: bool = False,
        greater_is_better: bool = True,
    ) -> Tuple[str, Scorer]:
        return name, Scorer(name, func, needs_proba, greater_is_better)

    entries = [
        # Regression
        reg("r2", m.r2_score),
        reg("rmse", m.rmse, greater_is_better=False),
        reg("mse", m.mse, greater_is_better=False),
        reg("mae", m.mae, greater_is_better=False),
        reg("median_ae", m.median_ae, greater_is_better=False),
        reg("ccc", m.ccc),
        # Classification on a decision
        reg("accuracy", m.accuracy),
        reg("balanced_accuracy", m.balanced_accuracy),
        reg("f1", m.f1_score),
        reg("mcc", m.matthews_corrcoef),
        reg("precision", m.precision),
        reg("recall", m.recall),
        reg("sensitivity", m.sensitivity),
        reg("specificity", m.specificity),
        reg("cohen_kappa", m.cohen_kappa),
        # Classification on a ranking or a probability
        reg("roc_auc", m.roc_auc, needs_proba=True),
        reg("pr_auc", m.pr_auc, needs_proba=True),
        reg("brier", m.brier_score, needs_proba=True, greater_is_better=False),
    ]
    return dict(entries)


def available_metrics() -> List[str]:
    """The metric names ``scoring=`` accepts.

    Examples
    --------
    >>> from qsarkit.validation import available_metrics
    >>> names = available_metrics()
    >>> "r2" in names, "roc_auc" in names, "rmse" in names
    (True, True, True)
    """
    return sorted(_registry())


_ScoringItem = Union[str, Scorer, Callable[..., float]]
Scoring = Union[None, _ScoringItem, Iterable[_ScoringItem]]


def _as_scorer(item: _ScoringItem) -> Scorer:
    if isinstance(item, Scorer):
        return item
    if isinstance(item, str):
        registry = _registry()
        try:
            return registry[item]
        except KeyError:
            raise ValueError(
                f"Unknown metric {item!r}. Available: {sorted(registry)}. "
                "Pass a callable, or qsarkit.validation.make_scorer(...) for a "
                "metric that needs probabilities or is a loss."
            ) from None
    if callable(item):
        # A bare callable is assumed to take (y_true, y_pred) and to improve
        # as it grows. make_scorer() is how the other cases are declared.
        return make_scorer(item)
    raise TypeError(
        f"scoring entries must be a metric name, a callable or a Scorer; "
        f"got {type(item).__name__}."
    )


def resolve_scoring(
    scoring: Scoring, default: _ScoringItem = "r2"
) -> Tuple[List[Scorer], bool]:
    """Normalize ``scoring`` into scorers, and say whether it was a single one.

    Returns
    -------
    scorers : list of Scorer
    single : bool
        ``True`` when the caller asked for one metric, so results should be
        reported as scalars rather than arrays. A one-element *list* counts
        as several: ``scoring=["r2"]`` returns an array of length one, which
        is what keeps a caller's code from changing shape when it adds a
        second metric.
    """
    if scoring is None:
        return [_as_scorer(default)], True
    if isinstance(scoring, (str, Scorer)) or callable(scoring):
        return [_as_scorer(scoring)], True  # type: ignore[arg-type]
    try:
        items = list(scoring)  # type: ignore[arg-type]
    except TypeError:
        raise TypeError(
            f"scoring must be a metric name, a callable, a Scorer, or an "
            f"iterable of those; got {type(scoring).__name__}."
        ) from None
    if not items:
        raise ValueError("scoring is empty; pass at least one metric.")
    return [_as_scorer(item) for item in items], False


def predictions_for(
    estimator: Any,
    X: "npt.NDArray[Any]",
    scorers: Sequence[Scorer],
) -> Tuple[Optional["npt.NDArray[Any]"], Optional["npt.NDArray[Any]"]]:
    """Compute only the outputs the scorers actually need.

    ``predict`` and ``predict_proba`` are each called at most once, so asking
    for six metrics costs no more model evaluations than asking for one.
    """
    y_pred = None
    y_score = None
    if any(not s.needs_proba for s in scorers):
        y_pred = np.asarray(estimator.predict(X))
    if any(s.needs_proba for s in scorers):
        y_score = positive_class_scores(estimator, X)
    return y_pred, y_score


def positive_class_scores(estimator: Any, X: "npt.NDArray[Any]") -> "npt.NDArray[Any]":
    """The positive-class probability, or the decision function as a fallback."""
    if hasattr(estimator, "predict_proba"):
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        if proba.ndim == 2 and proba.shape[1] == 1:
            return proba[:, 0]
        if proba.ndim == 1:
            return proba
        raise ValueError(
            f"A probability-based metric needs a binary predict_proba, but "
            f"this estimator returned shape {proba.shape}. Score a "
            f"multi-class model one class at a time."
        )
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(X), dtype=np.float64).ravel()
    raise ValueError(
        f"{type(estimator).__name__} has neither predict_proba nor "
        "decision_function, so a probability-based metric cannot be computed. "
        "Choose a metric that scores predictions, such as 'mcc' or 'rmse'."
    )


def score_all(
    scorers: Sequence[Scorer],
    y_true: "npt.NDArray[Any]",
    y_pred: Optional["npt.NDArray[Any]"],
    y_score: Optional["npt.NDArray[Any]"],
) -> "npt.NDArray[np.float64]":
    """Apply every scorer to predictions that were computed once."""
    values = np.empty(len(scorers), dtype=np.float64)
    for i, scorer in enumerate(scorers):
        prediction = y_score if scorer.needs_proba else y_pred
        if prediction is None:  # pragma: no cover - predictions_for prevents this
            raise ValueError(f"no prediction available for metric {scorer.name!r}")
        values[i] = scorer(y_true, prediction)
    return values


def score_estimator(
    scorers: Sequence[Scorer],
    estimator: Any,
    X: "npt.NDArray[Any]",
    y_true: "npt.NDArray[Any]",
) -> "npt.NDArray[np.float64]":
    """Fit-free scoring of an already-fitted estimator."""
    y_pred, y_score = predictions_for(estimator, X, scorers)
    return score_all(scorers, y_true, y_pred, y_score)


def unwrap(values: "npt.NDArray[np.float64]", single: bool) -> Any:
    """A float for one metric, an array for several."""
    if single:
        return float(values[0])
    return values


def metric_names(scorers: Sequence[Scorer], single: bool) -> Any:
    """The reported metric name, or the tuple of names."""
    if single:
        return scorers[0].name
    return tuple(s.name for s in scorers)
