r"""Choosing the decision threshold for a QSAR classifier.

``predict()`` cuts at 0.5. That default is almost never the right choice
for QSAR classification, for two reasons.

The first is class imbalance. On a screening set that is 2% active, a
model can reach 98% accuracy by calling everything inactive, and 0.5 will
push it most of the way there. The threshold that maximizes a
*balanced* criterion sits far lower.

The second is asymmetric cost. Missing an active and chasing a false
positive are not equally expensive, and the ratio differs between a
lead-finding campaign and a toxicity screen. That ratio is a property of
your project, not of the model, so it belongs in the threshold rather than
hidden inside the metric.

References
----------
- Youden, W. J. (1950). "Index for Rating Diagnostic Tests." Cancer, 3(1),
  32-35. https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
- Matthews, B. W. (1975). "Comparison of the Predicted and Observed
  Secondary Structure of T4 Phage Lysozyme." Biochim. Biophys. Acta,
  405(2), 442-451. https://doi.org/10.1016/0005-2795(75)90109-9
- Chicco, D. & Jurman, G. (2020). "The Advantages of the Matthews
  Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary
  Classification Evaluation." BMC Genomics, 21, 6.
  https://doi.org/10.1186/s12864-019-6413-7
- Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is More
  Informative than the ROC Plot When Evaluating Binary Classifiers on
  Imbalanced Datasets." PLoS ONE, 10(3), e0118432.
  https://doi.org/10.1371/journal.pone.0118432
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "threshold_sweep",
    "optimal_threshold",
    "threshold_report",
]

_Criterion = Literal[
    "youden", "f1", "mcc", "balanced_accuracy", "cost", "precision", "recall"
]


def _check_binary(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    pos_label: Optional[Any] = None,
) -> Tuple["npt.NDArray[np.int_]", "npt.NDArray[np.float64]"]:
    """Validate a binary label / score pair, dropping non-finite scores.

    Parameters
    ----------
    y_true, y_score : array-like
    pos_label : optional
        Which label counts as positive. Required for non-numeric labels:
        picking one by sort order would silently make ``"inactive"`` the
        positive class in a set labelled ``{"active", "inactive"}``, which
        inverts every metric without any error.

    Returns
    -------
    (ndarray of int, ndarray of float)
        Labels mapped to 0/1, and the matching finite scores.
    """
    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError(
            f"y_true has {labels.size} entries but y_score has {scores.size}."
        )
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if labels.size == 0:
        raise ValueError("No finite scores to threshold.")

    classes = np.unique(labels)
    if classes.size != 2:
        raise ValueError(
            f"Threshold selection needs exactly two classes, got "
            f"{classes.size} ({classes[:5]}). For a multiclass model, "
            "choose a threshold per class one-vs-rest."
        )

    if pos_label is not None:
        if pos_label not in classes:
            raise ValueError(
                f"pos_label={pos_label!r} is not one of the labels present "
                f"({[c.item() if hasattr(c, 'item') else c for c in classes]})."
            )
        positive = pos_label
    elif np.issubdtype(labels.dtype, np.number) or labels.dtype == bool:
        # For numeric or boolean labels the larger value is the positive
        # class by universal convention (1 over 0, True over False).
        positive = classes[-1]
    else:
        raise ValueError(
            f"Labels {[str(c) for c in classes]} are not numeric, so which one is "
            "'positive' cannot be inferred -- taking the later one "
            "alphabetically would make 'inactive' the positive class in an "
            "{'active', 'inactive'} set. Pass pos_label explicitly, or "
            "encode the labels as 0/1."
        )
    return (labels == positive).astype(int), scores


def threshold_sweep(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    n_thresholds: Optional[int] = None,
    pos_label: Optional[Any] = None,
) -> Dict[str, "npt.NDArray[np.float64]"]:
    """Every operating point of a binary classifier, as arrays.

    Evaluates the confusion matrix at each candidate threshold, so a
    criterion can be maximized or a curve plotted without refitting.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels. Numeric or boolean labels need no further
        specification -- the larger value is the positive class.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities. Only their order matters.
    n_thresholds : int, optional
        Evaluate this many evenly-spaced quantiles of the score instead of
        every distinct value. Use it on large sets, where the default
        gives one column per unique score.
    pos_label : optional
        Which label is the positive class. Required for string labels.

    Returns
    -------
    dict
        ``thresholds`` and, aligned with it, ``tp``, ``fp``, ``tn``,
        ``fn``, ``sensitivity`` (recall, TPR), ``specificity``,
        ``precision``, ``f1``, ``mcc``, ``balanced_accuracy``,
        ``youden_j`` and ``accuracy``.

    Raises
    ------
    ValueError
        If the inputs mismatch, or do not contain exactly two classes.

    Notes
    -----
    A prediction is positive when ``score >= threshold``, so the sweep
    includes a threshold above every score (predict nothing) to make the
    degenerate end of the curve explicit rather than absent.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import threshold_sweep
    >>> y = np.array([0, 0, 1, 1])
    >>> scores = np.array([0.1, 0.4, 0.6, 0.9])
    >>> sweep = threshold_sweep(y, scores)
    >>> best = int(np.argmax(sweep["youden_j"]))
    >>> float(sweep["thresholds"][best]), float(sweep["youden_j"][best])
    (0.6, 1.0)

    Sensitivity falls and specificity rises as the threshold increases:

    >>> bool(np.all(np.diff(sweep["sensitivity"]) <= 0))
    True
    >>> bool(np.all(np.diff(sweep["specificity"]) >= 0))
    True

    References
    ----------
    - Fawcett, T. (2006). "An Introduction to ROC Analysis." Pattern
      Recognit. Lett., 27(8), 861-874.
      https://doi.org/10.1016/j.patrec.2005.10.010
    """
    labels, scores = _check_binary(y_true, y_score, pos_label=pos_label)

    if n_thresholds is not None:
        if n_thresholds < 2:
            raise ValueError(
                f"n_thresholds must be at least 2, got {n_thresholds}."
            )
        candidates = np.unique(
            np.quantile(scores, np.linspace(0.0, 1.0, n_thresholds))
        )
    else:
        candidates = np.unique(scores)
    # One threshold above the maximum, so "predict nothing" is on the curve.
    thresholds = np.concatenate([candidates, [np.nextafter(candidates[-1], np.inf)]])

    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:  # pragma: no cover - guarded above
        raise ValueError("Both classes must be present.")

    # Vectorized over thresholds: predicted[i, j] is sample j at threshold i.
    predicted = scores[None, :] >= thresholds[:, None]
    tp = (predicted & (labels == 1)[None, :]).sum(axis=1).astype(np.float64)
    fp = (predicted & (labels == 0)[None, :]).sum(axis=1).astype(np.float64)
    fn = positives - tp
    tn = negatives - fp

    with np.errstate(divide="ignore", invalid="ignore"):
        sensitivity = np.divide(tp, positives)
        specificity = np.divide(tn, negatives)
        precision = np.where(tp + fp > 0, np.divide(tp, tp + fp), 0.0)
        f1 = np.where(
            (2 * tp + fp + fn) > 0, np.divide(2 * tp, 2 * tp + fp + fn), 0.0
        )
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = np.where(
            denominator > 0,
            np.divide(tp * tn - fp * fn, np.where(denominator > 0, denominator, 1.0)),
            0.0,
        )

    return {
        "thresholds": thresholds,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "mcc": mcc,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "youden_j": sensitivity + specificity - 1.0,
        "accuracy": (tp + tn) / float(labels.size),
    }


def optimal_threshold(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    criterion: _Criterion = "youden",
    cost_fn: float = 1.0,
    cost_fp: float = 1.0,
    min_precision: Optional[float] = None,
    min_recall: Optional[float] = None,
    n_thresholds: Optional[int] = None,
    pos_label: Optional[Any] = None,
) -> Dict[str, Any]:
    r"""Choose the decision threshold that best serves a stated objective.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities.
    criterion : str, default "youden"
        What "best" means:

        ``"youden"``
            Maximize Youden's :math:`J = \mathrm{sensitivity} +
            \mathrm{specificity} - 1`. Treats both error types as equally
            costly and both classes as equally important; the usual
            default when you have no cost information.
        ``"mcc"``
            Maximize the Matthews correlation coefficient. The most
            informative single number on imbalanced data, because it uses
            all four cells of the confusion matrix.
        ``"f1"``
            Maximize F1. Ignores true negatives, so it suits screening
            where the inactive majority is uninteresting.
        ``"balanced_accuracy"``
            Maximize the mean of sensitivity and specificity.
        ``"cost"``
            Minimize ``cost_fn * FN + cost_fp * FP``. The honest choice
            when the two errors have different consequences.
        ``"precision"`` / ``"recall"``
            Maximize the other of the pair subject to ``min_precision`` or
            ``min_recall``. For "find me 200 compounds to test, as pure as
            possible" and its mirror image.
    cost_fn, cost_fp : float, default 1.0
        Relative cost of a false negative and a false positive. Used by
        ``criterion="cost"``.
    min_precision : float, optional
        Required with ``criterion="precision"``: the floor precision must
        clear, among which recall is maximized.
    min_recall : float, optional
        Required with ``criterion="recall"``.
    n_thresholds : int, optional
        Passed to :func:`threshold_sweep`.
    pos_label : optional
        Which label is the positive class. Required for string labels.

    Returns
    -------
    dict
        ``threshold``, the ``criterion`` used, its ``score``, and the
        confusion matrix and derived rates at that threshold.

    Raises
    ------
    ValueError
        If ``criterion`` is unknown, a required constraint is missing, or
        no threshold satisfies the constraint.

    Notes
    -----
    Select the threshold on validation data, never on the test set.
    A threshold tuned on the same data it is scored on is a fitted
    parameter, and the resulting performance is optimistic in exactly the
    way an untuned 0.5 cut is not.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import optimal_threshold
    >>> y = np.array([0, 0, 0, 0, 1, 1])
    >>> scores = np.array([0.1, 0.2, 0.3, 0.55, 0.6, 0.8])
    >>> best = optimal_threshold(y, scores, criterion="youden")
    >>> float(best["threshold"]), round(best["score"], 3)
    (0.6, 1.0)

    On an imbalanced set the best threshold is nowhere near 0.5:

    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(1000, dtype=int); y[:30] = 1
    >>> scores = rng.beta(2, 8, size=1000) + y * 0.25
    >>> chosen = optimal_threshold(y, scores, criterion="mcc")
    >>> bool(chosen["threshold"] < 0.5)
    True

    Asymmetric costs move it. Making a missed active ten times as
    expensive as a false positive lowers the bar:

    >>> cheap = optimal_threshold(y, scores, criterion="cost",
    ...                           cost_fn=1.0, cost_fp=1.0)
    >>> costly = optimal_threshold(y, scores, criterion="cost",
    ...                            cost_fn=10.0, cost_fp=1.0)
    >>> bool(costly["threshold"] <= cheap["threshold"])
    True

    And a hard constraint is respected rather than traded away:

    >>> pure = optimal_threshold(y, scores, criterion="precision",
    ...                          min_precision=0.3)
    >>> bool(pure["precision"] >= 0.3)
    True

    An unreachable constraint is reported, with the best actually
    available, rather than quietly relaxed:

    >>> optimal_threshold(y, scores, criterion="precision", min_precision=0.9)
    Traceback (most recent call last):
        ...
    ValueError: No threshold reaches precision 0.9. The best available is ...

    References
    ----------
    - Youden, W. J. (1950). Cancer, 3(1), 32-35.
      https://doi.org/10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
    - Chicco, D. & Jurman, G. (2020). BMC Genomics, 21, 6.
      https://doi.org/10.1186/s12864-019-6413-7
    - Elkan, C. (2001). "The Foundations of Cost-Sensitive Learning."
      IJCAI 2001, 973-978.
    """
    sweep = threshold_sweep(
        y_true, y_score, n_thresholds=n_thresholds, pos_label=pos_label
    )

    direct = {
        "youden": "youden_j",
        "f1": "f1",
        "mcc": "mcc",
        "balanced_accuracy": "balanced_accuracy",
    }
    if criterion in direct:
        values = sweep[direct[criterion]]
        # Ties go to the lower threshold, which keeps more actives.
        best = int(np.argmax(values))
        score = float(values[best])
    elif criterion == "cost":
        if cost_fn < 0 or cost_fp < 0:
            raise ValueError(
                f"Costs must be non-negative, got cost_fn={cost_fn}, "
                f"cost_fp={cost_fp}."
            )
        total = cost_fn * sweep["fn"] + cost_fp * sweep["fp"]
        best = int(np.argmin(total))
        score = float(total[best])
    elif criterion == "precision":
        if min_precision is None:
            raise ValueError(
                "criterion='precision' needs min_precision: it maximizes "
                "recall subject to precision staying above that floor."
            )
        # Precision is only meaningful where something was predicted positive.
        eligible = (sweep["precision"] >= min_precision) & (sweep["tp"] + sweep["fp"] > 0)
        if not eligible.any():
            raise ValueError(
                f"No threshold reaches precision {min_precision}. The best "
                f"available is {float(sweep['precision'].max()):.3f}."
            )
        masked = np.where(eligible, sweep["sensitivity"], -np.inf)
        best = int(np.argmax(masked))
        score = float(sweep["sensitivity"][best])
    elif criterion == "recall":
        if min_recall is None:
            raise ValueError(
                "criterion='recall' needs min_recall: it maximizes precision "
                "subject to recall staying above that floor."
            )
        eligible = sweep["sensitivity"] >= min_recall
        if not eligible.any():
            raise ValueError(
                f"No threshold reaches recall {min_recall}. The best "
                f"available is {float(sweep['sensitivity'].max()):.3f}."
            )
        masked = np.where(eligible, sweep["precision"], -np.inf)
        best = int(np.argmax(masked))
        score = float(sweep["precision"][best])
    else:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Choose from 'youden', 'mcc', "
            "'f1', 'balanced_accuracy', 'cost', 'precision' or 'recall'."
        )

    return {
        "threshold": float(sweep["thresholds"][best]),
        "criterion": criterion,
        "score": score,
        "tp": int(sweep["tp"][best]),
        "fp": int(sweep["fp"][best]),
        "tn": int(sweep["tn"][best]),
        "fn": int(sweep["fn"][best]),
        "sensitivity": float(sweep["sensitivity"][best]),
        "specificity": float(sweep["specificity"][best]),
        "precision": float(sweep["precision"][best]),
        "f1": float(sweep["f1"][best]),
        "mcc": float(sweep["mcc"][best]),
        "balanced_accuracy": float(sweep["balanced_accuracy"][best]),
        "accuracy": float(sweep["accuracy"][best]),
    }


def threshold_report(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    n_thresholds: Optional[int] = None,
    pos_label: Optional[Any] = None,
) -> Dict[str, Any]:
    """Compare what each criterion would choose, against the 0.5 default.

    The most useful output when you do not yet know which criterion you
    want: it shows how much the choice actually matters on your data, and
    what the untuned 0.5 cut is costing.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities.
    n_thresholds : int, optional
        Passed to :func:`threshold_sweep`.
    pos_label : optional
        Which label is the positive class. Required for string labels.

    Returns
    -------
    dict
        One entry per criterion (``youden``, ``mcc``, ``f1``,
        ``balanced_accuracy``), plus ``default_0.5`` evaluated at the
        conventional cut, ``base_rate``, and ``roc_auc``/``pr_auc`` for
        the threshold-free picture.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import threshold_report
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(600, dtype=int); y[:30] = 1
    >>> scores = rng.beta(2, 8, size=600) + y * 0.3
    >>> report = threshold_report(y, scores)
    >>> sorted(k for k in report if isinstance(report[k], dict))
    ['balanced_accuracy', 'default_0.5', 'f1', 'mcc', 'youden']
    >>> report["youden"]["mcc"] > report["default_0.5"]["mcc"]
    True

    The default cut can miss nearly every active on imbalanced data:

    >>> report["default_0.5"]["sensitivity"] < report["youden"]["sensitivity"]
    True

    References
    ----------
    - Saito, T. & Rehmsmeier, M. (2015). PLoS ONE, 10(3), e0118432.
      https://doi.org/10.1371/journal.pone.0118432
    """
    from qsarkit.metrics._classification import pr_auc, roc_auc

    labels, scores = _check_binary(y_true, y_score, pos_label=pos_label)
    report: Dict[str, Any] = {
        "base_rate": float(labels.mean()),
        "n_samples": int(labels.size),
        "roc_auc": float(roc_auc(labels, scores)),
        "pr_auc": float(pr_auc(labels, scores)),
    }
    for criterion in ("youden", "mcc", "f1", "balanced_accuracy"):
        report[criterion] = optimal_threshold(
            labels, scores, criterion=criterion, n_thresholds=n_thresholds
        )

    sweep = threshold_sweep(labels, scores, n_thresholds=n_thresholds)
    at_half = int(np.searchsorted(sweep["thresholds"], 0.5, side="left"))
    at_half = min(at_half, sweep["thresholds"].size - 1)
    report["default_0.5"] = {
        "threshold": 0.5,
        "sensitivity": float(sweep["sensitivity"][at_half]),
        "specificity": float(sweep["specificity"][at_half]),
        "precision": float(sweep["precision"][at_half]),
        "f1": float(sweep["f1"][at_half]),
        "mcc": float(sweep["mcc"][at_half]),
        "balanced_accuracy": float(sweep["balanced_accuracy"][at_half]),
        "accuracy": float(sweep["accuracy"][at_half]),
    }
    return report
