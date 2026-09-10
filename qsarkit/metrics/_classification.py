"""Classification metrics used to judge QSAR classifiers and virtual screens.

Besides the standard confusion-matrix statistics this module implements the
early-recognition metrics used in virtual screening (enrichment factor,
RIE and BEDROC), which reward a ranking that places actives at the very
top of the list rather than merely separating the classes on average.

References
----------
- Matthews, B. W. (1975). "Comparison of the Predicted and Observed
  Secondary Structure of T4 Phage Lysozyme." Biochim. Biophys. Acta,
  405(2), 442-451. https://doi.org/10.1016/0005-2795(75)90109-9
- Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales."
  Educ. Psychol. Meas., 20(1), 37-46.
  https://doi.org/10.1177/001316446002000104
- Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
  Methods: Good and Bad Metrics for the 'Early Recognition' Problem."
  J. Chem. Inf. Model., 47(2), 488-508.
  https://doi.org/10.1021/ci600426e
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "confusion_counts",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1_score",
    "matthews_corrcoef",
    "cohen_kappa",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "enrichment_factor",
    "robust_initial_enhancement",
    "bedroc",
]


def _check_binary_labels(
    y_true: ArrayLike, y_pred: ArrayLike
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Coerce a pair of binary label vectors to ``{0, 1}`` integer arrays."""
    yt = np.asarray(y_true).ravel()
    yp = np.asarray(y_pred).ravel()
    if yt.shape[0] != yp.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length, "
            f"got {yt.shape[0]} and {yp.shape[0]}."
        )
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    out = []
    for arr, name in ((yt, "y_true"), (yp, "y_pred")):
        ints = arr.astype(np.int64)
        if not np.array_equal(ints, arr.astype(np.float64)):
            raise ValueError(f"{name} must contain integer class labels.")
        extra = set(np.unique(ints)) - {0, 1}
        if extra:
            raise ValueError(
                f"{name} must be binary with labels in {{0, 1}}; got extra "
                f"labels {sorted(extra)}. Encode the active/positive class as 1."
            )
        out.append(ints)
    return out[0], out[1]


def _check_scores(
    y_true: ArrayLike, y_score: ArrayLike
) -> Tuple[NDArray[np.int_], NDArray[np.float64]]:
    """Coerce a binary label vector plus a continuous score vector."""
    yt = np.asarray(y_true).ravel().astype(np.int64)
    extra = set(np.unique(yt)) - {0, 1}
    if extra:
        raise ValueError(
            f"y_true must be binary with labels in {{0, 1}}; got {sorted(extra)}."
        )
    ys = np.asarray(y_score, dtype=np.float64).ravel()
    if yt.shape[0] != ys.shape[0]:
        raise ValueError(
            f"y_true and y_score must have the same length, "
            f"got {yt.shape[0]} and {ys.shape[0]}."
        )
    if yt.size == 0:
        raise ValueError("y_true is empty.")
    if not np.all(np.isfinite(ys)):
        raise ValueError("y_score contains NaN or infinite values.")
    return yt, ys


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> Dict[str, int]:
    """Return the binary confusion-matrix counts as a dictionary.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels in ``{0, 1}``; ``1``
        denotes the active/positive class.

    Returns
    -------
    dict
        Keys ``"tp"``, ``"tn"``, ``"fp"``, ``"fn"``.

    Examples
    --------
    >>> from qsarkit.metrics import confusion_counts
    >>> confusion_counts([1, 1, 0, 0], [1, 0, 0, 0]) == {
    ...     "tp": 1, "tn": 2, "fp": 0, "fn": 1}
    True

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
      JMLR, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
    """
    yt, yp = _check_binary_labels(y_true, y_pred)
    return {
        "tp": int(np.sum((yt == 1) & (yp == 1))),
        "tn": int(np.sum((yt == 0) & (yp == 0))),
        "fp": int(np.sum((yt == 0) & (yp == 1))),
        "fn": int(np.sum((yt == 1) & (yp == 0))),
    }


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of correctly classified samples.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        ``(TP + TN) / n``.

    Examples
    --------
    >>> from qsarkit.metrics import accuracy
    >>> accuracy([1, 1, 0, 0], [1, 0, 0, 0])
    0.75

    References
    ----------
    - Pedregosa et al. (2011). JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    yt, yp = _check_binary_labels(y_true, y_pred)
    return float(np.mean(yt == yp))


def sensitivity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """True-positive rate ``TP / (TP + FN)`` (recall of the active class).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        The sensitivity; ``0.0`` when there are no positives.

    Examples
    --------
    >>> from qsarkit.metrics import sensitivity
    >>> sensitivity([1, 1, 0, 0], [1, 0, 0, 0])
    0.5

    References
    ----------
    - Altman, D. G. & Bland, J. M. (1994). "Diagnostic Tests. 1:
      Sensitivity and Specificity." BMJ, 308(6943), 1552.
      https://doi.org/10.1136/bmj.308.6943.1552
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fn"]
    return 0.0 if denom == 0 else c["tp"] / denom


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """True-negative rate ``TN / (TN + FP)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        The specificity; ``0.0`` when there are no negatives.

    Examples
    --------
    >>> from qsarkit.metrics import specificity
    >>> specificity([1, 1, 0, 0], [1, 0, 0, 0])
    1.0

    References
    ----------
    - Altman, D. G. & Bland, J. M. (1994). BMJ, 308(6943), 1552.
      https://doi.org/10.1136/bmj.308.6943.1552
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["tn"] + c["fp"]
    return 0.0 if denom == 0 else c["tn"] / denom


def recall(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Alias of :func:`sensitivity`.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        ``TP / (TP + FN)``.

    Examples
    --------
    >>> from qsarkit.metrics import recall
    >>> recall([1, 1, 0, 0], [1, 0, 0, 0])
    0.5

    References
    ----------
    - Pedregosa et al. (2011). JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    return sensitivity(y_true, y_pred)


def precision(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Positive predictive value ``TP / (TP + FP)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        The precision; ``0.0`` when nothing is predicted positive.

    Examples
    --------
    >>> from qsarkit.metrics import precision
    >>> precision([1, 1, 0, 0], [1, 0, 0, 0])
    1.0

    References
    ----------
    - Pedregosa et al. (2011). JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    c = confusion_counts(y_true, y_pred)
    denom = c["tp"] + c["fp"]
    return 0.0 if denom == 0 else c["tp"] / denom


def f1_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Harmonic mean of precision and recall.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        ``2 * P * R / (P + R)``; ``0.0`` when both are zero.

    Examples
    --------
    >>> from qsarkit.metrics import f1_score
    >>> round(f1_score([1, 1, 0, 0], [1, 0, 0, 0]), 4)
    0.6667

    References
    ----------
    - van Rijsbergen, C. J. (1979). *Information Retrieval*, 2nd ed.
      Butterworths. https://www.dcs.gla.ac.uk/Keith/Preface.html
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 0.0 if (p + r) == 0 else float(2 * p * r / (p + r))


def balanced_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean of sensitivity and specificity.

    Preferred over plain accuracy for the strongly imbalanced datasets that
    are the norm in QSAR classification (e.g. toxicity endpoints with 5%
    actives), where a trivial majority classifier already scores high
    accuracy but only 0.5 balanced accuracy.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        ``(sensitivity + specificity) / 2``.

    Examples
    --------
    >>> from qsarkit.metrics import balanced_accuracy
    >>> balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0])
    0.75

    References
    ----------
    - Brodersen, K. H., Ong, C. S., Stephan, K. E. & Buhmann, J. M. (2010).
      "The Balanced Accuracy and Its Posterior Distribution." ICPR 2010,
      3121-3124. https://doi.org/10.1109/ICPR.2010.764
    """
    return float((sensitivity(y_true, y_pred) + specificity(y_true, y_pred)) / 2.0)


def matthews_corrcoef(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Matthews correlation coefficient (MCC).

    ``MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))``

    MCC is a correlation coefficient between observed and predicted binary
    classifications; it is high only when all four confusion-matrix
    quadrants are good, which is why it is the recommended single-number
    summary for imbalanced QSAR classification.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        The MCC in ``[-1, 1]``; ``0.0`` if any marginal is degenerate.

    Examples
    --------
    >>> from qsarkit.metrics import matthews_corrcoef
    >>> matthews_corrcoef([1, 1, 0, 0], [1, 1, 0, 0])
    1.0

    References
    ----------
    - Matthews, B. W. (1975). Biochim. Biophys. Acta, 405(2), 442-451.
      https://doi.org/10.1016/0005-2795(75)90109-9
    - Chicco, D. & Jurman, G. (2020). "The Advantages of the Matthews
      Correlation Coefficient (MCC) over F1 Score and Accuracy in Binary
      Classification Evaluation." BMC Genomics, 21, 6.
      https://doi.org/10.1186/s12864-019-6413-7
    """
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    num = float(tp * tn - fp * fn)
    denom = float(tp + fp) * float(tp + fn) * float(tn + fp) * float(tn + fn)
    return 0.0 if denom == 0.0 else float(num / np.sqrt(denom))


def cohen_kappa(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Cohen's kappa: agreement corrected for chance.

    ``kappa = (p_o - p_e) / (1 - p_e)`` where ``p_o`` is the observed
    agreement and ``p_e`` the agreement expected from the marginal label
    frequencies.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels.

    Returns
    -------
    float
        Cohen's kappa; ``1.0`` when observed agreement is perfect *and*
        chance agreement is degenerate (``p_e == 1``).

    Examples
    --------
    >>> from qsarkit.metrics import cohen_kappa
    >>> cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    1.0

    References
    ----------
    - Cohen, J. (1960). "A Coefficient of Agreement for Nominal Scales."
      Educ. Psychol. Meas., 20(1), 37-46.
      https://doi.org/10.1177/001316446002000104
    """
    yt, yp = _check_binary_labels(y_true, y_pred)
    n = yt.shape[0]
    p_o = float(np.mean(yt == yp))
    p_e = 0.0
    for label in (0, 1):
        p_e += (np.sum(yt == label) / n) * (np.sum(yp == label) / n)
    if p_e == 1.0:
        # p_e == 1 is only attainable when yt and yp are both degenerate
        # (all-0 or all-1) with matching composition, which forces p_o == 1.
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def roc_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the receiver-operating-characteristic curve.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,)
        Continuous scores (higher = more likely active), e.g.
        ``predict_proba(X)[:, 1]``.

    Returns
    -------
    float
        The ROC-AUC.

    Raises
    ------
    ValueError
        If only one class is present in ``y_true``.

    Examples
    --------
    >>> from qsarkit.metrics import roc_auc
    >>> roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    1.0

    References
    ----------
    - Hanley, J. A. & McNeil, B. J. (1982). "The Meaning and Use of the Area
      under a Receiver Operating Characteristic (ROC) Curve." Radiology,
      143(1), 29-36. https://doi.org/10.1148/radiology.143.1.7063747
    """
    from sklearn.metrics import roc_auc_score

    yt, ys = _check_scores(y_true, y_score)
    if len(np.unique(yt)) < 2:
        raise ValueError("ROC-AUC is undefined when y_true has a single class.")
    return float(roc_auc_score(yt, ys))


def pr_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Area under the precision-recall curve (average precision).

    More informative than ROC-AUC when actives are rare, because the
    precision axis is sensitive to the large number of true negatives that
    ROC-AUC dilutes away.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,)
        Continuous scores (higher = more likely active).

    Returns
    -------
    float
        The average precision.

    Raises
    ------
    ValueError
        If only one class is present in ``y_true``.

    Examples
    --------
    >>> from qsarkit.metrics import pr_auc
    >>> pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    1.0

    References
    ----------
    - Davis, J. & Goadrich, M. (2006). "The Relationship between
      Precision-Recall and ROC Curves." ICML 2006, 233-240.
      https://doi.org/10.1145/1143844.1143874
    """
    from sklearn.metrics import average_precision_score

    yt, ys = _check_scores(y_true, y_score)
    if len(np.unique(yt)) < 2:
        raise ValueError("PR-AUC is undefined when y_true has a single class.")
    return float(average_precision_score(yt, ys))


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Brier score: mean squared error of the predicted probabilities.

    Measures calibration as well as discrimination; lower is better.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class, in ``[0, 1]``.

    Returns
    -------
    float
        ``mean((y_prob - y_true) ** 2)``.

    Examples
    --------
    >>> from qsarkit.metrics import brier_score
    >>> brier_score([0, 1], [0.0, 1.0])
    0.0

    References
    ----------
    - Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of
      Probability." Mon. Weather Rev., 78(1), 1-3.
      https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
    """
    yt, yp = _check_scores(y_true, y_prob)
    return float(np.mean((yp - yt) ** 2))


def _active_ranks(
    yt: NDArray[np.int_], ys: NDArray[np.float64]
) -> NDArray[np.float64]:
    """1-based ranks of the actives after sorting scores in descending order.

    Ties are broken deterministically by original index, which matches the
    behaviour of the reference BEDROC implementations (RDKit, ``croc``).
    """
    order = np.argsort(-ys, kind="stable")
    ranks = np.empty(ys.shape[0], dtype=np.float64)
    ranks[order] = np.arange(1, ys.shape[0] + 1, dtype=np.float64)
    return ranks[yt == 1]


def enrichment_factor(
    y_true: ArrayLike, y_score: ArrayLike, fraction: float = 0.01
) -> float:
    """Enrichment factor at a given fraction of the ranked list.

    ``EF(chi) = (n_actives_in_top / n_top) / (n_actives_total / N)``

    An EF of 10 at 1% means the screen finds ten times as many actives in
    the top 1% as random selection would.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,)
        Continuous scores; the list is ranked in decreasing score order.
    fraction : float, default=0.01
        Fraction ``chi`` of the ranked list to inspect, in ``(0, 1]``.
        The top-``k`` cut-off is ``max(1, round(fraction * N))``.

    Returns
    -------
    float
        The enrichment factor. Its maximum attainable value is
        ``min(1 / fraction, N / n_actives)``.

    Raises
    ------
    ValueError
        If ``fraction`` is outside ``(0, 1]`` or there are no actives.

    Examples
    --------
    >>> from qsarkit.metrics import enrichment_factor
    >>> enrichment_factor([1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                   [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ...                   fraction=0.2)
    5.0

    References
    ----------
    - Truchon, J.-F. & Bayly, C. I. (2007). J. Chem. Inf. Model., 47(2),
      488-508. https://doi.org/10.1021/ci600426e
    - Bender, A. & Glen, R. C. (2005). "A Discussion of Measures of
      Enrichment in Virtual Screening." J. Chem. Inf. Model., 45(5),
      1369-1375. https://doi.org/10.1021/ci0500177
    """
    yt, ys = _check_scores(y_true, y_score)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    n = yt.shape[0]
    n_actives = int(np.sum(yt == 1))
    if n_actives == 0:
        raise ValueError("The enrichment factor is undefined without actives.")
    k = max(1, int(round(fraction * n)))
    order = np.argsort(-ys, kind="stable")
    hits = int(np.sum(yt[order[:k]] == 1))
    return float((hits / k) / (n_actives / n))


def robust_initial_enhancement(
    y_true: ArrayLike, y_score: ArrayLike, alpha: float = 20.0
) -> float:
    """Robust initial enhancement (RIE) of Sheridan et al.

    ::

        RIE = sum_i exp(-alpha * r_i / N)
              / [ (n/N) * (1 - exp(-alpha)) / (exp(alpha/N) - 1) ]

    where ``r_i`` are the 1-based ranks of the ``n`` actives among ``N``
    compounds. The exponential weight makes RIE a continuous, threshold-free
    generalization of the enrichment factor: ``alpha`` sets how sharply
    early ranks are rewarded (the top ``1/alpha`` of the list carries most
    of the weight). RIE is 1 for a random ranking.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,)
        Continuous scores; the list is ranked in decreasing score order.
    alpha : float, default=20.0
        Exponential weighting parameter; must be positive.

    Returns
    -------
    float
        The RIE value.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or there are no actives.

    Examples
    --------
    >>> from qsarkit.metrics import robust_initial_enhancement
    >>> scores = [1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    >>> labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> robust_initial_enhancement(labels, scores, alpha=20.0) > 1.0
    True

    References
    ----------
    - Sheridan, R. P., Singh, S. B., Fluder, E. M. & Kearsley, S. K. (2001).
      "Protocols for Bridging the Peptide to Nonpeptide Gap in Topological
      Similarity Searches." J. Chem. Inf. Comput. Sci., 41(5), 1395-1406.
      https://doi.org/10.1021/ci0100144
    - Truchon, J.-F. & Bayly, C. I. (2007). J. Chem. Inf. Model., 47(2),
      488-508. https://doi.org/10.1021/ci600426e
    """
    yt, ys = _check_scores(y_true, y_score)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    n_total = yt.shape[0]
    ranks = _active_ranks(yt, ys)
    n_actives = ranks.shape[0]
    if n_actives == 0:
        raise ValueError("RIE is undefined without actives.")
    numerator = float(np.sum(np.exp(-alpha * ranks / n_total)))
    ratio = n_actives / n_total
    denominator = ratio * (1.0 - np.exp(-alpha)) / (np.exp(alpha / n_total) - 1.0)
    return float(numerator / denominator)


def bedroc(y_true: ArrayLike, y_score: ArrayLike, alpha: float = 20.0) -> float:
    """Boltzmann-enhanced discrimination of ROC (BEDROC).

    BEDROC rescales :func:`robust_initial_enhancement` onto ``[0, 1]``,
    removing RIE's dependence on the fraction of actives::

        BEDROC = RIE * Ra * sinh(alpha/2)
                 / (cosh(alpha/2) - cosh(alpha/2 - alpha*Ra))
                 + 1 / (1 - exp(alpha * (1 - Ra)))

    with ``Ra = n_actives / N``. A perfect early-recognition ranking gives
    ~1, a random ranking gives ~``Ra`` and the worst ranking ~0. The
    default ``alpha=20`` concentrates 80% of the weight in the top 8% of
    the list, the usual choice in virtual-screening benchmarks.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary ground-truth labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,)
        Continuous scores; the list is ranked in decreasing score order.
    alpha : float, default=20.0
        Early-recognition weighting parameter; must be positive.

    Returns
    -------
    float
        The BEDROC score.

    Raises
    ------
    ValueError
        If ``alpha <= 0``, there are no actives, or every compound is
        active (``Ra == 1``, for which the metric is undefined).

    Examples
    --------
    >>> from qsarkit.metrics import bedroc
    >>> labels = [1] * 5 + [0] * 95
    >>> scores = list(range(100, 0, -1))
    >>> bedroc(labels, scores, alpha=20.0) > 0.99
    True

    References
    ----------
    - Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
      Methods: Good and Bad Metrics for the 'Early Recognition' Problem."
      J. Chem. Inf. Model., 47(2), 488-508.
      https://doi.org/10.1021/ci600426e
    """
    yt, ys = _check_scores(y_true, y_score)
    if alpha <= 0:
        raise ValueError(f"alpha must be positive, got {alpha}.")
    n_total = yt.shape[0]
    n_actives = int(np.sum(yt == 1))
    if n_actives == 0:
        raise ValueError("BEDROC is undefined without actives.")
    ra = n_actives / n_total
    if ra == 1.0:
        raise ValueError("BEDROC is undefined when every compound is active.")

    rie = robust_initial_enhancement(yt, ys, alpha=alpha)
    half = alpha / 2.0
    scale = ra * np.sinh(half) / (np.cosh(half) - np.cosh(half - alpha * ra))
    offset = 1.0 / (1.0 - np.exp(alpha * (1.0 - ra)))
    return float(rie * scale + offset)
