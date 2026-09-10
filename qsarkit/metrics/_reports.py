"""Aggregate QSAR regression/classification metric report builders.

These convenience functions bundle the individual metrics in this module
into a single dictionary, matching the "appropriate measures of
goodness-of-fit, robustness and predictivity" reporting expected by OECD
principle 4 (see :mod:`qsarkit.validation`).

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models," OECD Series on Testing
  and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from numpy.typing import ArrayLike

from qsarkit.metrics._classification import (
    accuracy,
    balanced_accuracy,
    cohen_kappa,
    confusion_counts,
    f1_score,
    matthews_corrcoef,
    pr_auc,
    precision,
    recall,
    roc_auc,
    sensitivity,
    specificity,
)
from qsarkit.metrics._regression import (
    average_r2m,
    ccc,
    delta_r2m,
    golbraikh_tropsha_criteria,
    mae,
    q2_f1,
    q2_f2,
    q2_f3,
    r2_score,
    rmse,
)

__all__ = ["qsar_regression_report", "qsar_classification_report"]


def qsar_regression_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: Optional[ArrayLike] = None,
    q2: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute the standard bundle of QSAR regression validation statistics.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_test,)
        Observed and predicted responses of the (external) test set.
    y_train : array-like of shape (n_train,), optional
        Training-set responses. When given, the training-mean-normalized
        ``Q^2_F1`` and ``Q^2_F3`` statistics are included.
    q2 : float, optional
        Cross-validated ``Q^2`` of the training set, forwarded to
        :func:`~qsarkit.metrics.golbraikh_tropsha_criteria`.

    Returns
    -------
    dict
        Keys ``"r2"``, ``"rmse"``, ``"mae"``, ``"ccc"``, ``"q2_f2"``,
        ``"average_r2m"``, ``"delta_r2m"``, ``"golbraikh_tropsha"`` and,
        when ``y_train`` is given, ``"q2_f1"`` and ``"q2_f3"``.

    Examples
    --------
    >>> from qsarkit.metrics import qsar_regression_report
    >>> report = qsar_regression_report(
    ...     [1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 2.9, 4.05],
    ...     y_train=[0.0, 1.0, 2.0, 3.0, 4.0], q2=0.9,
    ... )
    >>> report["golbraikh_tropsha"]["passed"]
    True

    References
    ----------
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    - Golbraikh, A. & Tropsha, A. (2002). J. Mol. Graph. Model., 20(4),
      269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
    - OECD (2007). ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """
    report: Dict[str, Any] = {
        "r2": r2_score(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "ccc": ccc(y_true, y_pred),
        "q2_f2": q2_f2(y_true, y_pred),
        "average_r2m": average_r2m(y_true, y_pred),
        "delta_r2m": delta_r2m(y_true, y_pred),
    }
    if y_train is not None:
        report["q2_f1"] = q2_f1(y_true, y_pred, y_train)
        report["q2_f3"] = q2_f3(y_true, y_pred, y_train)
    report["golbraikh_tropsha"] = golbraikh_tropsha_criteria(y_true, y_pred, q2=q2)
    return report


def qsar_classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: Optional[ArrayLike] = None,
) -> Dict[str, Any]:
    """Compute the standard bundle of QSAR classification validation statistics.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Binary ground-truth and predicted labels in ``{0, 1}``.
    y_score : array-like of shape (n_samples,), optional
        Continuous scores (e.g. ``predict_proba(X)[:, 1]``). When given,
        ROC-AUC and PR-AUC are included.

    Returns
    -------
    dict
        Keys ``"confusion"``, ``"accuracy"``, ``"balanced_accuracy"``,
        ``"sensitivity"``, ``"specificity"``, ``"precision"``,
        ``"recall"``, ``"f1"``, ``"mcc"``, ``"cohen_kappa"`` and, when
        ``y_score`` is given, ``"roc_auc"`` and ``"pr_auc"``.

    Examples
    --------
    >>> from qsarkit.metrics import qsar_classification_report
    >>> report = qsar_classification_report(
    ...     [1, 1, 0, 0], [1, 0, 0, 0], y_score=[0.9, 0.4, 0.2, 0.1],
    ... )
    >>> report["mcc"] > 0
    True

    References
    ----------
    - Chicco, D. & Jurman, G. (2020). BMC Genomics, 21, 6.
      https://doi.org/10.1186/s12864-019-6413-7
    - OECD (2007). ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """
    report: Dict[str, Any] = {
        "confusion": confusion_counts(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
    }
    if y_score is not None:
        report["roc_auc"] = roc_auc(y_true, y_score)
        report["pr_auc"] = pr_auc(y_true, y_score)
    return report
