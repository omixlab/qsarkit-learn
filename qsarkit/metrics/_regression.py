"""Regression metrics used to judge QSAR models.

The module covers the classical statistics (RMSE, MAE, R^2) plus the
QSAR-specific external-predictivity coefficients (Q^2_F1, Q^2_F2, Q^2_F3),
Lin's concordance correlation coefficient, Roy's r_m^2 family and the
Golbraikh-Tropsha acceptability criteria.

References
----------
- Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
  Definition of the Q^2 Parameter for QSAR Validation." J. Chem. Inf.
  Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
- Schuurmann, G., Ebert, R.-U., Chen, J., Wang, B. & Kuhne, R. (2008).
  "External Validation and Prediction Employing the Predictive Squared
  Correlation Coefficient - Test Set Activity Mean vs Training Set Activity
  Mean." J. Chem. Inf. Model., 48(11), 2140-2145.
  https://doi.org/10.1021/ci800253u
- Lin, L. I.-K. (1989). "A Concordance Correlation Coefficient to Evaluate
  Reproducibility." Biometrics, 45(1), 255-268.
  https://doi.org/10.2307/2532051
- Roy, K., Chakraborty, P., Mitra, I., Ojha, P. K., Kar, S. & Das, R. N.
  (2013). "Some Case Studies on Application of r_m^2 Metrics for Judging
  Quality of QSAR Predictions." Chemom. Intell. Lab. Syst., 118, 200-210.
  https://doi.org/10.1016/j.chemolab.2012.05.007
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q^2!" J. Mol. Graph.
  Model., 20(4), 269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import ArrayLike

from qsarkit.metrics._common import as_float_1d, check_pair

__all__ = [
    "mse",
    "rmse",
    "rmsep",
    "mae",
    "median_ae",
    "bias",
    "press",
    "see",
    "r2_score",
    "adjusted_r2_score",
    "ccc",
    "q2_f1",
    "q2_f2",
    "q2_f3",
    "k_slope",
    "k_prime_slope",
    "r0_squared",
    "r0_prime_squared",
    "r2m",
    "r2m_prime",
    "delta_r2m",
    "average_r2m",
    "golbraikh_tropsha_criteria",
]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean squared error.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``mean((y_true - y_pred) ** 2)``.

    Examples
    --------
    >>> from qsarkit.metrics import mse
    >>> round(mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]), 4)
    1.3333

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
      JMLR, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(np.mean((yt - yp) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``sqrt(mean((y_true - y_pred) ** 2))``, in the units of ``y``.

    Examples
    --------
    >>> from qsarkit.metrics import rmse
    >>> rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    1.0

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). *Molecular Descriptors for
      Chemoinformatics*, 2nd ed. Wiley-VCH.
      https://doi.org/10.1002/9783527628766
    """
    return float(np.sqrt(mse(y_true, y_pred)))


def rmsep(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root mean squared error of prediction (RMSE on an external set).

    Numerically identical to :func:`rmse`; the separate name is retained
    because the QSAR literature consistently distinguishes RMSEC
    (calibration/training), RMSECV (cross-validation) and RMSEP
    (external prediction), and reporting code reads better when the
    intent is explicit.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses of the *external* test set.

    Returns
    -------
    float
        The root mean squared error of prediction.

    Examples
    --------
    >>> from qsarkit.metrics import rmsep
    >>> rmsep([1.0, 2.0], [1.5, 2.5])
    0.5

    References
    ----------
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """
    return rmse(y_true, y_pred)


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean absolute error.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``mean(|y_true - y_pred|)``.

    Examples
    --------
    >>> from qsarkit.metrics import mae
    >>> mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0])
    0.6666666666666666

    References
    ----------
    - Pedregosa et al. (2011). JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def median_ae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Median absolute error (outlier-robust location of the error).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``median(|y_true - y_pred|)``.

    Examples
    --------
    >>> from qsarkit.metrics import median_ae
    >>> median_ae([1.0, 2.0, 3.0], [1.0, 2.0, 30.0])
    0.0

    References
    ----------
    - Pedregosa et al. (2011). JMLR, 12, 2825-2830.
      https://jmlr.org/papers/v12/pedregosa11a.html
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(np.median(np.abs(yt - yp)))


def bias(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Systematic error (mean signed residual ``mean(y_pred - y_true)``).

    A non-zero bias indicates the model systematically over- (positive) or
    under-predicts (negative) the endpoint, which RMSE alone hides.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The mean signed residual.

    Examples
    --------
    >>> from qsarkit.metrics import bias
    >>> bias([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    1.0

    References
    ----------
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(np.mean(yp - yt))


def press(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Predictive residual sum of squares, ``sum((y_true - y_pred) ** 2)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and (cross-validated or external) predicted responses.

    Returns
    -------
    float
        The PRESS statistic.

    Examples
    --------
    >>> from qsarkit.metrics import press
    >>> press([1.0, 2.0, 3.0], [1.0, 2.0, 5.0])
    4.0

    References
    ----------
    - Allen, D. M. (1974). "The Relationship Between Variable Selection and
      Data Augmentation and a Method for Prediction." Technometrics, 16(1),
      125-127. https://doi.org/10.1080/00401706.1974.10489157
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(np.sum((yt - yp) ** 2))


def see(y_true: ArrayLike, y_pred: ArrayLike, n_parameters: int = 0) -> float:
    """Standard error of estimate, ``sqrt(RSS / (n - p - 1))``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and fitted responses (training set).
    n_parameters : int, default=0
        Number of model parameters ``p`` (excluding the intercept).

    Returns
    -------
    float
        The standard error of estimate.

    Raises
    ------
    ValueError
        If ``n_samples - n_parameters - 1 <= 0``.

    Examples
    --------
    >>> from qsarkit.metrics import see
    >>> round(see([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], n_parameters=1), 4)
    0.7071

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). *Molecular Descriptors for
      Chemoinformatics*. Wiley-VCH. https://doi.org/10.1002/9783527628766
    """
    yt, yp = check_pair(y_true, y_pred)
    dof = yt.shape[0] - int(n_parameters) - 1
    if dof <= 0:
        raise ValueError(
            "Degrees of freedom (n_samples - n_parameters - 1) must be positive; "
            f"got n_samples={yt.shape[0]}, n_parameters={n_parameters}."
        )
    return float(np.sqrt(np.sum((yt - yp) ** 2) / dof))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coefficient of determination ``R^2`` (fraction of variance explained).

    ``R^2 = 1 - RSS / TSS`` where ``TSS`` uses the mean of ``y_true``.
    Note this is *not* the squared Pearson correlation unless the
    predictions are unbiased and unit-slope; QSAR papers that report
    "R^2" for a test set usually mean this quantity (equivalently
    :func:`q2_f2`).

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The coefficient of determination. Can be negative.

    Raises
    ------
    ValueError
        If ``y_true`` has zero variance.

    Examples
    --------
    >>> from qsarkit.metrics import r2_score
    >>> r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """
    yt, yp = check_pair(y_true, y_pred)
    tss = float(np.sum((yt - yt.mean()) ** 2))
    if tss == 0.0:
        raise ValueError("R^2 is undefined when y_true has zero variance.")
    return float(1.0 - np.sum((yt - yp) ** 2) / tss)


def adjusted_r2_score(
    y_true: ArrayLike, y_pred: ArrayLike, n_features: int
) -> float:
    """R^2 penalized for the number of descriptors in the model.

    ``R^2_adj = 1 - (1 - R^2) * (n - 1) / (n - p - 1)``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and fitted responses.
    n_features : int
        Number of descriptors ``p`` used by the model.

    Returns
    -------
    float
        The adjusted coefficient of determination.

    Raises
    ------
    ValueError
        If ``n_samples - n_features - 1 <= 0``.

    Examples
    --------
    >>> from qsarkit.metrics import adjusted_r2_score
    >>> round(adjusted_r2_score([1.0, 2.0, 3.0, 4.5], [1.1, 2.0, 2.9, 4.4], 1), 3)
    0.993

    References
    ----------
    - Todeschini, R. & Consonni, V. (2009). *Molecular Descriptors for
      Chemoinformatics*. Wiley-VCH. https://doi.org/10.1002/9783527628766
    """
    yt, _ = check_pair(y_true, y_pred)
    n = yt.shape[0]
    dof = n - int(n_features) - 1
    if dof <= 0:
        raise ValueError(
            "Adjusted R^2 requires n_samples - n_features - 1 > 0; got "
            f"n_samples={n}, n_features={n_features}."
        )
    return float(1.0 - (1.0 - r2_score(yt, y_pred)) * (n - 1) / dof)


def ccc(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Lin's concordance correlation coefficient.

    ``CCC = 2 * cov(y, yhat) / (var(y) + var(yhat) + (mean(y) - mean(yhat))^2)``

    CCC simultaneously penalizes loss of precision (correlation) and loss of
    accuracy (deviation from the 45-degree line), which makes it stricter
    than the Pearson correlation and a recommended external-validation
    statistic for QSAR. Population (biased, ``ddof=0``) moments are used,
    as in the original paper.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The concordance correlation coefficient, in ``[-1, 1]``.

    Examples
    --------
    >>> from qsarkit.metrics import ccc
    >>> ccc([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Lin, L. I.-K. (1989). "A Concordance Correlation Coefficient to
      Evaluate Reproducibility." Biometrics, 45(1), 255-268.
      https://doi.org/10.2307/2532051
    """
    yt, yp = check_pair(y_true, y_pred)
    mt, mp = yt.mean(), yp.mean()
    vt = float(np.mean((yt - mt) ** 2))
    vp = float(np.mean((yp - mp) ** 2))
    cov = float(np.mean((yt - mt) * (yp - mp)))
    denom = vt + vp + (mt - mp) ** 2
    if denom == 0.0:
        return 1.0
    return float(2.0 * cov / denom)


def q2_f1(y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike) -> float:
    """External predictivity Q^2_F1 (Shi/Schuurmann formulation).

    ``Q^2_F1 = 1 - sum((y_ext - yhat_ext)^2) / sum((y_ext - mean(y_train))^2)``

    The residual sum of squares of the external set is normalized by its
    variation around the *training* set mean. This is the original
    "R^2_pred" of Shi et al. and is the most commonly reported form.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_test,)
        Observed and predicted responses of the external set.
    y_train : array-like of shape (n_train,)
        Observed responses of the training set (only its mean is used).

    Returns
    -------
    float
        The Q^2_F1 statistic.

    Raises
    ------
    ValueError
        If the normalizing sum of squares is zero.

    Examples
    --------
    >>> from qsarkit.metrics import q2_f1
    >>> q2_f1([2.0, 4.0], [2.0, 3.0], y_train=[0.0, 2.0, 4.0])
    0.75

    References
    ----------
    - Schuurmann, G., Ebert, R.-U., Chen, J., Wang, B. & Kuhne, R. (2008).
      J. Chem. Inf. Model., 48(11), 2140-2145.
      https://doi.org/10.1021/ci800253u
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """
    yt, yp = check_pair(y_true, y_pred)
    ytr = as_float_1d(y_train, "y_train")
    denom = float(np.sum((yt - ytr.mean()) ** 2))
    if denom == 0.0:
        raise ValueError(
            "Q^2_F1 is undefined: the external responses all equal the "
            "training-set mean."
        )
    return float(1.0 - np.sum((yt - yp) ** 2) / denom)


def q2_f2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """External predictivity Q^2_F2 (Schuurmann formulation).

    ``Q^2_F2 = 1 - sum((y_ext - yhat_ext)^2) / sum((y_ext - mean(y_ext))^2)``

    Uses the *external* set mean, making it identical to the ordinary
    coefficient of determination computed on the test set. It is the most
    conservative of the three and is invariant to the training/test split
    ratio.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_test,)
        Observed and predicted responses of the external set.

    Returns
    -------
    float
        The Q^2_F2 statistic.

    Examples
    --------
    >>> from qsarkit.metrics import q2_f2
    >>> q2_f2([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Schuurmann, G., Ebert, R.-U., Chen, J., Wang, B. & Kuhne, R. (2008).
      J. Chem. Inf. Model., 48(11), 2140-2145.
      https://doi.org/10.1021/ci800253u
    """
    return r2_score(y_true, y_pred)


def q2_f3(y_true: ArrayLike, y_pred: ArrayLike, y_train: ArrayLike) -> float:
    """External predictivity Q^2_F3 (Consonni/Todeschini formulation).

    ::

        Q^2_F3 = 1 - [sum((y_ext - yhat_ext)^2) / n_ext]
                   / [sum((y_train - mean(y_train))^2) / n_train]

    Both sums of squares are divided by their own sample size, which makes
    the statistic independent of the size of the external set and (per
    Consonni et al.) monotonically related to RMSEP for a fixed training
    set.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_test,)
        Observed and predicted responses of the external set.
    y_train : array-like of shape (n_train,)
        Observed responses of the training set.

    Returns
    -------
    float
        The Q^2_F3 statistic.

    Raises
    ------
    ValueError
        If ``y_train`` has zero variance.

    Examples
    --------
    >>> from qsarkit.metrics import q2_f3
    >>> q2_f3([2.0, 4.0], [2.0, 3.0], y_train=[0.0, 2.0, 4.0])
    0.8125

    References
    ----------
    - Consonni, V., Ballabio, D. & Todeschini, R. (2009). "Comments on the
      Definition of the Q^2 Parameter for QSAR Validation." J. Chem. Inf.
      Model., 49(7), 1669-1678. https://doi.org/10.1021/ci900115y
    """
    yt, yp = check_pair(y_true, y_pred)
    ytr = as_float_1d(y_train, "y_train")
    train_ms = float(np.sum((ytr - ytr.mean()) ** 2) / ytr.shape[0])
    if train_ms == 0.0:
        raise ValueError("Q^2_F3 is undefined when y_train has zero variance.")
    test_ms = float(np.sum((yt - yp) ** 2) / yt.shape[0])
    return float(1.0 - test_ms / train_ms)


def k_slope(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Slope ``k`` of the regression of predicted on observed through the origin.

    ``k = sum(y * yhat) / sum(y^2)``

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The through-origin slope ``k``. Golbraikh & Tropsha require
        ``0.85 <= k <= 1.15``.

    Raises
    ------
    ValueError
        If ``sum(y_true ** 2)`` is zero.

    Examples
    --------
    >>> from qsarkit.metrics import k_slope
    >>> k_slope([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). "Beware of q^2!" J. Mol. Graph.
      Model., 20(4), 269-276.
      https://doi.org/10.1016/S1093-3263(01)00123-1
    """
    yt, yp = check_pair(y_true, y_pred)
    denom = float(np.sum(yt**2))
    if denom == 0.0:
        raise ValueError("k is undefined when all observed values are zero.")
    return float(np.sum(yt * yp) / denom)


def k_prime_slope(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Slope ``k'`` of the regression of observed on predicted through the origin.

    ``k' = sum(y * yhat) / sum(yhat^2)``

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The through-origin slope ``k'``.

    Raises
    ------
    ValueError
        If ``sum(y_pred ** 2)`` is zero.

    Examples
    --------
    >>> from qsarkit.metrics import k_prime_slope
    >>> k_prime_slope([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
    0.5

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). J. Mol. Graph. Model., 20(4),
      269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
    """
    yt, yp = check_pair(y_true, y_pred)
    denom = float(np.sum(yp**2))
    if denom == 0.0:
        raise ValueError("k' is undefined when all predicted values are zero.")
    return float(np.sum(yt * yp) / denom)


def _pearson_r2(yt: np.ndarray, yp: np.ndarray) -> float:
    """Squared Pearson correlation between two validated arrays."""
    if yt.shape[0] < 2:
        raise ValueError("At least two samples are required for a correlation.")
    st = float(np.std(yt))
    sp = float(np.std(yp))
    if st == 0.0 or sp == 0.0:
        raise ValueError(
            "The squared correlation is undefined when y_true or y_pred is constant."
        )
    r = float(np.mean((yt - yt.mean()) * (yp - yp.mean())) / (st * sp))
    return r * r


def r0_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Determination coefficient of the through-origin fit ``yhat = k * y``.

    ``R0^2 = 1 - sum((yhat - k*y)^2) / sum((yhat - mean(yhat))^2)``

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The through-origin ``R0^2``.

    Examples
    --------
    >>> from qsarkit.metrics import r0_squared
    >>> r0_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). J. Mol. Graph. Model., 20(4),
      269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
    """
    yt, yp = check_pair(y_true, y_pred)
    k = k_slope(yt, yp)
    denom = float(np.sum((yp - yp.mean()) ** 2))
    if denom == 0.0:
        raise ValueError("R0^2 is undefined when y_pred is constant.")
    return float(1.0 - np.sum((yp - k * yt) ** 2) / denom)


def r0_prime_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Determination coefficient of the through-origin fit ``y = k' * yhat``.

    ``R0'^2 = 1 - sum((y - k'*yhat)^2) / sum((y - mean(y))^2)``

    This is the axis-swapped counterpart of :func:`r0_squared`.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The through-origin ``R0'^2``.

    Examples
    --------
    >>> from qsarkit.metrics import r0_prime_squared
    >>> r0_prime_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). J. Mol. Graph. Model., 20(4),
      269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
    """
    yt, yp = check_pair(y_true, y_pred)
    kp = k_prime_slope(yt, yp)
    denom = float(np.sum((yt - yt.mean()) ** 2))
    if denom == 0.0:
        raise ValueError("R0'^2 is undefined when y_true is constant.")
    return float(1.0 - np.sum((yt - kp * yp) ** 2) / denom)


def r2m(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Roy's ``r_m^2`` metric, ``r^2 * (1 - sqrt(|r^2 - r0^2|))``.

    ``r^2`` is the squared Pearson correlation and ``r0^2`` the
    through-origin determination coefficient (:func:`r0_squared`). A model
    is considered acceptable when the average of ``r_m^2`` and
    ``r_m'^2`` exceeds 0.5.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The ``r_m^2`` value.

    Examples
    --------
    >>> from qsarkit.metrics import r2m
    >>> r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Roy, K., Chakraborty, P., Mitra, I., Ojha, P. K., Kar, S. & Das, R. N.
      (2013). Chemom. Intell. Lab. Syst., 118, 200-210.
      https://doi.org/10.1016/j.chemolab.2012.05.007
    """
    yt, yp = check_pair(y_true, y_pred)
    r2 = _pearson_r2(yt, yp)
    return float(r2 * (1.0 - np.sqrt(abs(r2 - r0_squared(yt, yp)))))


def r2m_prime(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Roy's axis-swapped ``r_m'^2``, ``r^2 * (1 - sqrt(|r^2 - r0'^2|))``.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        The ``r_m'^2`` value.

    Examples
    --------
    >>> from qsarkit.metrics import r2m_prime
    >>> r2m_prime([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Roy, K. et al. (2013). Chemom. Intell. Lab. Syst., 118, 200-210.
      https://doi.org/10.1016/j.chemolab.2012.05.007
    """
    yt, yp = check_pair(y_true, y_pred)
    r2 = _pearson_r2(yt, yp)
    return float(r2 * (1.0 - np.sqrt(abs(r2 - r0_prime_squared(yt, yp)))))


def average_r2m(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean of ``r_m^2`` and ``r_m'^2``; should exceed 0.5.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``(r_m^2 + r_m'^2) / 2``.

    Examples
    --------
    >>> from qsarkit.metrics import average_r2m
    >>> average_r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    1.0

    References
    ----------
    - Roy, K. et al. (2013). Chemom. Intell. Lab. Syst., 118, 200-210.
      https://doi.org/10.1016/j.chemolab.2012.05.007
    """
    yt, yp = check_pair(y_true, y_pred)
    return float((r2m(yt, yp) + r2m_prime(yt, yp)) / 2.0)


def delta_r2m(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Absolute difference ``|r_m^2 - r_m'^2|``; should stay below 0.2.

    A large delta signals that the observed-vs-predicted relationship is
    strongly asymmetric, i.e. the model is systematically compressing or
    expanding the response range.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
        Observed and predicted responses.

    Returns
    -------
    float
        ``|r_m^2 - r_m'^2|``.

    Examples
    --------
    >>> from qsarkit.metrics import delta_r2m
    >>> delta_r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    0.0

    References
    ----------
    - Roy, K. et al. (2013). Chemom. Intell. Lab. Syst., 118, 200-210.
      https://doi.org/10.1016/j.chemolab.2012.05.007
    """
    yt, yp = check_pair(y_true, y_pred)
    return float(abs(r2m(yt, yp) - r2m_prime(yt, yp)))


def golbraikh_tropsha_criteria(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    q2: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate the five Golbraikh-Tropsha acceptability criteria.

    A QSAR model is deemed externally predictive when all of

    1. ``q^2 > 0.5`` (leave-one-out / cross-validated Q^2 of the training set);
    2. ``r^2 > 0.6`` (squared correlation between observed and predicted
       on the external set);
    3. ``(r^2 - R0^2) / r^2 < 0.1`` **or** ``(r^2 - R0'^2) / r^2 < 0.1``;
    4. ``0.85 <= k <= 1.15`` **or** ``0.85 <= k' <= 1.15``;
    5. ``|R0^2 - R0'^2| < 0.3``

    hold simultaneously.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_test,)
        Observed and predicted responses of the *external* validation set.
    q2 : float, optional
        Cross-validated Q^2 of the training set. If ``None``, criterion 1
        is reported as ``None`` and excluded from ``passed`` (the returned
        dict then also carries ``"q2_available": False``).

    Returns
    -------
    dict
        Keys ``r2``, ``r0_squared``, ``r0_prime_squared``, ``k``,
        ``k_prime``, ``q2``, the booleans ``criterion_1_q2``,
        ``criterion_2_r2``, ``criterion_3_r0``, ``criterion_4_slope``,
        ``criterion_5_delta_r0``, ``q2_available`` and the overall
        ``passed``.

    Examples
    --------
    >>> from qsarkit.metrics import golbraikh_tropsha_criteria
    >>> res = golbraikh_tropsha_criteria([1.0, 2.0, 3.0, 4.0],
    ...                                  [1.1, 2.0, 2.9, 4.05], q2=0.9)
    >>> res["passed"]
    True

    References
    ----------
    - Golbraikh, A. & Tropsha, A. (2002). "Beware of q^2!" J. Mol. Graph.
      Model., 20(4), 269-276.
      https://doi.org/10.1016/S1093-3263(01)00123-1
    - Tropsha, A., Gramatica, P. & Gombar, V. K. (2003). "The Importance of
      Being Earnest: Validation is the Absolute Essential for Successful
      Application and Interpretation of QSPR Models." QSAR Comb. Sci.,
      22(1), 69-77. https://doi.org/10.1002/qsar.200390007
    """
    yt, yp = check_pair(y_true, y_pred)
    r2 = _pearson_r2(yt, yp)
    r02 = r0_squared(yt, yp)
    r0p2 = r0_prime_squared(yt, yp)
    k = k_slope(yt, yp)
    kp = k_prime_slope(yt, yp)

    c1 = None if q2 is None else bool(q2 > 0.5)
    c2 = bool(r2 > 0.6)
    c3 = bool(((r2 - r02) / r2 < 0.1) or ((r2 - r0p2) / r2 < 0.1))
    c4 = bool((0.85 <= k <= 1.15) or (0.85 <= kp <= 1.15))
    c5 = bool(abs(r02 - r0p2) < 0.3)

    checks = [c2, c3, c4, c5] if c1 is None else [c1, c2, c3, c4, c5]
    return {
        "q2": None if q2 is None else float(q2),
        "r2": r2,
        "r0_squared": r02,
        "r0_prime_squared": r0p2,
        "k": k,
        "k_prime": kp,
        "delta_r0_squared": float(abs(r02 - r0p2)),
        "criterion_1_q2": c1,
        "criterion_2_r2": c2,
        "criterion_3_r0": c3,
        "criterion_4_slope": c4,
        "criterion_5_delta_r0": c5,
        "q2_available": q2 is not None,
        "passed": bool(all(checks)),
    }
