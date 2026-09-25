r"""Probability calibration and residual-distribution diagnostics.

A classifier that reports 0.9 should be right about 90% of the time. Most
are not: random forests are systematically under-confident at the extremes
and SVMs with Platt scaling can be badly miscalibrated on the imbalanced
datasets typical of virtual screening. ROC-AUC does not notice, because it
depends only on the *ranking* of the scores, not on their values -- a model
can have perfect AUC and useless probabilities.

That distinction matters whenever a probability is used as a number rather
than as a rank: to set a screening cutoff, to combine models, or to feed a
cost calculation.

The regression counterpart here is the Q-Q plot. Every regression metric in
:mod:`qsarkit.metrics` -- RMSE, :math:`R^2`, the Golbraikh-Tropsha
criteria -- assumes roughly normal, homoscedastic errors. When that fails,
the numbers still compute and quietly mean something else.

References
----------
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good Probabilities
  with Supervised Learning." ICML 2005, 625-632.
  https://doi.org/10.1145/1102351.1102430
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks."
  ICML 2017, 1321-1330. https://proceedings.mlr.press/v70/guo17a.html
- Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). "Obtaining Well
  Calibrated Probabilities Using Bayesian Binning." AAAI 2015, 2901-2907.
  https://doi.org/10.1609/aaai.v29i1.9602
- Brier, G. W. (1950). "Verification of Forecasts Expressed in Terms of
  Probability." Mon. Weather Rev., 78(1), 1-3.
  https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Wilk, M. B. & Gnanadesikan, R. (1968). "Probability Plotting Methods for
  the Analysis of Data." Biometrika, 55(1), 1-17.
  https://doi.org/10.1093/biomet/55.1.1
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Tuple

import numpy as np
import numpy.typing as npt

__all__ = [
    "calibration_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "calibration_report",
    "qq_data",
    "residual_normality",
]


def _check_probabilities(
    y_true: npt.ArrayLike, y_prob: npt.ArrayLike
) -> Tuple["npt.NDArray[np.float64]", "npt.NDArray[np.float64]"]:
    """Validate a binary label / probability pair."""
    labels = np.asarray(y_true, dtype=np.float64).ravel()
    probabilities = np.asarray(y_prob, dtype=np.float64).ravel()

    if labels.shape != probabilities.shape:
        raise ValueError(
            f"y_true has {labels.size} entries but y_prob has "
            f"{probabilities.size}."
        )
    if labels.size == 0:
        raise ValueError("Cannot assess calibration of an empty set.")

    unique = np.unique(labels[np.isfinite(labels)])
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(
            f"y_true must be binary 0/1 labels, got values {unique[:5]}. "
            "For a multiclass problem, assess each class one-vs-rest."
        )
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError(
            "y_prob must lie in [0, 1]. These look like decision-function "
            "scores rather than probabilities; convert them first (e.g. with "
            "predict_proba, or a sigmoid) -- calibration is a statement "
            "about probabilities and is meaningless for arbitrary scores."
        )
    return labels, probabilities


def calibration_curve(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> Dict[str, "npt.NDArray[np.float64]"]:
    """Observed frequency against predicted probability, per bin.

    The data behind a reliability diagram. A perfectly calibrated model
    lies on the diagonal: among the compounds it scored 0.7, 70% are
    active.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels, 0 or 1.
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class.
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        ``"uniform"`` splits [0, 1] into equal-width bins, which shows
        where on the probability scale the model is wrong. ``"quantile"``
        puts an equal number of samples in each bin, which gives every
        point a comparable error bar -- the better choice when predictions
        cluster near 0, as they do in virtual screening.

    Returns
    -------
    dict
        ``mean_predicted`` and ``observed_frequency`` (one entry per
        non-empty bin), ``counts``, and ``bin_edges``.

    Raises
    ------
    ValueError
        If the inputs are not a matching pair of binary labels and
        probabilities, or ``n_bins`` is below 2.

    Examples
    --------
    A perfectly calibrated set of predictions lies on the diagonal:

    >>> import numpy as np
    >>> from qsarkit.metrics import calibration_curve
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(size=4000)
    >>> y = (rng.uniform(size=4000) < p).astype(int)
    >>> curve = calibration_curve(y, p, n_bins=5)
    >>> bool(np.allclose(curve["mean_predicted"], curve["observed_frequency"],
    ...                  atol=0.05))
    True

    An over-confident model bends away from it:

    >>> squashed = np.clip(p * 1.6 - 0.3, 0, 1)
    >>> curve = calibration_curve(y, squashed, n_bins=5)
    >>> bool((curve["observed_frequency"][0] > curve["mean_predicted"][0]))
    True

    References
    ----------
    - Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good
      Probabilities with Supervised Learning." ICML 2005, 625-632.
      https://doi.org/10.1145/1102351.1102430
    """
    labels, probabilities = _check_probabilities(y_true, y_prob)
    if n_bins < 2:
        raise ValueError(f"n_bins must be at least 2, got {n_bins}.")
    if strategy not in ("uniform", "quantile"):
        raise ValueError(
            f"strategy must be 'uniform' or 'quantile', got {strategy!r}."
        )

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        edges = np.unique(
            np.quantile(probabilities, np.linspace(0.0, 1.0, n_bins + 1))
        )
        if edges.size < 2:  # every prediction identical
            edges = np.array([0.0, 1.0])

    # `right=True` with a lowered first edge puts a prediction of exactly 0
    # in the first bin rather than in a phantom bin 0.
    indices = np.digitize(probabilities, edges[1:-1], right=False)

    mean_predicted = []
    observed = []
    counts = []
    for b in range(len(edges) - 1):
        mask = indices == b
        if not mask.any():
            continue
        mean_predicted.append(float(probabilities[mask].mean()))
        observed.append(float(labels[mask].mean()))
        counts.append(int(mask.sum()))

    return {
        "mean_predicted": np.asarray(mean_predicted, dtype=np.float64),
        "observed_frequency": np.asarray(observed, dtype=np.float64),
        "counts": np.asarray(counts, dtype=np.float64),
        "bin_edges": edges,
    }


def expected_calibration_error(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> float:
    r"""Sample-weighted mean gap between confidence and accuracy.

    .. math::

        \mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N}
                       \bigl| \bar{p}_b - \bar{y}_b \bigr|

    where :math:`\bar{p}_b` is the mean predicted probability in bin
    :math:`b`, :math:`\bar{y}_b` the observed frequency, and :math:`n_b`
    the bin's size. 0 is perfect.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default 10
    strategy : {"uniform", "quantile"}, default "uniform"

    Returns
    -------
    float
        In [0, 1].

    Notes
    -----
    ECE depends on the binning, and a model can lower it by concentrating
    its predictions rather than by improving. Read it beside the curve
    from :func:`calibration_curve`, not on its own.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import expected_calibration_error
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(size=4000)
    >>> y = (rng.uniform(size=4000) < p).astype(int)
    >>> round(expected_calibration_error(y, p, n_bins=10), 2) < 0.05
    True

    A model whose probabilities are all shifted upward scores worse:

    >>> shifted = np.clip(p + 0.25, 0, 1)
    >>> expected_calibration_error(y, shifted) > expected_calibration_error(y, p)
    True

    References
    ----------
    - Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). "Obtaining
      Well Calibrated Probabilities Using Bayesian Binning." AAAI 2015,
      2901-2907. https://doi.org/10.1609/aaai.v29i1.9602
    - Guo, C. et al. (2017). "On Calibration of Modern Neural Networks."
      ICML 2017, 1321-1330.
      https://proceedings.mlr.press/v70/guo17a.html
    """
    curve = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    gaps = np.abs(curve["mean_predicted"] - curve["observed_frequency"])
    weights = curve["counts"] / curve["counts"].sum()
    return float(np.sum(weights * gaps))


def maximum_calibration_error(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
    min_count: int = 1,
) -> float:
    """Largest single-bin gap between confidence and accuracy.

    The worst case rather than the average. Useful when a decision will be
    made at one particular probability: an ECE of 0.02 is no comfort if
    the bin you actually threshold on is off by 0.3.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default 10
    strategy : {"uniform", "quantile"}, default "uniform"
    min_count : int, default 1
        Ignore bins with fewer samples than this. A bin holding two
        compounds can only report frequencies of 0, 0.5 or 1, so it
        produces a large gap by arithmetic rather than by miscalibration;
        raising this suppresses that artefact.

    Returns
    -------
    float
        In [0, 1]. ``0.0`` when no bin meets ``min_count``.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import (
    ...     expected_calibration_error, maximum_calibration_error)
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(size=4000)
    >>> y = (rng.uniform(size=4000) < p).astype(int)
    >>> mce = maximum_calibration_error(y, p, n_bins=10, min_count=20)
    >>> ece = expected_calibration_error(y, p, n_bins=10)
    >>> mce >= ece        # the worst bin is at least as bad as the average
    True

    References
    ----------
    - Naeini, M. P., Cooper, G. F. & Hauskrecht, M. (2015). AAAI 2015,
      2901-2907. https://doi.org/10.1609/aaai.v29i1.9602
    """
    if min_count < 1:
        raise ValueError(f"min_count must be at least 1, got {min_count}.")
    curve = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    keep = curve["counts"] >= min_count
    if not keep.any():
        return 0.0
    gaps = np.abs(
        curve["mean_predicted"][keep] - curve["observed_frequency"][keep]
    )
    return float(gaps.max())


def calibration_report(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: Literal["uniform", "quantile"] = "uniform",
) -> Dict[str, Any]:
    """Everything needed to judge whether probabilities can be believed.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default 10
    strategy : {"uniform", "quantile"}, default "uniform"

    Returns
    -------
    dict
        ``ece``, ``mce``, ``brier``, ``brier_skill_score``,
        ``mean_predicted``, ``observed_frequency``, ``base_rate``,
        ``n_samples`` and ``n_bins_used``.

    Notes
    -----
    ``brier_skill_score`` compares the Brier score against always
    predicting the base rate: positive means the model beats that
    baseline, 0 or below means it does not. On an imbalanced screening
    set a raw Brier score near 0.05 looks excellent and is often worse
    than the constant prediction -- the skill score is what exposes that.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import calibration_report
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(size=2000)
    >>> y = (rng.uniform(size=2000) < p).astype(int)
    >>> report = calibration_report(y, p, n_bins=10)
    >>> report["ece"] < 0.05
    True
    >>> report["brier_skill_score"] > 0
    True

    A constant prediction at the base rate has no skill at all:

    >>> flat = np.full(2000, y.mean())
    >>> round(calibration_report(y, flat)["brier_skill_score"], 6)
    0.0

    References
    ----------
    - Brier, G. W. (1950). Mon. Weather Rev., 78(1), 1-3.
      https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
    - Guo, C. et al. (2017). ICML 2017, 1321-1330.
      https://proceedings.mlr.press/v70/guo17a.html
    """
    labels, probabilities = _check_probabilities(y_true, y_prob)
    curve = calibration_curve(labels, probabilities, n_bins=n_bins, strategy=strategy)

    brier = float(np.mean((probabilities - labels) ** 2))
    base_rate = float(labels.mean())
    reference = float(np.mean((base_rate - labels) ** 2))
    skill = float(1.0 - brier / reference) if reference > 0 else 0.0

    return {
        "ece": expected_calibration_error(
            labels, probabilities, n_bins=n_bins, strategy=strategy
        ),
        "mce": maximum_calibration_error(
            labels, probabilities, n_bins=n_bins, strategy=strategy
        ),
        "brier": brier,
        "brier_skill_score": skill,
        "base_rate": base_rate,
        "mean_predicted": curve["mean_predicted"],
        "observed_frequency": curve["observed_frequency"],
        "n_samples": int(labels.size),
        "n_bins_used": int(curve["counts"].size),
    }


def qq_data(
    residuals: npt.ArrayLike, standardize: bool = True
) -> Dict[str, "npt.NDArray[np.float64]"]:
    """Theoretical against observed quantiles, for a normal Q-Q plot.

    Every regression metric here -- RMSE, :math:`R^2`, the
    Golbraikh-Tropsha criteria -- assumes roughly normal, homoscedastic
    errors. A Q-Q plot is the quickest check. Points on the diagonal mean
    normal residuals; an S-shape means heavy tails; a curve at one end
    means skew, usually from a handful of badly mispredicted compounds
    that RMSE alone will not name.

    Parameters
    ----------
    residuals : array-like of shape (n_samples,)
        Observed minus predicted. Non-finite entries are dropped.
    standardize : bool, default True
        Divide by the standard deviation, so the reference line is
        :math:`y = x` regardless of the residuals' scale.

    Returns
    -------
    dict
        ``theoretical_quantiles``, ``sample_quantiles`` (both sorted
        ascending) and ``reference_line`` as ``(slope, intercept)``.

    Raises
    ------
    ValueError
        If fewer than three finite residuals remain.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import qq_data
    >>> rng = np.random.default_rng(0)
    >>> data = qq_data(rng.normal(size=500))
    >>> corr = np.corrcoef(data["theoretical_quantiles"],
    ...                    data["sample_quantiles"])[0, 1]
    >>> bool(corr > 0.99)            # normal residuals track the diagonal
    True

    Heavy-tailed residuals do not:

    >>> heavy = qq_data(rng.standard_t(df=2, size=500))
    >>> bool(np.corrcoef(heavy["theoretical_quantiles"],
    ...                  heavy["sample_quantiles"])[0, 1] < corr)
    True

    References
    ----------
    - Wilk, M. B. & Gnanadesikan, R. (1968). "Probability Plotting Methods
      for the Analysis of Data." Biometrika, 55(1), 1-17.
      https://doi.org/10.1093/biomet/55.1.1
    - Blom, G. (1958). "Statistical Estimates and Transformed
      Beta-Variables." Wiley.
    """
    from scipy import stats

    values = np.asarray(residuals, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError(
            f"Need at least 3 finite residuals for a Q-Q plot, got "
            f"{values.size}."
        )

    if standardize:
        spread = float(values.std(ddof=1))
        values = (values - values.mean()) / spread if spread > 0 else values - values.mean()

    ordered = np.sort(values)
    n = ordered.size
    # Blom's plotting positions: (i - 3/8) / (n + 1/4), the standard choice
    # for a normal Q-Q plot and near-unbiased for the expected order
    # statistics.
    positions = (np.arange(1, n + 1) - 0.375) / (n + 0.25)
    theoretical = stats.norm.ppf(positions)

    # Reference line through the first and third quartiles, which is robust
    # to the outliers a Q-Q plot exists to reveal.
    q1_t, q3_t = np.quantile(theoretical, [0.25, 0.75])
    q1_s, q3_s = np.quantile(ordered, [0.25, 0.75])
    slope = (q3_s - q1_s) / (q3_t - q1_t) if q3_t != q1_t else 1.0
    intercept = q1_s - slope * q1_t

    return {
        "theoretical_quantiles": theoretical,
        "sample_quantiles": ordered,
        "reference_line": np.asarray([slope, intercept], dtype=np.float64),
    }


def residual_normality(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> Dict[str, float]:
    """Test whether regression residuals are normal and homoscedastic.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)

    Returns
    -------
    dict
        ``shapiro_statistic`` and ``shapiro_p`` (normality; p below 0.05
        argues against it), ``skew``, ``excess_kurtosis``,
        ``heteroscedasticity_r`` (Spearman correlation between the fitted
        value and the absolute residual -- non-zero means error size
        depends on the prediction), and ``n``.

    Notes
    -----
    Shapiro-Wilk cuts both ways at QSAR sample sizes, and neither
    direction should be read as a verdict:

    * On a few dozen compounds it has little power, so ``shapiro_p > 0.05``
      is weak evidence of normality rather than a clearance.
    * On a few hundred it starts rejecting samples that *are* normal, for
      departures far too small to affect an RMSE.

    The skew, kurtosis and heteroscedasticity terms describe the shape
    rather than testing a hypothesis, which is more useful here -- and the
    Q-Q plot from :func:`qq_data` more useful still, because it shows
    *where* the departure is.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.metrics import residual_normality
    >>> rng = np.random.default_rng(0)
    >>> truth = rng.normal(size=500)
    >>> report = residual_normality(truth, truth + rng.normal(scale=0.1, size=500))
    >>> report["shapiro_p"] > 0.05         # residuals are normal
    True
    >>> abs(report["skew"]) < 0.2
    True

    A badly skewed residual distribution is caught by every term at once:

    >>> skewed = residual_normality(np.zeros(300), -rng.exponential(size=300))
    >>> skewed["shapiro_p"] < 0.001, abs(skewed["skew"]) > 1
    (True, True)

    Errors that grow with the prediction show up as heteroscedasticity:

    >>> fitted = np.linspace(1, 10, 300)
    >>> noisy = fitted + rng.normal(scale=fitted * 0.3)
    >>> abs(residual_normality(noisy, fitted)["heteroscedasticity_r"]) > 0.2
    True

    References
    ----------
    - Shapiro, S. S. & Wilk, M. B. (1965). "An Analysis of Variance Test
      for Normality." Biometrika, 52(3-4), 591-611.
      https://doi.org/10.1093/biomet/52.3-4.591
    - Breusch, T. S. & Pagan, A. R. (1979). "A Simple Test for
      Heteroscedasticity and Random Coefficient Variation."
      Econometrica, 47(5), 1287-1294. https://doi.org/10.2307/1911963
    """
    from scipy import stats

    truth = np.asarray(y_true, dtype=np.float64).ravel()
    predicted = np.asarray(y_pred, dtype=np.float64).ravel()
    if truth.shape != predicted.shape:
        raise ValueError(
            f"y_true has {truth.size} entries but y_pred has {predicted.size}."
        )
    finite = np.isfinite(truth) & np.isfinite(predicted)
    truth, predicted = truth[finite], predicted[finite]
    if truth.size < 3:
        raise ValueError(
            f"Need at least 3 finite pairs, got {truth.size}."
        )

    residuals = truth - predicted

    # Shapiro-Wilk is undefined below 3 points and unreliable above ~5000;
    # scipy warns rather than failing, so cap the sample it sees.
    sample = residuals
    if sample.size > 5000:
        sample = np.random.default_rng(0).choice(sample, 5000, replace=False)
    if float(np.std(sample)) == 0.0:
        shapiro_statistic, shapiro_p = 1.0, 1.0
    else:
        result = stats.shapiro(sample)
        shapiro_statistic, shapiro_p = float(result.statistic), float(result.pvalue)

    if np.std(predicted) == 0.0 or np.std(np.abs(residuals)) == 0.0:
        hetero = 0.0
    else:
        hetero = float(stats.spearmanr(predicted, np.abs(residuals)).statistic)

    return {
        "shapiro_statistic": shapiro_statistic,
        "shapiro_p": shapiro_p,
        "skew": float(stats.skew(residuals)),
        "excess_kurtosis": float(stats.kurtosis(residuals)),
        "heteroscedasticity_r": hetero,
        "n": int(residuals.size),
    }
