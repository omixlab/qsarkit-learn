"""Calibration diagnostics for regression uncertainty estimates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go

__all__ = ["UncertaintyCalibration"]


class UncertaintyCalibration:
    """Assess whether predicted uncertainties mean what they claim.

    An uncertainty estimate is only useful if it is *calibrated*: when a
    model says +/-0.5, the true value should land inside that interval
    about as often as the nominal level promises. Models routinely fail
    this — deep ensembles are typically overconfident — and a
    well-ranked but miscalibrated uncertainty will silently break any
    downstream decision rule with an absolute threshold.

    Two distinct properties are measured here, and a good estimator needs
    both:

    - **Calibration** (ENCE, miscalibration area, coverage curve): are
      the magnitudes right?
    - **Ranking** (Spearman correlation of uncertainty with absolute
      error): do higher-uncertainty predictions actually err more?

    Parameters
    ----------
    n_bins : int, default 10
        Number of equal-count bins used for the binned statistics.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> y_true = rng.normal(size=500)
    >>> sigma = np.full(500, 1.0)
    >>> y_pred = y_true + rng.normal(scale=1.0, size=500)
    >>> cal = UncertaintyCalibration()
    >>> report = cal.report(y_true, y_pred, sigma)
    >>> 0.0 <= report["ence"] < 1.0
    True

    References
    ----------
    - Levi, D. et al. (2022). "Evaluating and Calibrating Uncertainty
      Prediction in Regression Tasks." Sensors, 22(15), 5540.
      https://doi.org/10.3390/s22155540
    - Kuleshov, V., Fenner, N. & Ermon, S. (2018). "Accurate
      Uncertainties for Deep Learning Using Calibrated Regression."
      ICML 2018. https://arxiv.org/abs/1807.00263
    - Scalia, G. et al. (2020). "Evaluating Scalable Uncertainty
      Estimation Methods for Deep Learning-Based Molecular Property
      Prediction." J. Chem. Inf. Model., 60(6), 2697-2717.
      https://doi.org/10.1021/acs.jcim.9b00975
    - Tran, K. et al. (2020). "Methods for Comparing Uncertainty
      Quantifications for Material Property Predictions." Mach. Learn.:
      Sci. Technol., 1, 025006. https://doi.org/10.1088/2632-2153/ab7e1a
    """

    def __init__(self, n_bins: int = 10) -> None:
        self.n_bins = n_bins

    def _validate(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
    ) -> Tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]:
        true = np.asarray(y_true, dtype=np.float64).ravel()
        pred = np.asarray(y_pred, dtype=np.float64).ravel()
        std = np.asarray(sigma, dtype=np.float64).ravel()
        if not (true.shape == pred.shape == std.shape):
            raise ValueError(
                f"Shape mismatch: y_true {true.shape}, y_pred {pred.shape}, "
                f"sigma {std.shape}."
            )
        if np.any(std < 0):
            raise ValueError("sigma must be non-negative.")
        if self.n_bins < 1:
            raise ValueError(f"n_bins must be positive, got {self.n_bins}.")
        return true, pred, std

    def ence(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
    ) -> float:
        """Expected Normalized Calibration Error.

        Samples are binned by predicted uncertainty; within each bin the
        root-mean-square error is compared to the mean predicted sigma.
        ENCE is the mean relative discrepancy — 0 is perfect.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)

        Returns
        -------
        float
            Non-negative; smaller is better.

        References
        ----------
        - Levi, D. et al. (2022). Sensors, 22(15), 5540.
          https://doi.org/10.3390/s22155540
        """
        true, pred, std = self._validate(y_true, y_pred, sigma)
        order = np.argsort(std)
        errors = np.abs(true - pred)[order]
        sorted_std = std[order]

        total = 0.0
        counted = 0
        for chunk_err, chunk_std in zip(
            np.array_split(errors, self.n_bins),
            np.array_split(sorted_std, self.n_bins),
        ):
            if chunk_err.size == 0:
                continue
            rmse = float(np.sqrt(np.mean(chunk_err**2)))
            mean_std = float(np.mean(chunk_std))
            if mean_std > 0:
                total += abs(mean_std - rmse) / mean_std
                counted += 1
        return total / counted if counted else float("nan")

    def coverage_curve(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
        n_points: int = 20,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Observed coverage against nominal confidence level.

        For each nominal level, the fraction of true values falling
        inside the corresponding Gaussian interval. A perfectly
        calibrated model traces the diagonal; below it is overconfident.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)
        n_points : int, default 20
            Number of confidence levels sampled.

        Returns
        -------
        nominal, observed : ndarray of shape (n_points,)
        """
        from scipy.stats import norm

        true, pred, std = self._validate(y_true, y_pred, sigma)
        nominal = np.linspace(0.01, 0.99, n_points)
        safe_std = np.where(std > 0, std, np.finfo(float).eps)
        z = np.abs(true - pred) / safe_std
        observed = np.array(
            [float(np.mean(z <= norm.ppf(0.5 + level / 2))) for level in nominal]
        )
        return nominal, observed

    def miscalibration_area(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
        n_points: int = 20,
    ) -> float:
        """Area between the coverage curve and the ideal diagonal.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)
        n_points : int, default 20

        Returns
        -------
        float
            0 is perfect calibration; the maximum is about 0.5.

        References
        ----------
        - Tran, K. et al. (2020). Mach. Learn.: Sci. Technol., 1, 025006.
          https://doi.org/10.1088/2632-2153/ab7e1a
        """
        nominal, observed = self.coverage_curve(y_true, y_pred, sigma, n_points)
        return float(np.trapezoid(np.abs(observed - nominal), nominal))

    def spearman_error_correlation(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
    ) -> float:
        """Rank correlation between predicted uncertainty and absolute error.

        Measures *ranking* quality rather than calibration: whether the
        model knows which predictions are worse, regardless of whether
        the magnitudes are right. An estimator can score well here and
        still be badly calibrated (and vice versa), which is why both
        are reported.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)

        Returns
        -------
        float
            Spearman rho in [-1, 1]; higher is better.
        """
        from scipy.stats import spearmanr

        true, pred, std = self._validate(y_true, y_pred, sigma)
        if np.allclose(std, std[0]):
            # Constant uncertainty carries no ranking information at all.
            return 0.0
        rho = spearmanr(std, np.abs(true - pred)).statistic
        return float(rho) if np.isfinite(rho) else 0.0

    def report(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
    ) -> Dict[str, float]:
        """Full calibration report.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)

        Returns
        -------
        dict
            ``ence``, ``miscalibration_area``,
            ``spearman_error_correlation``, ``coverage_68``,
            ``coverage_95`` (observed coverage at the nominal 1- and
            2-sigma levels), ``mean_sigma`` and ``rmse``.
        """
        from scipy.stats import norm

        true, pred, std = self._validate(y_true, y_pred, sigma)
        safe_std = np.where(std > 0, std, np.finfo(float).eps)
        z = np.abs(true - pred) / safe_std
        return {
            "ence": self.ence(true, pred, std),
            "miscalibration_area": self.miscalibration_area(true, pred, std),
            "spearman_error_correlation": self.spearman_error_correlation(
                true, pred, std
            ),
            "coverage_68": float(np.mean(z <= norm.ppf(0.84))),
            "coverage_95": float(np.mean(z <= norm.ppf(0.975))),
            "mean_sigma": float(np.mean(std)),
            "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        }

    def plot_calibration(
        self,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        sigma: npt.ArrayLike,
        n_points: int = 20,
    ) -> "go.Figure":
        """Plot the coverage curve against the ideal diagonal.

        Parameters
        ----------
        y_true, y_pred, sigma : array-like of shape (n_samples,)
        n_points : int, default 20

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        nominal, observed = self.coverage_curve(y_true, y_pred, sigma, n_points)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="ideal",
                line={"dash": "dash"},
            )
        )
        fig.add_trace(
            go.Scatter(x=nominal, y=observed, mode="lines+markers", name="observed")
        )
        fig.update_layout(
            title="Uncertainty calibration",
            xaxis_title="Nominal confidence level",
            yaxis_title="Observed coverage",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        )
        return fig
