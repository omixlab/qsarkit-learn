"""Applicability-domain reporting: coverage, accuracy-vs-coverage, Williams plot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

import numpy as np
import numpy.typing as npt

from qsarkit.applicability._domains import BaseApplicabilityDomain

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go

__all__ = ["ADAnalyzer"]


class ADAnalyzer:
    """Quantify what an applicability domain buys you in prediction accuracy.

    A domain definition is only useful if excluding the compounds it
    rejects actually improves accuracy on the ones it keeps. This class
    measures exactly that trade-off: as the domain is tightened, coverage
    falls and error should fall with it. A domain whose accuracy curve is
    flat is not carrying information, however statistically principled it
    looks.

    Parameters
    ----------
    domain : BaseApplicabilityDomain
        A fitted (or fittable) applicability-domain estimator.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.applicability import KNNApplicabilityDomain
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(60, 3))
    >>> analyzer = ADAnalyzer(KNNApplicabilityDomain(n_neighbors=3)).fit(X)
    >>> 0.0 <= analyzer.coverage(X) <= 1.0
    True

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of (Q)SAR
      Models." OECD Series on Testing and Assessment No. 69,
      ENV/JM/MONO(2007)2. https://doi.org/10.1787/9789264085442-en
    - Dragos, H., Gilles, M. & Alexandre, V. (2009). "Predicting the
      Predictability: A Unified Approach to the Applicability Domain
      Problem of QSAR Models." J. Chem. Inf. Model., 49(7), 1762-1776.
      https://doi.org/10.1021/ci9000579
    - Sheridan, R. P. (2012). "Three Useful Dimensions for Domain
      Applicability in QSAR Models Using Random Forest." J. Chem. Inf.
      Model., 52(3), 814-823. https://doi.org/10.1021/ci300004n
    """

    def __init__(self, domain: BaseApplicabilityDomain) -> None:
        self.domain = domain

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "ADAnalyzer":
        """Fit the wrapped domain on training descriptors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        ADAnalyzer
        """
        self.domain.fit(X, y)
        return self

    def coverage(self, X: npt.ArrayLike) -> float:
        """Fraction of ``X`` inside the domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        float
        """
        return self.domain.coverage(X)

    def report(
        self,
        X: npt.ArrayLike,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> Dict[str, float]:
        """Compare in-domain and out-of-domain prediction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Descriptors of the evaluated set.
        y_true : array-like of shape (n_samples,)
            Observed values.
        y_pred : array-like of shape (n_samples,)
            Model predictions.

        Returns
        -------
        dict
            ``coverage``, ``n_inside``, ``n_outside``, ``rmse_inside``,
            ``rmse_outside``, ``mae_inside``, ``mae_outside`` and
            ``rmse_ratio`` (outside/inside; > 1 means the domain is
            doing its job). Error entries are ``nan`` when the
            corresponding subset is empty.
        """
        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        if true.shape != pred.shape:
            raise ValueError(
                f"y_true has shape {true.shape} but y_pred has {pred.shape}."
            )
        inside = self.domain.predict(X)
        if inside.shape[0] != true.shape[0]:
            raise ValueError(
                f"X has {inside.shape[0]} samples but y_true has {true.shape[0]}."
            )

        def _rmse(mask: npt.NDArray[np.bool_]) -> float:
            if not mask.any():
                return float("nan")
            return float(np.sqrt(np.mean((true[mask] - pred[mask]) ** 2)))

        def _mae(mask: npt.NDArray[np.bool_]) -> float:
            if not mask.any():
                return float("nan")
            return float(np.mean(np.abs(true[mask] - pred[mask])))

        rmse_in, rmse_out = _rmse(inside), _rmse(~inside)
        return {
            "coverage": float(np.mean(inside)),
            "n_inside": int(inside.sum()),
            "n_outside": int((~inside).sum()),
            "rmse_inside": rmse_in,
            "rmse_outside": rmse_out,
            "mae_inside": _mae(inside),
            "mae_outside": _mae(~inside),
            "rmse_ratio": (
                rmse_out / rmse_in if rmse_in and np.isfinite(rmse_out) else float("nan")
            ),
        }

    def accuracy_vs_coverage(
        self,
        X: npt.ArrayLike,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        n_points: int = 20,
    ) -> "pd.DataFrame":
        """Trace prediction error as the domain is progressively tightened.

        Compounds are ranked by how far outside the domain they score;
        the curve then reports RMSE over the most-confident fraction at
        a series of coverage levels. A useful domain gives a curve that
        rises monotonically from left (strictest) to right (all compounds).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_true, y_pred : array-like of shape (n_samples,)
        n_points : int, default 20
            Number of coverage levels sampled.

        Returns
        -------
        pandas.DataFrame
            Columns ``coverage``, ``n_samples``, ``rmse``, ``mae``.
        """
        import pandas as pd

        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        scores = self.domain.score_samples(X)
        if not (len(scores) == len(true) == len(pred)):
            raise ValueError("X, y_true and y_pred must all have the same length.")
        if n_points < 1:
            raise ValueError(f"n_points must be positive, got {n_points}.")

        order = np.argsort(scores, kind="stable")
        n = len(order)
        counts = np.unique(
            np.clip(np.linspace(1, n, num=min(n_points, n)).astype(int), 1, n)
        )
        rows = []
        for k in counts:
            idx = order[:k]
            residual = true[idx] - pred[idx]
            rows.append(
                {
                    "coverage": k / n,
                    "n_samples": int(k),
                    "rmse": float(np.sqrt(np.mean(residual**2))),
                    "mae": float(np.mean(np.abs(residual))),
                }
            )
        return pd.DataFrame(rows, columns=["coverage", "n_samples", "rmse", "mae"])

    def plot_accuracy_vs_coverage(
        self,
        X: npt.ArrayLike,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        n_points: int = 20,
    ) -> "go.Figure":
        """Plot the accuracy-vs-coverage curve.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_true, y_pred : array-like of shape (n_samples,)
        n_points : int, default 20

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        curve = self.accuracy_vs_coverage(X, y_true, y_pred, n_points)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=curve["coverage"], y=curve["rmse"], mode="lines+markers", name="RMSE"
            )
        )
        fig.add_trace(
            go.Scatter(
                x=curve["coverage"], y=curve["mae"], mode="lines+markers", name="MAE"
            )
        )
        fig.update_layout(
            title=f"Accuracy vs coverage ({type(self.domain).__name__})",
            xaxis_title="Coverage (fraction of compounds retained)",
            yaxis_title="Prediction error",
        )
        return fig

    def williams_plot(
        self,
        X: npt.ArrayLike,
        y_true: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        residual_limit: float = 3.0,
    ) -> "go.Figure":
        """Williams plot: standardized residuals against leverage.

        The standard regulatory diagnostic. Points to the right of the
        vertical ``h*`` line are structural outliers (the model is
        extrapolating); points outside the horizontal +/-3 sigma lines are
        response outliers (the model is wrong). Both together mark
        predictions that should not be relied on.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y_true, y_pred : array-like of shape (n_samples,)
        residual_limit : float, default 3.0
            Standardized-residual warning level, in standard deviations.

        Returns
        -------
        plotly.graph_objects.Figure

        References
        ----------
        - Gramatica, P. (2007). QSAR Comb. Sci., 26(5), 694-701.
          https://doi.org/10.1002/qsar.200610151
        - OECD (2007). Guidance Document No. 69, ENV/JM/MONO(2007)2.
          https://doi.org/10.1787/9789264085442-en
        """
        import plotly.graph_objects as go

        from qsarkit.applicability._domains import LeverageAD

        true = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        residual = true - pred
        std = residual.std(ddof=1) if residual.size > 1 else 0.0
        standardized = residual / std if std > 0 else np.zeros_like(residual)

        leverage_ad = LeverageAD().fit(X)
        leverage = leverage_ad.score_samples(X)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=leverage,
                y=standardized,
                mode="markers",
                name="compounds",
                text=[f"index {i}" for i in range(len(standardized))],
                hovertemplate="%{text}<br>h=%{x:.3f}<br>std resid=%{y:.2f}<extra></extra>",
            )
        )
        fig.add_vline(
            x=leverage_ad.threshold_,
            line_dash="dash",
            annotation_text="h*",
        )
        fig.add_hline(y=residual_limit, line_dash="dash")
        fig.add_hline(y=-residual_limit, line_dash="dash")
        fig.update_layout(
            title="Williams plot",
            xaxis_title="Leverage (h)",
            yaxis_title="Standardized residual",
        )
        return fig
