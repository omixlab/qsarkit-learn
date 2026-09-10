"""Standard QSAR diagnostic plots, as Plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

__all__ = [
    "plot_predicted_vs_observed",
    "plot_residuals",
    "plot_williams",
    "plot_roc_curve",
    "plot_learning_curve",
    "plot_feature_importance",
    "figure_to_html",
]


def _as_arrays(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if true.shape != pred.shape:
        raise ValueError(
            f"y_true has shape {true.shape} but y_pred has {pred.shape}."
        )
    return true, pred


def plot_predicted_vs_observed(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    title: str = "Predicted vs observed",
    labels: Optional[Sequence[str]] = None,
) -> "go.Figure":
    """Scatter predictions against observations, with the identity line.

    The first plot to look at. A good model's points hug the diagonal;
    systematic curvature, fanning, or a slope visibly different from 1
    are all visible here and invisible in a single R² number.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Observed values.
    y_pred : array-like of shape (n_samples,)
        Predicted values.
    title : str
        Figure title.
    labels : sequence of str, optional
        Per-point hover labels, e.g. compound identifiers.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> fig = plot_predicted_vs_observed([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
    >>> len(fig.data)
    2

    References
    ----------
    - Gramatica, P. (2007). "Principles of QSAR Models Validation."
      QSAR Comb. Sci., 26(5), 694-701.
      https://doi.org/10.1002/qsar.200610151
    - Tropsha, A. (2010). "Best Practices for QSAR Model Development,
      Validation, and Exploitation." Mol. Inform., 29(6-7), 476-488.
      https://doi.org/10.1002/minf.201000061
    """
    import plotly.graph_objects as go

    true, pred = _as_arrays(y_true, y_pred)
    low = float(min(true.min(), pred.min()))
    high = float(max(true.max(), pred.max()))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[low, high], y=[low, high], mode="lines", name="identity",
            line={"dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=true, y=pred, mode="markers", name="compounds",
            text=list(labels) if labels is not None else None,
            hovertemplate=(
                "%{text}<br>observed %{x:.3f}<br>predicted %{y:.3f}<extra></extra>"
                if labels is not None
                else "observed %{x:.3f}<br>predicted %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title, xaxis_title="Observed", yaxis_title="Predicted"
    )
    return fig


def plot_residuals(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    title: str = "Residuals",
    standardized: bool = True,
) -> "go.Figure":
    """Plot residuals against predicted values.

    Where the predicted-vs-observed plot shows whether the model is
    right, this shows *how* it is wrong. Residuals should look like a
    structureless band around zero; a funnel means the error grows with
    potency, and a curve means a missing non-linear term.

    Parameters
    ----------
    y_true, y_pred : array-like of shape (n_samples,)
    title : str
        Figure title.
    standardized : bool, default True
        Divide residuals by their standard deviation, which puts the
        conventional +/-3 sigma warning lines on a meaningful scale.

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Gramatica, P. (2007). QSAR Comb. Sci., 26(5), 694-701.
      https://doi.org/10.1002/qsar.200610151
    - Draper, N. R. & Smith, H. (1998). "Applied Regression Analysis,"
      3rd ed. Wiley. https://doi.org/10.1002/9781118625590
    """
    import plotly.graph_objects as go

    true, pred = _as_arrays(y_true, y_pred)
    residual = true - pred
    if standardized:
        spread = float(residual.std(ddof=1)) if residual.size > 1 else 0.0
        residual = residual / spread if spread > 0 else residual

    fig = go.Figure(
        go.Scatter(x=pred, y=residual, mode="markers", name="residuals")
    )
    fig.add_hline(y=0.0)
    if standardized:
        for limit in (-3.0, 3.0):
            fig.add_hline(y=limit, line_dash="dash")
    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Standardized residual" if standardized else "Residual",
    )
    return fig


def plot_williams(
    leverage: npt.ArrayLike,
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    h_star: Optional[float] = None,
    residual_limit: float = 3.0,
    title: str = "Williams plot",
) -> "go.Figure":
    """Williams plot: standardized residuals against leverage.

    The standard regulatory diagnostic, and the one plot that separates
    the two distinct ways a prediction can be untrustworthy. Points to
    the right of ``h*`` are structural outliers the model is
    extrapolating to; points outside +/-3 sigma are response outliers the
    model simply gets wrong. A point in both regions should not be
    reported at all.

    Parameters
    ----------
    leverage : array-like of shape (n_samples,)
        Hat-matrix diagonal, e.g. from
        :class:`~qsarkit.applicability.LeverageAD`.
    y_true, y_pred : array-like of shape (n_samples,)
    h_star : float, optional
        Warning leverage. Defaults to the conventional ``3(p+1)/n``
        estimated from the data when omitted.
    residual_limit : float, default 3.0
        Standardized-residual warning level.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Gramatica, P. (2007). QSAR Comb. Sci., 26(5), 694-701.
      https://doi.org/10.1002/qsar.200610151
    - Atkinson, A. C. (1985). "Plots, Transformations and Regression."
      Oxford University Press.
    - OECD (2007). Guidance Document No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    """
    import plotly.graph_objects as go

    true, pred = _as_arrays(y_true, y_pred)
    h = np.asarray(leverage, dtype=np.float64).ravel()
    if h.shape != true.shape:
        raise ValueError(
            f"leverage has shape {h.shape} but y_true has {true.shape}."
        )

    residual = true - pred
    spread = float(residual.std(ddof=1)) if residual.size > 1 else 0.0
    standardized = residual / spread if spread > 0 else np.zeros_like(residual)

    fig = go.Figure(
        go.Scatter(
            x=h, y=standardized, mode="markers", name="compounds",
            text=[f"index {i}" for i in range(len(h))],
            hovertemplate="%{text}<br>h=%{x:.3f}<br>std resid=%{y:.2f}<extra></extra>",
        )
    )
    if h_star is None and h.size:
        # Recover p from the mean leverage: sum(h) = p for a hat matrix.
        p = max(1.0, float(h.sum()))
        h_star = float(3.0 * p / len(h))
    if h_star is not None:
        fig.add_vline(x=h_star, line_dash="dash", annotation_text="h*")
    for limit in (-residual_limit, residual_limit):
        fig.add_hline(y=limit, line_dash="dash")
    fig.update_layout(
        title=title, xaxis_title="Leverage (h)", yaxis_title="Standardized residual"
    )
    return fig


def plot_roc_curve(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    title: str = "ROC curve",
) -> "go.Figure":
    """Receiver operating characteristic curve with its AUC.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Scores or predicted probabilities.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Fawcett, T. (2006). "An Introduction to ROC Analysis." Pattern
      Recognit. Lett., 27(8), 861-874.
      https://doi.org/10.1016/j.patrec.2005.10.010
    - Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
      Methods." J. Chem. Inf. Model., 47(2), 488-508.
      https://doi.org/10.1021/ci600426e
    """
    import plotly.graph_objects as go
    from sklearn.metrics import auc, roc_curve

    labels = np.asarray(y_true).ravel()
    scores = np.asarray(y_score, dtype=np.float64).ravel()
    if labels.shape != scores.shape:
        raise ValueError(
            f"y_true has shape {labels.shape} but y_score has {scores.shape}."
        )
    fpr, tpr, _ = roc_curve(labels, scores)
    area = float(auc(fpr, tpr))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash"}
        )
    )
    fig.add_trace(
        go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC = {area:.3f})")
    )
    fig.update_layout(
        title=title,
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        xaxis_range=[0, 1],
        yaxis_range=[0, 1],
    )
    return fig


def plot_learning_curve(
    train_sizes: npt.ArrayLike,
    train_scores: npt.ArrayLike,
    test_scores: npt.ArrayLike,
    title: str = "Learning curve",
) -> "go.Figure":
    """Plot training and validation score against training-set size.

    Answers whether more data would help. A gap that stays wide as the
    curves flatten means the model is over-fitting and needs
    regularization, not compounds; curves still rising means measuring
    more compounds is the better investment.

    Parameters
    ----------
    train_sizes : array-like of shape (n_points,)
    train_scores : array-like of shape (n_points,) or (n_points, n_folds)
    test_scores : array-like of shape (n_points,) or (n_points, n_folds)
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Perlich, C. (2010). "Learning Curves in Machine Learning." In
      Encyclopedia of Machine Learning. Springer.
      https://doi.org/10.1007/978-0-387-30164-8_452
    - scikit-learn learning curve documentation:
      https://scikit-learn.org/stable/modules/learning_curve.html
    """
    import plotly.graph_objects as go

    sizes = np.asarray(train_sizes, dtype=np.float64).ravel()
    fig = go.Figure()
    for scores, name in ((train_scores, "training"), (test_scores, "validation")):
        arr = np.asarray(scores, dtype=np.float64)
        mean = arr.mean(axis=1) if arr.ndim == 2 else arr
        error = (
            {"type": "data", "array": arr.std(axis=1)} if arr.ndim == 2 else None
        )
        if mean.shape != sizes.shape:
            raise ValueError(
                f"{name} scores have shape {mean.shape} but train_sizes has "
                f"{sizes.shape}."
            )
        fig.add_trace(
            go.Scatter(
                x=sizes, y=mean, mode="lines+markers", name=name, error_y=error
            )
        )
    fig.update_layout(
        title=title, xaxis_title="Training set size", yaxis_title="Score"
    )
    return fig


def plot_feature_importance(
    names: Sequence[str],
    importances: npt.ArrayLike,
    errors: Optional[npt.ArrayLike] = None,
    top_n: int = 20,
    title: str = "Feature importance",
) -> "go.Figure":
    """Horizontal bar chart of the most important descriptors.

    Parameters
    ----------
    names : sequence of str
        Feature names.
    importances : array-like of shape (n_features,)
    errors : array-like of shape (n_features,), optional
        Error bars, e.g. the standard deviation across permutation
        repeats.
    top_n : int, default 20
        Number of features shown.
    title : str
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure

    References
    ----------
    - Breiman, L. (2001). "Random Forests." Mach. Learn., 45, 5-32.
      https://doi.org/10.1023/A:1010933404324
    - Lundberg, S. M. et al. (2020). "From Local Explanations to Global
      Understanding with Explainable AI for Trees." Nat. Mach. Intell.,
      2, 56-67. https://doi.org/10.1038/s42256-019-0138-9
    """
    import plotly.graph_objects as go

    values = np.asarray(importances, dtype=np.float64).ravel()
    if len(names) != len(values):
        raise ValueError(
            f"names has length {len(names)} but importances has {len(values)}."
        )
    order = np.argsort(values)[-top_n:]
    error_bars = None
    if errors is not None:
        error_array = np.asarray(errors, dtype=np.float64).ravel()
        error_bars = {"type": "data", "array": error_array[order]}

    fig = go.Figure(
        go.Bar(
            x=values[order],
            y=[names[i] for i in order],
            orientation="h",
            error_x=error_bars,
        )
    )
    fig.update_layout(title=title, xaxis_title="Importance", yaxis_title="Feature")
    return fig


def figure_to_html(
    figure: "go.Figure", include_plotlyjs: Union[str, bool] = "cdn"
) -> str:
    """Render a figure as an embeddable HTML fragment.

    Parameters
    ----------
    figure : plotly.graph_objects.Figure
    include_plotlyjs : str or bool, default "cdn"
        Passed to Plotly. ``"cdn"`` keeps reports small; ``True`` inlines
        the library so the report works offline; ``False`` omits it,
        which is what every figure after the first in a document wants.

    Returns
    -------
    str
        An HTML ``<div>`` fragment.
    """
    return str(
        figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs)
    )
