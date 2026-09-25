"""Standard QSAR diagnostic plots, as Plotly figures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # pragma: no cover
    import plotly.graph_objects as go

__all__ = [
    "plot_calibration_curve",
    "plot_qq",
    "plot_threshold_sweep",
    "plot_precision_recall",
    "plot_atom_contributions",
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

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_residuals
    >>> rng = np.random.default_rng(0)
    >>> truth = rng.normal(size=50)
    >>> figure = plot_residuals(truth, truth + rng.normal(scale=0.2, size=50))
    >>> type(figure).__name__
    'Figure'

    Standardized residuals put the conventional +/-3 sigma lines on a
    fixed scale, so an outlier is visible without knowing the endpoint's
    units:

    >>> raw = plot_residuals(truth, truth, standardized=False)
    >>> raw.layout.yaxis.title.text is not None
    True

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

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.applicability import LeverageAD
    >>> from qsarkit.reporting import plot_williams
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(40, 4))
    >>> truth = X[:, 0] * 2 + rng.normal(scale=0.2, size=40)
    >>> leverage = LeverageAD().fit(X).score_samples(X)
    >>> figure = plot_williams(leverage, truth, truth + rng.normal(scale=0.2, size=40))
    >>> type(figure).__name__
    'Figure'

    The two lines are what make it a Williams plot: ``h_star`` marks the
    leverage threshold and ``residual_limit`` the residual one, so a point
    beyond either is influential, an outlier, or both.

    >>> figure = plot_williams(leverage, truth, truth, h_star=0.3, residual_limit=2.5)
    >>> len(figure.layout.shapes) >= 2
    True

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

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_roc_curve
    >>> scores = np.linspace(1.0, 0.0, 100)
    >>> labels = np.zeros(100); labels[:10] = 1     # actives ranked first
    >>> figure = plot_roc_curve(labels, scores)
    >>> type(figure).__name__
    'Figure'
    >>> figure.data[1].name                          # AUC shown in the legend
    'ROC (AUC = 1.000)'

    On an imbalanced screening set prefer
    :func:`plot_precision_recall`: ROC's specificity axis is dominated by
    the inactive majority, so a model can look excellent while its
    top-ranked compounds are mostly false positives.

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

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_learning_curve
    >>> sizes = np.array([10, 20, 40, 80])
    >>> train = np.array([[0.99, 0.98], [0.97, 0.96], [0.95, 0.94], [0.93, 0.92]])
    >>> test = np.array([[0.40, 0.45], [0.55, 0.58], [0.68, 0.70], [0.74, 0.75]])
    >>> figure = plot_learning_curve(sizes, train, test)
    >>> type(figure).__name__
    'Figure'

    Read the *gap*, not the level. A training score far above the
    validation score that stays apart as data is added means the model is
    memorizing; converging curves mean more data would help.

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

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_feature_importance
    >>> names = ["MolWt", "MolLogP", "TPSA", "NumHDonors"]
    >>> figure = plot_feature_importance(names, [0.4, 0.3, 0.2, 0.1])
    >>> type(figure).__name__
    'Figure'

    Error bars turn a ranking into a claim you can judge. Two features
    whose intervals overlap are not distinguishable by this much data:

    >>> figure = plot_feature_importance(
    ...     names, [0.4, 0.3, 0.2, 0.1], errors=[0.05, 0.08, 0.06, 0.03])
    >>> len(figure.data)
    1

    ``top_n`` truncates a long list, which is the usual case with
    fingerprints:

    >>> many = [f"bit_{i}" for i in range(500)]
    >>> figure = plot_feature_importance(many, np.linspace(0, 1, 500), top_n=10)
    >>> len(figure.data[0].y)
    10

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

def plot_calibration_curve(
    y_true: npt.ArrayLike,
    y_prob: npt.ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
    title: str = "Calibration (reliability diagram)",
) -> "go.Figure":
    """Reliability diagram: observed frequency against predicted probability.

    A perfectly calibrated classifier lies on the diagonal. Above it the
    model is under-confident, below it over-confident. Bubble size shows
    how many compounds each point rests on, because a bin holding three
    compounds says very little.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_prob : array-like of shape (n_samples,)
        Predicted probability of the positive class.
    n_bins : int, default 10
        Number of bins.
    strategy : {"uniform", "quantile"}, default "uniform"
        Binning strategy; see
        :func:`~qsarkit.metrics.calibration_curve`.
    title : str, default "Calibration (reliability diagram)"

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_calibration_curve
    >>> rng = np.random.default_rng(0)
    >>> p = rng.uniform(size=500)
    >>> y = (rng.uniform(size=500) < p).astype(int)
    >>> figure = plot_calibration_curve(y, p)
    >>> type(figure).__name__
    'Figure'

    Examples
    --------
    >>> from qsarkit.reporting import figure_to_html, plot_roc_curve
    >>> import numpy as np
    >>> figure = plot_roc_curve(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    >>> html = figure_to_html(figure)
    >>> "plotly" in html
    True

    Plotly's JavaScript must be included exactly once per page, so pass
    ``include_plotlyjs=False`` for every figure after the first:

    >>> first = figure_to_html(figure)
    >>> rest = figure_to_html(figure, include_plotlyjs=False)
    >>> len(first) > len(rest)
    True

    References
    ----------
    - Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting Good
      Probabilities with Supervised Learning." ICML 2005, 625-632.
      https://doi.org/10.1145/1102351.1102430
    """
    import plotly.graph_objects as go

    from qsarkit.metrics import calibration_curve, expected_calibration_error

    curve = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)  # type: ignore[arg-type]
    ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins, strategy=strategy)  # type: ignore[arg-type]

    counts = curve["counts"]
    # Area-proportional markers, floored so a sparse bin is still visible.
    sizes = 8.0 + 22.0 * np.sqrt(counts / counts.max()) if counts.size else counts

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="perfect calibration",
            line={"dash": "dash", "color": "#9AA5B1"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=curve["mean_predicted"],
            y=curve["observed_frequency"],
            mode="lines+markers",
            name="model",
            marker={"size": sizes, "color": "#41618F"},
            customdata=counts,
            hovertemplate=(
                "predicted %{x:.3f}<br>observed %{y:.3f}"
                "<br>%{customdata:.0f} compounds<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{title} — ECE {ece:.3f}",
        xaxis_title="mean predicted probability",
        yaxis_title="observed frequency of the positive class",
        xaxis={"range": [-0.02, 1.02]},
        yaxis={"range": [-0.02, 1.02], "scaleanchor": "x", "scaleratio": 1},
    )
    return figure


def plot_qq(
    residuals: npt.ArrayLike,
    title: str = "Normal Q-Q plot of residuals",
) -> "go.Figure":
    r"""Normal Q-Q plot, for checking the assumption behind every RMSE.

    Points on the line mean normal residuals. An S-shape means heavy
    tails; a bend at one end means skew, usually from a few badly
    mispredicted compounds that :math:`R^2` will not name.

    Parameters
    ----------
    residuals : array-like of shape (n_samples,)
        Observed minus predicted.
    title : str, default "Normal Q-Q plot of residuals"

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_qq
    >>> rng = np.random.default_rng(0)
    >>> figure = plot_qq(rng.normal(size=200))
    >>> type(figure).__name__
    'Figure'

    References
    ----------
    - Wilk, M. B. & Gnanadesikan, R. (1968). "Probability Plotting Methods
      for the Analysis of Data." Biometrika, 55(1), 1-17.
      https://doi.org/10.1093/biomet/55.1.1
    """
    import plotly.graph_objects as go

    from qsarkit.metrics import qq_data

    data = qq_data(residuals)
    theoretical = data["theoretical_quantiles"]
    slope, intercept = data["reference_line"]

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=slope * theoretical + intercept,
            mode="lines",
            name="normal reference",
            line={"dash": "dash", "color": "#9AA5B1"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=data["sample_quantiles"],
            mode="markers",
            name="residuals",
            marker={"size": 6, "color": "#41618F"},
        )
    )
    figure.update_layout(
        title=title,
        xaxis_title="theoretical quantile (standard normal)",
        yaxis_title="observed quantile (standardized residual)",
    )
    return figure


def plot_threshold_sweep(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    criteria: Optional[Sequence[str]] = None,
    title: str = "Criterion against decision threshold",
    pos_label: Optional[Any] = None,
) -> "go.Figure":
    """How each selection criterion varies with the decision threshold.

    Shows what the conventional 0.5 cut costs, and whether the optimum is
    a sharp peak or a broad plateau -- a broad one means the exact
    threshold hardly matters, which is worth knowing before tuning it.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities.
    criteria : sequence of str, optional
        Which curves to draw. Defaults to ``("youden_j", "mcc", "f1",
        "balanced_accuracy")``; any key of
        :func:`~qsarkit.metrics.threshold_sweep` is allowed.
    title : str, default "Criterion against decision threshold"
    pos_label : optional
        Which label is the positive class. Required for string labels.

    Returns
    -------
    plotly.graph_objects.Figure

    Raises
    ------
    ValueError
        If a requested criterion is not produced by the sweep.

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_threshold_sweep
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(300, dtype=int); y[:30] = 1
    >>> scores = rng.beta(2, 8, size=300) + y * 0.3
    >>> figure = plot_threshold_sweep(y, scores)
    >>> type(figure).__name__
    'Figure'

    References
    ----------
    - Chicco, D. & Jurman, G. (2020). "The Advantages of the Matthews
      Correlation Coefficient (MCC) over F1 Score and Accuracy."
      BMC Genomics, 21, 6. https://doi.org/10.1186/s12864-019-6413-7
    """
    import plotly.graph_objects as go

    from qsarkit.metrics import threshold_sweep

    sweep = threshold_sweep(y_true, y_score, pos_label=pos_label)  # type: ignore[arg-type]
    wanted = tuple(criteria) if criteria else (
        "youden_j",
        "mcc",
        "f1",
        "balanced_accuracy",
    )
    unknown = [name for name in wanted if name not in sweep]
    if unknown:
        raise ValueError(
            f"Unknown criteria {unknown}. Available: "
            f"{sorted(k for k in sweep if k != 'thresholds')}."
        )

    figure = go.Figure()
    for name in wanted:
        figure.add_trace(
            go.Scatter(
                x=sweep["thresholds"],
                y=sweep[name],
                mode="lines",
                name=name.replace("_", " "),
            )
        )
    figure.add_vline(
        x=0.5,
        line={"dash": "dot", "color": "#9AA5B1"},
        annotation_text="default 0.5",
    )
    figure.update_layout(
        title=title,
        xaxis_title="decision threshold",
        yaxis_title="criterion value",
    )
    return figure


def plot_precision_recall(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    title: str = "Precision-recall curve",
    pos_label: Optional[Any] = None,
) -> "go.Figure":
    """Precision-recall curve, with the base rate as the honest baseline.

    Preferable to ROC on imbalanced data: ROC's specificity axis is
    dominated by the inactive majority, so a model can look excellent
    while its top-ranked compounds are mostly false positives. The
    baseline here is the base rate, which is what random ranking achieves.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary labels.
    y_score : array-like of shape (n_samples,)
        Scores or probabilities.
    title : str, default "Precision-recall curve"
    pos_label : optional
        Which label is the positive class. Required for string labels; see
        :func:`~qsarkit.metrics.threshold_sweep`.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from qsarkit.reporting import plot_precision_recall
    >>> rng = np.random.default_rng(0)
    >>> y = np.zeros(300, dtype=int); y[:30] = 1
    >>> scores = rng.beta(2, 8, size=300) + y * 0.3
    >>> figure = plot_precision_recall(y, scores)
    >>> type(figure).__name__
    'Figure'

    References
    ----------
    - Saito, T. & Rehmsmeier, M. (2015). "The Precision-Recall Plot Is
      More Informative than the ROC Plot When Evaluating Binary
      Classifiers on Imbalanced Datasets." PLoS ONE, 10(3), e0118432.
      https://doi.org/10.1371/journal.pone.0118432
    - Davis, J. & Goadrich, M. (2006). "The Relationship Between
      Precision-Recall and ROC Curves." ICML 2006, 233-240.
      https://doi.org/10.1145/1143844.1143874
    """
    import plotly.graph_objects as go

    from qsarkit.metrics import pr_auc, threshold_sweep
    from qsarkit.metrics._thresholds import _check_binary

    # Resolve the positive class once, and use the 0/1 form everywhere after:
    # pr_auc takes integer labels, and re-deriving "which class is positive"
    # in two places is how the two ends of a plot come to disagree.
    labels, scores = _check_binary(y_true, y_score, pos_label=pos_label)
    sweep = threshold_sweep(labels, scores)
    base_rate = float(labels.mean())
    area = float(pr_auc(labels, scores))

    # Order by recall so the curve is drawn left to right.
    order = np.argsort(sweep["sensitivity"])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[base_rate, base_rate],
            mode="lines",
            name=f"random ({base_rate:.3f})",
            line={"dash": "dash", "color": "#9AA5B1"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["sensitivity"][order],
            y=sweep["precision"][order],
            mode="lines",
            name="model",
            line={"color": "#41618F"},
            customdata=sweep["thresholds"][order],
            hovertemplate=(
                "recall %{x:.3f}<br>precision %{y:.3f}"
                "<br>threshold %{customdata:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=f"{title} — PR-AUC {area:.3f}",
        xaxis_title="recall (sensitivity)",
        yaxis_title="precision",
        xaxis={"range": [-0.02, 1.02]},
        yaxis={"range": [-0.02, 1.02]},
    )
    return figure


def plot_atom_contributions(
    mol: Any,
    atom_weights: npt.ArrayLike,
    title: str = "Per-atom contributions",
    size: Tuple[int, int] = (450, 450),
) -> str:
    """Draw atom-level attributions on the structure, as an RDKit SVG.

    A thin wrapper over
    :func:`~qsarkit.explainability.draw_atom_weights`, provided here so a
    report can assemble every figure from one module. Unlike the other
    plotting functions this returns SVG text rather than a Plotly figure,
    because the depiction is a chemical drawing and RDKit draws those
    properly.

    Parameters
    ----------
    mol : Mol
        The molecule.
    atom_weights : array-like of shape (n_atoms,)
        Per-atom attribution, e.g. from
        :class:`~qsarkit.explainability.AttributionAtomMapper`.
    title : str, default "Per-atom contributions"
        Prepended as an SVG ``<title>``, which becomes the tooltip.
    size : tuple of int, default (450, 450)
        Image size in pixels.

    Returns
    -------
    str
        SVG text, ready to embed in an HTML report or display in a
        notebook.

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from qsarkit.reporting import plot_atom_contributions
    >>> mol = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
    >>> svg = plot_atom_contributions(mol, np.linspace(-1, 1, mol.GetNumAtoms()))
    >>> "<svg" in svg
    True

    References
    ----------
    - Riniker, S. & Landrum, G. A. (2013). "Similarity Maps." J.
      Cheminform., 5, 43. https://doi.org/10.1186/1758-2946-5-43
    """
    from qsarkit.explainability import draw_atom_weights

    svg = draw_atom_weights(mol, atom_weights, size=size, fmt="svg")
    # Insert an accessible title, which RDKit's drawer does not emit.
    marker = ">"
    index = svg.find("<svg")
    if index != -1:
        close = svg.find(marker, index)
        if close != -1:
            svg = (
                svg[: close + 1]
                + f"<title>{title}</title>"
                + svg[close + 1 :]
            )
    return svg
