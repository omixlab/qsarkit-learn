"""Feature-importance explainers: SHAP, permutation, LIME, partial dependence."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt

from qsarkit.base.exceptions import ModelNotFittedError

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go

__all__ = [
    "PermutationImportance",
    "SHAPExplainer",
    "LIMEExplainer",
    "PartialDependence",
]


class PermutationImportance:
    """Feature importance by measuring the damage from shuffling a column.

    Model-agnostic and honest: a feature matters exactly to the degree
    that destroying its relationship with the target degrades held-out
    performance. Unlike a tree model's built-in ``feature_importances_``,
    which is computed on training data and is biased toward
    high-cardinality features, this is measured on data the model has
    not seen.

    The caveat worth knowing: with correlated descriptors — which is the
    normal situation in QSAR — permuting one of a correlated pair leaves
    the model able to recover the signal from its partner, so both look
    unimportant. Cluster correlated descriptors before interpreting, or
    read the result as "importance given the others are present".

    Parameters
    ----------
    n_repeats : int, default 10
        Shuffles per feature. More repeats reduce the variance of the
        estimate.
    scoring : str, optional
        scikit-learn scorer name. Defaults to the estimator's own score.
    random_state : int, optional
        Seed.
    n_jobs : int, optional
        Parallel jobs.

    Attributes
    ----------
    importances_mean_ : ndarray
        Mean drop in score per feature.
    importances_std_ : ndarray
        Standard deviation across repeats.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X, y = make_regression(n_samples=60, n_features=5, n_informative=2,
    ...                        random_state=0)
    >>> model = RandomForestRegressor(n_estimators=10, random_state=0).fit(X, y)
    >>> imp = PermutationImportance(random_state=0).fit(model, X, y)
    >>> imp.importances_mean_.shape
    (5,)

    References
    ----------
    - Breiman, L. (2001). "Random Forests." Mach. Learn., 45, 5-32.
      https://doi.org/10.1023/A:1010933404324
    - Fisher, A., Rudin, C. & Dominici, F. (2019). "All Models Are Wrong,
      but Many Are Useful: Learning a Variable's Importance." J. Mach.
      Learn. Res., 20(177), 1-81.
      https://jmlr.org/papers/v20/18-760.html
    - Strobl, C. et al. (2008). "Conditional Variable Importance for
      Random Forests." BMC Bioinformatics, 9, 307.
      https://doi.org/10.1186/1471-2105-9-307
    - scikit-learn permutation importance documentation:
      https://scikit-learn.org/stable/modules/permutation_importance.html
    """

    importances_mean_: npt.NDArray[np.float64]
    importances_std_: npt.NDArray[np.float64]

    def __init__(
        self,
        n_repeats: int = 10,
        scoring: Optional[str] = None,
        random_state: Optional[int] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        estimator: Any,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "PermutationImportance":
        """Measure importance on the supplied (ideally held-out) data.

        Parameters
        ----------
        estimator : fitted sklearn estimator
        X : array-like of shape (n_samples, n_features)
            Evaluation data. Use a test set, not the training set.
        y : array-like of shape (n_samples,)
        feature_names : sequence of str, optional

        Returns
        -------
        PermutationImportance
        """
        from sklearn.inspection import permutation_importance

        X_arr = np.asarray(X, dtype=np.float64)
        result = permutation_importance(
            estimator, X_arr, np.asarray(y),
            n_repeats=self.n_repeats, scoring=self.scoring,
            random_state=self.random_state, n_jobs=self.n_jobs,
        )
        self.importances_mean_ = np.asarray(
            result.importances_mean, dtype=np.float64
        )
        self.importances_std_ = np.asarray(
            result.importances_std, dtype=np.float64
        )
        self.feature_names_ = list(feature_names) if feature_names is not None else [
            f"x{i}" for i in range(X_arr.shape[1])
        ]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "importances_mean_"):
            raise ModelNotFittedError(
                "PermutationImportance must be fitted before use."
            )

    def to_dataframe(self, top_n: Optional[int] = None) -> "pd.DataFrame":
        """Importances as a ranked table.

        Parameters
        ----------
        top_n : int, optional
            Keep only the ``top_n`` most important features.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature``, ``importance``, ``std``, descending.
        """
        import pandas as pd

        self._check_fitted()
        frame = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "importance": self.importances_mean_,
                "std": self.importances_std_,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        return frame.head(top_n) if top_n else frame

    def plot(self, top_n: int = 20) -> "go.Figure":
        """Horizontal bar chart of the most important features.

        Parameters
        ----------
        top_n : int, default 20

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        frame = self.to_dataframe(top_n).iloc[::-1]
        fig = go.Figure(
            go.Bar(
                x=frame["importance"], y=frame["feature"], orientation="h",
                error_x={"type": "data", "array": frame["std"]},
            )
        )
        fig.update_layout(
            title="Permutation importance",
            xaxis_title="Mean decrease in score",
            yaxis_title="Feature",
        )
        return fig


class SHAPExplainer:
    """SHAP values: the game-theoretic attribution of a prediction.

    SHAP assigns each feature the payoff it contributes to a prediction,
    averaged over all orderings in which features could be added. That
    construction gives it the properties ad-hoc attributions lack — the
    contributions sum exactly to the prediction minus the base value
    (local accuracy), and a feature the model ignores always gets zero.

    The explainer is chosen from the model type: ``TreeExplainer`` for
    forests and boosted trees (exact and fast), ``LinearExplainer`` for
    linear models, and ``KernelExplainer`` otherwise (model-agnostic but
    slow, so it samples the background set).

    Requires the ``explainability`` extra.

    Parameters
    ----------
    model : fitted sklearn estimator
    explainer_type : {"auto", "tree", "linear", "kernel"}, default "auto"
        Which SHAP explainer to use.
    background : array-like, optional
        Background dataset for the kernel/linear explainers. A sample of
        the training data; 100 rows is usually enough.
    n_background : int, default 100
        How many background rows to sample when ``background`` is a full
        training set.
    random_state : int, optional

    Examples
    --------
    >>> import pytest
    >>> shap = pytest.importorskip("shap")  # doctest: +SKIP

    References
    ----------
    - Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to
      Interpreting Model Predictions." NeurIPS 2017.
      https://arxiv.org/abs/1705.07874
    - Lundberg, S. M. et al. (2020). "From Local Explanations to Global
      Understanding with Explainable AI for Trees." Nat. Mach. Intell.,
      2, 56-67. https://doi.org/10.1038/s42256-019-0138-9
    - Shapley, L. S. (1953). "A Value for n-Person Games." Contributions
      to the Theory of Games, 2(28), 307-317.
      https://doi.org/10.1515/9781400881970-018
    - Rodriguez-Perez, R. & Bajorath, J. (2020). "Interpretation of
      Machine Learning Models Using Shapley Values." J. Comput. Aided
      Mol. Des., 34, 1013-1026.
      https://doi.org/10.1007/s10822-020-00314-0
    """

    def __init__(
        self,
        model: Any,
        explainer_type: Literal["auto", "tree", "linear", "kernel"] = "auto",
        background: Optional[npt.ArrayLike] = None,
        n_background: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        self.model = model
        self.explainer_type = explainer_type
        self.background = background
        self.n_background = n_background
        self.random_state = random_state

    def _resolve_type(self) -> str:
        """Pick an explainer from the model class when set to auto."""
        if self.explainer_type != "auto":
            return self.explainer_type
        name = type(self.model).__name__.lower()
        if any(k in name for k in ("forest", "tree", "boost", "xgb", "lgbm", "gradient")):
            return "tree"
        if any(k in name for k in ("linear", "ridge", "lasso", "elastic", "logistic")):
            return "linear"
        return "kernel"

    def _sample_background(self) -> npt.NDArray[np.float64]:
        if self.background is None:
            raise ValueError(
                f"The {self._resolve_type()} explainer needs a `background` "
                "dataset (a sample of the training data)."
            )
        arr = np.asarray(self.background, dtype=np.float64)
        if len(arr) <= self.n_background:
            return arr
        rng = np.random.RandomState(self.random_state)
        return arr[rng.choice(len(arr), self.n_background, replace=False)]

    def _build(self) -> Any:
        from qsarkit.base import require

        shap = require("shap")
        kind = self._resolve_type()
        if kind == "tree":
            return shap.TreeExplainer(self.model)
        if kind == "linear":
            return shap.LinearExplainer(self.model, self._sample_background())
        if kind == "kernel":
            return shap.KernelExplainer(
                self.model.predict, self._sample_background()
            )
        raise ValueError(
            "explainer_type must be 'auto', 'tree', 'linear' or 'kernel', "
            f"got {self.explainer_type!r}."
        )

    @property
    def explainer(self) -> Any:
        """The lazily-constructed SHAP explainer."""
        if not hasattr(self, "_explainer"):
            self._explainer = self._build()
        return self._explainer

    def shap_values(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """SHAP values for each sample and feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            For multiclass models, the values for the positive class.
        """
        values = self.explainer.shap_values(np.asarray(X, dtype=np.float64))
        if isinstance(values, list):
            # Multiclass returns one array per class; report the last,
            # which is the positive class for a binary problem.
            values = values[-1]
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        return arr

    def global_importance(
        self, X: npt.ArrayLike, feature_names: Optional[Sequence[str]] = None
    ) -> "pd.DataFrame":
        """Mean absolute SHAP value per feature — a global ranking.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : sequence of str, optional

        Returns
        -------
        pandas.DataFrame
            Columns ``feature``, ``importance``, descending.
        """
        import pandas as pd

        values = self.shap_values(X)
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(values.shape[1])]
        )
        return (
            pd.DataFrame(
                {"feature": names, "importance": np.abs(values).mean(axis=0)}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def explain_one(
        self,
        x: npt.ArrayLike,
        feature_names: Optional[Sequence[str]] = None,
        top_n: int = 10,
    ) -> "pd.DataFrame":
        """Per-feature contributions to a single prediction.

        Parameters
        ----------
        x : array-like of shape (n_features,) or (1, n_features)
        feature_names : sequence of str, optional
        top_n : int, default 10
            Number of largest-magnitude contributions to report.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature``, ``value``, ``shap_value``, ordered by
            absolute contribution.
        """
        import pandas as pd

        arr = np.atleast_2d(np.asarray(x, dtype=np.float64))
        values = self.shap_values(arr)[0]
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(len(values))]
        )
        frame = pd.DataFrame(
            {"feature": names, "value": arr[0], "shap_value": values}
        )
        return (
            frame.reindex(frame["shap_value"].abs().sort_values(ascending=False).index)
            .head(top_n)
            .reset_index(drop=True)
        )

    def plot_importance(
        self,
        X: npt.ArrayLike,
        feature_names: Optional[Sequence[str]] = None,
        top_n: int = 20,
    ) -> "go.Figure":
        """Bar chart of mean absolute SHAP value per feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature_names : sequence of str, optional
        top_n : int, default 20

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        frame = self.global_importance(X, feature_names).head(top_n).iloc[::-1]
        fig = go.Figure(
            go.Bar(x=frame["importance"], y=frame["feature"], orientation="h")
        )
        fig.update_layout(
            title="SHAP feature importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
        )
        return fig


class LIMEExplainer:
    """LIME: explain one prediction with a local surrogate model.

    Perturbs the molecule's descriptors, records what the model predicts
    for each perturbation, and fits a sparse linear model to that local
    neighbourhood. The surrogate's coefficients are the explanation.

    Compared with SHAP, LIME is faster and easier to read but its
    explanations are not guaranteed to be self-consistent: the answer
    depends on the perturbation distribution and kernel width, and
    re-running can give a different story. Prefer SHAP where the
    attribution has to be defensible.

    Requires the ``explainability`` extra.

    Parameters
    ----------
    model : fitted sklearn estimator
    training_data : array-like
        Data defining the perturbation distribution.
    feature_names : sequence of str, optional
    mode : {"regression", "classification"}, default "regression"
    n_samples : int, default 5000
        Perturbations per explanation.
    random_state : int, optional

    Examples
    --------
    >>> import pytest
    >>> lime = pytest.importorskip("lime")  # doctest: +SKIP

    References
    ----------
    - Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I
      Trust You?: Explaining the Predictions of Any Classifier." KDD
      2016, 1135-1144. https://doi.org/10.1145/2939672.2939778
    - Alvarez-Melis, D. & Jaakkola, T. S. (2018). "On the Robustness of
      Interpretability Methods." arXiv:1806.08049.
      https://arxiv.org/abs/1806.08049
    """

    def __init__(
        self,
        model: Any,
        training_data: npt.ArrayLike,
        feature_names: Optional[Sequence[str]] = None,
        mode: Literal["regression", "classification"] = "regression",
        n_samples: int = 5000,
        random_state: Optional[int] = None,
    ) -> None:
        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.mode = mode
        self.n_samples = n_samples
        self.random_state = random_state

    @property
    def explainer(self) -> Any:
        """The lazily-constructed LIME tabular explainer."""
        if not hasattr(self, "_explainer"):
            from qsarkit.base import require

            require("lime")
            from lime.lime_tabular import (  # type: ignore[import-not-found]
                LimeTabularExplainer,
            )

            arr = np.asarray(self.training_data, dtype=np.float64)
            names = (
                list(self.feature_names)
                if self.feature_names is not None
                else [f"x{i}" for i in range(arr.shape[1])]
            )
            self._explainer = LimeTabularExplainer(
                arr, feature_names=names, mode=self.mode,
                random_state=self.random_state, discretize_continuous=False,
            )
        return self._explainer

    def explain_one(
        self, x: npt.ArrayLike, top_n: int = 10
    ) -> "pd.DataFrame":
        """Explain a single prediction.

        Parameters
        ----------
        x : array-like of shape (n_features,)
        top_n : int, default 10
            Number of features in the local surrogate.

        Returns
        -------
        pandas.DataFrame
            Columns ``feature``, ``weight``, ordered by absolute weight.
        """
        import pandas as pd

        predict = (
            self.model.predict_proba
            if self.mode == "classification"
            else self.model.predict
        )
        explanation = self.explainer.explain_instance(
            np.asarray(x, dtype=np.float64).ravel(),
            predict,
            num_features=top_n,
            num_samples=self.n_samples,
        )
        pairs = explanation.as_list()
        return pd.DataFrame(pairs, columns=["feature", "weight"])


class PartialDependence:
    """Partial dependence: the model's average response to one descriptor.

    Sweeps a descriptor across its range, averaging the model's
    prediction over the observed distribution of the others. The
    resulting curve shows the shape of the model's dependence — whether
    logP acts linearly, saturates, or has an optimum — which a single
    importance number cannot express.

    Its known blind spot is extrapolation: averaging over the marginal
    distribution evaluates the model at descriptor combinations that
    never occur (a molecule with MW 100 and 40 rotatable bonds), so read
    the curve only across the range where the data are dense.

    Parameters
    ----------
    model : fitted sklearn estimator
    grid_resolution : int, default 50
        Points sampled across each feature's range.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> X, y = make_regression(n_samples=50, n_features=4, random_state=0)
    >>> model = RandomForestRegressor(n_estimators=5, random_state=0).fit(X, y)
    >>> grid, avg = PartialDependence(model).compute(X, feature=0)
    >>> grid.shape == avg.shape
    True

    References
    ----------
    - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient
      Boosting Machine." Ann. Stat., 29(5), 1189-1232.
      https://doi.org/10.1214/aos/1013203451
    - Apley, D. W. & Zhu, J. (2020). "Visualizing the Effects of
      Predictor Variables in Black Box Supervised Learning Models."
      J. R. Stat. Soc. B, 82(4), 1059-1086.
      https://doi.org/10.1111/rssb.12377
    - scikit-learn partial dependence documentation:
      https://scikit-learn.org/stable/modules/partial_dependence.html
    """

    def __init__(self, model: Any, grid_resolution: int = 50) -> None:
        self.model = model
        self.grid_resolution = grid_resolution

    def compute(
        self, X: npt.ArrayLike, feature: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Partial-dependence curve for one feature.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature : int
            Column index.

        Returns
        -------
        grid : ndarray of shape (grid_resolution,)
            Feature values swept.
        average : ndarray of shape (grid_resolution,)
            Mean prediction at each grid point.
        """
        arr = np.asarray(X, dtype=np.float64)
        if not 0 <= feature < arr.shape[1]:
            raise ValueError(
                f"feature index {feature} out of range for {arr.shape[1]} features."
            )
        if self.grid_resolution < 2:
            raise ValueError(
                f"grid_resolution must be at least 2, got {self.grid_resolution}."
            )

        column = arr[:, feature]
        grid = np.linspace(column.min(), column.max(), self.grid_resolution)
        averages = np.empty(self.grid_resolution, dtype=np.float64)
        probe = arr.copy()
        for i, value in enumerate(grid):
            probe[:, feature] = value
            averages[i] = float(np.mean(self.model.predict(probe)))
        return grid, averages

    def plot(
        self,
        X: npt.ArrayLike,
        feature: int,
        feature_name: Optional[str] = None,
    ) -> "go.Figure":
        """Plot the partial-dependence curve.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        feature : int
        feature_name : str, optional

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        grid, average = self.compute(X, feature)
        name = feature_name or f"feature {feature}"
        fig = go.Figure(go.Scatter(x=grid, y=average, mode="lines"))
        # A rug of the observed values marks where the curve is supported
        # by data and where it is extrapolating.
        fig.add_trace(
            go.Scatter(
                x=np.asarray(X, dtype=np.float64)[:, feature],
                y=np.full(len(np.asarray(X)), average.min()),
                mode="markers",
                marker={"symbol": "line-ns-open", "size": 6},
                name="observed values",
            )
        )
        fig.update_layout(
            title=f"Partial dependence on {name}",
            xaxis_title=name,
            yaxis_title="Average prediction",
        )
        return fig
