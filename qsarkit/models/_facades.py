"""Unified ``name``-dispatching QSAR regressor/classifier facades."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.multiclass import unique_labels

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.models._gaussian_process import GaussianProcessQSAR
from qsarkit.models._neural_network import NeuralNetworkQSAR
from qsarkit.models._pls import PLSRegressor
from qsarkit.models._random_forest import RandomForestQSAR
from qsarkit.models._svm import SVMQSAR
from qsarkit.models._tanimoto_kernel import TanimotoKernel
from qsarkit.neighbors import JaccardKNeighborsClassifier, JaccardKNeighborsRegressor

__all__ = ["QSARRegressor", "QSARClassifier"]

_BuiltinName = Literal[
    "rf", "svm", "gbm", "xgboost", "lightgbm", "knn", "pls",
    "ridge", "lasso", "elasticnet", "mlp", "gp",
]
#: Either one of the built-in names, or any scikit-learn-compatible
#: estimator class or instance (XGBoost, LightGBM, CatBoost, your own).
_ModelName = Union[_BuiltinName, Any]

_VALID_NAMES: Tuple[str, ...] = (
    "rf", "svm", "gbm", "xgboost", "lightgbm", "knn", "pls",
    "ridge", "lasso", "elasticnet", "mlp", "gp",
)


def _with_defaults(params: Dict[str, Any], **defaults: Any) -> Dict[str, Any]:
    """The facade's defaults, overridable by the caller's ``model_params``.

    ``model_params`` is the documented way to configure a backend, so passing
    a keyword the facade also sets has to work. Splatting both into the
    constructor instead raises ``TypeError: got multiple values for keyword
    argument``, which is a confusing way to reject a documented use --
    ``QSARClassifier("rf", model_params={"n_estimators": 100})`` hit exactly
    that. The caller's value wins.
    """
    merged = dict(defaults)
    merged.update(params)
    return merged


def _invalid_name_error(name: str) -> ValueError:
    return ValueError(f"name must be one of {_VALID_NAMES}, got {name!r}.")


def _is_estimator_like(obj: Any) -> bool:
    """True for anything exposing the scikit-learn ``fit``/``predict`` pair.

    Deliberately duck-typed rather than an ``isinstance`` check against
    :class:`~sklearn.base.BaseEstimator`: XGBoost, LightGBM and CatBoost
    all ship estimators that follow the protocol without inheriting from
    it, and so do plenty of in-house wrappers.
    """
    target = obj if isinstance(obj, type) else type(obj)
    return callable(getattr(target, "fit", None)) and callable(
        getattr(target, "predict", None)
    )


def _describe_estimator(obj: Any) -> str:
    """Readable name for an estimator name, class or instance."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, type):
        return obj.__name__
    return type(obj).__name__


class _CustomEstimatorMixin:
    """Shared handling of user-supplied estimators and per-method kwargs.

    Both facades accept, in place of a built-in ``name``, any object
    following the scikit-learn protocol -- an instance, or a class to be
    constructed from ``model_args``/``model_params``. Extra keyword
    arguments for ``fit``, ``predict`` and ``predict_proba`` are carried
    separately, because those belong to the call rather than to the
    constructor and some libraries only accept them there (XGBoost's
    ``eval_set``, LightGBM's ``callbacks``, CatBoost's ``verbose``).
    """

    __slots__ = ()

    # Supplied by the concrete facades. Declared here so the mixin type
    # checks on its own; both subclasses set them in __init__.
    name: _ModelName
    model_args: Optional[Sequence[Any]]
    model_params: Optional[Dict[str, Any]]

    def _build_custom(self) -> Any:
        """Instantiate or clone the user-supplied estimator."""
        from sklearn.base import clone

        estimator = self.name
        args = tuple(self.model_args or ())
        params = dict(self.model_params or {})

        if isinstance(estimator, type):
            return estimator(*args, **params)

        if args:
            raise ValueError(
                "model_args applies only when an estimator *class* is passed. "
                f"Got an instance of {type(estimator).__name__}; configure it "
                "directly, or pass the class instead."
            )
        # Clone so repeated fits never mutate the object the caller handed
        # us -- the same contract sklearn's own meta-estimators follow.
        # Non-sklearn estimators without get_params are used as given.
        try:
            built = clone(estimator)
        except (TypeError, RuntimeError):
            built = estimator
        if params:
            try:
                built.set_params(**params)
            except (AttributeError, ValueError) as exc:
                raise ValueError(
                    f"Could not apply model_params to "
                    f"{type(estimator).__name__}: {exc}. Pass the estimator "
                    "class instead of an instance to have them handed to the "
                    "constructor."
                ) from exc
        return built

    def _validate_name(self) -> None:
        """Reject a ``name`` that is neither a known string nor estimator-like."""
        if isinstance(self.name, str):
            if self.name not in _VALID_NAMES:
                raise _invalid_name_error(self.name)
            return
        if not _is_estimator_like(self.name):
            raise ValueError(
                f"name must be one of {_VALID_NAMES}, or an object following "
                "the scikit-learn API (a class or instance with .fit and "
                f".predict). Got {type(self.name).__name__!r}, which has "
                f"{'no fit' if not hasattr(self.name, 'fit') else 'no predict'} "
                "method."
            )

    @property
    def _uses_custom_estimator(self) -> bool:
        return not isinstance(self.name, str)


class _PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """Binary partial least squares discriminant analysis (PLS-DA).

    Internal helper backing ``QSARClassifier(name="pls")``: fits
    :class:`~qsarkit.models.PLSRegressor` on a 0/1-encoded binary target
    and thresholds the continuous PLS output at 0.5 to obtain class
    predictions, with :meth:`predict_proba` given by a clipped version of
    that same continuous score. PLS-DA is the standard way chemometricians
    turn a regression tool (PLS) into a classifier when the number of
    descriptors greatly exceeds the number of compounds, a regime where
    logistic regression is unstable but PLS's latent-variable projection
    remains well behaved.

    Parameters
    ----------
    n_components : int, default 2
    scale : bool, default True
    max_iter : int, default 500
    tol : float, default 1e-6
    copy : bool, default True
        Forwarded verbatim to the internal :class:`~qsarkit.models.PLSRegressor`.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        The two class labels seen during :meth:`fit`.
    pls_ : PLSRegressor
        The fitted internal PLS regressor.

    References
    ----------
    - Barker, M. & Rayens, W. (2003). "Partial least squares for
      discrimination." J. Chemometrics, 17(3), 166-173.
      https://doi.org/10.1002/cem.785
    """

    classes_: npt.NDArray[np.generic]
    pls_: PLSRegressor

    def __init__(
        self,
        n_components: int = 2,
        *,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-6,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "_PLSDAClassifier":
        """Fit the underlying PLS regressor on a 0/1-encoded target."""
        y_arr = np.asarray(y)
        self.classes_ = unique_labels(y_arr)
        if len(self.classes_) != 2:
            raise ValueError(
                "PLS-DA (name='pls') supports binary classification only, "
                f"got {len(self.classes_)} classes."
            )
        y_binary = np.searchsorted(self.classes_, y_arr).astype(np.float64)
        self.pls_ = PLSRegressor(
            n_components=self.n_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
            copy=self.copy,
        ).fit(X, y_binary)
        return self

    def _score(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(self.pls_.predict(X), dtype=np.float64).ravel()

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Predict the class whose side of the 0.5 threshold the score falls on."""
        idx = (self._score(X) >= 0.5).astype(np.intp)
        return self.classes_[idx]

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities from the clipped continuous PLS score."""
        p1 = np.clip(self._score(X), 0.0, 1.0)
        return np.column_stack([1.0 - p1, p1])


class QSARRegressor(_CustomEstimatorMixin, RegressorMixin, BaseEstimator):
    """Unified, ``name``-dispatching QSAR regressor facade.

    Wraps a broad menu of regression algorithms behind one scikit-learn
    estimator, so pipelines and benchmarking code can sweep across
    algorithm families by changing a single string rather than importing
    and configuring each estimator by hand.

    Parameters
    ----------
    name : {"rf", "svm", "gbm", "xgboost", "lightgbm", "knn", "pls", \
"ridge", "lasso", "elasticnet", "mlp", "gp"}
        Which algorithm to build. One-line rationale for each default:

        - ``"rf"``: :class:`~qsarkit.models.RandomForestQSAR` — robust,
          scale-insensitive default for tabular molecular descriptors
          (Svetnik et al. 2003).
        - ``"svm"``: :class:`~qsarkit.models.SVMQSAR` — RBF support-vector
          regression, strong on non-linear SAR with few hundred compounds
          (Burbidge et al. 2001).
        - ``"gbm"``: ``HistGradientBoostingRegressor`` — fast, regularized
          gradient boosting with no extra install required.
        - ``"xgboost"``: ``xgboost.XGBRegressor`` if installed, otherwise
          the ``"gbm"`` fallback with a warning (Chen & Guestrin 2016).
        - ``"lightgbm"``: ``lightgbm.LGBMRegressor`` if installed,
          otherwise the ``"gbm"`` fallback with a warning (Ke et al. 2017).
        - ``"knn"``: :class:`~qsarkit.neighbors.JaccardKNeighborsRegressor`
          — similarity-based read-across on fingerprints.
        - ``"pls"``: :class:`~qsarkit.models.PLSRegressor` — handles more
          descriptors than compounds via latent-variable projection
          (Wold et al. 2001).
        - ``"ridge"``: ``Ridge`` — L2-regularized linear baseline
          (Hoerl & Kennard 1970).
        - ``"lasso"``: ``Lasso`` — L1-regularized, sparse linear model
          for descriptor selection (Tibshirani 1996).
        - ``"elasticnet"``: ``ElasticNet`` — L1/L2 compromise, robust to
          correlated descriptors (Zou & Hastie 2005).
        - ``"mlp"``: :class:`~qsarkit.models.NeuralNetworkQSAR` — non-linear
          feed-forward network (Winkler 2004).
        - ``"gp"``: :class:`~qsarkit.models.GaussianProcessQSAR` — Tanimoto-
          kernel Gaussian process with predictive uncertainty
          (Ralaivola et al. 2005).
    random_state : int, optional
        Seed forwarded to the underlying estimator, where applicable.
        Ignored when ``name`` is an estimator instance, which is used as
        configured.
    model_params : dict, optional
        Keyword arguments for the underlying estimator's constructor,
        overriding any default (e.g. ``model_params={"n_estimators": 200}``
        for ``name="rf"``). When ``name`` is an *instance*, these are
        applied with ``set_params``.
    model_args : sequence, optional
        Positional arguments for the constructor. Only meaningful when
        ``name`` is an estimator *class*; passing them alongside an
        instance raises, since the instance is already built.
    fit_params : dict, optional
        Extra keyword arguments passed to the estimator's ``fit``.
        Some libraries only accept certain options there rather than in
        the constructor -- XGBoost's ``eval_set``, LightGBM's
        ``callbacks``, CatBoost's ``verbose``, or a ``sample_weight``
        array.
    predict_params : dict, optional
        Extra keyword arguments passed to the estimator's ``predict``.

    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted underlying estimator.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=40, n_features=5, random_state=0)
    >>> model = QSARRegressor(name="rf", random_state=0).fit(X, y)
    >>> model.predict(X).shape
    (40,)

    ``name`` also accepts any object following the scikit-learn API, so
    the facade is not limited to the built-in menu. An instance is used
    as configured:

    >>> from sklearn.linear_model import Ridge
    >>> model = QSARRegressor(Ridge(alpha=2.0)).fit(X, y)
    >>> type(model.estimator_).__name__, model.estimator_.alpha
    ('Ridge', 2.0)

    The instance you pass is cloned, never mutated, so one configured
    template can seed several models:

    >>> template = Ridge(alpha=1.0)
    >>> _ = QSARRegressor(template).fit(X, y)
    >>> hasattr(template, "coef_")       # still unfitted
    False

    A *class* is constructed from ``model_args``/``model_params``:

    >>> model = QSARRegressor(Ridge, model_params={"alpha": 5.0}).fit(X, y)
    >>> model.estimator_.alpha
    5.0

    ``fit_params`` reaches arguments the constructor does not take -- the
    same mechanism serves ``sample_weight`` here and ``eval_set`` for
    XGBoost or CatBoost:

    >>> import numpy as np
    >>> weights = np.linspace(0.5, 1.5, len(y))
    >>> model = QSARRegressor(Ridge, fit_params={"sample_weight": weights}).fit(X, y)
    >>> model.predict(X).shape
    (40,)

    References
    ----------
    - Breiman, L. (2001). "Random Forests." Machine Learning, 45(1),
      5-32. https://doi.org/10.1023/A:1010933404324
    - Cortes, C. & Vapnik, V. (1995). "Support-Vector Networks." Machine
      Learning, 20(3), 273-297. https://doi.org/10.1007/BF00994018
    - Chen, T. & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting
      System." KDD 2016. https://doi.org/10.1145/2939672.2939785
    - Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient
      Boosting Decision Tree." NeurIPS 2017.
    - Wold, S., Sjostrom, M. & Eriksson, L. (2001). Chemometrics and
      Intelligent Laboratory Systems, 58(2), 109-130.
      https://doi.org/10.1016/S0169-7439(01)00155-1
    - Hoerl, A. E. & Kennard, R. W. (1970). "Ridge Regression: Biased
      Estimation for Nonorthogonal Problems." Technometrics, 12(1),
      55-67. https://doi.org/10.1080/00401706.1970.10488634
    - Tibshirani, R. (1996). "Regression Shrinkage and Selection via the
      Lasso." J. R. Stat. Soc. B, 58(1), 267-288.
      https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
    - Zou, H. & Hastie, T. (2005). "Regularization and Variable Selection
      via the Elastic Net." J. R. Stat. Soc. B, 67(2), 301-320.
      https://doi.org/10.1111/j.1467-9868.2005.00503.x
    - Winkler, D. A. (2004). "Neural Networks as Robust Tools in Drug
      Design and Analysis." Mol. Biotechnol., 27(2), 139-167.
      https://doi.org/10.1385/MB:27:2:139
    - Ralaivola, L. et al. (2005). "Graph Kernels for Chemical
      Informatics." Neural Networks, 18(8), 1093-1110.
      https://doi.org/10.1016/j.neunet.2005.07.009
    """

    estimator_: BaseEstimator

    def __init__(
        self,
        name: _ModelName,
        random_state: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None,
        model_args: Optional[Sequence[Any]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        predict_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.random_state = random_state
        self.model_params = model_params
        self.model_args = model_args
        self.fit_params = fit_params
        self.predict_params = predict_params

    def _build(self) -> BaseEstimator:
        if self._uses_custom_estimator:
            return self._build_custom()

        params = dict(self.model_params or {})
        name = self.name

        if name == "rf":
            return RandomForestQSAR(**_with_defaults(params, random_state=self.random_state))
        if name == "svm":
            return SVMQSAR(**params)
        if name == "gbm":
            return HistGradientBoostingRegressor(
                **_with_defaults(params, random_state=self.random_state)
            )
        if name == "xgboost":
            try:
                import xgboost as xgb
            except ImportError:
                warnings.warn(
                    "xgboost is not installed; falling back to sklearn "
                    "HistGradientBoostingRegressor. Install qsarkit-learn[boosting] "
                    "for XGBoost.",
                    stacklevel=2,
                )
                return HistGradientBoostingRegressor(
                    **_with_defaults(params, random_state=self.random_state)
                )
            return xgb.XGBRegressor(**_with_defaults(params, random_state=self.random_state))
        if name == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                warnings.warn(
                    "lightgbm is not installed; falling back to sklearn "
                    "HistGradientBoostingRegressor. Install qsarkit-learn[boosting] "
                    "for LightGBM.",
                    stacklevel=2,
                )
                return HistGradientBoostingRegressor(
                    **_with_defaults(params, random_state=self.random_state)
                )
            return lgb.LGBMRegressor(**_with_defaults(params, random_state=self.random_state))
        if name == "knn":
            return JaccardKNeighborsRegressor(**params)
        if name == "pls":
            return PLSRegressor(**params)
        if name == "ridge":
            return Ridge(**_with_defaults(params, random_state=self.random_state))
        if name == "lasso":
            return Lasso(**_with_defaults(params, random_state=self.random_state))
        if name == "elasticnet":
            return ElasticNet(**_with_defaults(params, random_state=self.random_state))
        if name == "mlp":
            return NeuralNetworkQSAR(**_with_defaults(params, random_state=self.random_state))
        if name == "gp":
            return GaussianProcessQSAR(**_with_defaults(params, random_state=self.random_state))
        raise _invalid_name_error(name)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "QSARRegressor":
        """Build (from ``name``) and fit the underlying regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        QSARRegressor
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``name`` is not a recognized algorithm name.
        """
        self._validate_name()
        self.estimator_ = self._build()
        self.estimator_.fit(X, y, **(self.fit_params or {}))
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Predict by delegating to the fitted underlying regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return np.asarray(
            self.estimator_.predict(X, **(self.predict_params or {})),
            dtype=np.float64,
        )


class QSARClassifier(_CustomEstimatorMixin, ClassifierMixin, BaseEstimator):
    """Unified, ``name``-dispatching QSAR classifier facade.

    The classification counterpart of :class:`QSARRegressor`: wraps a
    broad menu of classification algorithms behind one scikit-learn
    estimator, selected by a single ``name`` string.

    Parameters
    ----------
    name : {"rf", "svm", "gbm", "xgboost", "lightgbm", "knn", "pls", \
"ridge", "lasso", "elasticnet", "mlp", "gp"}
        Which algorithm to build. One-line rationale for each default:

        - ``"rf"``: ``RandomForestClassifier`` — robust default ensemble
          for tabular fingerprints/descriptors (Svetnik et al. 2003).
        - ``"svm"``: ``SVC`` (RBF, ``probability=True``) — strong
          non-linear classifier on molecular fingerprints
          (Cortes & Vapnik 1995; Burbidge et al. 2001).
        - ``"gbm"``: ``HistGradientBoostingClassifier`` — fast,
          regularized gradient boosting, no extra install.
        - ``"xgboost"``: ``xgboost.XGBClassifier`` if installed,
          otherwise the ``"gbm"`` fallback with a warning
          (Chen & Guestrin 2016).
        - ``"lightgbm"``: ``lightgbm.LGBMClassifier`` if installed,
          otherwise the ``"gbm"`` fallback with a warning (Ke et al. 2017).
        - ``"knn"``: :class:`~qsarkit.neighbors.JaccardKNeighborsClassifier`
          — similarity-based read-across on fingerprints.
        - ``"pls"``: an internal binary PLS-DA wrapper around
          :class:`~qsarkit.models.PLSRegressor` (Barker & Rayens 2003).
        - ``"ridge"``: ``RidgeClassifier`` — fast linear baseline; **has
          no native** ``predict_proba`` (calling it raises
          ``AttributeError``).
        - ``"lasso"``: ``LogisticRegression(penalty="l1", solver="liblinear")``
          — sparse linear classifier (Tibshirani 1996).
        - ``"elasticnet"``: ``LogisticRegression(penalty="elasticnet", \
solver="saga")`` — L1/L2 compromise (Zou & Hastie 2005).
        - ``"mlp"``: ``MLPClassifier`` — non-linear feed-forward network
          (Winkler 2004).
        - ``"gp"``: ``GaussianProcessClassifier`` with a
          :class:`~qsarkit.models.TanimotoKernel` — probabilistic
          fingerprint-similarity classifier (Ralaivola et al. 2005).
    random_state : int, optional
        Seed forwarded to the underlying estimator, where applicable.
        Ignored when ``name`` is an estimator instance, which is used as
        configured.
    model_params : dict, optional
        Keyword arguments for the underlying estimator's constructor,
        overriding any default (e.g. ``model_params={"n_estimators": 200}``
        for ``name="rf"``). When ``name`` is an *instance*, these are
        applied with ``set_params``.
    model_args : sequence, optional
        Positional arguments for the constructor. Only meaningful when
        ``name`` is an estimator *class*; passing them alongside an
        instance raises, since the instance is already built.
    fit_params : dict, optional
        Extra keyword arguments passed to the estimator's ``fit``.
        Some libraries only accept certain options there rather than in
        the constructor -- XGBoost's ``eval_set``, LightGBM's
        ``callbacks``, CatBoost's ``verbose``, or a ``sample_weight``
        array.
    predict_params : dict, optional
        Extra keyword arguments passed to the estimator's ``predict``.
    predict_proba_params : dict, optional
        Extra keyword arguments passed to the estimator's
        ``predict_proba``.

    Attributes
    ----------
    estimator_ : BaseEstimator
        The fitted underlying scikit-learn (or xgboost/lightgbm)
        estimator.
    classes_ : ndarray
        Class labels, taken from the fitted underlying estimator.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=40, n_features=5, random_state=0)
    >>> model = QSARClassifier(name="rf", random_state=0).fit(X, y)
    >>> model.predict(X).shape
    (40,)

    As for :class:`QSARRegressor`, ``name`` accepts any estimator
    following the scikit-learn API -- CatBoost, XGBoost, LightGBM or your
    own -- as a class or an instance:

    >>> from sklearn.tree import DecisionTreeClassifier
    >>> model = QSARClassifier(
    ...     DecisionTreeClassifier, model_params={"max_depth": 3}, random_state=0
    ... ).fit(X, y)
    >>> type(model.estimator_).__name__
    'DecisionTreeClassifier'
    >>> model.predict_proba(X).shape
    (40, 2)

    ``classes_`` is taken from the estimator when it exposes one, and
    otherwise from the labels seen during ``fit``, so a wrapper that
    omits the attribute still works:

    >>> model.classes_.tolist()
    [0, 1]

    References
    ----------
    - Breiman, L. (2001). Machine Learning, 45(1), 5-32.
      https://doi.org/10.1023/A:1010933404324
    - Cortes, C. & Vapnik, V. (1995). Machine Learning, 20(3), 273-297.
      https://doi.org/10.1007/BF00994018
    - Chen, T. & Guestrin, C. (2016). KDD 2016.
      https://doi.org/10.1145/2939672.2939785
    - Ke, G. et al. (2017). NeurIPS 2017.
    - Barker, M. & Rayens, W. (2003). J. Chemometrics, 17(3), 166-173.
      https://doi.org/10.1002/cem.785
    - Hoerl, A. E. & Kennard, R. W. (1970). Technometrics, 12(1), 55-67.
      https://doi.org/10.1080/00401706.1970.10488634
    - Tibshirani, R. (1996). J. R. Stat. Soc. B, 58(1), 267-288.
      https://doi.org/10.1111/j.2517-6161.1996.tb02080.x
    - Zou, H. & Hastie, T. (2005). J. R. Stat. Soc. B, 67(2), 301-320.
      https://doi.org/10.1111/j.1467-9868.2005.00503.x
    - Winkler, D. A. (2004). Mol. Biotechnol., 27(2), 139-167.
      https://doi.org/10.1385/MB:27:2:139
    - Ralaivola, L. et al. (2005). Neural Networks, 18(8), 1093-1110.
      https://doi.org/10.1016/j.neunet.2005.07.009
    """

    estimator_: BaseEstimator
    classes_: npt.NDArray[np.generic]

    def __init__(
        self,
        name: _ModelName,
        random_state: Optional[int] = None,
        model_params: Optional[Dict[str, Any]] = None,
        model_args: Optional[Sequence[Any]] = None,
        fit_params: Optional[Dict[str, Any]] = None,
        predict_params: Optional[Dict[str, Any]] = None,
        predict_proba_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.random_state = random_state
        self.model_params = model_params
        self.model_args = model_args
        self.fit_params = fit_params
        self.predict_params = predict_params
        self.predict_proba_params = predict_proba_params

    def _build(self) -> BaseEstimator:
        if self._uses_custom_estimator:
            return self._build_custom()

        params = dict(self.model_params or {})
        name = self.name

        if name == "rf":
            return RandomForestClassifier(
                **_with_defaults(params, n_estimators=500, random_state=self.random_state)
            )
        if name == "svm":
            return SVC(**_with_defaults(params, kernel="rbf", probability=True))
        if name == "gbm":
            return HistGradientBoostingClassifier(
                **_with_defaults(params, random_state=self.random_state)
            )
        if name == "xgboost":
            try:
                import xgboost as xgb
            except ImportError:
                warnings.warn(
                    "xgboost is not installed; falling back to sklearn "
                    "HistGradientBoostingClassifier. Install qsarkit-learn[boosting] "
                    "for XGBoost.",
                    stacklevel=2,
                )
                return HistGradientBoostingClassifier(
                    **_with_defaults(params, random_state=self.random_state)
                )
            return xgb.XGBClassifier(**_with_defaults(params, random_state=self.random_state))
        if name == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError:
                warnings.warn(
                    "lightgbm is not installed; falling back to sklearn "
                    "HistGradientBoostingClassifier. Install qsarkit-learn[boosting] "
                    "for LightGBM.",
                    stacklevel=2,
                )
                return HistGradientBoostingClassifier(
                    **_with_defaults(params, random_state=self.random_state)
                )
            return lgb.LGBMClassifier(**_with_defaults(params, random_state=self.random_state))
        if name == "knn":
            return JaccardKNeighborsClassifier(**params)
        if name == "pls":
            return _PLSDAClassifier(**params)
        if name == "ridge":
            return RidgeClassifier(**_with_defaults(params, random_state=self.random_state))
        if name == "lasso":
            return LogisticRegression(
                penalty="l1",
                solver="liblinear",
                random_state=self.random_state,
                **params,
            )
        if name == "elasticnet":
            return LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                random_state=self.random_state,
                **params,
            )
        if name == "mlp":
            return MLPClassifier(**params)
        if name == "gp":
            # Tanimoto suits fingerprints; on signed descriptors it is not
            # positive semi-definite, so `_build` is given the training data
            # in fit() and picks accordingly.
            return GaussianProcessClassifier(
                kernel=self._default_gp_kernel(), random_state=self.random_state,
                **params,
            )
        raise _invalid_name_error(name)

    def _default_gp_kernel(self) -> Any:
        """Tanimoto for non-negative data, RBF otherwise."""
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        data = getattr(self, "_fit_X", None)
        if data is not None and data.size and np.all(data >= 0):
            return TanimotoKernel()
        return RBF() + WhiteKernel()

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "QSARClassifier":
        """Build (from ``name``) and fit the underlying classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        QSARClassifier
            The fitted estimator.

        Raises
        ------
        ValueError
            If ``name`` is not a recognized algorithm name.
        """
        self._validate_name()
        # Recorded so `_build` can choose a data-appropriate GP kernel.
        self._fit_X = np.asarray(X, dtype=np.float64)
        self.estimator_ = self._build()
        self.estimator_.fit(X, y, **(self.fit_params or {}))
        # A custom estimator need not expose `classes_` (CatBoost names it
        # `classes_` only after fit; some wrappers omit it entirely), so
        # fall back to the labels actually seen.
        found = getattr(self.estimator_, "classes_", None)
        self.classes_ = np.asarray(unique_labels(y) if found is None else found)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "estimator_"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Predict by delegating to the fitted underlying classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return np.asarray(self.estimator_.predict(X, **(self.predict_params or {})))

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities, by delegating to the fitted classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)

        Raises
        ------
        ModelNotFittedError
            If called before :meth:`fit`.
        AttributeError
            If the underlying estimator has no ``predict_proba`` (e.g.
            ``name="ridge"``, whose ``RidgeClassifier`` backend has no
            native probability estimates).
        """
        self._check_fitted()
        if not hasattr(self.estimator_, "predict_proba"):
            raise AttributeError(
                f"The '{_describe_estimator(self.name)}' backend "
                f"({type(self.estimator_).__name__}) does not support "
                "predict_proba."
            )
        return np.asarray(
            self.estimator_.predict_proba(X, **(self.predict_proba_params or {})),
            dtype=np.float64,
        )
