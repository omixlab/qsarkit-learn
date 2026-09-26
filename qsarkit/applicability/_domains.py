"""Applicability domain estimators (OECD validation principle 3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Literal, Optional, Sequence

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.neighbors import tanimoto_similarity_matrix

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go

__all__ = [
    "BaseApplicabilityDomain",
    "LeverageAD",
    "DistanceToModelAD",
    "KNNApplicabilityDomain",
    "RangeAD",
    "BoundingBoxAD",
    "PCABoundingBoxAD",
    "ConvexHullAD",
    "TanimotoSimilarityAD",
    "KernelDensityAD",
    "IsolationForestAD",
    "EnsembleAD",
]


class BaseApplicabilityDomain(BaseEstimator, ABC):
    """Common interface for every applicability-domain estimator.

    OECD validation principle 3 requires a QSAR model to declare *the
    chemical space in which its predictions are reliable*. Every subclass
    answers that question with the same three methods:

    - ``fit(X)`` learns the domain from the training descriptors.
    - ``score_samples(X)`` returns a continuous "how far outside" score,
      where **larger means further outside the domain**.
    - ``predict(X)`` returns a boolean array: True = inside the domain.

    ``decision_function(X)`` is provided for scikit-learn compatibility
    and returns ``threshold_ - score``, so positive means inside — the
    sign convention sklearn's outlier detectors use.

    References
    ----------
    - OECD (2007). "Guidance Document on the Validation of (Quantitative)
      Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
      Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
      https://doi.org/10.1787/9789264085442-en
    - Sahigara, F. et al. (2012). "Comparison of Different Approaches to
      Define the Applicability Domain of QSAR Models." Molecules, 17(5),
      4791-4810. https://doi.org/10.3390/molecules17054791
    - Jaworska, J., Nikolova-Jeliazkova, N. & Aldenberg, T. (2005). "QSAR
      Applicability Domain Estimation by Projection of the Training Set
      in Descriptor Space: A Review." ATLA, 33(5), 445-459.
      https://doi.org/10.1177/026119290503300508
    - Netzeva, T. I. et al. (2005). "Current Status of Methods for
      Defining the Applicability Domain of (Q)SARs." ATLA, 33(2), 155-173.
      https://doi.org/10.1177/026119290503300209
    """

    threshold_: float
    n_features_in_: int

    @abstractmethod
    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "BaseApplicabilityDomain":
        """Learn the domain from training descriptors."""

    @abstractmethod
    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return per-sample distance-from-domain scores (larger = further out)."""

    def _check_fitted(self) -> None:
        if not hasattr(self, "threshold_"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before use."
            )

    def _validate(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if hasattr(self, "n_features_in_") and arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {arr.shape[1]} features, expected {self.n_features_in_}."
            )
        return arr

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Return True for samples inside the applicability domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of bool of shape (n_samples,)
        """
        return np.asarray(self.score_samples(X) <= self.threshold_, dtype=np.bool_)

    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Signed margin to the domain boundary; positive means inside.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        self._check_fitted()
        return np.asarray(self.threshold_ - self.score_samples(X), dtype=np.float64)

    def coverage(self, X: npt.ArrayLike) -> float:
        """Fraction of ``X`` that falls inside the domain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        float
            Value in [0, 1].
        """
        return float(np.mean(self.predict(X)))


class LeverageAD(BaseApplicabilityDomain):
    """Leverage (hat-matrix) applicability domain — the Williams-plot method.

    The leverage of a compound is its diagonal element of the hat matrix
    ``H = X(X'X)^-1 X'``, i.e. how much influence it exerts on the fitted
    regression. Compounds whose leverage exceeds the warning value
    ``h* = 3(p+1)/n`` (p = descriptors, n = training compounds) sit in a
    sparse region of descriptor space where the model is extrapolating.

    This is the domain definition assumed by the Williams plot
    (standardized residual vs leverage), the standard regulatory
    presentation of QSAR reliability.

    Parameters
    ----------
    threshold_factor : float, default 3.0
        Numerator factor in ``h* = factor * (p+1)/n``. 3 is the
        conventional warning leverage; 2 is sometimes used for large
        training sets.

    Attributes
    ----------
    threshold_ : float
        The computed ``h*``.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(50, 3))
    >>> ad = LeverageAD().fit(X)
    >>> bool(ad.predict(np.zeros((1, 3)))[0])
    True

    References
    ----------
    - Gramatica, P. (2007). "Principles of QSAR Models Validation:
      Internal and External." QSAR Comb. Sci., 26(5), 694-701.
      https://doi.org/10.1002/qsar.200610151
    - Eriksson, L. et al. (2003). "Methods for Reliability and
      Uncertainty Assessment and for Applicability Evaluations of
      Classification- and Regression-Based QSARs." Environ. Health
      Perspect., 111(10), 1361-1375. https://doi.org/10.1289/ehp.5758
    - Atkinson, A. C. (1985). "Plots, Transformations and Regression."
      Oxford University Press.
    """

    def __init__(self, threshold_factor: float = 3.0) -> None:
        self.threshold_factor = threshold_factor

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "LeverageAD":
        """Compute ``(X'X)^-1`` and the warning leverage from training data."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        n_samples, n_features = arr.shape
        if n_samples == 0:
            raise ValueError("Cannot fit on an empty training set.")

        self.n_features_in_ = n_features
        # Pseudo-inverse rather than inv: descriptor matrices are routinely
        # rank-deficient (correlated descriptors), where inv() would raise.
        self._xtx_inv = np.linalg.pinv(arr.T @ arr)
        self.threshold_ = self.threshold_factor * (n_features + 1) / n_samples
        self._X_train = arr
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Leverage ``h_i = x_i' (X'X)^-1 x_i`` for each sample."""
        self._check_fitted()
        arr = self._validate(X)
        return np.asarray(
            np.einsum("ij,jk,ik->i", arr, self._xtx_inv, arr), dtype=np.float64
        )


class DistanceToModelAD(BaseApplicabilityDomain):
    """Distance-to-centroid applicability domain.

    Scores each compound by its distance to the centroid of the training
    set, with the boundary set at a percentile of the training
    distribution. Mahalanobis distance accounts for descriptor
    correlation and scale, which plain Euclidean distance does not.

    Parameters
    ----------
    metric : {"euclidean", "mahalanobis", "cityblock"}, default "euclidean"
        Distance measure.
    percentile : float, default 95.0
        Percentile of the training distance distribution used as the
        domain boundary.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(50, 3))
    >>> ad = DistanceToModelAD(metric="mahalanobis").fit(X)
    >>> bool(ad.predict(np.full((1, 3), 50.0))[0])
    False

    References
    ----------
    - Jaworska, J., Nikolova-Jeliazkova, N. & Aldenberg, T. (2005). ATLA,
      33(5), 445-459. https://doi.org/10.1177/026119290503300508
    - Mahalanobis, P. C. (1936). "On the Generalised Distance in
      Statistics." Proc. Natl. Inst. Sci. India, 2(1), 49-55.
    - Sahigara, F. et al. (2012). Molecules, 17(5), 4791-4810.
      https://doi.org/10.3390/molecules17054791
    """

    def __init__(
        self,
        metric: Literal["euclidean", "mahalanobis", "cityblock"] = "euclidean",
        percentile: float = 95.0,
    ) -> None:
        self.metric = metric
        self.percentile = percentile

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "DistanceToModelAD":
        """Learn the centroid, covariance and distance threshold."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {self.percentile}.")
        if self.metric not in ("euclidean", "mahalanobis", "cityblock"):
            raise ValueError(
                "metric must be 'euclidean', 'mahalanobis' or 'cityblock', "
                f"got {self.metric!r}."
            )

        self.n_features_in_ = arr.shape[1]
        self._centroid = arr.mean(axis=0)
        if self.metric == "mahalanobis":
            self._cov_inv = np.linalg.pinv(np.cov(arr, rowvar=False).reshape(
                self.n_features_in_, self.n_features_in_
            ))
        self.threshold_ = float(
            np.percentile(self._distances(arr), self.percentile)
        )
        return self

    def _distances(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        centered = arr - self._centroid
        if self.metric == "euclidean":
            return np.asarray(np.linalg.norm(centered, axis=1), dtype=np.float64)
        if self.metric == "cityblock":
            return np.asarray(np.abs(centered).sum(axis=1), dtype=np.float64)
        squared = np.einsum("ij,jk,ik->i", centered, self._cov_inv, centered)
        return np.asarray(np.sqrt(np.maximum(squared, 0.0)), dtype=np.float64)

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Distance from each sample to the training centroid."""
        self._check_fitted()
        return self._distances(self._validate(X))


class KNNApplicabilityDomain(BaseApplicabilityDomain):
    """k-nearest-neighbour applicability domain.

    Scores a compound by the mean distance to its ``k`` nearest training
    neighbours, with the boundary at a percentile of the training
    distribution. Unlike leverage or centroid distance this makes no
    assumption that the training set forms a single convex cloud, so it
    handles the clustered, multi-series datasets typical of real QSAR
    work — which is why it is usually the best-performing AD definition
    in comparative studies.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbours averaged.
    metric : {"euclidean", "tanimoto"}, default "euclidean"
        ``"tanimoto"`` uses Jaccard distance and is the right choice for
        binary fingerprints.
    percentile : float, default 95.0
        Percentile of the training score distribution used as the
        boundary.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(50, 3))
    >>> ad = KNNApplicabilityDomain(n_neighbors=3).fit(X)
    >>> bool(ad.predict(np.full((1, 3), 50.0))[0])
    False

    References
    ----------
    - Sahigara, F. et al. (2013). "Defining a Novel k-Nearest Neighbours
      Approach to Assess the Applicability Domain of a QSAR Model for
      Reliable Predictions." J. Cheminform., 5, 27.
      https://doi.org/10.1186/1758-2946-5-27
    - Sahigara, F. et al. (2012). Molecules, 17(5), 4791-4810.
      https://doi.org/10.3390/molecules17054791
    - Sheridan, R. P. et al. (2004). "Similarity to Molecules in the
      Training Set Is a Good Discriminator for Prediction Accuracy in
      QSAR." J. Chem. Inf. Comput. Sci., 44(6), 1912-1928.
      https://doi.org/10.1021/ci049782w
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: Literal["euclidean", "tanimoto"] = "euclidean",
        percentile: float = 95.0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.percentile = percentile

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "KNNApplicabilityDomain":
        """Store the training set and calibrate the distance threshold."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")
        if self.n_neighbors >= arr.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} must be smaller than the "
                f"training set size ({arr.shape[0]})."
            )
        if self.metric not in ("euclidean", "tanimoto"):
            raise ValueError(
                f"metric must be 'euclidean' or 'tanimoto', got {self.metric!r}."
            )
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {self.percentile}.")

        self.n_features_in_ = arr.shape[1]
        self._X_train = arr
        # Calibrate on the training set itself, excluding each point's
        # zero-distance self-match.
        self.threshold_ = float(
            np.percentile(self._mean_knn_distance(arr, exclude_self=True), self.percentile)
        )
        return self

    def _pairwise(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.metric == "tanimoto":
            return 1.0 - tanimoto_similarity_matrix(arr, self._X_train)
        # cdist, not a broadcast subtraction. Materialising
        # `arr[:, None, :] - X_train[None, :, :]` allocates
        # (n_query, n_train, n_features) floats: on 5000 training compounds
        # described by 2048-bit fingerprints that is over 100 TB for a single
        # call, so the domain died on any realistic dataset while working
        # fine on the two dozen molecules in the documentation.
        from scipy.spatial.distance import cdist

        return np.asarray(cdist(arr, self._X_train, metric="euclidean"), dtype=np.float64)

    def _mean_knn_distance(
        self, arr: npt.NDArray[np.float64], exclude_self: bool = False
    ) -> npt.NDArray[np.float64]:
        dist = self._pairwise(arr)
        k = self.n_neighbors + 1 if exclude_self else self.n_neighbors
        partitioned = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        ordered = np.sort(partitioned, axis=1)
        if exclude_self:
            ordered = ordered[:, 1:]
        return np.asarray(ordered.mean(axis=1), dtype=np.float64)

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Mean distance to the ``k`` nearest training compounds."""
        self._check_fitted()
        return self._mean_knn_distance(self._validate(X))


class RangeAD(BaseApplicabilityDomain):
    """Per-descriptor range (bounding-box) applicability domain.

    The simplest and most conservative definition: a compound is inside
    the domain only if every descriptor falls within the training range,
    optionally widened by a tolerance. Cheap and completely transparent —
    which is why regulators like it — but it accepts the empty corners of
    the bounding box, so it over-estimates the domain in high dimensions.

    Parameters
    ----------
    tolerance : float, default 0.0
        Fractional widening of each descriptor's range, relative to that
        descriptor's training span. 0.1 widens each side by 10%.

    Attributes
    ----------
    threshold_ : float
        Always 0.0: the score counts range violations, so any violation
        puts a compound outside.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0.0, 0.0], [1.0, 1.0]])
    >>> ad = RangeAD().fit(X)
    >>> bool(ad.predict(np.array([[0.5, 0.5]]))[0])
    True
    >>> bool(ad.predict(np.array([[9.0, 0.5]]))[0])
    False

    References
    ----------
    - Jaworska, J., Nikolova-Jeliazkova, N. & Aldenberg, T. (2005). ATLA,
      33(5), 445-459. https://doi.org/10.1177/026119290503300508
    - Netzeva, T. I. et al. (2005). ATLA, 33(2), 155-173.
      https://doi.org/10.1177/026119290503300209
    """

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def fit(self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None) -> "RangeAD":
        """Record the per-descriptor training range."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")
        if self.tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {self.tolerance}.")

        self.n_features_in_ = arr.shape[1]
        low, high = arr.min(axis=0), arr.max(axis=0)
        pad = (high - low) * self.tolerance
        self._low, self._high = low - pad, high + pad
        self.threshold_ = 0.0
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Number of descriptors falling outside the training range."""
        self._check_fitted()
        arr = self._validate(X)
        violations = (arr < self._low) | (arr > self._high)
        return np.asarray(violations.sum(axis=1), dtype=np.float64)


# The bounding-box AD is the range AD; the alias is the name used in much
# of the AD literature, kept so either term finds the class.
BoundingBoxAD = RangeAD


class PCABoundingBoxAD(BaseApplicabilityDomain):
    """Bounding box in principal-component space.

    Projects onto the leading principal components before applying a
    range test. Because PCs are uncorrelated and ordered by variance,
    this fits the training cloud far more tightly than a bounding box in
    the raw (correlated) descriptor space, while staying just as cheap to
    evaluate.

    Parameters
    ----------
    n_components : int or float, default 0.95
        Passed to ``sklearn.decomposition.PCA``: an int selects that many
        components, a float in (0, 1) selects enough to retain that
        fraction of variance.
    tolerance : float, default 0.0
        Fractional widening of each component's range.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(50, 4))
    >>> ad = PCABoundingBoxAD(n_components=2).fit(X)
    >>> bool(ad.predict(np.full((1, 4), 50.0))[0])
    False

    References
    ----------
    - Jaworska, J., Nikolova-Jeliazkova, N. & Aldenberg, T. (2005). ATLA,
      33(5), 445-459. https://doi.org/10.1177/026119290503300508
    - Jolliffe, I. T. (2002). "Principal Component Analysis," 2nd ed.
      Springer. https://doi.org/10.1007/b98835
    - scikit-learn PCA documentation:
      https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    """

    def __init__(
        self, n_components: Any = 0.95, tolerance: float = 0.0
    ) -> None:
        self.n_components = n_components
        self.tolerance = tolerance

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "PCABoundingBoxAD":
        """Fit the PCA projection and the per-component range."""
        from sklearn.decomposition import PCA

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")

        self.n_features_in_ = arr.shape[1]
        self._pca = PCA(n_components=self.n_components).fit(arr)
        self._box = RangeAD(tolerance=self.tolerance).fit(self._pca.transform(arr))
        self.threshold_ = 0.0
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Number of principal components falling outside the training range."""
        self._check_fitted()
        return self._box.score_samples(self._pca.transform(self._validate(X)))


class ConvexHullAD(BaseApplicabilityDomain):
    """Convex-hull applicability domain.

    A compound is inside the domain if it lies within the convex hull of
    the training set — the tightest interpolation region there is, with
    no empty corners. The hull becomes intractable above roughly ten
    dimensions (and needs more points than dimensions to exist at all),
    so this class projects onto the leading principal components first.

    Parameters
    ----------
    n_components : int, default 3
        Number of principal components the hull is built in.
    tolerance : float, default 1e-10
        Numerical slack when testing hull inequalities.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(50, 3))
    >>> ad = ConvexHullAD(n_components=2).fit(X)
    >>> bool(ad.predict(np.full((1, 3), 50.0))[0])
    False

    References
    ----------
    - Jaworska, J., Nikolova-Jeliazkova, N. & Aldenberg, T. (2005). ATLA,
      33(5), 445-459. https://doi.org/10.1177/026119290503300508
    - Barber, C. B., Dobkin, D. P. & Huhdanpaa, H. (1996). "The Quickhull
      Algorithm for Convex Hulls." ACM Trans. Math. Softw., 22(4),
      469-483. https://doi.org/10.1145/235815.235821
    """

    def __init__(self, n_components: int = 3, tolerance: float = 1e-10) -> None:
        self.n_components = n_components
        self.tolerance = tolerance

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "ConvexHullAD":
        """Build the convex hull of the projected training set."""
        from scipy.spatial import ConvexHull
        from sklearn.decomposition import PCA

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        n_components = min(self.n_components, arr.shape[1])
        if arr.shape[0] <= n_components:
            raise ValueError(
                f"A convex hull in {n_components} dimensions needs more than "
                f"{n_components} points; got {arr.shape[0]}."
            )

        self.n_features_in_ = arr.shape[1]
        self._pca = PCA(n_components=n_components).fit(arr)
        self._hull = ConvexHull(self._pca.transform(arr))
        self.threshold_ = 0.0
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Largest positive violation of any hull face inequality (0 = inside)."""
        self._check_fitted()
        projected = self._pca.transform(self._validate(X))
        # Hull equations are [normal | offset] with normal.x + offset <= 0 inside.
        equations = self._hull.equations
        violations = projected @ equations[:, :-1].T + equations[:, -1]
        return np.asarray(
            np.maximum(violations.max(axis=1) - self.tolerance, 0.0), dtype=np.float64
        )


class TanimotoSimilarityAD(BaseApplicabilityDomain):
    """Fingerprint-similarity applicability domain.

    Declares a compound inside the domain when its Tanimoto similarity to
    the nearest (or mean of the ``k`` nearest) training compound reaches a
    threshold. This is the domain definition that speaks the language
    chemists use — "is there anything like this in the training set?" —
    and the one to prefer whenever the model is built on fingerprints.

    Parameters
    ----------
    threshold : float, default 0.3
        Minimum similarity required to be inside the domain. The
        conventional ECFP4 value for "meaningfully similar" is 0.3-0.4
        for AD purposes (much lower than the 0.7 used for hit expansion,
        because the question is coverage, not equivalence).
    n_neighbors : int, default 1
        Number of nearest training compounds averaged. 1 uses the single
        nearest neighbour.

    Attributes
    ----------
    threshold_ : float
        Stored as a *distance* (``1 - threshold``) to match the base
        class's "larger is further out" convention.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=float)
    >>> ad = TanimotoSimilarityAD(threshold=0.5).fit(X)
    >>> bool(ad.predict(np.array([[1, 1, 0, 0]], dtype=float))[0])
    True
    >>> bool(ad.predict(np.array([[0, 0, 0, 1]], dtype=float))[0])
    False

    References
    ----------
    - Sheridan, R. P. et al. (2004). J. Chem. Inf. Comput. Sci., 44(6),
      1912-1928. https://doi.org/10.1021/ci049782w
    - Tetko, I. V. et al. (2008). "Critical Assessment of QSAR Models of
      Environmental Toxicity against Tetrahymena Pyriformis." J. Chem.
      Inf. Model., 48(9), 1733-1746. https://doi.org/10.1021/ci800151m
    - Bajusz, D., Racz, A. & Heberger, K. (2015). J. Cheminform., 7, 20.
      https://doi.org/10.1186/s13321-015-0069-3
    """

    def __init__(self, threshold: float = 0.3, n_neighbors: int = 1) -> None:
        self.threshold = threshold
        self.n_neighbors = n_neighbors

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "TanimotoSimilarityAD":
        """Store the training fingerprints."""
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}.")
        if self.n_neighbors > arr.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the training set size "
                f"({arr.shape[0]})."
            )

        self.n_features_in_ = arr.shape[1]
        self._X_train = arr
        self.threshold_ = 1.0 - self.threshold
        return self

    def similarity_to_training(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Mean Tanimoto similarity to the ``k`` nearest training compounds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Similarities in [0, 1].
        """
        self._check_fitted()
        sim = tanimoto_similarity_matrix(self._validate(X), self._X_train)
        top = np.sort(sim, axis=1)[:, -self.n_neighbors:]
        return np.asarray(top.mean(axis=1), dtype=np.float64)

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Tanimoto *distance* to the nearest training compounds."""
        return np.asarray(1.0 - self.similarity_to_training(X), dtype=np.float64)


class KernelDensityAD(BaseApplicabilityDomain):
    """Kernel-density applicability domain.

    Estimates the training-set density in descriptor space and puts the
    boundary at a low-density percentile. Unlike leverage or centroid
    distance this handles multi-modal training sets — several distinct
    chemical series — without declaring the sparse space between the
    clusters to be inside the domain.

    Parameters
    ----------
    bandwidth : float or str, default "scott"
        Kernel bandwidth, or a rule-of-thumb name passed through to
        ``sklearn.neighbors.KernelDensity`` when numeric.
    kernel : str, default "gaussian"
        Kernel name accepted by ``KernelDensity``.
    percentile : float, default 5.0
        Training log-density percentile used as the boundary; samples
        below it are outside.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(60, 2))
    >>> ad = KernelDensityAD().fit(X)
    >>> bool(ad.predict(np.full((1, 2), 50.0))[0])
    False

    References
    ----------
    - Sahigara, F. et al. (2012). Molecules, 17(5), 4791-4810.
      https://doi.org/10.3390/molecules17054791
    - Silverman, B. W. (1986). "Density Estimation for Statistics and
      Data Analysis." Chapman and Hall.
      https://doi.org/10.1201/9781315140919
    - Scott, D. W. (1992). "Multivariate Density Estimation." Wiley.
      https://doi.org/10.1002/9780470316849
    """

    def __init__(
        self,
        bandwidth: Any = "scott",
        kernel: str = "gaussian",
        percentile: float = 5.0,
    ) -> None:
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.percentile = percentile

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "KernelDensityAD":
        """Fit the density estimate and its low-density boundary."""
        from sklearn.neighbors import KernelDensity

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")
        if not 0.0 <= self.percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {self.percentile}.")

        self.n_features_in_ = arr.shape[1]
        self._kde = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(arr)
        # Score is negative log-density, so "larger = further outside".
        self.threshold_ = float(
            np.percentile(-self._kde.score_samples(arr), 100.0 - self.percentile)
        )
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Negative log-density under the fitted kernel density estimate."""
        self._check_fitted()
        return np.asarray(-self._kde.score_samples(self._validate(X)), dtype=np.float64)


class IsolationForestAD(BaseApplicabilityDomain):
    """Isolation-Forest applicability domain.

    Treats "outside the domain" as "easy to isolate": a tree ensemble
    partitions the descriptor space at random, and points separated in
    few splits are anomalies. It is nonparametric, handles multi-modal
    and non-convex training sets, and scales to large high-dimensional
    descriptor matrices where hull- and density-based definitions break
    down.

    Parameters
    ----------
    contamination : float, default 0.05
        Expected fraction of training compounds treated as outliers,
        which sets the boundary.
    n_estimators : int, default 100
        Number of trees.
    random_state : int, optional
        Seed for reproducibility.

    Attributes
    ----------
    threshold_ : float
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(80, 3))
    >>> ad = IsolationForestAD(random_state=0).fit(X)
    >>> bool(ad.predict(np.full((1, 3), 50.0))[0])
    False

    References
    ----------
    - Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2008). "Isolation Forest."
      IEEE ICDM 2008, 413-422. https://doi.org/10.1109/ICDM.2008.17
    - Liu, F. T., Ting, K. M. & Zhou, Z.-H. (2012). "Isolation-Based
      Anomaly Detection." ACM Trans. Knowl. Discov. Data, 6(1), 1-39.
      https://doi.org/10.1145/2133360.2133363
    - scikit-learn IsolationForest documentation:
      https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: Optional[int] = None,
    ) -> None:
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "IsolationForestAD":
        """Fit the isolation forest on the training descriptors."""
        from sklearn.ensemble import IsolationForest

        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty training set.")

        self.n_features_in_ = arr.shape[1]
        self._forest = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        ).fit(arr)
        # sklearn's decision_function is positive for inliers; negate so
        # that larger means further outside, then the boundary sits at 0.
        self.threshold_ = 0.0
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Negated isolation-forest decision function (larger = more anomalous)."""
        self._check_fitted()
        return np.asarray(
            -self._forest.decision_function(self._validate(X)), dtype=np.float64
        )


class EnsembleAD(BaseApplicabilityDomain):
    """Consensus applicability domain over several AD definitions.

    Different AD definitions disagree, and each has a characteristic
    failure mode — leverage assumes a single elliptical cloud, bounding
    boxes accept empty corners, k-NN is sensitive to ``k``. Requiring
    agreement among several gives a more honest domain than trusting any
    one, and the fraction of members that agree is itself a graded
    confidence score.

    Parameters
    ----------
    estimators : sequence of BaseApplicabilityDomain, optional
        Members of the ensemble. Defaults to leverage, k-NN and range.
    voting : {"majority", "unanimous", "any"}, default "majority"
        How member votes combine into the final in-domain decision.

    Attributes
    ----------
    threshold_ : float
        Fraction-outside boundary implied by ``voting``.
    n_features_in_ : int

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.RandomState(0).normal(size=(60, 3))
    >>> ad = EnsembleAD().fit(X)
    >>> bool(ad.predict(np.full((1, 3), 50.0))[0])
    False

    References
    ----------
    - Sahigara, F. et al. (2012). Molecules, 17(5), 4791-4810.
      https://doi.org/10.3390/molecules17054791
    - Sushko, I. et al. (2010). "Applicability Domains for Classification
      Problems: Benchmarking of Distance to Models for Ames
      Mutagenicity." J. Chem. Inf. Model., 50(12), 2094-2111.
      https://doi.org/10.1021/ci100253r
    - Hanser, T. et al. (2016). "Applicability Domain: Towards a More
      Formal Definition." SAR QSAR Environ. Res., 27(11), 865-881.
      https://doi.org/10.1080/1062936X.2016.1250229
    """

    def __init__(
        self,
        estimators: Optional[Sequence[BaseApplicabilityDomain]] = None,
        voting: Literal["majority", "unanimous", "any"] = "majority",
    ) -> None:
        self.estimators = estimators
        self.voting = voting

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "EnsembleAD":
        """Fit every member on the same training data."""
        if self.voting not in ("majority", "unanimous", "any"):
            raise ValueError(
                f"voting must be 'majority', 'unanimous' or 'any', got {self.voting!r}."
            )
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")

        members: List[BaseApplicabilityDomain] = (
            list(self.estimators)
            if self.estimators is not None
            else [LeverageAD(), KNNApplicabilityDomain(), RangeAD()]
        )
        if not members:
            raise ValueError("EnsembleAD needs at least one member estimator.")

        self._members = [m.fit(arr) for m in members]
        self.n_features_in_ = arr.shape[1]
        self.threshold_ = {
            "majority": 0.5,
            "unanimous": 0.0,
            "any": 1.0 - 1e-9,
        }[self.voting]
        return self

    def score_samples(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Fraction of member estimators calling each sample out-of-domain."""
        self._check_fitted()
        arr = self._validate(X)
        outside = np.array([~m.predict(arr) for m in self._members], dtype=np.float64)
        return np.asarray(outside.mean(axis=0), dtype=np.float64)

    def member_predictions(self, X: npt.ArrayLike) -> "pd.DataFrame":
        """Per-member in-domain decisions, for diagnosing disagreement.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        pandas.DataFrame
            One boolean column per member, named after its class.
        """
        import pandas as pd

        self._check_fitted()
        arr = self._validate(X)
        return pd.DataFrame(
            {type(m).__name__: m.predict(arr) for m in self._members}
        )
