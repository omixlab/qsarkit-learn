"""k-NN classifier and regressor using Jaccard/Tanimoto distance."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.neighbors._search import JaccardNeighborSearch

__all__ = ["JaccardKNeighborsClassifier", "JaccardKNeighborsRegressor"]

_Weights = Literal["uniform", "distance", "similarity"]


def _neighbor_weights(
    distances: npt.NDArray[np.float64], weights: _Weights
) -> npt.NDArray[np.float64]:
    """Turn neighbour distances into per-neighbour voting weights."""
    if weights == "uniform":
        return np.ones_like(distances)
    if weights == "similarity":
        # Tanimoto similarity itself as the weight - the natural choice
        # for fingerprint neighbourhoods, and bounded in [0, 1].
        return 1.0 - distances
    if weights == "distance":
        with np.errstate(divide="ignore"):
            w = 1.0 / distances
        # An exact match (distance 0) must dominate its row entirely.
        exact = np.isinf(w)
        rows_with_exact = exact.any(axis=1)
        w[rows_with_exact] = exact[rows_with_exact].astype(np.float64)
        return w
    raise ValueError(
        f"weights must be 'uniform', 'distance' or 'similarity', got {weights!r}."
    )


class _BaseJaccardKNN(BaseEstimator):
    """Shared fit/neighbour machinery for the Jaccard k-NN estimators."""

    _search: JaccardNeighborSearch
    n_features_in_: int

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: _Weights = "uniform",
        chunk_size: int = 1024,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.chunk_size = chunk_size

    def _fit_search(self, X: npt.ArrayLike, y: npt.ArrayLike) -> npt.NDArray[np.float64]:
        X_arr = np.asarray(X, dtype=np.float64)
        y_arr = np.asarray(y)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X_arr.shape}.")
        if len(X_arr) != len(y_arr):
            raise ValueError(
                f"X has {len(X_arr)} samples but y has {len(y_arr)}."
            )
        if self.n_neighbors > len(X_arr):
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the training set size "
                f"({len(X_arr)})."
            )
        self._search = JaccardNeighborSearch(
            n_neighbors=self.n_neighbors, chunk_size=self.chunk_size
        ).fit(X_arr)
        self.n_features_in_ = X_arr.shape[1]
        return y_arr

    def _neighbors(
        self, X: npt.ArrayLike
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
        if not hasattr(self, "_search"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling predict()."
            )
        result = self._search.kneighbors(X, n_neighbors=self.n_neighbors)
        assert isinstance(result, tuple)
        return result

    def kneighbors(
        self, X: npt.ArrayLike, n_neighbors: Optional[int] = None
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]]:
        """Expose the underlying neighbour query (distances, indices)."""
        if not hasattr(self, "_search"):
            raise ModelNotFittedError(
                f"{type(self).__name__} must be fitted before calling kneighbors()."
            )
        result = self._search.kneighbors(
            X, n_neighbors=n_neighbors or self.n_neighbors
        )
        assert isinstance(result, tuple)
        return result


class JaccardKNeighborsClassifier(ClassifierMixin, _BaseJaccardKNN):
    """k-nearest-neighbours classifier under Jaccard/Tanimoto distance.

    The canonical similarity-based QSAR classifier: a compound is
    predicted from the observed classes of its most structurally similar
    training neighbours. Using Tanimoto rather than Euclidean distance
    matters because fingerprints are sparse binary vectors, where
    Euclidean distance is dominated by the (huge) number of jointly-absent
    bits — exactly the bits that carry no chemical information.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbours voting on each prediction.
    weights : {"uniform", "distance", "similarity"}, default "uniform"
        Vote weighting. ``"similarity"`` weights each neighbour by its
        Tanimoto coefficient; ``"distance"`` by ``1/d`` (with exact
        matches taking the row outright).
    chunk_size : int, default 1024
        Query block size for the similarity computation.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Sorted class labels seen during :meth:`fit`.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 0]])
    >>> y = np.array([1, 1, 0, 0])
    >>> clf = JaccardKNeighborsClassifier(n_neighbors=1).fit(X, y)
    >>> int(clf.predict(np.array([[1, 1, 0, 0]]))[0])
    1

    References
    ----------
    - Cover, T. & Hart, P. (1967). "Nearest Neighbor Pattern
      Classification." IEEE Trans. Inf. Theory, 13(1), 21-27.
      https://doi.org/10.1109/TIT.1967.1053964
    - Bajusz, D., Racz, A. & Heberger, K. (2015). "Why is Tanimoto Index
      an Appropriate Choice for Fingerprint-Based Similarity
      Calculations?" J. Cheminform., 7, 20.
      https://doi.org/10.1186/s13321-015-0069-3
    - Willett, P. (2006). "Similarity-Based Virtual Screening Using 2D
      Fingerprints." Drug Discov. Today, 11(23-24), 1046-1053.
      https://doi.org/10.1016/j.drudis.2006.10.005
    - scikit-learn classifier API:
      https://scikit-learn.org/stable/developers/develop.html
    """

    classes_: npt.NDArray[np.generic]

    def fit(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> "JaccardKNeighborsClassifier":
        """Store training fingerprints and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training fingerprints.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        JaccardKNeighborsClassifier
            The fitted estimator.
        """
        y_arr = self._fit_search(X, y)
        self.classes_ = unique_labels(y_arr)
        self._y_encoded = np.searchsorted(self.classes_, y_arr)
        return self

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Class probabilities as the weighted neighbour vote share.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries, n_classes)
            Rows sum to 1.
        """
        distances, indices = self._neighbors(X)
        w = _neighbor_weights(distances, self.weights)
        neighbor_classes = self._y_encoded[indices]

        proba = np.zeros((len(neighbor_classes), len(self.classes_)), dtype=np.float64)
        for c in range(len(self.classes_)):
            proba[:, c] = (w * (neighbor_classes == c)).sum(axis=1)
        totals = proba.sum(axis=1, keepdims=True)
        # All-zero weights (every neighbour at similarity 0) -> uniform.
        proba = np.where(totals > 0, proba / np.where(totals > 0, totals, 1.0),
                         1.0 / len(self.classes_))
        return proba

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Predict the majority (weighted) class of each query's neighbours.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries,)
            Predicted labels drawn from ``classes_``.
        """
        if not hasattr(self, "classes_"):
            raise ModelNotFittedError(
                "JaccardKNeighborsClassifier must be fitted before calling "
                "predict()."
            )
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class JaccardKNeighborsRegressor(RegressorMixin, _BaseJaccardKNN):
    """k-nearest-neighbours regressor under Jaccard/Tanimoto distance.

    Predicts a continuous endpoint (pIC50, logS, ...) as the weighted
    mean of its structurally nearest training neighbours — the numerical
    form of read-across, and a strong similarity-only baseline that any
    fitted QSAR model should be expected to beat.

    Parameters
    ----------
    n_neighbors : int, default 5
        Number of neighbours averaged for each prediction.
    weights : {"uniform", "distance", "similarity"}, default "uniform"
        Averaging weights, as in :class:`JaccardKNeighborsClassifier`.
    chunk_size : int, default 1024
        Query block size for the similarity computation.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]])
    >>> y = np.array([7.0, 6.5, 4.0])
    >>> reg = JaccardKNeighborsRegressor(n_neighbors=1).fit(X, y)
    >>> float(reg.predict(np.array([[1, 1, 0, 0]]))[0])
    7.0

    References
    ----------
    - Cover, T. & Hart, P. (1967). IEEE Trans. Inf. Theory, 13(1), 21-27.
      https://doi.org/10.1109/TIT.1967.1053964
    - Sheridan, R. P. et al. (2004). "Similarity to Molecules in the
      Training Set Is a Good Discriminator for Prediction Accuracy in
      QSAR." J. Chem. Inf. Comput. Sci., 44(6), 1912-1928.
      https://doi.org/10.1021/ci049782w
    - Bajusz, D., Racz, A. & Heberger, K. (2015). J. Cheminform., 7, 20.
      https://doi.org/10.1186/s13321-015-0069-3
    """

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> "JaccardKNeighborsRegressor":
        """Store training fingerprints and endpoint values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training fingerprints.
        y : array-like of shape (n_samples,)
            Continuous target values.

        Returns
        -------
        JaccardKNeighborsRegressor
            The fitted estimator.
        """
        self._y = self._fit_search(X, y).astype(np.float64)
        return self

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Predict as the weighted mean of the neighbours' target values.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)

        Returns
        -------
        ndarray of shape (n_queries,)
            Predicted values.
        """
        distances, indices = self._neighbors(X)
        w = _neighbor_weights(distances, self.weights)
        totals = w.sum(axis=1, keepdims=True)
        # No usable similarity anywhere -> fall back to an unweighted mean.
        w = np.where(totals > 0, w, 1.0)
        totals = np.where(totals > 0, totals, w.sum(axis=1, keepdims=True))
        return np.asarray((w * self._y[indices]).sum(axis=1) / totals.ravel())
