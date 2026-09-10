"""Fingerprint similarity search with Jaccard/Tanimoto distance."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.neighbors._distance import tanimoto_similarity_matrix

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["JaccardNeighborSearch"]


class JaccardNeighborSearch(BaseEstimator):
    """Nearest-neighbour search over fingerprints under Jaccard/Tanimoto distance.

    This is the similarity-search primitive behind chemical-similarity
    applicability domains, read-across, and virtual screening triage. It
    is deliberately exact (brute force over a vectorized similarity
    matrix) rather than approximate: chemical fingerprint spaces are
    high-dimensional and sparse, where tree-based indices degrade to
    linear scans anyway.

    Parameters
    ----------
    n_neighbors : int, default 5
        Default number of neighbours returned by :meth:`kneighbors`.
    chunk_size : int, default 1024
        Number of query rows processed per block, bounding peak memory
        at ``chunk_size x n_train`` floats.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The fitted reference fingerprints.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]])
    >>> search = JaccardNeighborSearch(n_neighbors=2).fit(X)
    >>> dist, idx = search.kneighbors(np.array([[1, 1, 0, 0]]))
    >>> int(idx[0, 0])
    0
    >>> float(dist[0, 0])
    0.0

    References
    ----------
    - Willett, P., Barnard, J. M. & Downs, G. M. (1998). "Chemical
      Similarity Searching." J. Chem. Inf. Comput. Sci., 38(6), 983-996.
      https://doi.org/10.1021/ci9800211
    - Bajusz, D., Racz, A. & Heberger, K. (2015). "Why is Tanimoto Index
      an Appropriate Choice for Fingerprint-Based Similarity
      Calculations?" J. Cheminform., 7, 20.
      https://doi.org/10.1186/s13321-015-0069-3
    - scikit-learn nearest-neighbours documentation:
      https://scikit-learn.org/stable/modules/neighbors.html
    """

    X_: npt.NDArray[np.float64]
    n_features_in_: int

    def __init__(self, n_neighbors: int = 5, chunk_size: int = 1024) -> None:
        self.n_neighbors = n_neighbors
        self.chunk_size = chunk_size

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "JaccardNeighborSearch":
        """Store the reference fingerprint matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Reference fingerprints.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        JaccardNeighborSearch
            The fitted estimator.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError("Cannot fit on an empty reference set.")
        self.X_ = arr
        self.n_features_in_ = arr.shape[1]
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "X_"):
            raise ModelNotFittedError(
                "JaccardNeighborSearch must be fitted before querying."
            )

    def _similarity(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        self._check_fitted()
        query = np.asarray(X, dtype=np.float64)
        if query.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {query.shape}.")
        if query.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {query.shape[1]} features, expected {self.n_features_in_}."
            )
        blocks = [
            tanimoto_similarity_matrix(query[i : i + self.chunk_size], self.X_)
            for i in range(0, query.shape[0], self.chunk_size)
        ]
        return np.vstack(blocks)

    def kneighbors(
        self,
        X: npt.ArrayLike,
        n_neighbors: Optional[int] = None,
        return_distance: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.intp]] | npt.NDArray[np.intp]:
        """Find the ``k`` most similar reference fingerprints for each query.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query fingerprints.
        n_neighbors : int, optional
            Overrides ``self.n_neighbors`` for this call.
        return_distance : bool, default True
            If True return ``(distances, indices)``, else just ``indices``.

        Returns
        -------
        distances : ndarray of shape (n_queries, n_neighbors)
            Jaccard distances, ascending.
        indices : ndarray of shape (n_queries, n_neighbors)
            Indices into the fitted reference set.
        """
        k = self.n_neighbors if n_neighbors is None else n_neighbors
        sim = self._similarity(X)
        n_ref = sim.shape[1]
        if k > n_ref:
            raise ValueError(
                f"n_neighbors={k} exceeds the number of reference samples ({n_ref})."
            )

        # argpartition gives the top-k cheaply; sort only that slice.
        part = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sim.shape[0])[:, None]
        order = np.argsort(-sim[rows, part], axis=1, kind="stable")
        indices = part[rows, order]
        if not return_distance:
            return indices
        distances = 1.0 - sim[rows, indices]
        return distances, indices

    def radius_neighbors(
        self, X: npt.ArrayLike, radius: float = 0.3
    ) -> Tuple[list[npt.NDArray[np.float64]], list[npt.NDArray[np.intp]]]:
        """Find all reference fingerprints within ``radius`` Jaccard distance.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query fingerprints.
        radius : float, default 0.3
            Maximum Jaccard distance, i.e. minimum Tanimoto similarity
            of ``1 - radius``.

        Returns
        -------
        distances : list of ndarray
            Per-query distances to the in-radius neighbours, ascending.
        indices : list of ndarray
            Per-query reference indices.
        """
        if not 0.0 <= radius <= 1.0:
            raise ValueError(f"radius must be in [0, 1], got {radius}.")
        sim = self._similarity(X)
        dist = 1.0 - sim
        out_d: list[npt.NDArray[np.float64]] = []
        out_i: list[npt.NDArray[np.intp]] = []
        for row in dist:
            hits = np.flatnonzero(row <= radius)
            order = np.argsort(row[hits], kind="stable")
            out_i.append(hits[order])
            out_d.append(row[hits][order])
        return out_d, out_i

    def similarity_search(
        self,
        X: npt.ArrayLike,
        threshold: float = 0.7,
        max_hits: Optional[int] = None,
    ) -> list[list[Tuple[int, float]]]:
        """Classic similarity search: hits above a Tanimoto ``threshold``.

        Parameters
        ----------
        X : array-like of shape (n_queries, n_features)
            Query fingerprints.
        threshold : float, default 0.7
            Minimum Tanimoto similarity for a hit. 0.7 on ECFP4 is the
            long-standing rule-of-thumb cutoff for "similar" molecules.
        max_hits : int, optional
            Truncate each query's hit list to this many best hits.

        Returns
        -------
        list of list of (index, similarity)
            Per-query hits sorted by descending similarity.

        References
        ----------
        - Maggiora, G. et al. (2014). "Molecular Similarity in Medicinal
          Chemistry." J. Med. Chem., 57(8), 3186-3204.
          https://doi.org/10.1021/jm401411z
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}.")
        sim = self._similarity(X)
        results: list[list[Tuple[int, float]]] = []
        for row in sim:
            hits = np.flatnonzero(row >= threshold)
            order = np.argsort(-row[hits], kind="stable")
            ranked = [(int(hits[j]), float(row[hits[j]])) for j in order]
            results.append(ranked if max_hits is None else ranked[:max_hits])
        return results
