"""Taylor-Butina clustering with a scikit-learn estimator API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin

from qsarkit.neighbors._distance import jaccard_distance_matrix

if TYPE_CHECKING:  # pragma: no cover
    from rdkit.Chem import Mol

__all__ = ["ButinaClustering"]


class ButinaClustering(BaseEstimator, ClusterMixin):
    """Taylor-Butina sphere-exclusion clustering of fingerprints.

    The standard clustering algorithm of cheminformatics. Unlike k-means
    it needs no ``n_clusters``, produces deterministic results, and its
    single parameter (``cutoff``) is a chemically meaningful similarity
    threshold. The algorithm:

    1. Count, for every molecule, how many neighbours fall within the
       distance ``cutoff`` — its "neighbour count".
    2. Take the molecule with the largest count as a cluster centroid,
       and assign all of its unassigned neighbours to that cluster.
    3. Repeat with the next-largest unassigned molecule until none remain.

    Molecules that end up alone form singleton clusters, which is
    informative: a large singleton fraction means the library is
    structurally diverse (or the cutoff is too tight).

    Parameters
    ----------
    cutoff : float, default 0.35
        Maximum Jaccard distance for two molecules to be neighbours,
        i.e. a Tanimoto similarity of ``1 - cutoff``. The 0.35 default
        (Tanimoto 0.65) is the customary value for ECFP4.
    metric : {"jaccard", "precomputed"}, default "jaccard"
        ``"jaccard"`` computes Tanimoto distances from fingerprints;
        ``"precomputed"`` treats ``X`` as a square distance matrix.
    reordering : bool, default False
        If True, re-sort the remaining candidates by neighbour count
        after each cluster is formed (the "reordering" variant, which
        tends to give tighter clusters at higher cost).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster index of each sample, ordered by descending cluster size.
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Index of the centroid molecule of each cluster.
    n_clusters_ : int
        Number of clusters found.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1]])
    >>> model = ButinaClustering(cutoff=0.5).fit(X)
    >>> model.n_clusters_ >= 1
    True
    >>> model.labels_.shape
    (4,)

    References
    ----------
    - Butina, D. (1999). "Unsupervised Data Base Clustering Based on
      Daylight's Fingerprint and Tanimoto Similarity: A Fast and
      Automated Way to Cluster Small and Large Data Sets."
      J. Chem. Inf. Comput. Sci., 39(4), 747-750.
      https://doi.org/10.1021/ci9803381
    - Taylor, R. (1995). "Simulation Analysis of Experimental Design
      Strategies for Screening Random Compounds as Potential New Drugs
      and Agrochemicals." J. Chem. Inf. Comput. Sci., 35(1), 59-67.
      https://doi.org/10.1021/ci00023a009
    - RDKit ``rdSimDivPickers``/``Butina`` documentation:
      https://www.rdkit.org/docs/source/rdkit.ML.Cluster.Butina.html
    """

    labels_: npt.NDArray[np.intp]
    cluster_centers_indices_: npt.NDArray[np.intp]
    n_clusters_: int

    def __init__(
        self,
        cutoff: float = 0.35,
        metric: str = "jaccard",
        reordering: bool = False,
    ) -> None:
        self.cutoff = cutoff
        self.metric = metric
        self.reordering = reordering

    def _distance_matrix(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.metric == "precomputed":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    "With metric='precomputed', X must be square; got shape "
                    f"{arr.shape}."
                )
            return arr
        if self.metric == "jaccard":
            return jaccard_distance_matrix(arr)
        raise ValueError(
            f"metric must be 'jaccard' or 'precomputed', got {self.metric!r}."
        )

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "ButinaClustering":
        """Cluster the fingerprints (or precomputed distance matrix).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
            Fingerprints, or a square distance matrix when
            ``metric="precomputed"``.
        y : ignored
            Present for scikit-learn API compatibility.

        Returns
        -------
        ButinaClustering
            The fitted estimator.
        """
        if not 0.0 <= self.cutoff <= 1.0:
            raise ValueError(f"cutoff must be in [0, 1], got {self.cutoff}.")

        dist = self._distance_matrix(X)
        n = dist.shape[0]
        if n == 0:
            raise ValueError("Cannot cluster an empty dataset.")

        within = dist <= self.cutoff
        np.fill_diagonal(within, False)
        neighbor_counts = within.sum(axis=1)

        assigned = np.zeros(n, dtype=bool)
        labels = np.full(n, -1, dtype=np.intp)
        centroids: list[int] = []
        clusters: list[list[int]] = []

        candidates = list(np.argsort(-neighbor_counts, kind="stable"))
        while True:
            if self.reordering:
                # Recount neighbours among still-unassigned molecules only.
                live = ~assigned
                if not live.any():
                    break
                counts = (within & live[None, :]).sum(axis=1)
                counts[assigned] = -1
                seed = int(np.argmax(counts))
                if counts[seed] < 0:
                    break
            else:
                seed = -1
                while candidates:
                    nxt = int(candidates.pop(0))
                    if not assigned[nxt]:
                        seed = nxt
                        break
                if seed < 0:
                    break

            members = [seed]
            assigned[seed] = True
            for j in np.flatnonzero(within[seed]):
                if not assigned[j]:
                    assigned[j] = True
                    members.append(int(j))
            centroids.append(seed)
            clusters.append(members)

            if assigned.all():
                break

        # Relabel so cluster 0 is the largest (stable, deterministic).
        order = sorted(range(len(clusters)), key=lambda i: (-len(clusters[i]), centroids[i]))
        for new_label, old in enumerate(order):
            labels[clusters[old]] = new_label

        self.labels_ = labels
        self.cluster_centers_indices_ = np.array(
            [centroids[i] for i in order], dtype=np.intp
        )
        self.n_clusters_ = len(clusters)
        return self

    def fit_predict(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.intp]:
        """Fit and return ``labels_``.

        Parameters
        ----------
        X : array-like
        y : ignored

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster labels.
        """
        return self.fit(X, y).labels_
