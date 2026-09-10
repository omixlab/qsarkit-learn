"""scikit-learn wrappers around RDKit's diversity pickers and clustering tools."""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClusterMixin

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.neighbors._distance import (
    jaccard_distance_matrix,
    tanimoto_similarity_matrix,
)

__all__ = [
    "SphereExclusionClustering",
    "MaxMinPicker",
    "HierarchicalClustering",
]


class SphereExclusionClustering(BaseEstimator, ClusterMixin):
    """Sphere-exclusion clustering (RDKit ``LeaderPicker`` semantics).

    Greedily selects "leader" molecules such that no two leaders are
    within ``cutoff`` distance of each other, then assigns every
    remaining molecule to its nearest leader. Unlike Butina it does not
    need the full neighbour-count pass, so it scales to very large
    libraries, and it guarantees a minimum inter-centroid distance —
    which is what makes it the usual choice for picking a diverse
    screening subset.

    Parameters
    ----------
    cutoff : float, default 0.35
        Minimum Jaccard distance between any two leaders.
    metric : {"jaccard", "precomputed"}, default "jaccard"
        Distance source, as in :class:`~qsarkit.cluster.ButinaClustering`.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster assignment of each sample.
    cluster_centers_indices_ : ndarray of shape (n_clusters,)
        Indices of the selected leaders.
    n_clusters_ : int
        Number of leaders selected.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 1]])
    >>> model = SphereExclusionClustering(cutoff=0.5).fit(X)
    >>> model.labels_.shape
    (3,)

    References
    ----------
    - Hudson, B. D. et al. (1996). "Parameter Based Methods for Compound
      Selection from Chemical Databases." Quant. Struct.-Act. Relat.,
      15(4), 285-289. https://doi.org/10.1002/qsar.19960150402
    - Gobbi, A. & Lee, M.-L. (2003). "DISE: Directed Sphere Exclusion."
      J. Chem. Inf. Comput. Sci., 43(1), 317-323.
      https://doi.org/10.1021/ci025554v
    - RDKit ``rdSimDivPickers.LeaderPicker`` documentation:
      https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html
    """

    labels_: npt.NDArray[np.intp]
    cluster_centers_indices_: npt.NDArray[np.intp]
    n_clusters_: int

    def __init__(self, cutoff: float = 0.35, metric: str = "jaccard") -> None:
        self.cutoff = cutoff
        self.metric = metric

    def _distance_matrix(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.metric == "precomputed":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"With metric='precomputed', X must be square; got {arr.shape}."
                )
            return arr
        if self.metric == "jaccard":
            return jaccard_distance_matrix(arr)
        raise ValueError(
            f"metric must be 'jaccard' or 'precomputed', got {self.metric!r}."
        )

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "SphereExclusionClustering":
        """Select leaders and assign every sample to its nearest one.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored

        Returns
        -------
        SphereExclusionClustering
            The fitted estimator.
        """
        if not 0.0 <= self.cutoff <= 1.0:
            raise ValueError(f"cutoff must be in [0, 1], got {self.cutoff}.")

        dist = self._distance_matrix(X)
        n = dist.shape[0]
        if n == 0:
            raise ValueError("Cannot cluster an empty dataset.")

        leaders: list[int] = []
        for i in range(n):
            if all(dist[i, leader] > self.cutoff for leader in leaders):
                leaders.append(i)

        centers = np.array(leaders, dtype=np.intp)
        self.labels_ = np.asarray(
            np.argmin(dist[:, centers], axis=1), dtype=np.intp
        )
        self.cluster_centers_indices_ = centers
        self.n_clusters_ = len(leaders)
        return self

    def fit_predict(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.intp]:
        """Fit and return ``labels_``."""
        return self.fit(X, y).labels_


class MaxMinPicker(BaseEstimator):
    """MaxMin diverse-subset selection (RDKit ``MaxMinPicker`` semantics).

    Iteratively picks the molecule whose minimum distance to the already
    picked set is largest, producing a maximally spread-out subset. This
    is the standard way to choose a diverse plate from a large library,
    and the diversity-sampling primitive used by
    :mod:`qsarkit.active_learning`.

    Parameters
    ----------
    n_to_pick : int, default 10
        Size of the diverse subset to select.
    metric : {"jaccard", "precomputed"}, default "jaccard"
        Distance source.
    seed_index : int, optional
        Index of the first pick. When ``None`` (default) the molecule
        farthest from the dataset centroid is used, which makes the
        result deterministic without a random seed.

    Attributes
    ----------
    picks_ : ndarray of shape (n_to_pick,)
        Indices of the selected molecules, in selection order.
    min_distances_ : ndarray of shape (n_to_pick,)
        The MaxMin distance achieved at each pick — a monotonically
        non-increasing diversity profile of the selection.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1]])
    >>> picker = MaxMinPicker(n_to_pick=2).fit(X)
    >>> len(picker.picks_)
    2

    References
    ----------
    - Ashton, M. et al. (2002). "Identification of Diverse Database
      Subsets using Property-Based and Fragment-Based Molecular
      Descriptions." Quant. Struct.-Act. Relat., 21(6), 598-604.
      https://doi.org/10.1002/qsar.200290002
    - Higgs, R. E. et al. (1997). "A Genetic Algorithm Approach to
      Similarity-Based Compound Selection." J. Chem. Inf. Comput. Sci.,
      37(5), 861-870. https://doi.org/10.1021/ci9702858
    - RDKit ``rdSimDivPickers.MaxMinPicker`` documentation:
      https://www.rdkit.org/docs/source/rdkit.SimDivFilters.rdSimDivPickers.html
    """

    picks_: npt.NDArray[np.intp]
    min_distances_: npt.NDArray[np.float64]

    def __init__(
        self,
        n_to_pick: int = 10,
        metric: str = "jaccard",
        seed_index: Optional[int] = None,
    ) -> None:
        self.n_to_pick = n_to_pick
        self.metric = metric
        self.seed_index = seed_index

    def fit(self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None) -> "MaxMinPicker":
        """Select the diverse subset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_samples)
        y : ignored

        Returns
        -------
        MaxMinPicker
            The fitted picker.
        """
        arr = np.asarray(X, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {arr.shape}.")
        if self.metric == "precomputed":
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(
                    f"With metric='precomputed', X must be square; got {arr.shape}."
                )
            dist = arr
        elif self.metric == "jaccard":
            dist = jaccard_distance_matrix(arr)
        else:
            raise ValueError(
                f"metric must be 'jaccard' or 'precomputed', got {self.metric!r}."
            )

        n = dist.shape[0]
        if self.n_to_pick > n:
            raise ValueError(
                f"n_to_pick={self.n_to_pick} exceeds the dataset size ({n})."
            )
        if self.n_to_pick <= 0:
            raise ValueError(f"n_to_pick must be positive, got {self.n_to_pick}.")

        if self.seed_index is None:
            first = int(np.argmax(dist.sum(axis=1)))
        else:
            if not 0 <= self.seed_index < n:
                raise ValueError(
                    f"seed_index={self.seed_index} out of range for {n} samples."
                )
            first = self.seed_index

        picks = [first]
        achieved = [float("inf")]
        min_dist = dist[first].copy()
        min_dist[first] = -np.inf

        while len(picks) < self.n_to_pick:
            nxt = int(np.argmax(min_dist))
            achieved.append(float(min_dist[nxt]))
            picks.append(nxt)
            min_dist = np.minimum(min_dist, dist[nxt])
            min_dist[picks] = -np.inf

        self.picks_ = np.array(picks, dtype=np.intp)
        self.min_distances_ = np.array(achieved, dtype=np.float64)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the rows of ``X`` corresponding to the picks.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_to_pick, n_features)
        """
        if not hasattr(self, "picks_"):
            raise ModelNotFittedError("MaxMinPicker must be fitted before transform().")
        return np.asarray(X, dtype=np.float64)[self.picks_]

    def fit_transform(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.float64]:
        """Fit then return the picked rows."""
        return self.fit(X, y).transform(X)


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """Agglomerative clustering on Tanimoto distances.

    A thin wrapper that computes the Jaccard distance matrix and hands it
    to scikit-learn's ``AgglomerativeClustering`` with
    ``metric="precomputed"``, so that hierarchical clustering can be used
    on fingerprints with the chemically correct distance. Ward linkage is
    not available for precomputed distances; use ``"average"`` (the
    default here, equivalent to RDKit's UPGMA option), ``"complete"`` or
    ``"single"``.

    Parameters
    ----------
    n_clusters : int, optional, default 2
        Number of clusters. Pass ``None`` together with
        ``distance_threshold`` to cut the dendrogram by distance instead.
    linkage : {"average", "complete", "single"}, default "average"
        Linkage criterion.
    distance_threshold : float, optional
        Distance at which to cut the dendrogram. Requires
        ``n_clusters=None``.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    n_clusters_ : int
        Number of clusters found.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]])
    >>> model = HierarchicalClustering(n_clusters=2).fit(X)
    >>> model.n_clusters_
    2

    References
    ----------
    - Sokal, R. R. & Michener, C. D. (1958). "A Statistical Method for
      Evaluating Systematic Relationships" (UPGMA). Univ. Kansas Sci.
      Bull., 38, 1409-1438.
    - Downs, G. M. & Barnard, J. M. (2002). "Clustering Methods and Their
      Uses in Computational Chemistry." Rev. Comput. Chem., 18, 1-40.
      https://doi.org/10.1002/0471433519.ch1
    - scikit-learn ``AgglomerativeClustering`` documentation:
      https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
    """

    labels_: npt.NDArray[np.intp]
    n_clusters_: int

    def __init__(
        self,
        n_clusters: Optional[int] = 2,
        linkage: str = "average",
        distance_threshold: Optional[float] = None,
    ) -> None:
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.distance_threshold = distance_threshold

    def fit(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> "HierarchicalClustering":
        """Cluster fingerprints by agglomerative linkage on Tanimoto distance.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        HierarchicalClustering
            The fitted estimator.
        """
        from sklearn.cluster import AgglomerativeClustering

        if self.linkage == "ward":
            raise ValueError(
                "Ward linkage requires Euclidean distances and cannot be used "
                "with precomputed Tanimoto distances; use 'average', "
                "'complete' or 'single'."
            )
        dist = jaccard_distance_matrix(X)
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            metric="precomputed",
            linkage=self.linkage,
            distance_threshold=self.distance_threshold,
        )
        self.labels_ = np.asarray(model.fit_predict(dist), dtype=np.intp)
        self.n_clusters_ = int(self.labels_.max()) + 1
        return self

    def fit_predict(
        self, X: npt.ArrayLike, y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.intp]:
        """Fit and return ``labels_``."""
        return self.fit(X, y).labels_
