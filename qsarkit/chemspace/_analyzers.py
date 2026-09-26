"""Chemical space analysis: embedding, diversity, clustering, scaffolds."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from qsarkit.base.exceptions import ModelNotFittedError
from qsarkit.chemspace._fingerprints import (
    bemis_murcko_smiles,
    compute_fingerprints,
    fingerprints_to_array,
)

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd
    import plotly.graph_objects as go
    from rdkit.Chem import Mol

__all__ = [
    "ChemicalSpaceAnalyzer",
    "DiversityAnalyzer",
    "ClusterAnalyzer",
    "NearestNeighborAnalyzer",
    "ScaffoldAnalyzer",
    "ChemicalSpaceCoverage",
]


def _features(
    mols: Sequence[Any], radius: int = 2, n_bits: int = 2048
) -> npt.NDArray[np.float64]:
    """ECFP bit matrix for a molecule sequence."""
    return fingerprints_to_array(compute_fingerprints(mols, radius, n_bits))


class ChemicalSpaceAnalyzer:
    """Project a compound collection into two dimensions for inspection.

    Chemical space is high-dimensional and sparse, so any 2D picture of
    it is a lossy projection — but the *right* projection answers real
    questions: whether a library covers one region or several, whether
    the test set sits inside the training set's cloud, whether a
    screening hit is an outlier.

    The three methods answer different questions and are not
    interchangeable. PCA preserves global variance and its axes are
    interpretable, but it flattens the non-linear structure fingerprints
    actually have. t-SNE and UMAP preserve local neighbourhoods and give
    the familiar island plots, but between-cluster distances in those
    plots are *not* meaningful — reading them as chemical distance is the
    commonest misuse of the technique.

    Parameters
    ----------
    method : {"pca", "tsne", "mds", "umap"}, default "pca"
        Projection method. ``"umap"`` needs the optional ``umap-learn``
        package.
    n_components : int, default 2
        Output dimensionality.
    metric : {"jaccard", "euclidean"}, default "jaccard"
        Distance used by the neighbourhood methods. Jaccard/Tanimoto is
        correct for fingerprints.
    random_state : int, optional
        Seed.
    **kwargs
        Forwarded to the underlying estimator (``perplexity``,
        ``n_neighbors``, ...).

    Attributes
    ----------
    embedding_ : ndarray of shape (n_molecules, n_components)
        The projected coordinates.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1", "CCC")]
    >>> analyzer = ChemicalSpaceAnalyzer(random_state=0).fit(mols)
    >>> analyzer.embedding_.shape
    (4, 2)

    References
    ----------
    - van der Maaten, L. & Hinton, G. (2008). "Visualizing Data Using
      t-SNE." J. Mach. Learn. Res., 9, 2579-2605.
      https://jmlr.org/papers/v9/vandermaaten08a.html
    - McInnes, L., Healy, J. & Melville, J. (2018). "UMAP: Uniform
      Manifold Approximation and Projection." arXiv:1802.03426.
      https://arxiv.org/abs/1802.03426
    - Wattenberg, M., Viegas, F. & Johnson, I. (2016). "How to Use t-SNE
      Effectively." Distill. https://doi.org/10.23915/distill.00002
    - Osolodkin, D. I. et al. (2015). "Progress in Visual
      Representations of Chemical Space." Expert Opin. Drug Discov.,
      10(9), 959-973. https://doi.org/10.1517/17460441.2015.1060216
    """

    embedding_: npt.NDArray[np.float64]

    def __init__(
        self,
        method: Literal["pca", "tsne", "mds", "umap"] = "pca",
        n_components: int = 2,
        metric: Literal["jaccard", "euclidean"] = "jaccard",
        random_state: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.method = method
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.kwargs = kwargs

    def _distances(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        from scipy.spatial.distance import cdist

        from qsarkit.neighbors import jaccard_distance_matrix

        if self.metric == "jaccard":
            return jaccard_distance_matrix(X)
        if self.metric == "euclidean":
            return np.asarray(cdist(X, X, metric="euclidean"), dtype=np.float64)
        raise ValueError(
            f"metric must be 'jaccard' or 'euclidean', got {self.metric!r}."
        )

    def fit(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> "ChemicalSpaceAnalyzer":
        """Project the molecules.

        Parameters
        ----------
        mols : sequence of Mol or array-like
            Molecules, or a precomputed feature matrix.
        y : ignored

        Returns
        -------
        ChemicalSpaceAnalyzer
        """
        X = self._featurize(mols)
        if len(X) < 2:
            raise ValueError(
                f"Need at least 2 molecules to embed, got {len(X)}."
            )

        # A copy, because the branches below consume entries with pop(). Popping
        # from self.kwargs mutates a constructor argument: the user's
        # `perplexity` was honoured by the first fit() and silently forgotten by
        # the second, so two identical calls on one object disagreed.
        options = dict(self.kwargs)

        if self.method == "pca":
            from sklearn.decomposition import PCA

            model = PCA(
                n_components=min(self.n_components, *X.shape),
                random_state=self.random_state,
                **options,
            )
            self.embedding_ = np.asarray(model.fit_transform(X), dtype=np.float64)
            self.explained_variance_ratio_ = model.explained_variance_ratio_
        elif self.method == "tsne":
            from sklearn.manifold import TSNE

            # Perplexity must stay below the sample count or sklearn raises.
            perplexity = options.pop(
                "perplexity", min(30.0, max(2.0, (len(X) - 1) / 3.0))
            )
            model = TSNE(
                n_components=self.n_components,
                metric="precomputed",
                init="random",
                perplexity=perplexity,
                random_state=self.random_state,
                **options,
            )
            self.embedding_ = np.asarray(
                model.fit_transform(self._distances(X)), dtype=np.float64
            )
        elif self.method == "mds":
            from sklearn.manifold import MDS

            # `n_init` and `init` are passed explicitly because sklearn is
            # changing both defaults; pinning them keeps results stable
            # across versions rather than shifting under the user.
            model = MDS(
                n_components=self.n_components,
                dissimilarity="precomputed",
                n_init=options.pop("n_init", 4),
                init=options.pop("init", "random"),
                random_state=self.random_state,
                **options,
            )
            self.embedding_ = np.asarray(
                model.fit_transform(self._distances(X)), dtype=np.float64
            )
        elif self.method == "umap":
            from qsarkit.base import require

            umap = require("umap")
            model = umap.UMAP(
                n_components=self.n_components,
                metric="jaccard" if self.metric == "jaccard" else "euclidean",
                random_state=self.random_state,
                **options,
            )
            self.embedding_ = np.asarray(model.fit_transform(X), dtype=np.float64)
        else:
            raise ValueError(
                "method must be 'pca', 'tsne', 'mds' or 'umap', "
                f"got {self.method!r}."
            )
        return self

    @staticmethod
    def _featurize(mols: Sequence[Any]) -> npt.NDArray[np.float64]:
        """Accept either molecules or a ready feature matrix."""
        from rdkit import Chem

        items = list(mols)
        if items and isinstance(items[0], Chem.Mol):
            return _features(items)
        return np.asarray(items, dtype=np.float64)

    def fit_transform(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.float64]:
        """Project and return the coordinates."""
        return self.fit(mols, y).embedding_

    def trustworthiness(
        self,
        mols: Sequence[Any],
        n_neighbors: Any = 5,
        subsample: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Any:
        """How much of the projection's local structure is real.

        Report this with any t-SNE or UMAP figure. Those methods produce
        convincing islands whose between-cluster distances mean nothing, and
        the picture looks the same whether or not the neighbourhoods survived
        the projection.

        Parameters
        ----------
        mols : sequence of Mol or array-like
            The same molecules or feature matrix passed to :meth:`fit`. They
            are not retained by ``fit``, because a fingerprint matrix for a
            screening library is large enough that keeping a copy is a real
            cost.
        n_neighbors : int or iterable of int, default 5
            Neighbourhood size, or several; several return an array in order.
        subsample : int, optional
            Score this many randomly chosen compounds. Trustworthiness needs
            the full pairwise distance matrix, so a large collection may need
            one.
        random_state : int, optional
            Seed for ``subsample``.

        Returns
        -------
        float or ndarray

        Raises
        ------
        AttributeError
            If called before :meth:`fit`.

        Examples
        --------
        >>> from rdkit import Chem
        >>> smiles = ["CCO", "CCN", "CCC", "c1ccccc1", "c1ccccc1O", "CCCl"]
        >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
        >>> analyzer = ChemicalSpaceAnalyzer(
        ...     method="pca", random_state=0).fit(mols)
        >>> score = analyzer.trustworthiness(mols, n_neighbors=2)
        >>> 0.0 <= score <= 1.0
        True
        """
        if not hasattr(self, "embedding_"):
            raise AttributeError(
                "Call fit() before trustworthiness(): there is no projection "
                "to score yet."
            )
        from qsarkit.chemspace._quality import projection_trustworthiness

        return projection_trustworthiness(
            self._featurize(mols),
            self.embedding_,
            n_neighbors=n_neighbors,
            metric=self.metric,
            subsample=subsample,
            random_state=random_state,
        )

    def plot(
        self,
        color: Optional[npt.ArrayLike] = None,
        labels: Optional[Sequence[str]] = None,
        title: str = "Chemical space",
    ) -> "go.Figure":
        """Scatter the embedding, optionally coloured by a property.

        Parameters
        ----------
        color : array-like, optional
            Per-molecule value used for colour, e.g. activity.
        labels : sequence of str, optional
            Hover labels.
        title : str
            Figure title.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as go

        if not hasattr(self, "embedding_"):
            raise ModelNotFittedError(
                "ChemicalSpaceAnalyzer must be fitted before plotting."
            )
        coords = self.embedding_
        marker: Dict[str, Any] = {"size": 7}
        if color is not None:
            marker.update(
                {
                    "color": np.asarray(color, dtype=np.float64),
                    "colorscale": "Viridis",
                    "showscale": True,
                }
            )
        fig = go.Figure(
            go.Scatter(
                x=coords[:, 0],
                y=coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(coords)),
                mode="markers",
                marker=marker,
                text=list(labels) if labels is not None else None,
            )
        )
        axis = "PC" if self.method == "pca" else self.method.upper()
        fig.update_layout(
            title=title, xaxis_title=f"{axis} 1", yaxis_title=f"{axis} 2"
        )
        return fig


class DiversityAnalyzer:
    """Quantify how diverse a compound collection is.

    "Diverse" needs a definition before it can be measured, and the
    available ones disagree. Mean pairwise distance rewards a few far-out
    outliers; scaffold count rewards structural variety regardless of
    distance; internal diversity is bounded and comparable between sets
    of different sizes. All three are reported, because a library can
    score well on one and poorly on another and the difference is
    informative.

    Parameters
    ----------
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1", "CCCCCC")]
    >>> report = DiversityAnalyzer().analyze(mols)
    >>> 0.0 <= report["mean_pairwise_distance"] <= 1.0
    True

    References
    ----------
    - Waldman, M., Li, H. & Hassan, M. (2000). "Novel Algorithms for the
      Optimization of Molecular Diversity of Combinatorial Libraries."
      J. Mol. Graph. Model., 18(4-5), 412-426.
      https://doi.org/10.1016/S1093-3263(00)00071-2
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known
      Drugs. 1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - Benhenda, M. (2017). "ChemGAN Challenge for Drug Discovery: Can AI
      Reproduce Natural Chemical Diversity?" arXiv:1708.08227.
      https://arxiv.org/abs/1708.08227
    - Shannon, C. E. (1948). "A Mathematical Theory of Communication."
      Bell Syst. Tech. J., 27(3), 379-423.
      https://doi.org/10.1002/j.1538-7305.1948.tb01338.x
    """

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        self.radius = radius
        self.n_bits = n_bits

    def analyze(self, mols: Sequence[Any]) -> Dict[str, float]:
        """Compute the diversity measures.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        dict
            ``n_molecules``, ``mean_pairwise_distance``,
            ``median_pairwise_distance``, ``internal_diversity``,
            ``n_scaffolds`` (acyclic molecules excluded -- they have
            no framework), ``acyclic_fraction``,
            ``scaffold_diversity`` (scaffolds per
            molecule), ``scaffold_entropy`` and ``bit_entropy``.
        """
        from qsarkit.neighbors import tanimoto_similarity_matrix

        if len(mols) < 2:
            raise ValueError(
                f"Diversity needs at least 2 molecules, got {len(mols)}."
            )
        X = _features(mols, self.radius, self.n_bits)
        similarity = tanimoto_similarity_matrix(X)
        upper = np.triu_indices(len(mols), k=1)
        distances = 1.0 - similarity[upper]

        # Acyclic molecules have no Bemis-Murcko framework, and
        # `bemis_murcko_smiles` returns "" for them. Counting that empty
        # string as a scaffold would inflate the count and let a library
        # of straight-chain molecules look scaffold-diverse. ScaffoldAnalyzer
        # excludes it too, so the two agree on what `n_scaffolds` means.
        scaffolds = [bemis_murcko_smiles(m) for m in mols]
        counts = Counter(s for s in scaffolds if s)
        n_cyclic = sum(counts.values())
        proportions = np.array(
            [c / n_cyclic for c in counts.values()], dtype=np.float64
        )
        scaffold_entropy = (
            float(-(proportions * np.log(proportions)).sum()) if n_cyclic else 0.0
        )

        # Entropy of the bit-set frequencies: a library that sets the same
        # bits in every molecule has low entropy however far apart its
        # members look by any single distance measure.
        frequencies = X.mean(axis=0)
        active = frequencies[(frequencies > 0) & (frequencies < 1)]
        bit_entropy = (
            float(
                -(
                    active * np.log(active)
                    + (1 - active) * np.log(1 - active)
                ).sum()
                / len(frequencies)
            )
            if active.size
            else 0.0
        )

        return {
            "n_molecules": float(len(mols)),
            "mean_pairwise_distance": float(distances.mean()),
            "median_pairwise_distance": float(np.median(distances)),
            "internal_diversity": float(distances.mean()),
            "min_pairwise_distance": float(distances.min()),
            "n_scaffolds": float(len(counts)),
            "scaffold_diversity": float(len(counts) / len(mols)),
            "acyclic_fraction": float(
                sum(1 for s in scaffolds if not s) / len(mols)
            ),
            "scaffold_entropy": scaffold_entropy,
            "bit_entropy": bit_entropy,
        }

    def compare(
        self, libraries: Dict[str, Sequence[Any]]
    ) -> "pd.DataFrame":
        """Compare several libraries on every diversity measure.

        Parameters
        ----------
        libraries : dict
            ``name -> molecules``.

        Returns
        -------
        pandas.DataFrame
            One row per library.
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for name, mols in libraries.items():
            record: Dict[str, Any] = {"library": name}
            record.update(self.analyze(mols))
            rows.append(record)
        return pd.DataFrame(rows)


class ClusterAnalyzer:
    """Cluster a compound collection and describe the result.

    Delegates to :mod:`qsarkit.cluster` for the cheminformatics methods
    and to scikit-learn for the general ones, then summarizes what came
    out: how many clusters, how big, and how many singletons. The
    singleton fraction is the number worth watching — a library that
    clusters into mostly singletons is either genuinely diverse or being
    clustered at too tight a cutoff.

    Parameters
    ----------
    method : {"butina", "sphere_exclusion", "hierarchical", "kmeans", "dbscan"}, default "butina"
        Clustering algorithm.
    cutoff : float, default 0.35
        Distance cutoff for the cheminformatics methods.
    n_clusters : int, optional
        Cluster count for ``"hierarchical"`` and ``"kmeans"``.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.
    random_state : int, optional
        Seed for k-means.

    Attributes
    ----------
    labels_ : ndarray of shape (n_molecules,)
        Cluster assignment.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("CCO", "CCN", "c1ccccc1", "c1ccccc1C")]
    >>> analyzer = ClusterAnalyzer(cutoff=0.5).fit(mols)
    >>> analyzer.labels_.shape
    (4,)

    References
    ----------
    - Butina, D. (1999). "Unsupervised Data Base Clustering Based on
      Daylight's Fingerprint and Tanimoto Similarity." J. Chem. Inf.
      Comput. Sci., 39(4), 747-750. https://doi.org/10.1021/ci9803381
    - Downs, G. M. & Barnard, J. M. (2002). "Clustering Methods and Their
      Uses in Computational Chemistry." Rev. Comput. Chem., 18, 1-40.
      https://doi.org/10.1002/0471433519.ch1
    """

    labels_: npt.NDArray[np.intp]

    def __init__(
        self,
        method: Literal[
            "butina", "sphere_exclusion", "hierarchical", "kmeans", "dbscan"
        ] = "butina",
        cutoff: float = 0.35,
        n_clusters: Optional[int] = None,
        radius: int = 2,
        n_bits: int = 2048,
        random_state: Optional[int] = None,
    ) -> None:
        self.method = method
        self.cutoff = cutoff
        self.n_clusters = n_clusters
        self.radius = radius
        self.n_bits = n_bits
        self.random_state = random_state

    def fit(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> "ClusterAnalyzer":
        """Cluster the molecules.

        Parameters
        ----------
        mols : sequence of Mol
        y : ignored

        Returns
        -------
        ClusterAnalyzer
        """
        from qsarkit.cluster import (
            ButinaClustering,
            HierarchicalClustering,
            SphereExclusionClustering,
        )

        X = _features(mols, self.radius, self.n_bits)
        if self.method == "butina":
            self.labels_ = ButinaClustering(cutoff=self.cutoff).fit(X).labels_
        elif self.method == "sphere_exclusion":
            self.labels_ = (
                SphereExclusionClustering(cutoff=self.cutoff).fit(X).labels_
            )
        elif self.method == "hierarchical":
            self.labels_ = (
                HierarchicalClustering(n_clusters=self.n_clusters or 2)
                .fit(X)
                .labels_
            )
        elif self.method == "kmeans":
            from sklearn.cluster import KMeans

            self.labels_ = np.asarray(
                KMeans(
                    n_clusters=self.n_clusters or 2, n_init=10,
                    random_state=self.random_state,
                ).fit_predict(X),
                dtype=np.intp,
            )
        elif self.method == "dbscan":
            from sklearn.cluster import DBSCAN

            from qsarkit.neighbors import jaccard_distance_matrix

            self.labels_ = np.asarray(
                DBSCAN(eps=self.cutoff, min_samples=2, metric="precomputed")
                .fit_predict(jaccard_distance_matrix(X)),
                dtype=np.intp,
            )
        else:
            raise ValueError(
                "method must be 'butina', 'sphere_exclusion', 'hierarchical', "
                f"'kmeans' or 'dbscan', got {self.method!r}."
            )
        return self

    def fit_predict(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> npt.NDArray[np.intp]:
        """Cluster and return the labels."""
        return self.fit(mols, y).labels_

    def summary(self) -> Dict[str, float]:
        """Describe the clustering.

        Returns
        -------
        dict
            ``n_clusters``, ``n_singletons``, ``singleton_fraction``,
            ``largest_cluster``, ``mean_cluster_size``. DBSCAN's noise
            label (-1) is counted as singletons.
        """
        if not hasattr(self, "labels_"):
            raise ModelNotFittedError(
                "ClusterAnalyzer must be fitted before calling summary()."
            )
        labels = self.labels_
        real = labels[labels >= 0]
        sizes = np.bincount(real) if real.size else np.array([], dtype=int)
        sizes = sizes[sizes > 0]
        n_noise = int((labels < 0).sum())
        singletons = int((sizes == 1).sum()) + n_noise
        return {
            "n_clusters": float(len(sizes)),
            "n_singletons": float(singletons),
            "singleton_fraction": float(singletons / len(labels)) if len(labels) else 0.0,
            "largest_cluster": float(sizes.max()) if sizes.size else 0.0,
            "mean_cluster_size": float(sizes.mean()) if sizes.size else 0.0,
        }

    def cluster_members(self) -> Dict[int, List[int]]:
        """Molecule indices grouped by cluster.

        Returns
        -------
        dict
            ``cluster label -> list of molecule indices``.
        """
        if not hasattr(self, "labels_"):
            raise ModelNotFittedError(
                "ClusterAnalyzer must be fitted before calling cluster_members()."
            )
        groups: Dict[int, List[int]] = {}
        for i, label in enumerate(self.labels_):
            groups.setdefault(int(label), []).append(i)
        return groups


class NearestNeighborAnalyzer:
    """Nearest-neighbour statistics for a compound collection.

    Underpins two everyday questions: how close is this new compound to
    anything we already have (novelty), and how self-similar is this
    library (redundancy). A screening set whose members are all each
    other's near neighbours is smaller than its compound count suggests.

    Parameters
    ----------
    n_neighbors : int, default 1
        Neighbours considered.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.

    Examples
    --------
    >>> from rdkit import Chem
    >>> library = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1")]
    >>> analyzer = NearestNeighborAnalyzer().fit(library)
    >>> float(analyzer.nearest_similarity([Chem.MolFromSmiles("CCO")])[0])
    1.0

    References
    ----------
    - Sheridan, R. P. et al. (2004). "Similarity to Molecules in the
      Training Set Is a Good Discriminator for Prediction Accuracy in
      QSAR." J. Chem. Inf. Comput. Sci., 44(6), 1912-1928.
      https://doi.org/10.1021/ci049782w
    - Willett, P. (2006). "Similarity-Based Virtual Screening Using 2D
      Fingerprints." Drug Discov. Today, 11(23-24), 1046-1053.
      https://doi.org/10.1016/j.drudis.2006.10.005
    """

    def __init__(
        self, n_neighbors: int = 1, radius: int = 2, n_bits: int = 2048
    ) -> None:
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.n_bits = n_bits

    def fit(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> "NearestNeighborAnalyzer":
        """Store the reference library.

        Parameters
        ----------
        mols : sequence of Mol
        y : ignored

        Returns
        -------
        NearestNeighborAnalyzer
        """
        if not mols:
            raise ValueError("Cannot fit on an empty library.")
        self._X = _features(mols, self.radius, self.n_bits)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "_X"):
            raise ModelNotFittedError(
                "NearestNeighborAnalyzer must be fitted before querying."
            )

    def nearest_similarity(
        self, mols: Sequence[Any], exclude_self: bool = False
    ) -> npt.NDArray[np.float64]:
        """Similarity of each query to its nearest library member.

        Parameters
        ----------
        mols : sequence of Mol
            Query molecules.
        exclude_self : bool, default False
            Ignore each query's own row, which is what you want when the
            queries *are* the library and you are measuring internal
            redundancy. Only the matching position is masked, not every
            perfect match -- masking those would hide exact duplicates,
            which is precisely what such a measurement is looking for.
            Requires the query set to be the fitted library.

        Returns
        -------
        ndarray of shape (n_queries,)

        Raises
        ------
        ValueError
            If ``exclude_self`` is set but the query set is not the same
            size as the library, so there is no "self" to exclude.
        """
        from qsarkit.neighbors import tanimoto_similarity_matrix

        self._check_fitted()
        similarity = tanimoto_similarity_matrix(
            _features(mols, self.radius, self.n_bits), self._X
        )
        if exclude_self:
            if similarity.shape[0] != similarity.shape[1]:
                raise ValueError(
                    "exclude_self=True compares the library against itself, "
                    f"but {similarity.shape[0]} queries were given for a "
                    f"library of {similarity.shape[1]}."
                )
            similarity = similarity.copy()
            np.fill_diagonal(similarity, -np.inf)
        top = np.sort(similarity, axis=1)[:, -self.n_neighbors :]
        return np.asarray(top.mean(axis=1), dtype=np.float64)

    def novelty(self, mols: Sequence[Any]) -> npt.NDArray[np.float64]:
        """Novelty of each query: ``1 - nearest similarity``.

        Parameters
        ----------
        mols : sequence of Mol

        Returns
        -------
        ndarray of shape (n_queries,)
            0 means an exact match in the library; 1 means nothing alike.
        """
        return np.asarray(1.0 - self.nearest_similarity(mols), dtype=np.float64)

    def redundancy(self, mols: Sequence[Any], threshold: float = 0.9) -> float:
        """Fraction of a set having a near-duplicate elsewhere in it.

        Parameters
        ----------
        mols : sequence of Mol
        threshold : float, default 0.9
            Similarity above which two molecules count as duplicates.

        Returns
        -------
        float
            In [0, 1].
        """
        self._check_fitted()
        nearest = self.nearest_similarity(mols, exclude_self=True)
        finite = nearest[np.isfinite(nearest)]
        return float((finite >= threshold).mean()) if finite.size else 0.0


class ScaffoldAnalyzer:
    """Bemis-Murcko scaffold analysis of a compound collection.

    Scaffolds are how medicinal chemists actually partition a library,
    and the scaffold distribution says something a compound count cannot:
    a 10,000-compound set built on twelve scaffolds is a different asset
    from one built on three thousand.

    Parameters
    ----------
    generic : bool, default False
        Reduce scaffolds to their carbon skeleton, which merges
    heteroatom-substituted variants of the same ring system.

    Examples
    --------
    >>> from rdkit import Chem
    >>> mols = [Chem.MolFromSmiles(s) for s in
    ...         ("c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCO")]
    >>> analyzer = ScaffoldAnalyzer().fit(mols)
    >>> analyzer.n_scaffolds
    2

    Toluene and ethylbenzene share the benzene framework; picoline
    contributes a second. Ethanol is acyclic, so it has no Bemis-Murcko
    framework at all and is not counted as a scaffold of its own --
    ``summary()`` reports it separately instead:

    >>> analyzer.summary()["acyclic_fraction"]
    0.25

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known
      Drugs. 1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    - Schuffenhauer, A. et al. (2007). "The Scaffold Tree." J. Chem. Inf.
      Model., 47(1), 47-58. https://doi.org/10.1021/ci600338x
    - Langdon, S. R., Brown, N. & Blagg, J. (2011). "Scaffold Diversity
      of Exemplified Medicinal Chemistry Space." J. Chem. Inf. Model.,
      51(9), 2174-2185. https://doi.org/10.1021/ci2001428
    """

    scaffolds_: List[str]

    def __init__(self, generic: bool = False) -> None:
        self.generic = generic

    def fit(
        self, mols: Sequence[Any], y: Optional[npt.ArrayLike] = None
    ) -> "ScaffoldAnalyzer":
        """Extract the scaffold of every molecule.

        Parameters
        ----------
        mols : sequence of Mol
        y : ignored

        Returns
        -------
        ScaffoldAnalyzer
        """
        self.scaffolds_ = [
            bemis_murcko_smiles(m, generic=self.generic) for m in mols
        ]
        self.counts_ = Counter(self.scaffolds_)
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "scaffolds_"):
            raise ModelNotFittedError(
                "ScaffoldAnalyzer must be fitted before use."
            )

    @property
    def n_scaffolds(self) -> int:
        """Number of distinct scaffolds, excluding the acyclic empty one."""
        self._check_fitted()
        return len([s for s in self.counts_ if s])

    def most_common(self, n: int = 10) -> List[Tuple[str, int]]:
        """The most frequent scaffolds.

        Parameters
        ----------
        n : int, default 10

        Returns
        -------
        list of (scaffold_smiles, count)
        """
        self._check_fitted()
        return [(s, c) for s, c in self.counts_.most_common() if s][:n]

    def groups(self) -> Dict[str, List[int]]:
        """Molecule indices grouped by scaffold.

        Returns
        -------
        dict
            ``scaffold SMILES -> list of molecule indices``.
        """
        self._check_fitted()
        out: Dict[str, List[int]] = {}
        for i, scaffold in enumerate(self.scaffolds_):
            out.setdefault(scaffold, []).append(i)
        return out

    def summary(self) -> Dict[str, float]:
        """Describe the scaffold distribution.

        Returns
        -------
        dict
            ``n_molecules``, ``n_scaffolds``, ``scaffold_diversity``,
            ``n_singleton_scaffolds``, ``largest_scaffold_group``,
            ``acyclic_fraction`` and ``top_scaffold_share`` (the fraction
            of the library sitting on its single most common scaffold —
            the quickest way to spot a library that is really one series).
        """
        self._check_fitted()
        n = len(self.scaffolds_)
        cyclic = {s: c for s, c in self.counts_.items() if s}
        sizes = np.array(list(cyclic.values()), dtype=np.float64)
        return {
            "n_molecules": float(n),
            "n_scaffolds": float(len(cyclic)),
            "scaffold_diversity": float(len(cyclic) / n) if n else 0.0,
            "n_singleton_scaffolds": float((sizes == 1).sum()) if sizes.size else 0.0,
            "largest_scaffold_group": float(sizes.max()) if sizes.size else 0.0,
            "acyclic_fraction": float(self.counts_.get("", 0) / n) if n else 0.0,
            "top_scaffold_share": float(sizes.max() / n) if sizes.size and n else 0.0,
        }

    def to_dataframe(self) -> "pd.DataFrame":
        """Scaffold frequency table.

        Returns
        -------
        pandas.DataFrame
            Columns ``scaffold``, ``count``, ``fraction``, descending.
        """
        import pandas as pd

        self._check_fitted()
        n = len(self.scaffolds_)
        rows = [
            {"scaffold": s, "count": c, "fraction": c / n}
            for s, c in self.counts_.most_common()
            if s
        ]
        return pd.DataFrame(rows, columns=["scaffold", "count", "fraction"])


class ChemicalSpaceCoverage:
    """Compare the chemical space covered by two collections.

    The practical question behind a library purchase or a virtual screen:
    how much of the target space does this set actually reach, and how
    much of it is already covered by what we own?

    Parameters
    ----------
    threshold : float, default 0.7
        Similarity at which a reference compound counts as covered. 0.7
        on ECFP4 is the conventional "similar enough" cutoff.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Fingerprint length.

    Examples
    --------
    >>> from rdkit import Chem
    >>> a = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN")]
    >>> b = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1")]
    >>> report = ChemicalSpaceCoverage().compare(a, b)
    >>> 0.0 <= report["coverage_of_reference"] <= 1.0
    True

    References
    ----------
    - Maggiora, G. & Shanmugasundaram, V. (2011). "Molecular Similarity
      Measures." Methods Mol. Biol., 672, 39-100.
      https://doi.org/10.1007/978-1-60761-839-3_2
    - Willett, P. (2006). Drug Discov. Today, 11(23-24), 1046-1053.
      https://doi.org/10.1016/j.drudis.2006.10.005
    """

    def __init__(
        self, threshold: float = 0.7, radius: int = 2, n_bits: int = 2048
    ) -> None:
        self.threshold = threshold
        self.radius = radius
        self.n_bits = n_bits

    def compare(
        self, query: Sequence[Any], reference: Sequence[Any]
    ) -> Dict[str, float]:
        """Measure how well ``query`` covers ``reference``.

        Parameters
        ----------
        query : sequence of Mol
            The collection being assessed.
        reference : sequence of Mol
            The space to be covered.

        Returns
        -------
        dict
            ``coverage_of_reference`` (fraction of reference compounds
            with a similar query compound), ``mean_nearest_similarity``,
            ``n_novel_in_query`` (query compounds unlike anything in the
            reference) and ``novel_fraction``.
        """
        from qsarkit.neighbors import tanimoto_similarity_matrix

        if not query or not reference:
            raise ValueError("Both collections must be non-empty.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1], got {self.threshold}."
            )

        similarity = tanimoto_similarity_matrix(
            _features(reference, self.radius, self.n_bits),
            _features(query, self.radius, self.n_bits),
        )
        nearest_to_reference = similarity.max(axis=1)
        nearest_to_query = similarity.max(axis=0)
        return {
            "coverage_of_reference": float(
                (nearest_to_reference >= self.threshold).mean()
            ),
            "mean_nearest_similarity": float(nearest_to_reference.mean()),
            "n_novel_in_query": float((nearest_to_query < self.threshold).sum()),
            "novel_fraction": float((nearest_to_query < self.threshold).mean()),
        }
