"""Fingerprint similarity search and k-NN estimators under Jaccard/Tanimoto distance."""

from qsarkit.neighbors._distance import (
    is_binary,
    jaccard_distance,
    jaccard_distance_matrix,
    jaccard_similarity,
    tanimoto_similarity_matrix,
)
from qsarkit.neighbors._knn import (
    JaccardKNeighborsClassifier,
    JaccardKNeighborsRegressor,
)
from qsarkit.neighbors._search import JaccardNeighborSearch

__all__ = [
    "jaccard_distance",
    "jaccard_similarity",
    "jaccard_distance_matrix",
    "tanimoto_similarity_matrix",
    "is_binary",
    "JaccardNeighborSearch",
    "JaccardKNeighborsClassifier",
    "JaccardKNeighborsRegressor",
]
