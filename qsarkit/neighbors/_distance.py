"""Jaccard/Tanimoto distance kernels for binary and count fingerprints.

For binary vectors the Jaccard distance is exactly the complement of the
Tanimoto coefficient used throughout cheminformatics, which is why the
two names are used interchangeably here::

    T(a, b) = |a AND b| / |a OR b|          (Tanimoto / Jaccard similarity)
    d(a, b) = 1 - T(a, b)                   (Jaccard distance)

For count (non-binary) vectors the generalized MinMax form is used, which
reduces to the binary expression when all counts are 0/1::

    T(a, b) = sum(min(a_i, b_i)) / sum(max(a_i, b_i))

References
----------
- Jaccard, P. (1901). "Etude comparative de la distribution florale dans
  une portion des Alpes et des Jura." Bull. Soc. Vaudoise Sci. Nat., 37,
  547-579. https://doi.org/10.5169/seals-266450
- Rogers, D. J. & Tanimoto, T. T. (1960). "A Computer Program for
  Classifying Plants." Science, 132(3434), 1115-1118.
  https://doi.org/10.1126/science.132.3434.1115
- Bajusz, D., Racz, A. & Heberger, K. (2015). "Why is Tanimoto Index an
  Appropriate Choice for Fingerprint-Based Similarity Calculations?"
  J. Cheminform., 7, 20. https://doi.org/10.1186/s13321-015-0069-3
- Willett, P., Barnard, J. M. & Downs, G. M. (1998). "Chemical Similarity
  Searching." J. Chem. Inf. Comput. Sci., 38(6), 983-996.
  https://doi.org/10.1021/ci9800211
- RDKit ``DataStructs`` documentation:
  https://www.rdkit.org/docs/source/rdkit.DataStructs.html
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "jaccard_distance",
    "jaccard_similarity",
    "tanimoto_similarity_matrix",
    "jaccard_distance_matrix",
    "is_binary",
]


def is_binary(X: npt.ArrayLike) -> bool:
    """Return True if every entry of ``X`` is 0 or 1.

    Parameters
    ----------
    X : array-like
        Array to inspect.

    Returns
    -------
    bool
        Whether the array holds only 0/1 values.

    Examples
    --------
    >>> is_binary(np.array([[0, 1], [1, 0]]))
    True
    >>> is_binary(np.array([[0, 2]]))
    False
    """
    arr = np.asarray(X)
    return bool(np.all((arr == 0) | (arr == 1)))


def jaccard_similarity(u: npt.ArrayLike, v: npt.ArrayLike) -> float:
    """Tanimoto/Jaccard similarity between two fingerprint vectors.

    Uses the binary expression when both vectors are binary and the
    generalized MinMax expression otherwise.

    Parameters
    ----------
    u, v : array-like of shape (n_features,)
        Fingerprint vectors.

    Returns
    -------
    float
        Similarity in [0, 1]. Two all-zero vectors are defined to have
        similarity 1.0 (they are identical), following the convention
        used by RDKit's ``BulkTanimotoSimilarity`` for empty fingerprints.

    Examples
    --------
    >>> round(jaccard_similarity([1, 1, 0, 0], [1, 0, 0, 0]), 4)
    0.5
    """
    a = np.asarray(u, dtype=np.float64).ravel()
    b = np.asarray(v, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}.")

    union = float(np.maximum(a, b).sum())
    if union == 0.0:
        return 1.0
    return float(np.minimum(a, b).sum() / union)


def jaccard_distance(u: npt.ArrayLike, v: npt.ArrayLike) -> float:
    """Jaccard (1 - Tanimoto) distance between two fingerprint vectors.

    This is the callable to hand to scikit-learn estimators that accept
    ``metric=<callable>``.

    Parameters
    ----------
    u, v : array-like of shape (n_features,)
        Fingerprint vectors.

    Returns
    -------
    float
        Distance in [0, 1].

    Examples
    --------
    >>> round(jaccard_distance([1, 1, 0, 0], [1, 0, 0, 0]), 4)
    0.5
    """
    return 1.0 - jaccard_similarity(u, v)


def tanimoto_similarity_matrix(
    X: npt.ArrayLike, Y: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """Vectorized pairwise Tanimoto/Jaccard similarity matrix.

    Computes all pairs at once with matrix algebra rather than a Python
    loop, which is what makes fingerprint similarity searches over tens
    of thousands of molecules practical.

    For binary inputs the intersection is ``X @ Y.T`` and the union is
    ``|x| + |y| - intersection``. For count inputs the MinMax form is
    computed in chunks.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Fingerprint matrix.
    Y : array-like of shape (n_samples_Y, n_features), optional
        Second fingerprint matrix. Defaults to ``X``.

    Returns
    -------
    ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise similarities in [0, 1].

    Examples
    --------
    >>> S = tanimoto_similarity_matrix(np.array([[1, 1, 0], [1, 0, 0]]))
    >>> S.shape
    (2, 2)
    >>> bool(np.allclose(np.diag(S), 1.0))
    True
    """
    a = np.asarray(X, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {a.shape}.")
    b = a if Y is None else np.asarray(Y, dtype=np.float64)
    if b.ndim != 2:
        raise ValueError(f"Y must be 2-dimensional, got shape {b.shape}.")
    if a.shape[1] != b.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: X has {a.shape[1]}, Y has {b.shape[1]}."
        )

    if is_binary(a) and is_binary(b):
        intersection = a @ b.T
        union = a.sum(axis=1)[:, None] + b.sum(axis=1)[None, :] - intersection
    else:
        # MinMax over count vectors: sum(min)/sum(max), computed row-wise
        # to keep peak memory at O(n_Y * n_features) instead of O(n_X * n_Y * n_features).
        intersection = np.empty((a.shape[0], b.shape[0]), dtype=np.float64)
        union = np.empty_like(intersection)
        for i in range(a.shape[0]):
            row = a[i][None, :]
            intersection[i] = np.minimum(row, b).sum(axis=1)
            union[i] = np.maximum(row, b).sum(axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        sim = np.where(union > 0, intersection / union, 1.0)
    return np.clip(sim, 0.0, 1.0)


def jaccard_distance_matrix(
    X: npt.ArrayLike, Y: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """Pairwise Jaccard distance matrix (``1 - tanimoto_similarity_matrix``).

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
    Y : array-like of shape (n_samples_Y, n_features), optional

    Returns
    -------
    ndarray of shape (n_samples_X, n_samples_Y)
        Pairwise distances in [0, 1].

    Examples
    --------
    >>> D = jaccard_distance_matrix(np.array([[1, 1, 0], [1, 0, 0]]))
    >>> bool(np.allclose(np.diag(D), 0.0))
    True
    """
    return 1.0 - tanimoto_similarity_matrix(X, Y)
