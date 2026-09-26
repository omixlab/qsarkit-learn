"""How much of a projection can be believed.

A 2D picture of chemical space invites conclusions about neighbourhoods, and
a projection that does not support them looks exactly like one that does.
Trustworthiness quantifies the difference: it measures how many of each
point's neighbours in the projection were also its neighbours in the original
space, penalised by how far away they really were. It runs from 0 to 1, and a
value near 1 means the local structure in the picture is local structure in the
fingerprint space rather than an artefact of the embedding.

This matters most for exactly the methods people plot: t-SNE and UMAP produce
convincing islands whose *between*-cluster distances carry no meaning, and
reading those distances as chemical distance is the commonest misuse of the
technique. Reporting trustworthiness alongside the figure states what the
figure can support.

References
----------
- Venna, J. & Kaski, S. (2001). "Neighborhood Preservation in Nonlinear
  Projection Methods: An Experimental Study." ICANN 2001, 485-491.
  https://doi.org/10.1007/3-540-44668-0_68
- van der Maaten, L. & Hinton, G. (2008). "Visualizing Data Using t-SNE."
  J. Mach. Learn. Res., 9, 2579-2605.
  https://jmlr.org/papers/v9/vandermaaten08a.html
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

__all__ = ["projection_trustworthiness"]


def projection_trustworthiness(
    X: npt.ArrayLike,
    embedding: npt.ArrayLike,
    n_neighbors: Union[int, Iterable[int]] = 5,
    metric: str = "jaccard",
    subsample: Optional[int] = None,
    random_state: Optional[int] = None,
) -> Any:
    """How well ``embedding`` preserves the neighbourhoods of ``X``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The original space -- a fingerprint or descriptor matrix.
    embedding : array-like of shape (n_samples, n_components)
        The projected coordinates, as returned by
        :meth:`~qsarkit.chemspace.ChemicalSpaceAnalyzer.fit_transform`.
    n_neighbors : int or iterable of int, default 5
        The neighbourhood size to score. Pass several and an array is
        returned in the order given, because trustworthiness is a function of
        the scale you ask about: a projection can preserve a compound's five
        nearest analogues while misplacing its series.
    metric : str, default "jaccard"
        Distance in the *original* space. Jaccard/Tanimoto is correct for
        binary fingerprints; pass ``"euclidean"`` for descriptors. This must
        match the metric the projection was built with, or the score measures
        agreement between two different questions.
    subsample : int, optional
        Score a random subset of this many samples. Trustworthiness needs the
        full pairwise distance matrix, which is O(n^2) in memory -- 20 000
        compounds is 3.2 GB in float64 -- so a subsample is often the only
        way to compute it at all. It is an estimate of the same quantity, and
        seeding it with ``random_state`` makes it reproducible.
    random_state : int, optional
        Seed for ``subsample``.

    Returns
    -------
    float or ndarray
        A float for one neighbourhood size, an array in the requested order
        for several.

    Raises
    ------
    ValueError
        If the two inputs disagree on the sample count, if a neighbourhood
        size is not at least 1, or if it is too large for the sample --
        trustworthiness is undefined once ``n_neighbors`` reaches half the
        samples, since every point is then a neighbour of every other.

    Examples
    --------
    A projection of a small set, scored at two scales:

    >>> import numpy as np
    >>> from qsarkit.chemspace import (
    ...     ChemicalSpaceAnalyzer, projection_trustworthiness)
    >>> rng = np.random.default_rng(0)
    >>> X = (rng.random((60, 32)) > 0.7).astype(float)
    >>> embedding = ChemicalSpaceAnalyzer(
    ...     method="pca", metric="euclidean", random_state=0).fit_transform(X)
    >>> score = projection_trustworthiness(
    ...     X, embedding, n_neighbors=5, metric="euclidean")
    >>> 0.0 <= score <= 1.0
    True
    >>> scores = projection_trustworthiness(
    ...     X, embedding, n_neighbors=[5, 10], metric="euclidean")
    >>> scores.shape
    (2,)
    """
    from sklearn.manifold import trustworthiness as _trustworthiness

    X_arr = np.asarray(X, dtype=np.float64)
    embedding_arr = np.asarray(embedding, dtype=np.float64)
    if X_arr.ndim != 2 or embedding_arr.ndim != 2:
        raise ValueError(
            f"X and embedding must both be 2-dimensional, got shapes "
            f"{X_arr.shape} and {embedding_arr.shape}."
        )
    if len(X_arr) != len(embedding_arr):
        raise ValueError(
            f"X has {len(X_arr)} samples but embedding has "
            f"{len(embedding_arr)}; they must describe the same compounds."
        )

    # UMAP can return non-finite coordinates for points its fuzzy graph left
    # disconnected, and t-SNE can do the same on degenerate input. sklearn
    # would report this as a bare "Input X contains NaN" from inside a
    # nearest-neighbour search, which does not say which input or why.
    for name, array in (("X", X_arr), ("embedding", embedding_arr)):
        if not np.isfinite(array).all():
            bad = int(np.count_nonzero(~np.isfinite(array).all(axis=1)))
            raise ValueError(
                f"{name} contains non-finite values in {bad} of {len(array)} "
                f"rows, so neighbourhoods are undefined there. For an "
                f"embedding this usually means the projection left those "
                f"points disconnected -- UMAP does this on sparse "
                f"fingerprints; raising `n_neighbors` often fixes it. Drop "
                f"the affected rows from both arrays to score the rest."
            )

    # Narrowed explicitly rather than with a blanket ignore, so the two
    # branches stay checked.
    if isinstance(n_neighbors, (int, np.integer)):
        single = True
        sizes = [int(n_neighbors)]
    else:
        single = False
        sizes = [int(k) for k in n_neighbors]
    if not sizes:
        raise ValueError("n_neighbors is empty; pass at least one neighbourhood size.")

    if subsample is not None and subsample < len(X_arr):
        if subsample < 3:
            raise ValueError(f"subsample must be at least 3, got {subsample}.")
        rng = np.random.default_rng(random_state)
        keep = rng.choice(len(X_arr), size=subsample, replace=False)
        X_arr = X_arr[keep]
        embedding_arr = embedding_arr[keep]

    # Fingerprints arrive as 0/1 floats, and sklearn converts them to boolean
    # for a set metric while warning about it on every call. Converting here
    # makes the intent explicit and keeps the warning out of the caller's
    # output, which otherwise fills with one line per neighbourhood size.
    _BOOLEAN_METRICS = frozenset(
        {"jaccard", "dice", "kulsinski", "rogerstanimoto", "russellrao",
         "sokalmichener", "sokalsneath", "yule", "matching"}
    )
    scored_X = X_arr.astype(bool) if metric in _BOOLEAN_METRICS else X_arr

    n_samples = len(X_arr)
    values = np.empty(len(sizes), dtype=np.float64)
    for i, k in enumerate(sizes):
        if k < 1:
            raise ValueError(f"n_neighbors must be at least 1, got {k}.")
        if k >= n_samples / 2:
            raise ValueError(
                f"n_neighbors={k} is too large for {n_samples} samples: "
                f"trustworthiness is undefined at or above half the sample "
                f"count. Use n_neighbors < {int(n_samples // 2)}, or raise "
                f"`subsample`."
            )
        values[i] = float(
            _trustworthiness(scored_X, embedding_arr, n_neighbors=k, metric=metric)
        )

    return float(values[0]) if single else values
