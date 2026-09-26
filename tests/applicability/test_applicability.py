"""Scale behaviour of the k-NN applicability domain."""

from __future__ import annotations

import numpy as np

from qsarkit.applicability import KNNApplicabilityDomain




class TestKNNDomainScales:
    """The k-NN domain has to work on a real dataset, not just a demo one.

    ``_pairwise`` used to compute euclidean distances by materialising
    ``arr[:, None, :] - X_train[None, :, :]``, an array of shape
    (n_query, n_train, n_features). That is fine for the two dozen molecules
    in the documentation and impossible for a screening set: 5000 training
    compounds described by 2048-bit fingerprints need over 100 TB for one
    call, and the process was killed rather than raising.
    """

    def test_distances_match_the_explicit_formulation(self):
        rng = np.random.default_rng(0)
        X_train, X_query = rng.random((40, 8)), rng.random((15, 8))

        domain = KNNApplicabilityDomain(n_neighbors=3).fit(X_train)
        scores = domain.score_samples(X_query)

        distances = np.linalg.norm(X_query[:, None, :] - X_train[None, :, :], axis=2)
        nearest = np.sort(np.partition(distances, kth=2, axis=1)[:, :3], axis=1)
        assert np.allclose(scores, nearest.mean(axis=1))

    def test_a_realistic_matrix_does_not_exhaust_memory(self):
        """The broadcast version needed ~2.4 GB here; this needs ~2.4 MB."""
        rng = np.random.default_rng(0)
        X_train = (rng.random((1000, 1024)) > 0.9).astype(float)
        X_query = (rng.random((300, 1024)) > 0.9).astype(float)

        domain = KNNApplicabilityDomain(n_neighbors=5).fit(X_train)
        inside = domain.predict(X_query)

        assert inside.shape == (300,)
        assert inside.dtype == bool

    def test_the_tanimoto_metric_scales_too(self):
        rng = np.random.default_rng(0)
        X_train = (rng.random((1000, 1024)) > 0.9).astype(float)
        X_query = (rng.random((300, 1024)) > 0.9).astype(float)

        domain = KNNApplicabilityDomain(n_neighbors=5, metric="tanimoto").fit(X_train)
        assert domain.predict(X_query).shape == (300,)
