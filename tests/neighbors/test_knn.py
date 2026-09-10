from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.model_selection import cross_val_score

from qsarkit.base import ModelNotFittedError
from qsarkit.neighbors import (
    JaccardKNeighborsClassifier,
    JaccardKNeighborsRegressor,
    JaccardNeighborSearch,
)


@pytest.fixture(scope="module")
def fingerprints():
    # two well-separated bit patterns, so nearest neighbours are obvious
    rng = np.random.RandomState(0)
    cluster_a = np.zeros((10, 32))
    cluster_a[:, :8] = 1
    cluster_a[np.arange(10), rng.randint(8, 16, 10)] = 1
    cluster_b = np.zeros((10, 32))
    cluster_b[:, 16:24] = 1
    cluster_b[np.arange(10), rng.randint(24, 32, 10)] = 1
    X = np.vstack([cluster_a, cluster_b])
    y_reg = np.concatenate([np.full(10, 1.0), np.full(10, 9.0)])
    y_clf = np.concatenate([np.zeros(10, dtype=int), np.ones(10, dtype=int)])
    return X, y_reg, y_clf


class TestJaccardNeighborSearch:
    def test_finds_itself_first(self, fingerprints):
        X, _, _ = fingerprints
        search = JaccardNeighborSearch(n_neighbors=3).fit(X)
        dist, idx = search.kneighbors(X[:1])
        assert idx[0, 0] == 0
        assert dist[0, 0] == pytest.approx(0.0)

    def test_neighbours_come_from_the_same_cluster(self, fingerprints):
        X, _, _ = fingerprints
        search = JaccardNeighborSearch(n_neighbors=5).fit(X)
        _, idx = search.kneighbors(X[:1])
        assert (idx[0] < 10).all()

    def test_distances_are_sorted(self, fingerprints):
        X, _, _ = fingerprints
        dist, _ = JaccardNeighborSearch(n_neighbors=5).fit(X).kneighbors(X[:3])
        assert np.all(np.diff(dist, axis=1) >= -1e-12)

    def test_return_distance_false(self, fingerprints):
        X, _, _ = fingerprints
        idx = JaccardNeighborSearch(n_neighbors=3).fit(X).kneighbors(
            X[:2], return_distance=False
        )
        assert idx.shape == (2, 3)

    def test_n_neighbors_override(self, fingerprints):
        X, _, _ = fingerprints
        _, idx = JaccardNeighborSearch(n_neighbors=3).fit(X).kneighbors(
            X[:1], n_neighbors=7
        )
        assert idx.shape == (1, 7)

    def test_too_many_neighbours_raises(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ValueError, match="exceeds the number of reference"):
            JaccardNeighborSearch(n_neighbors=99).fit(X).kneighbors(X[:1])

    def test_radius_neighbours(self, fingerprints):
        X, _, _ = fingerprints
        dists, idxs = JaccardNeighborSearch().fit(X).radius_neighbors(
            X[:1], radius=0.5
        )
        assert len(dists) == 1 and len(idxs) == 1
        assert np.all(dists[0] <= 0.5)

    def test_radius_rejects_out_of_range(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ValueError, match="radius must be in"):
            JaccardNeighborSearch().fit(X).radius_neighbors(X[:1], radius=2.0)

    def test_similarity_search_returns_ranked_hits(self, fingerprints):
        X, _, _ = fingerprints
        hits = JaccardNeighborSearch().fit(X).similarity_search(
            X[:1], threshold=0.5
        )
        assert hits[0][0] == (0, pytest.approx(1.0))
        similarities = [s for _, s in hits[0]]
        assert similarities == sorted(similarities, reverse=True)

    def test_similarity_search_max_hits(self, fingerprints):
        X, _, _ = fingerprints
        hits = JaccardNeighborSearch().fit(X).similarity_search(
            X[:1], threshold=0.0, max_hits=3
        )
        assert len(hits[0]) == 3

    def test_similarity_search_rejects_bad_threshold(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ValueError, match="threshold must be in"):
            JaccardNeighborSearch().fit(X).similarity_search(X[:1], threshold=5.0)

    def test_chunking_gives_the_same_answer(self, fingerprints):
        X, _, _ = fingerprints
        whole = JaccardNeighborSearch(n_neighbors=3, chunk_size=1000).fit(X)
        chunked = JaccardNeighborSearch(n_neighbors=3, chunk_size=2).fit(X)
        assert np.array_equal(
            whole.kneighbors(X)[1], chunked.kneighbors(X)[1]
        )

    def test_requires_fitting(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ModelNotFittedError):
            JaccardNeighborSearch().kneighbors(X)

    def test_rejects_empty_training_set(self):
        with pytest.raises(ValueError, match="empty reference set"):
            JaccardNeighborSearch().fit(np.zeros((0, 4)))

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            JaccardNeighborSearch().fit(np.zeros(5))

    def test_rejects_feature_mismatch(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ValueError, match="expected 32"):
            JaccardNeighborSearch().fit(X).kneighbors(np.zeros((1, 7)))


class TestJaccardKNeighborsClassifier:
    def test_recovers_the_cluster_labels(self, fingerprints):
        X, _, y = fingerprints
        clf = JaccardKNeighborsClassifier(n_neighbors=3).fit(X, y)
        assert (clf.predict(X) == y).mean() >= 0.9

    def test_is_a_sklearn_classifier(self):
        assert is_classifier(JaccardKNeighborsClassifier())

    def test_probabilities_sum_to_one(self, fingerprints):
        X, _, y = fingerprints
        proba = JaccardKNeighborsClassifier(n_neighbors=3).fit(X, y).predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_classes_are_recorded(self, fingerprints):
        X, _, y = fingerprints
        assert JaccardKNeighborsClassifier().fit(X, y).classes_.tolist() == [0, 1]

    @pytest.mark.parametrize("weights", ["uniform", "distance", "similarity"])
    def test_weighting_schemes(self, weights, fingerprints):
        X, _, y = fingerprints
        clf = JaccardKNeighborsClassifier(n_neighbors=3, weights=weights).fit(X, y)
        assert clf.predict(X).shape == y.shape

    def test_rejects_unknown_weights(self, fingerprints):
        X, _, y = fingerprints
        clf = JaccardKNeighborsClassifier(weights="bogus").fit(X, y)
        with pytest.raises(ValueError, match="weights must be"):
            clf.predict(X)

    def test_exact_match_dominates_with_distance_weighting(self, fingerprints):
        X, _, y = fingerprints
        clf = JaccardKNeighborsClassifier(n_neighbors=5, weights="distance").fit(X, y)
        # every training point is its own exact match, so it must predict itself
        assert (clf.predict(X) == y).all()

    def test_score_method(self, fingerprints):
        X, _, y = fingerprints
        assert JaccardKNeighborsClassifier(n_neighbors=3).fit(X, y).score(X, y) >= 0.9

    def test_works_in_cross_validation(self, fingerprints):
        X, _, y = fingerprints
        scores = cross_val_score(
            JaccardKNeighborsClassifier(n_neighbors=1), X, y, cv=3
        )
        assert scores.shape == (3,)

    def test_clone_roundtrip(self):
        clf = JaccardKNeighborsClassifier(n_neighbors=7, weights="similarity")
        assert clone(clf).get_params() == clf.get_params()

    def test_requires_fitting(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ModelNotFittedError):
            JaccardKNeighborsClassifier().predict(X)
        with pytest.raises(ModelNotFittedError):
            JaccardKNeighborsClassifier().kneighbors(X)

    def test_rejects_length_mismatch(self, fingerprints):
        X, _, y = fingerprints
        with pytest.raises(ValueError, match="but y has"):
            JaccardKNeighborsClassifier().fit(X, y[:5])

    def test_rejects_too_many_neighbours(self, fingerprints):
        X, _, y = fingerprints
        with pytest.raises(ValueError, match="exceeds the training set size"):
            JaccardKNeighborsClassifier(n_neighbors=99).fit(X, y)

    def test_rejects_non_2d(self, fingerprints):
        _, _, y = fingerprints
        with pytest.raises(ValueError, match="2-dimensional"):
            JaccardKNeighborsClassifier().fit(np.zeros(20), y)


class TestJaccardKNeighborsRegressor:
    def test_recovers_the_cluster_values(self, fingerprints):
        X, y, _ = fingerprints
        reg = JaccardKNeighborsRegressor(n_neighbors=3).fit(X, y)
        assert np.corrcoef(reg.predict(X), y)[0, 1] > 0.9

    def test_is_a_sklearn_regressor(self):
        assert is_regressor(JaccardKNeighborsRegressor())

    def test_exact_match_returns_its_own_value(self, fingerprints):
        X, y, _ = fingerprints
        reg = JaccardKNeighborsRegressor(n_neighbors=1).fit(X, y)
        assert reg.predict(X) == pytest.approx(y)

    @pytest.mark.parametrize("weights", ["uniform", "distance", "similarity"])
    def test_weighting_schemes(self, weights, fingerprints):
        X, y, _ = fingerprints
        reg = JaccardKNeighborsRegressor(n_neighbors=3, weights=weights).fit(X, y)
        assert np.isfinite(reg.predict(X)).all()

    def test_predictions_stay_within_the_target_range(self, fingerprints):
        X, y, _ = fingerprints
        pred = JaccardKNeighborsRegressor(n_neighbors=5).fit(X, y).predict(X)
        assert pred.min() >= y.min() - 1e-9
        assert pred.max() <= y.max() + 1e-9

    def test_score_method(self, fingerprints):
        X, y, _ = fingerprints
        assert JaccardKNeighborsRegressor(n_neighbors=3).fit(X, y).score(X, y) > 0.8

    def test_falls_back_when_nothing_is_similar(self):
        # a query sharing no bits with training has zero similarity to all
        X = np.zeros((4, 8))
        X[:, :4] = 1
        y = np.array([1.0, 2.0, 3.0, 4.0])
        reg = JaccardKNeighborsRegressor(n_neighbors=2, weights="similarity").fit(X, y)
        query = np.zeros((1, 8))
        query[0, 4:] = 1
        assert np.isfinite(reg.predict(query)).all()

    def test_works_in_cross_validation(self, fingerprints):
        X, y, _ = fingerprints
        scores = cross_val_score(
            JaccardKNeighborsRegressor(n_neighbors=1), X, y, cv=3
        )
        assert scores.shape == (3,)

    def test_clone_roundtrip(self):
        reg = JaccardKNeighborsRegressor(n_neighbors=7, weights="distance")
        assert clone(reg).get_params() == reg.get_params()

    def test_requires_fitting(self, fingerprints):
        X, _, _ = fingerprints
        with pytest.raises(ModelNotFittedError):
            JaccardKNeighborsRegressor().predict(X)

    def test_kneighbors_is_exposed(self, fingerprints):
        X, y, _ = fingerprints
        dist, idx = JaccardKNeighborsRegressor(n_neighbors=3).fit(X, y).kneighbors(X[:2])
        assert dist.shape == idx.shape == (2, 3)
