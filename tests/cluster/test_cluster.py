from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone

from qsarkit.cluster import (
    ButinaClustering,
    HierarchicalClustering,
    MaxMinPicker,
    SphereExclusionClustering,
)
from qsarkit.neighbors import jaccard_distance_matrix


@pytest.fixture(scope="module")
def two_clusters():
    """Two well-separated fingerprint groups: bits 0-7 vs bits 16-23."""
    rng = np.random.RandomState(0)
    a = np.zeros((10, 32))
    a[:, :8] = 1
    a[np.arange(10), rng.randint(8, 16, 10)] = 1
    b = np.zeros((10, 32))
    b[:, 16:24] = 1
    b[np.arange(10), rng.randint(24, 32, 10)] = 1
    return np.vstack([a, b])


CLUSTERERS = [
    pytest.param(ButinaClustering, {}, id="butina"),
    pytest.param(SphereExclusionClustering, {}, id="sphere-exclusion"),
    pytest.param(HierarchicalClustering, {"n_clusters": 2}, id="hierarchical"),
]


class TestClustererContract:
    @pytest.mark.parametrize("cls, kwargs", CLUSTERERS)
    def test_labels_cover_every_sample(self, cls, kwargs, two_clusters):
        model = cls(**kwargs).fit(two_clusters)
        assert model.labels_.shape == (len(two_clusters),)

    @pytest.mark.parametrize("cls, kwargs", CLUSTERERS)
    def test_recovers_the_two_groups(self, cls, kwargs, two_clusters):
        labels = cls(**kwargs).fit_predict(two_clusters)
        # whatever the label numbering, the first 10 must not mix with the last 10
        assert set(labels[:10]).isdisjoint(set(labels[10:]))

    @pytest.mark.parametrize("cls, kwargs", CLUSTERERS)
    def test_is_deterministic(self, cls, kwargs, two_clusters):
        a = cls(**kwargs).fit_predict(two_clusters)
        b = cls(**kwargs).fit_predict(two_clusters)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("cls, kwargs", CLUSTERERS)
    def test_clone_roundtrip(self, cls, kwargs):
        model = cls(**kwargs)
        assert clone(model).get_params() == model.get_params()

    @pytest.mark.parametrize("cls, kwargs", CLUSTERERS)
    def test_rejects_non_2d(self, cls, kwargs):
        with pytest.raises(ValueError):
            cls(**kwargs).fit(np.zeros(10))


class TestButinaClustering:
    def test_cutoff_controls_cluster_count(self, two_clusters):
        tight = ButinaClustering(cutoff=0.05).fit(two_clusters).n_clusters_
        loose = ButinaClustering(cutoff=0.95).fit(two_clusters).n_clusters_
        assert tight >= loose

    def test_cluster_zero_is_the_largest(self, two_clusters):
        model = ButinaClustering(cutoff=0.5).fit(two_clusters)
        sizes = [int((model.labels_ == k).sum()) for k in range(model.n_clusters_)]
        assert sizes == sorted(sizes, reverse=True)

    def test_centroids_are_real_samples(self, two_clusters):
        model = ButinaClustering(cutoff=0.5).fit(two_clusters)
        assert len(model.cluster_centers_indices_) == model.n_clusters_
        assert model.cluster_centers_indices_.max() < len(two_clusters)

    def test_reordering_variant_runs(self, two_clusters):
        model = ButinaClustering(cutoff=0.5, reordering=True).fit(two_clusters)
        assert model.labels_.shape == (len(two_clusters),)
        assert set(model.labels_[:10]).isdisjoint(set(model.labels_[10:]))

    def test_precomputed_distances(self, two_clusters):
        distances = jaccard_distance_matrix(two_clusters)
        from_fps = ButinaClustering(cutoff=0.5).fit_predict(two_clusters)
        from_dist = ButinaClustering(
            cutoff=0.5, metric="precomputed"
        ).fit_predict(distances)
        assert np.array_equal(from_fps, from_dist)

    def test_precomputed_requires_a_square_matrix(self):
        with pytest.raises(ValueError, match="must be square"):
            ButinaClustering(metric="precomputed").fit(np.zeros((3, 5)))

    def test_rejects_unknown_metric(self, two_clusters):
        with pytest.raises(ValueError, match="metric must be"):
            ButinaClustering(metric="cosine").fit(two_clusters)

    def test_rejects_bad_cutoff(self, two_clusters):
        with pytest.raises(ValueError, match="cutoff must be in"):
            ButinaClustering(cutoff=1.5).fit(two_clusters)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty dataset"):
            ButinaClustering().fit(np.zeros((0, 4)))

    def test_identical_molecules_form_one_cluster(self):
        X = np.tile(np.array([1.0, 1, 0, 0]), (5, 1))
        assert ButinaClustering(cutoff=0.1).fit(X).n_clusters_ == 1


class TestSphereExclusionClustering:
    def test_leaders_are_mutually_distant(self, two_clusters):
        model = SphereExclusionClustering(cutoff=0.5).fit(two_clusters)
        centres = model.cluster_centers_indices_
        distances = jaccard_distance_matrix(two_clusters[centres])
        off_diagonal = distances[~np.eye(len(centres), dtype=bool)]
        if off_diagonal.size:
            assert off_diagonal.min() > 0.5

    def test_cutoff_controls_leader_count(self, two_clusters):
        tight = SphereExclusionClustering(cutoff=0.05).fit(two_clusters)
        loose = SphereExclusionClustering(cutoff=0.95).fit(two_clusters)
        assert tight.n_clusters_ >= loose.n_clusters_

    def test_precomputed_distances(self, two_clusters):
        distances = jaccard_distance_matrix(two_clusters)
        model = SphereExclusionClustering(
            cutoff=0.5, metric="precomputed"
        ).fit(distances)
        assert model.labels_.shape == (len(two_clusters),)

    def test_precomputed_requires_a_square_matrix(self):
        with pytest.raises(ValueError, match="must be square"):
            SphereExclusionClustering(metric="precomputed").fit(np.zeros((3, 5)))

    def test_rejects_unknown_metric(self, two_clusters):
        with pytest.raises(ValueError, match="metric must be"):
            SphereExclusionClustering(metric="cosine").fit(two_clusters)

    def test_rejects_bad_cutoff(self, two_clusters):
        with pytest.raises(ValueError, match="cutoff must be in"):
            SphereExclusionClustering(cutoff=-1).fit(two_clusters)

    def test_rejects_empty_input(self):
        with pytest.raises(ValueError, match="empty dataset"):
            SphereExclusionClustering().fit(np.zeros((0, 4)))


class TestMaxMinPicker:
    def test_picks_the_requested_number(self, two_clusters):
        assert len(MaxMinPicker(n_to_pick=5).fit(two_clusters).picks_) == 5

    def test_picks_are_unique(self, two_clusters):
        picks = MaxMinPicker(n_to_pick=8).fit(two_clusters).picks_
        assert len(set(picks.tolist())) == 8

    def test_spans_both_clusters(self, two_clusters):
        picks = MaxMinPicker(n_to_pick=2).fit(two_clusters).picks_
        assert (picks < 10).any() and (picks >= 10).any()

    def test_diversity_profile_is_non_increasing(self, two_clusters):
        distances = MaxMinPicker(n_to_pick=6).fit(two_clusters).min_distances_
        # the first entry is inf (no prior picks); the rest must not rise
        assert np.all(np.diff(distances[1:]) <= 1e-9)

    def test_seed_index_controls_the_first_pick(self, two_clusters):
        assert MaxMinPicker(n_to_pick=3, seed_index=7).fit(two_clusters).picks_[0] == 7

    def test_is_deterministic_without_a_seed(self, two_clusters):
        a = MaxMinPicker(n_to_pick=4).fit(two_clusters).picks_
        b = MaxMinPicker(n_to_pick=4).fit(two_clusters).picks_
        assert np.array_equal(a, b)

    def test_transform_returns_the_picked_rows(self, two_clusters):
        picker = MaxMinPicker(n_to_pick=3).fit(two_clusters)
        assert picker.transform(two_clusters).shape == (3, two_clusters.shape[1])

    def test_fit_transform(self, two_clusters):
        assert MaxMinPicker(n_to_pick=3).fit_transform(two_clusters).shape == (
            3, two_clusters.shape[1]
        )

    def test_transform_requires_fitting(self, two_clusters):
        from qsarkit.base import ModelNotFittedError

        with pytest.raises(ModelNotFittedError):
            MaxMinPicker().transform(two_clusters)

    def test_precomputed_distances(self, two_clusters):
        distances = jaccard_distance_matrix(two_clusters)
        from_fps = MaxMinPicker(n_to_pick=4).fit(two_clusters).picks_
        from_dist = MaxMinPicker(n_to_pick=4, metric="precomputed").fit(
            distances
        ).picks_
        assert np.array_equal(from_fps, from_dist)

    def test_rejects_too_many_picks(self, two_clusters):
        with pytest.raises(ValueError, match="exceeds the dataset size"):
            MaxMinPicker(n_to_pick=999).fit(two_clusters)

    def test_rejects_non_positive_picks(self, two_clusters):
        with pytest.raises(ValueError, match="must be positive"):
            MaxMinPicker(n_to_pick=0).fit(two_clusters)

    def test_rejects_out_of_range_seed(self, two_clusters):
        with pytest.raises(ValueError, match="out of range"):
            MaxMinPicker(n_to_pick=2, seed_index=999).fit(two_clusters)

    def test_rejects_unknown_metric(self, two_clusters):
        with pytest.raises(ValueError, match="metric must be"):
            MaxMinPicker(metric="cosine").fit(two_clusters)

    def test_precomputed_requires_a_square_matrix(self):
        with pytest.raises(ValueError, match="must be square"):
            MaxMinPicker(metric="precomputed").fit(np.zeros((3, 5)))


class TestHierarchicalClustering:
    def test_requested_cluster_count(self, two_clusters):
        assert HierarchicalClustering(n_clusters=3).fit(two_clusters).n_clusters_ == 3

    @pytest.mark.parametrize("linkage", ["average", "complete", "single"])
    def test_linkage_options(self, linkage, two_clusters):
        model = HierarchicalClustering(n_clusters=2, linkage=linkage).fit(
            two_clusters
        )
        assert set(model.labels_[:10]).isdisjoint(set(model.labels_[10:]))

    def test_ward_linkage_is_rejected(self, two_clusters):
        # Ward needs Euclidean distances and cannot use a precomputed
        # Tanimoto matrix; failing loudly beats a silently wrong tree
        with pytest.raises(ValueError, match="Ward linkage requires"):
            HierarchicalClustering(linkage="ward").fit(two_clusters)

    def test_distance_threshold_mode(self, two_clusters):
        model = HierarchicalClustering(
            n_clusters=None, distance_threshold=0.5
        ).fit(two_clusters)
        assert model.n_clusters_ >= 1
