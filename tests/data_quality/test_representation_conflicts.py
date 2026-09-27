"""The ceiling a representation puts on any model fitted to it."""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.data_quality import representation_conflicts


class TestClassification:
    def test_a_conflicting_pair_costs_exactly_one_prediction(self):
        X = np.array([[1, 0], [1, 0], [0, 1], [0, 0]])
        y = np.array([1, 0, 1, 0])
        report = representation_conflicts(X, y)

        assert report.n_collision_groups == 1
        assert report.n_conflicting_groups == 1
        assert report.n_compounds_in_conflict == 2
        assert report.n_irreducible_errors == 1
        assert report.max_accuracy == pytest.approx(0.75)

    def test_a_collision_with_agreeing_labels_is_not_a_conflict(self):
        """Identical features are only a problem when the labels differ."""
        X = np.array([[1, 0], [1, 0], [0, 1]])
        y = np.array([1, 1, 0])
        report = representation_conflicts(X, y)

        assert report.n_collision_groups == 1
        assert report.n_conflicting_groups == 0
        assert report.n_irreducible_errors == 0
        assert report.max_accuracy == pytest.approx(1.0)

    def test_a_clean_dataset_has_no_ceiling(self):
        report = representation_conflicts(np.eye(4), np.array([0, 1, 0, 1]))
        assert report.n_conflicting_groups == 0
        assert report.max_accuracy == pytest.approx(1.0)
        assert report.max_balanced_accuracy == pytest.approx(1.0)

    def test_the_majority_label_is_the_one_that_minimises_error(self):
        X = np.zeros((5, 2))
        y = np.array([1, 1, 1, 0, 0])
        report = representation_conflicts(X, y)

        conflict = report.conflicts[0]
        assert conflict.majority_label == 1
        assert conflict.n_misassigned == 2

    def test_balanced_accuracy_weights_the_rare_class(self):
        """The optimal label for a group is not always the majority one.

        With 2 positives and 18 negatives, a group holding one of each is
        better answered "positive": it gains 1/2 of the positive recall and
        costs 1/18 of the negative recall.
        """
        X = np.zeros((20, 2))
        X[:, 0] = np.arange(20)          # every row distinct ...
        X[0] = X[1] = [99, 99]            # ... except one conflicting pair
        y = np.array([1, 0] + [0] * 17 + [1])

        report = representation_conflicts(X, y)
        assert report.n_conflicting_groups == 1
        # Accuracy would prefer the negative label; balanced accuracy does not.
        assert report.max_balanced_accuracy > report.max_accuracy

    def test_conflicts_are_sorted_largest_first(self):
        X = np.array([[1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [3, 3]])
        y = np.array([1, 0, 1, 1, 0, 1])
        report = representation_conflicts(X, y)
        sizes = [len(c.indices) for c in report.conflicts]
        assert sizes == sorted(sizes, reverse=True)

    def test_the_cap_limits_detail_not_the_counts(self):
        X = np.repeat(np.arange(10), 2).reshape(-1, 1)
        y = np.tile([0, 1], 10)
        full = representation_conflicts(X, y, max_conflicts=None)
        capped = representation_conflicts(X, y, max_conflicts=2)

        assert len(capped.conflicts) == 2
        assert len(full.conflicts) == 10
        assert capped.n_conflicting_groups == full.n_conflicting_groups == 10
        assert capped.n_irreducible_errors == full.n_irreducible_errors


class TestRegression:
    def test_the_floor_is_the_within_group_spread(self):
        X = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        y = np.array([5.0, 7.0, 3.0])
        report = representation_conflicts(X, y, task="regression")

        assert report.n_conflicting_groups == 1
        # Predicting the group mean (6.0) leaves residuals of ±1 on two of
        # three samples: sqrt(2/3).
        assert report.irreducible_rmse == pytest.approx(np.sqrt(2 / 3), abs=1e-6)
        assert report.max_accuracy is None

    def test_tolerance_forgives_assay_noise(self):
        X = np.array([[1.0], [1.0]])
        y = np.array([5.0, 5.2])
        assert representation_conflicts(
            X, y, task="regression", tolerance=0.5
        ).n_conflicting_groups == 0
        assert representation_conflicts(
            X, y, task="regression", tolerance=0.1
        ).n_conflicting_groups == 1


class TestGuards:
    def test_a_one_dimensional_matrix_is_refused(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            representation_conflicts(np.arange(4), np.arange(4))

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="rows but y has"):
            representation_conflicts(np.eye(4), np.arange(3))

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(ValueError, match="task must be"):
            representation_conflicts(np.eye(2), np.arange(2), task="ranking")

    def test_a_negative_tolerance_is_refused(self):
        with pytest.raises(ValueError, match="non-negative"):
            representation_conflicts(np.eye(2), np.arange(2), tolerance=-1.0)

    def test_an_empty_dataset_is_refused(self):
        with pytest.raises(ValueError, match="empty"):
            representation_conflicts(np.empty((0, 3)), np.empty(0))


class TestOnFingerprints:
    def test_stereoisomers_collide_under_a_topological_fingerprint(self):
        """The case the Tox21 analysis turns up: same bits, opposite label."""
        from rdkit import Chem

        from qsarkit.representation import MorganFingerprint

        smiles = [
            r"C(=C/c1ccccc1)\c1ccccc1",   # trans-stilbene
            r"C(=C\c1ccccc1)\c1ccccc1",   # cis-stilbene
            "c1ccccc1O",
        ]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        X = MorganFingerprint(radius=2, n_bits=2048).transform(mols)
        y = np.array([1, 0, 0])

        report = representation_conflicts(X, y)
        assert report.n_conflicting_groups == 1
        assert report.n_irreducible_errors == 1
        assert report.max_accuracy == pytest.approx(2 / 3)
