from __future__ import annotations

import numpy as np
import pytest

from qsarkit.neighbors import (
    is_binary,
    jaccard_distance,
    jaccard_distance_matrix,
    jaccard_similarity,
    tanimoto_similarity_matrix,
)


def test_is_binary():
    assert is_binary(np.array([[0, 1], [1, 0]]))
    assert not is_binary(np.array([[0, 2]]))
    assert not is_binary(np.array([[0.5]]))


def test_similarity_hand_computed():
    # a = {0,1}, b = {0}: intersection 1, union 2 -> 0.5
    assert jaccard_similarity([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_distance_complements_similarity():
    u, v = [1, 1, 0, 1], [1, 0, 1, 1]
    assert jaccard_distance(u, v) == pytest.approx(1.0 - jaccard_similarity(u, v))


def test_identical_vectors():
    assert jaccard_similarity([1, 0, 1], [1, 0, 1]) == pytest.approx(1.0)
    assert jaccard_distance([1, 0, 1], [1, 0, 1]) == pytest.approx(0.0)


def test_disjoint_vectors():
    assert jaccard_similarity([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(0.0)


def test_all_zero_vectors_are_identical_by_convention():
    assert jaccard_similarity([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)


def test_count_vectors_use_minmax():
    # min = [1,2], max = [3,2] -> 3/5
    assert jaccard_similarity([1, 2], [3, 2]) == pytest.approx(0.6)


def test_minmax_reduces_to_binary_form():
    u, v = [1, 1, 0, 0], [1, 0, 1, 0]
    assert jaccard_similarity(u, v) == pytest.approx(1 / 3)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        jaccard_similarity([1, 0], [1, 0, 1])


def test_matrix_matches_rdkit_bulk_tanimoto():
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    smis = ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1", "CCN", "CCOC"]
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smis]
    X = np.array([list(fp) for fp in fps], dtype=float)

    ours = tanimoto_similarity_matrix(X)
    theirs = np.array([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps])
    assert np.allclose(ours, theirs)


def test_matrix_is_symmetric_with_unit_diagonal():
    X = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=float)
    S = tanimoto_similarity_matrix(X)
    assert np.allclose(S, S.T)
    assert np.allclose(np.diag(S), 1.0)


def test_matrix_two_argument_form():
    X = np.array([[1, 1, 0, 0]], dtype=float)
    Y = np.array([[1, 0, 0, 0], [0, 0, 1, 1]], dtype=float)
    S = tanimoto_similarity_matrix(X, Y)
    assert S.shape == (1, 2)
    assert S[0, 0] == pytest.approx(0.5)
    assert S[0, 1] == pytest.approx(0.0)


def test_count_matrix_path():
    X = np.array([[1, 2], [3, 2]], dtype=float)
    S = tanimoto_similarity_matrix(X)
    assert S[0, 1] == pytest.approx(0.6)
    assert np.allclose(np.diag(S), 1.0)


def test_all_zero_row_in_matrix():
    X = np.array([[0, 0, 0], [1, 1, 0]], dtype=float)
    S = tanimoto_similarity_matrix(X)
    assert S[0, 0] == pytest.approx(1.0)
    assert S[0, 1] == pytest.approx(0.0)


def test_distance_matrix_complements_similarity():
    X = np.array([[1, 1, 0], [1, 0, 1]], dtype=float)
    assert np.allclose(
        jaccard_distance_matrix(X), 1.0 - tanimoto_similarity_matrix(X)
    )


@pytest.mark.parametrize(
    "bad, msg",
    [
        (np.array([1, 0, 1]), "2-dimensional"),
    ],
)
def test_matrix_rejects_non_2d(bad, msg):
    with pytest.raises(ValueError, match=msg):
        tanimoto_similarity_matrix(bad)


def test_matrix_rejects_non_2d_y():
    with pytest.raises(ValueError, match="Y must be 2-dimensional"):
        tanimoto_similarity_matrix(np.array([[1, 0]]), np.array([1, 0]))


def test_matrix_rejects_feature_mismatch():
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        tanimoto_similarity_matrix(np.array([[1, 0]]), np.array([[1, 0, 1]]))
