from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.chemspace import (
    bemis_murcko_smiles,
    compute_fingerprints,
    fingerprints_to_array,
    morgan_generator,
    tanimoto_matrix,
)

SMILES = ["CCO", "CCN", "c1ccccc1", "c1ccccc1C", "CC(=O)Oc1ccccc1C(=O)O"]


@pytest.fixture(scope="module")
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


class TestFingerprints:
    def test_generator_respects_n_bits(self, mols):
        gen = morgan_generator(radius=2, n_bits=512)
        assert len(gen.GetFingerprint(mols[0])) == 512

    def test_compute_fingerprints_one_per_molecule(self, mols):
        assert len(compute_fingerprints(mols)) == len(mols)

    def test_fingerprints_to_array_shape(self, mols):
        arr = fingerprints_to_array(compute_fingerprints(mols, n_bits=256))
        assert arr.shape == (len(mols), 256)

    def test_array_is_binary(self, mols):
        arr = fingerprints_to_array(compute_fingerprints(mols, n_bits=128))
        assert set(np.unique(arr).tolist()) <= {0.0, 1.0}

    def test_radius_changes_the_fingerprint(self, mols):
        small = fingerprints_to_array(compute_fingerprints(mols, radius=1, n_bits=512))
        large = fingerprints_to_array(compute_fingerprints(mols, radius=3, n_bits=512))
        assert not np.array_equal(small, large)

    def test_is_deterministic(self, mols):
        a = fingerprints_to_array(compute_fingerprints(mols))
        b = fingerprints_to_array(compute_fingerprints(mols))
        assert np.array_equal(a, b)


class TestTanimotoMatrix:
    def test_square_matrix_with_unit_diagonal(self, mols):
        sim = tanimoto_matrix(compute_fingerprints(mols))
        assert sim.shape == (len(mols), len(mols))
        assert np.allclose(np.diag(sim), 1.0)

    def test_is_symmetric(self, mols):
        sim = tanimoto_matrix(compute_fingerprints(mols))
        assert np.allclose(sim, sim.T)

    def test_values_are_bounded(self, mols):
        sim = tanimoto_matrix(compute_fingerprints(mols))
        assert np.all((sim >= 0) & (sim <= 1))

    def test_two_argument_form(self, mols):
        fps = compute_fingerprints(mols)
        sim = tanimoto_matrix(fps[:2], fps)
        assert sim.shape == (2, len(mols))

    def test_matches_rdkit_bulk_tanimoto(self, mols):
        from rdkit import DataStructs

        fps = compute_fingerprints(mols)
        ours = tanimoto_matrix(fps)
        theirs = np.array(
            [DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in fps]
        )
        assert np.allclose(ours, theirs)

    def test_related_molecules_are_more_similar(self, mols):
        # benzene vs toluene should beat benzene vs ethanol
        sim = tanimoto_matrix(compute_fingerprints(mols))
        assert sim[2, 3] > sim[2, 0]


class TestScaffolds:
    def test_extracts_the_benzene_scaffold(self):
        assert bemis_murcko_smiles(Chem.MolFromSmiles("c1ccccc1C")) == "c1ccccc1"

    def test_acyclic_molecule_has_an_empty_scaffold(self):
        assert bemis_murcko_smiles(Chem.MolFromSmiles("CCO")) == ""

    def test_generic_scaffold_strips_elements(self):
        generic = bemis_murcko_smiles(
            Chem.MolFromSmiles("c1ccncc1C"), generic=True
        )
        assert "n" not in generic and "N" not in generic

    def test_analogues_share_a_scaffold(self):
        a = bemis_murcko_smiles(Chem.MolFromSmiles("c1ccccc1C"))
        b = bemis_murcko_smiles(Chem.MolFromSmiles("c1ccccc1CC"))
        assert a == b
