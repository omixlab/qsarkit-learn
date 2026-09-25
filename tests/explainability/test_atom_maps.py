"""Tests for projecting feature attributions onto atoms."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.explainability import (
    AttributionAtomMapper,
    bit_atom_environments,
    bit_weights_to_atom_weights,
    draw_atom_weights,
)
from qsarkit.representation import MorganFingerprint


@pytest.fixture(scope="module")
def mol():
    return Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")


@pytest.fixture(scope="module")
def symmetric():
    """Chlorobenzene: atoms 2/6 and 3/5 are symmetry-equivalent."""
    return Chem.MolFromSmiles("Clc1ccccc1")


class TestBitEnvironments:
    def test_reports_environments_for_the_bits_a_molecule_sets(self, mol):
        environments = bit_atom_environments(mol, n_bits=256)
        assert environments
        assert all(isinstance(bit, int) for bit in environments)

    def test_every_central_atom_is_a_real_atom(self, mol):
        environments = bit_atom_environments(mol, n_bits=256)
        centres = {atom for envs in environments.values() for atom, _ in envs}
        assert max(centres) < mol.GetNumAtoms()

    def test_radius_never_exceeds_the_requested_one(self, mol):
        environments = bit_atom_environments(mol, radius=2, n_bits=256)
        radii = {r for envs in environments.values() for _, r in envs}
        assert max(radii) <= 2

    def test_a_wider_fingerprint_sets_at_least_as_many_bits(self, mol):
        narrow = bit_atom_environments(mol, n_bits=64)
        wide = bit_atom_environments(mol, n_bits=4096)
        assert len(wide) >= len(narrow)

    def test_feature_invariants_give_a_different_map(self, mol):
        ecfp = bit_atom_environments(mol, n_bits=1024)
        fcfp = bit_atom_environments(mol, n_bits=1024, use_features=True)
        assert set(ecfp) != set(fcfp)

    def test_a_single_atom_molecule_works(self):
        environments = bit_atom_environments(Chem.MolFromSmiles("C"), n_bits=64)
        assert all(r == 0 for envs in environments.values() for _, r in envs)


class TestBitToAtomWeights:
    def test_one_weight_per_atom(self, mol):
        weights = bit_weights_to_atom_weights(
            mol, np.ones(256), n_bits=256
        )
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_a_single_radius_zero_bit_lands_on_one_atom(self, mol):
        chlorine = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"][0]
        bits = np.zeros(256)
        for bit, envs in bit_atom_environments(mol, n_bits=256).items():
            if (chlorine, 0) in envs:
                bits[bit] = 1.0
        weights = bit_weights_to_atom_weights(mol, bits, n_bits=256)
        assert int(np.argmax(weights)) == chlorine
        assert int((weights != 0).sum()) == 1

    def test_zero_attributions_produce_zero_weights(self, mol):
        weights = bit_weights_to_atom_weights(mol, np.zeros(256), n_bits=256)
        assert np.all(weights == 0.0)

    def test_negative_attributions_are_preserved(self, mol):
        weights = bit_weights_to_atom_weights(mol, -np.ones(256), n_bits=256)
        assert np.all(weights <= 0.0)
        assert np.any(weights < 0.0)

    def test_symmetry_equivalent_atoms_get_equal_weight(self, symmetric):
        """A correctness check on the mapping, not a coincidence."""
        weights = bit_weights_to_atom_weights(
            symmetric, np.ones(1024), n_bits=1024
        )
        assert weights[2] == pytest.approx(weights[6])
        assert weights[3] == pytest.approx(weights[5])

    def test_center_puts_a_whole_bit_on_one_atom(self, mol):
        """With one attributed bit, "center" is a spike and "uniform" a spread.

        (With *every* bit attributed the comparison reverses, because a
        peripheral atom then accumulates shares from many environments --
        so the distinction only shows on a single bit.)
        """
        # Find a bit whose environment spans several atoms.
        wide_bit = next(
            bit
            for bit, envs in bit_atom_environments(mol, n_bits=256).items()
            if any(radius >= 1 for _, radius in envs)
        )
        bits = np.zeros(256)
        bits[wide_bit] = 1.0

        centred = bit_weights_to_atom_weights(
            mol, bits, n_bits=256, distribution="center"
        )
        uniform = bit_weights_to_atom_weights(
            mol, bits, n_bits=256, distribution="uniform"
        )
        assert int((centred != 0).sum()) < int((uniform != 0).sum())
        assert centred.max() > uniform.max()
        # Both account for the same total attribution.
        assert centred.sum() == pytest.approx(uniform.sum())

    def test_radius_weighted_downweights_diffuse_environments(self, mol):
        bits = np.ones(256)
        uniform = bit_weights_to_atom_weights(
            mol, bits, n_bits=256, distribution="uniform"
        )
        scaled = bit_weights_to_atom_weights(
            mol, bits, n_bits=256, distribution="radius_weighted"
        )
        assert scaled.sum() < uniform.sum()

    def test_rejects_a_mismatched_vector_length(self, mol):
        with pytest.raises(ValueError, match="n_bits"):
            bit_weights_to_atom_weights(mol, np.ones(128), n_bits=256)

    def test_rejects_an_unknown_distribution(self, mol):
        with pytest.raises(ValueError, match="distribution"):
            bit_weights_to_atom_weights(
                mol, np.ones(256), n_bits=256, distribution="magic"
            )


class TestAttributionAtomMapper:
    def test_reads_its_settings_from_the_fingerprint(self):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=3, n_bits=512))
        assert mapper.radius == 3
        assert mapper.n_bits == 512

    def test_maps_a_vector_onto_atoms(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        weights = mapper.atom_weights(mol, np.ones(256))
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_from_lime_accepts_a_plain_mapping(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        bits = list(bit_atom_environments(mol, n_bits=256))[:3]
        weights = mapper.from_lime(mol, {bit: 1.0 for bit in bits})
        assert np.any(weights != 0.0)

    def test_from_lime_accepts_an_explanation_object(self, mol):
        class _Explanation:
            def as_map(self):
                bits = list(bit_atom_environments(mol, n_bits=256))[:2]
                return {1: [(bit, 0.5) for bit in bits]}

        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        assert np.any(mapper.from_lime(mol, _Explanation()) != 0.0)

    def test_from_shap_handles_a_two_dimensional_result(self, mol):
        class _Explainer:
            def shap_values(self, X):
                return np.ones((1, 256))

        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        weights = mapper.from_shap(mol, _Explainer(), np.zeros((1, 256)))
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_from_shap_handles_a_per_class_result(self, mol):
        """Classification SHAP returns (n_samples, n_features, n_classes)."""
        class _Explainer:
            def shap_values(self, X):
                return np.ones((1, 256, 2))

        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        weights = mapper.from_shap(mol, _Explainer(), np.zeros((1, 256)))
        assert weights.shape == (mol.GetNumAtoms(),)

    def test_collision_rate_is_a_fraction(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256))
        assert 0.0 <= mapper.collision_rate(mol) <= 1.0

    def test_a_wider_fingerprint_collides_less(self, mol):
        """The honesty check on the picture: fewer collisions, sharper map."""
        narrow = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=32))
        wide = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=8192))
        assert wide.collision_rate(mol) <= narrow.collision_rate(mol)

    def test_repr_names_the_settings(self):
        text = repr(AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=256)))
        assert "256" in text and "radius=2" in text


class TestDrawing:
    def test_svg_is_returned_as_text(self, mol):
        svg = draw_atom_weights(mol, np.linspace(-1, 1, mol.GetNumAtoms()))
        assert isinstance(svg, str)
        assert "<svg" in svg

    def test_png_is_returned_as_bytes(self, mol):
        png = draw_atom_weights(
            mol, np.linspace(-1, 1, mol.GetNumAtoms()), fmt="png"
        )
        assert isinstance(png, bytes) and len(png) > 0

    def test_molecules_without_coordinates_are_handled(self, mol):
        """A molecule from SMILES has no conformer; the drawer needs one."""
        assert mol.GetNumConformers() == 0
        draw_atom_weights(mol, np.zeros(mol.GetNumAtoms()))

    def test_the_input_molecule_is_not_modified(self, mol):
        draw_atom_weights(mol, np.zeros(mol.GetNumAtoms()))
        assert mol.GetNumConformers() == 0

    def test_all_zero_weights_do_not_crash(self, mol):
        draw_atom_weights(mol, np.zeros(mol.GetNumAtoms()))

    def test_normalization_can_be_turned_off(self, mol):
        weights = np.linspace(-0.001, 0.001, mol.GetNumAtoms())
        assert draw_atom_weights(mol, weights, normalize=False)

    def test_size_is_respected(self, mol):
        svg = draw_atom_weights(
            mol, np.zeros(mol.GetNumAtoms()), size=(250, 180)
        )
        assert "250" in svg and "180" in svg

    def test_rejects_a_mismatched_weight_vector(self, mol):
        with pytest.raises(ValueError, match="11 atoms"):
            draw_atom_weights(mol, [0.1, 0.2])

    def test_rejects_an_unknown_format(self, mol):
        with pytest.raises(ValueError, match="fmt"):
            draw_atom_weights(mol, np.zeros(mol.GetNumAtoms()), fmt="jpeg")


class TestEndToEnd:
    def test_a_real_model_explanation_reaches_the_structure(self):
        """The whole point: bit attributions become a picture of a molecule."""
        from qsarkit.explainability import PermutationImportance
        from qsarkit.models import QSARRegressor

        mols = [Chem.MolFromSmiles(s) for s in (
            "CC(=O)Nc1ccccc1", "CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1",
            "CCO", "CCN", "CCC",
        )]
        y = np.array([6.2, 6.7, 6.5, 5.0, 5.1, 4.9])
        fingerprint = MorganFingerprint(radius=2, n_bits=256)
        X = fingerprint.transform(mols)
        model = QSARRegressor("rf", random_state=0).fit(X, y)

        importance = PermutationImportance(n_repeats=3, random_state=0).fit(
            model, X, y
        )
        mapper = AttributionAtomMapper(fingerprint)
        target = mols[1]
        weights = mapper.atom_weights(target, importance.importances_mean_)

        assert weights.shape == (target.GetNumAtoms(),)
        assert np.any(weights != 0.0)
        assert "<svg" in draw_atom_weights(target, weights)
