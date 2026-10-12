"""SHAP and LIME attributions reaching atoms, through the fingerprint's own bits.

The mapper used to re-derive the bit numbering from loose ``radius`` /
``n_bits`` / ``use_features`` values copied off the transformer. That silently
disagreed with the matrix the model was fitted on whenever another setting
changed the hashing — ``use_chirality`` being the clear case — so attributions
landed on the wrong atoms while the picture looked entirely plausible. It also
accepted fingerprints with no atom-environment correspondence at all.

``LIMEExplainer.explain_one`` returns a ``feature``/``weight`` frame whose
labels are strings like ``"x389"``; ``from_lime`` raised ``TypeError`` on it,
so the package's own LIME output could not be mapped onto a molecule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from qsarkit.explainability import AttributionAtomMapper, draw_atom_weights
from qsarkit.models import QSARClassifier
from qsarkit.representation import (
    MACCSKeysFingerprint,
    MorganFingerprint,
    RDKitFingerprint,
)

ACETANILIDE = "CC(=O)Nc1ccc(Cl)cc1"
SETTINGS = [
    ("plain", {}),
    ("chirality", {"use_chirality": True}),
    ("features", {"use_features": True}),
    ("counts", {"use_counts": True}),
]


@pytest.fixture(scope="module")
def mol():
    return Chem.MolFromSmiles(ACETANILIDE)


@pytest.fixture(scope="module")
def dataset():
    smiles = [
        "CCO", "c1ccccc1O", "CC(=O)O", "CCN", "c1ccccc1", "CCCCO",
        "c1ccccc1N", ACETANILIDE, "c1ccc2ccccc2c1", "CCCCCCO",
        "c1ccc(cc1)C=O", "CCCl",
    ] * 6
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    y = np.array([1 if "c1ccccc1" in s else 0 for s in smiles])
    return mols, y


class TestBitInfoMatchesTheTransform:
    """The provenance has to describe the matrix the model was fitted on."""

    @pytest.mark.parametrize("label, kwargs", SETTINGS)
    def test_the_bits_agree(self, mol, label, kwargs):
        fingerprint = MorganFingerprint(radius=2, n_bits=1024, **kwargs)
        reported = set(fingerprint.bit_info(mol))
        actual = {int(b) for b in fingerprint.transform([mol])[0].nonzero()[0]}
        assert reported == actual

    def test_every_environment_names_a_real_atom(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=1024)
        for envs in fingerprint.bit_info(mol).values():
            for atom, radius in envs:
                assert 0 <= atom < mol.GetNumAtoms()
                assert 0 <= radius <= 2

    def test_a_missing_molecule_is_refused(self):
        with pytest.raises(ValueError, match="needs a molecule"):
            MorganFingerprint().bit_info(None)


class TestAttributionLandsOnTheRightAtom:
    @pytest.mark.parametrize("label, kwargs", SETTINGS)
    def test_a_bit_centred_on_chlorine_weights_chlorine(self, mol, label, kwargs):
        """The case that was silently wrong for a chirality-aware fingerprint."""
        fingerprint = MorganFingerprint(radius=2, n_bits=512, **kwargs)
        mapper = AttributionAtomMapper(fingerprint, distribution="center")
        chlorine = next(
            a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"
        )
        bits = [
            bit
            for bit, envs in fingerprint.bit_info(mol).items()
            if (chlorine, 0) in envs
        ]
        assert bits, "the chlorine should set a radius-0 bit"

        weights = mapper.from_lime(mol, {bits[0]: 1.0})
        assert int(np.argmax(weights)) == chlorine

    def test_weight_spreads_over_the_environment(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=512)
        centre = AttributionAtomMapper(fingerprint, distribution="center")
        spread = AttributionAtomMapper(fingerprint, distribution="uniform")
        bit = next(
            bit for bit, envs in fingerprint.bit_info(mol).items()
            if any(r > 0 for _, r in envs)
        )
        assert (spread.from_lime(mol, {bit: 1.0}) != 0).sum() > (
            centre.from_lime(mol, {bit: 1.0}) != 0
        ).sum()


class TestFromLimeAcceptsWhatLimeProduces:
    def test_the_explainer_frame(self, dataset, mol):
        """`LIMEExplainer.explain_one` output, the form that used to fail."""
        pytest.importorskip("lime")
        from qsarkit.explainability import LIMEExplainer

        mols, y = dataset
        fingerprint = MorganFingerprint(radius=2, n_bits=512)
        X = fingerprint.transform(mols)
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 20}
        ).fit(X, y)

        explanation = LIMEExplainer(
            model, training_data=X, mode="classification", random_state=0
        ).explain_one(X[7], top_n=10)
        assert list(explanation.columns) == ["feature", "weight"]

        weights = AttributionAtomMapper(fingerprint).from_lime(mol, explanation)
        assert weights.shape == (mol.GetNumAtoms(),)
        assert np.isfinite(weights).all()

    def test_a_plain_mapping(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=256)
        bit = next(iter(fingerprint.bit_info(mol)))
        weights = AttributionAtomMapper(fingerprint).from_lime(mol, {bit: 1.0})
        assert weights.sum() > 0

    def test_pairs(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=256)
        bit = next(iter(fingerprint.bit_info(mol)))
        weights = AttributionAtomMapper(fingerprint).from_lime(mol, [(bit, 1.0)])
        assert weights.sum() > 0

    def test_an_x_prefixed_label(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=256)
        bit = next(iter(fingerprint.bit_info(mol)))
        frame = pd.DataFrame({"feature": [f"x{bit}"], "weight": [1.0]})
        weights = AttributionAtomMapper(fingerprint).from_lime(mol, frame)
        assert weights.sum() > 0

    def test_duplicate_features_accumulate(self, mol):
        """LIME can report a feature twice; the weights add rather than clobber."""
        fingerprint = MorganFingerprint(radius=2, n_bits=256)
        mapper = AttributionAtomMapper(fingerprint, distribution="center")
        bit = next(iter(fingerprint.bit_info(mol)))
        once = mapper.from_lime(mol, [(bit, 1.0)])
        twice = mapper.from_lime(mol, [(bit, 1.0), (bit, 1.0)])
        assert np.allclose(twice, 2 * once)


class TestGuards:
    def test_a_fingerprint_without_bit_info_is_refused(self):
        with pytest.raises(TypeError, match="cannot report which atoms"):
            AttributionAtomMapper(MACCSKeysFingerprint())

    def test_a_path_fingerprint_is_refused(self):
        """It has n_bits, so the old code accepted it and guessed Morgan bits."""
        with pytest.raises(TypeError, match="cannot report which atoms"):
            AttributionAtomMapper(RDKitFingerprint(n_bits=512))

    def test_an_out_of_range_bit_is_refused(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=512))
        with pytest.raises(ValueError, match="outside the 512-bit"):
            mapper.from_lime(mol, {9999: 1.0})

    def test_an_unreadable_feature_label_is_refused(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=512))
        frame = pd.DataFrame({"feature": ["molecular weight"], "weight": [1.0]})
        with pytest.raises(ValueError, match="cannot read a fingerprint bit"):
            mapper.from_lime(mol, frame)

    def test_a_mismatched_weight_vector_is_refused(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=512))
        with pytest.raises(ValueError, match="must describe the same feature"):
            mapper.atom_weights(mol, np.zeros(256))

    def test_nonsense_is_refused(self, mol):
        mapper = AttributionAtomMapper(MorganFingerprint(radius=2, n_bits=512))
        with pytest.raises(TypeError, match="cannot read a LIME explanation"):
            mapper.from_lime(mol, 42)


class TestBothExplainersReachADrawing:
    """The end the user sees: explainer -> atom weights -> rendered molecule."""

    @pytest.mark.parametrize("label, kwargs", SETTINGS)
    def test_shap_to_svg(self, dataset, mol, label, kwargs):
        pytest.importorskip("shap")
        from qsarkit.explainability import SHAPExplainer

        mols, y = dataset
        fingerprint = MorganFingerprint(radius=2, n_bits=512, **kwargs)
        X = fingerprint.transform(mols)
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 20}
        ).fit(X, y)

        explainer = SHAPExplainer(
            model, explainer_type="tree", background=X, n_background=20,
            random_state=0,
        )
        weights = AttributionAtomMapper(fingerprint).from_shap(
            mol, explainer, X[7:8], index=0
        )
        assert weights.shape == (mol.GetNumAtoms(),)
        svg = draw_atom_weights(mol, weights, fmt="svg")
        assert "svg" in svg[:400].lower()

    def test_lime_to_svg(self, dataset, mol):
        pytest.importorskip("lime")
        from qsarkit.explainability import LIMEExplainer

        mols, y = dataset
        fingerprint = MorganFingerprint(radius=2, n_bits=512)
        X = fingerprint.transform(mols)
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 20}
        ).fit(X, y)

        explanation = LIMEExplainer(
            model, training_data=X, mode="classification", random_state=0
        ).explain_one(X[7], top_n=10)
        weights = AttributionAtomMapper(fingerprint).from_lime(mol, explanation)
        svg = draw_atom_weights(mol, weights, fmt="svg")
        assert "svg" in svg[:400].lower()


class TestFromShapReadsTheRightAxis:
    """A raw shap explainer returns a list, one array per class.

    Picking the sample before reducing the class axis read class 1 as sample 1,
    so the drawing described a different molecule than the one passed in.
    """

    @pytest.fixture
    def setup(self, mol):
        fingerprint = MorganFingerprint(radius=2, n_bits=128)
        bits = sorted(fingerprint.bit_info(mol))
        mapper = AttributionAtomMapper(fingerprint, distribution="center")
        return fingerprint, bits, mapper

    def _one_hot(self, n_bits, bit):
        row = np.zeros(n_bits)
        row[bit] = 1.0
        return row

    def test_a_list_per_class(self, mol, setup):
        fingerprint, bits, mapper = setup

        class RawShap:
            def shap_values(self, X):
                # Two classes, three samples; only sample 2 weights bits[1].
                negative = np.zeros((3, 128))
                positive = np.zeros((3, 128))
                positive[2, bits[1]] = 1.0
                return [negative, positive]

        weights = mapper.from_shap(mol, RawShap(), np.zeros((3, 128)), index=2)
        expected = mapper.atom_weights(mol, self._one_hot(128, bits[1]))
        assert np.allclose(weights, expected)
        assert weights.sum() > 0

    def test_a_three_dimensional_array(self, mol, setup):
        fingerprint, bits, mapper = setup

        class RawShap:
            def shap_values(self, X):
                values = np.zeros((3, 128, 2))
                values[2, bits[1], 1] = 1.0
                return values

        weights = mapper.from_shap(mol, RawShap(), np.zeros((3, 128)), index=2)
        expected = mapper.atom_weights(mol, self._one_hot(128, bits[1]))
        assert np.allclose(weights, expected)

    def test_a_plain_regression_matrix(self, mol, setup):
        fingerprint, bits, mapper = setup

        class RawShap:
            def shap_values(self, X):
                values = np.zeros((3, 128))
                values[2, bits[1]] = 1.0
                return values

        weights = mapper.from_shap(mol, RawShap(), np.zeros((3, 128)), index=2)
        expected = mapper.atom_weights(mol, self._one_hot(128, bits[1]))
        assert np.allclose(weights, expected)
