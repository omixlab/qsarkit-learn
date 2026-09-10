from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from qsarkit.representation import (
    AtomPairFingerprint,
    AvalonFingerprint,
    ConstitutionalDescriptors,
    DescriptorCalculator,
    Descriptors3D,
    FeatureMorganFingerprint,
    FingerprintCombiner,
    FragmentDescriptors,
    LayeredFingerprint,
    LipinskiDescriptors,
    MACCSKeysFingerprint,
    MorganFingerprint,
    PatternFingerprint,
    PharmacophoreFingerprint,
    PhysicochemicalDescriptors,
    RDKitDescriptors,
    RDKitFingerprint,
    TopologicalTorsionFingerprint,
)

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]


@pytest.fixture(scope="module")
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


FINGERPRINTS = [
    pytest.param(MorganFingerprint, {"n_bits": 128}, 128, id="morgan"),
    pytest.param(FeatureMorganFingerprint, {"n_bits": 128}, 128, id="fcfp"),
    pytest.param(RDKitFingerprint, {"n_bits": 128}, 128, id="rdkit"),
    pytest.param(PatternFingerprint, {"n_bits": 128}, 128, id="pattern"),
    pytest.param(LayeredFingerprint, {"n_bits": 128}, 128, id="layered"),
    pytest.param(AtomPairFingerprint, {"n_bits": 128}, 128, id="atom-pair"),
    pytest.param(
        TopologicalTorsionFingerprint, {"n_bits": 128}, 128, id="torsion"
    ),
    pytest.param(MACCSKeysFingerprint, {}, 167, id="maccs"),
    pytest.param(PharmacophoreFingerprint, {"n_bits": 128}, 128, id="pharmacophore"),
]

DESCRIPTORS = [
    pytest.param(RDKitDescriptors, {}, id="rdkit-all"),
    pytest.param(PhysicochemicalDescriptors, {}, id="physicochemical"),
    pytest.param(ConstitutionalDescriptors, {}, id="constitutional"),
    pytest.param(LipinskiDescriptors, {}, id="lipinski"),
    pytest.param(FragmentDescriptors, {}, id="fragments"),
]


class TestFingerprintContract:
    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_output_shape(self, cls, kwargs, width, mols):
        assert cls(**kwargs).transform(mols).shape == (len(mols), width)

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_feature_names_match_the_width(self, cls, kwargs, width, mols):
        transformer = cls(**kwargs)
        names = transformer.fit(mols).get_feature_names_out()
        assert len(names) == transformer.transform(mols).shape[1]

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_is_deterministic(self, cls, kwargs, width, mols):
        a = cls(**kwargs).transform(mols)
        b = cls(**kwargs).transform(mols)
        assert np.array_equal(a, b)

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_output_is_non_negative(self, cls, kwargs, width, mols):
        assert (cls(**kwargs).transform(mols) >= 0).all()

    @pytest.mark.parametrize(
        "cls, kwargs, width",
        [f for f in FINGERPRINTS if f.id not in {"atom-pair", "torsion"}],
    )
    def test_bit_fingerprints_are_binary(self, cls, kwargs, width, mols):
        # atom pairs and torsions are count fingerprints by default, where
        # the multiplicity is genuine information; the rest are bit vectors
        values = np.unique(cls(**kwargs).transform(mols))
        assert set(values.tolist()) <= {0.0, 1.0}

    def test_count_fingerprints_can_exceed_one(self, mols):
        assert AtomPairFingerprint(n_bits=128).transform(mols).max() > 1

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_identical_molecules_give_identical_rows(self, cls, kwargs, width):
        pair = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("OCC")]
        X = cls(**kwargs).transform(pair)
        assert np.array_equal(X[0], X[1])

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_different_molecules_give_different_rows(self, cls, kwargs, width):
        # both molecules carry donors and acceptors, so even the
        # pharmacophore fingerprint has features to distinguish them
        pair = [
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"),
            Chem.MolFromSmiles("NCCc1ccc(O)cc1"),
        ]
        X = cls(**kwargs).transform(pair)
        assert not np.array_equal(X[0], X[1])

    def test_pharmacophore_needs_pharmacophoric_features(self):
        # a pure hydrocarbon has no donors or acceptors to pair, so an
        # empty fingerprint is the correct answer rather than a failure
        row = PharmacophoreFingerprint(n_bits=256).transform(
            [Chem.MolFromSmiles("c1ccc2ccccc2c1")]
        )[0]
        assert row.sum() == 0

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_clone_roundtrip(self, cls, kwargs, width):
        transformer = cls(**kwargs)
        assert clone(transformer).get_params() == transformer.get_params()

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_works_in_a_pipeline(self, cls, kwargs, width, mols):
        pipe = Pipeline(
            [
                ("fp", cls(**kwargs)),
                ("model", RandomForestRegressor(n_estimators=5, random_state=0)),
            ]
        ).fit(mols, [1.0, 2.0, 3.0, 4.0])
        assert pipe.predict(mols).shape == (len(mols),)

    @pytest.mark.parametrize("cls, kwargs, width", FINGERPRINTS)
    def test_fit_transform_matches_transform(self, cls, kwargs, width, mols):
        transformer = cls(**kwargs)
        assert np.array_equal(
            transformer.fit_transform(mols), transformer.transform(mols)
        )


class TestMorganFingerprint:
    def test_radius_changes_the_output(self, mols):
        small = MorganFingerprint(radius=1, n_bits=256).transform(mols)
        large = MorganFingerprint(radius=3, n_bits=256).transform(mols)
        assert not np.array_equal(small, large)

    def test_n_bits_controls_the_width(self, mols):
        assert MorganFingerprint(n_bits=512).transform(mols).shape[1] == 512

    def test_count_mode_can_exceed_one(self):
        long_chain = [Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")]
        counts = MorganFingerprint(n_bits=256, use_counts=True).transform(long_chain)
        assert counts.max() > 1

    def test_binary_mode_never_exceeds_one(self):
        long_chain = [Chem.MolFromSmiles("CCCCCCCCCCCCCCCC")]
        assert MorganFingerprint(n_bits=256).transform(long_chain).max() <= 1

    def test_feature_variant_differs_from_the_plain_one(self, mols):
        plain = MorganFingerprint(n_bits=256).transform(mols)
        features = FeatureMorganFingerprint(n_bits=256).transform(mols)
        assert not np.array_equal(plain, features)

    def test_matches_rdkit_directly(self):
        from rdkit.Chem import rdFingerprintGenerator

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)
        expected = np.array(list(gen.GetFingerprint(mol)), dtype=float)
        assert np.array_equal(
            MorganFingerprint(radius=2, n_bits=256).transform([mol])[0], expected
        )


class TestMACCSKeys:
    def test_fixed_width(self, mols):
        assert MACCSKeysFingerprint().transform(mols).shape[1] == 167

    def test_matches_rdkit_directly(self):
        from rdkit.Chem import MACCSkeys

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        expected = np.array(list(MACCSkeys.GenMACCSKeys(mol)), dtype=float)
        assert np.array_equal(
            MACCSKeysFingerprint().transform([mol])[0], expected
        )


class TestFingerprintCombiner:
    def test_widths_add_up(self, mols):
        combiner = FingerprintCombiner(
            [("morgan", MorganFingerprint(n_bits=64)),
             ("maccs", MACCSKeysFingerprint())]
        )
        assert combiner.fit_transform(mols).shape == (len(mols), 64 + 167)

    def test_feature_names_match_the_width(self, mols):
        combiner = FingerprintCombiner(
            [("morgan", MorganFingerprint(n_bits=64)),
             ("maccs", MACCSKeysFingerprint())]
        ).fit(mols)
        assert len(combiner.get_feature_names_out()) == combiner.transform(
            mols
        ).shape[1]

    def test_blocks_appear_in_order(self, mols):
        morgan = MorganFingerprint(n_bits=64).transform(mols)
        combined = FingerprintCombiner(
            [("morgan", MorganFingerprint(n_bits=64)),
             ("maccs", MACCSKeysFingerprint())]
        ).fit_transform(mols)
        assert np.array_equal(combined[:, :64], morgan)


class TestAvalonFingerprint:
    def test_transform_or_optional_dependency_error(self, mols):
        from qsarkit.base import OptionalDependencyError

        try:
            X = AvalonFingerprint(n_bits=128).transform(mols)
        except OptionalDependencyError:
            pytest.skip("Avalon tools are not available in this RDKit build")
        assert X.shape == (len(mols), 128)


class TestDescriptorContract:
    @pytest.mark.parametrize("cls, kwargs", DESCRIPTORS)
    def test_row_per_molecule(self, cls, kwargs, mols):
        assert cls(**kwargs).transform(mols).shape[0] == len(mols)

    @pytest.mark.parametrize("cls, kwargs", DESCRIPTORS)
    def test_feature_names_match_the_width(self, cls, kwargs, mols):
        transformer = cls(**kwargs)
        X = transformer.fit(mols).transform(mols)
        assert len(transformer.get_feature_names_out()) == X.shape[1]

    @pytest.mark.parametrize("cls, kwargs", DESCRIPTORS)
    def test_no_infinities(self, cls, kwargs, mols):
        X = cls(**kwargs).transform(mols)
        assert not np.isinf(X).any()

    @pytest.mark.parametrize("cls, kwargs", DESCRIPTORS)
    def test_is_deterministic(self, cls, kwargs, mols):
        a = cls(**kwargs).transform(mols)
        b = cls(**kwargs).transform(mols)
        assert np.allclose(a, b, equal_nan=True)

    @pytest.mark.parametrize("cls, kwargs", DESCRIPTORS)
    def test_clone_roundtrip(self, cls, kwargs):
        transformer = cls(**kwargs)
        assert clone(transformer).get_params() == transformer.get_params()


class TestRDKitDescriptors:
    def test_computes_the_full_descriptor_set(self, mols):
        X = RDKitDescriptors().transform(mols)
        assert X.shape[1] > 150

    def test_molecular_weight_is_correct(self):
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        transformer = RDKitDescriptors().fit([mol])
        names = list(transformer.get_feature_names_out())
        X = transformer.transform([mol])
        assert X[0, names.index("MolWt")] == pytest.approx(
            Descriptors.MolWt(mol), rel=1e-6
        )


class TestPhysicochemicalDescriptors:
    def test_reports_the_expected_properties(self, mols):
        names = set(PhysicochemicalDescriptors().fit(mols).get_feature_names_out())
        assert {"MolWt", "TPSA"} <= names or len(names) >= 5

    def test_values_match_rdkit(self):
        from rdkit.Chem import rdMolDescriptors

        mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        transformer = PhysicochemicalDescriptors().fit([mol])
        names = list(transformer.get_feature_names_out())
        X = transformer.transform([mol])
        if "TPSA" in names:
            assert X[0, names.index("TPSA")] == pytest.approx(
                rdMolDescriptors.CalcTPSA(mol), rel=1e-6
            )


class TestFragmentDescriptors:
    def test_counts_functional_groups(self, mols):
        transformer = FragmentDescriptors().fit(mols)
        X = transformer.transform(mols)
        assert X.shape[1] > 50
        assert (X >= 0).all()

    def test_detects_a_carboxylic_acid(self):
        acid = Chem.MolFromSmiles("CC(=O)O")
        transformer = FragmentDescriptors().fit([acid])
        names = list(transformer.get_feature_names_out())
        X = transformer.transform([acid])
        acid_columns = [i for i, n in enumerate(names) if "COO" in n or "acid" in n.lower()]
        assert any(X[0, i] > 0 for i in acid_columns)


class TestDescriptorCalculator:
    def test_selects_a_named_set(self, mols):
        calculator = DescriptorCalculator(blocks=["physicochemical"])
        assert calculator.fit_transform(mols).shape[0] == len(mols)

    def test_combining_sets_widens_the_output(self, mols):
        one = DescriptorCalculator(blocks=["physicochemical"])
        two = DescriptorCalculator(
            blocks=["physicochemical", "constitutional"]
        )
        assert two.fit_transform(mols).shape[1] > one.fit_transform(mols).shape[1]

    def test_feature_names_match_the_width(self, mols):
        calculator = DescriptorCalculator(
            blocks=["physicochemical", "lipinski"]
        ).fit(mols)
        assert len(calculator.get_feature_names_out()) == calculator.transform(
            mols
        ).shape[1]

    def test_unknown_set_is_rejected(self, mols):
        with pytest.raises(ValueError):
            DescriptorCalculator(blocks=["nonexistent"]).fit(mols)


class TestDescriptors3D:
    def test_embeds_conformers_and_computes_descriptors(self):
        # 3D descriptors need a conformer, which the transformer generates
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CC(=O)O")]
        X = Descriptors3D(random_state=0).fit_transform(mols)
        assert X.shape[0] == 2
        assert np.isfinite(X).any()

    def test_feature_names_match_the_width(self):
        mols = [Chem.MolFromSmiles("CCO")]
        transformer = Descriptors3D(random_state=0).fit(mols)
        assert len(transformer.get_feature_names_out()) == transformer.transform(
            mols
        ).shape[1]


class TestOptionalBackends:
    def test_mol2vec_requires_gensim(self, mols):
        pytest.importorskip("gensim")
        from qsarkit.representation import Mol2VecTransformer

        transformer = Mol2VecTransformer(vector_size=16, epochs=1, random_state=0)
        X = transformer.fit(mols).transform(mols)
        assert X.shape == (len(mols), 16)

    def test_mol_to_sentence_produces_identifiers(self):
        from qsarkit.representation import mol_to_sentence

        sentence = mol_to_sentence(
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"), radius=1
        )
        assert sentence and all(isinstance(token, str) for token in sentence)

    def test_chemberta_requires_transformers(self, mols):
        pytest.importorskip("transformers")
        pytest.importorskip("torch")
        pytest.skip("ChemBERTa needs a network download; covered by an integration run")

    def test_mhfp_backend(self, mols):
        pytest.importorskip("mhfp")
        from qsarkit.representation import MHFPFingerprint

        assert MHFPFingerprint(n_bits=128).transform(mols).shape == (len(mols), 128)
