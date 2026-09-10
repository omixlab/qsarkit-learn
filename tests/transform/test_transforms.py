from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from sklearn.base import clone

from qsarkit.base import InvalidMoleculeError, ModelNotFittedError
from qsarkit.transform import (
    DescriptorScaler,
    MolToSmiles,
    MoleculeFeatureUnion,
    NaNHandler,
    SmilesToMol,
    VarianceThresholdMol,
    make_qsar_pipeline,
)


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1", "CCN")]


@pytest.fixture
def with_nan():
    return np.array([[1.0, np.nan, 5.0], [3.0, 2.0, 5.0], [5.0, 4.0, 5.0]])


class TestSmilesToMol:
    def test_parses_smiles(self):
        mols = SmilesToMol().transform(["CCO", "c1ccccc1"])
        assert [Chem.MolToSmiles(m) for m in mols] == ["CCO", "c1ccccc1"]

    def test_bad_smiles_becomes_none(self):
        assert SmilesToMol().transform(["!!bad!!"]) == [None]

    def test_on_error_raise(self):
        with pytest.raises(InvalidMoleculeError, match="position 0"):
            SmilesToMol(on_error="raise").transform(["!!bad!!"])

    def test_sanitize_off_keeps_odd_structures(self):
        mols = SmilesToMol(sanitize=False).transform(["c1ccc1"])
        assert mols[0] is not None

    def test_fit_is_a_noop_and_returns_self(self):
        t = SmilesToMol()
        assert t.fit(["CCO"]) is t

    def test_clone_roundtrip(self):
        t = SmilesToMol(on_error="raise")
        assert clone(t).get_params() == t.get_params()


class TestMolToSmiles:
    def test_canonicalizes(self):
        assert MolToSmiles().transform([Chem.MolFromSmiles("OCC")]) == ["CCO"]

    def test_none_passes_through(self):
        assert MolToSmiles().transform([None]) == [None]

    def test_isomeric_flag(self):
        mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
        assert "@" in MolToSmiles(isomeric=True).transform([mol])[0]
        assert "@" not in MolToSmiles(isomeric=False).transform([mol])[0]

    def test_roundtrip_with_smiles_to_mol(self):
        smiles = ["CCO", "c1ccccc1"]
        assert MolToSmiles().transform(SmilesToMol().transform(smiles)) == smiles


class TestMoleculeFeatureUnion:
    def test_concatenates_widths(self, mols):
        from qsarkit.representation import MACCSKeysFingerprint, MorganFingerprint

        union = MoleculeFeatureUnion(
            [MorganFingerprint(n_bits=64), MACCSKeysFingerprint()]
        )
        X = union.fit_transform(mols)
        assert X.shape == (3, 64 + 167)

    def test_feature_names_match_width(self, mols):
        from qsarkit.representation import MACCSKeysFingerprint, MorganFingerprint

        union = MoleculeFeatureUnion(
            [MorganFingerprint(n_bits=64), MACCSKeysFingerprint()]
        ).fit(mols)
        assert len(union.get_feature_names_out()) == union.transform(mols).shape[1]

    def test_named_transformers_prefix_the_names(self, mols):
        from qsarkit.representation import MorganFingerprint

        union = MoleculeFeatureUnion([("ecfp", MorganFingerprint(n_bits=8))]).fit(mols)
        assert all(n.startswith("ecfp__") for n in union.get_feature_names_out())

    def test_requires_at_least_one_transformer(self, mols):
        with pytest.raises(ValueError, match="at least one transformer"):
            MoleculeFeatureUnion([]).fit(mols)

    def test_requires_fitting(self, mols):
        from qsarkit.representation import MorganFingerprint

        union = MoleculeFeatureUnion([MorganFingerprint(n_bits=8)])
        with pytest.raises(ModelNotFittedError):
            union.transform(mols)
        with pytest.raises(ModelNotFittedError):
            union.get_feature_names_out()


class TestNaNHandler:
    def test_median_imputation(self, with_nan):
        out = NaNHandler(strategy="median").fit_transform(with_nan)
        assert np.isfinite(out).all()
        assert out[0, 1] == pytest.approx(3.0)

    def test_mean_imputation(self, with_nan):
        out = NaNHandler(strategy="mean").fit_transform(with_nan)
        assert out[0, 1] == pytest.approx(3.0)

    def test_constant_imputation(self, with_nan):
        out = NaNHandler(strategy="constant", fill_value=-1.0).fit_transform(with_nan)
        assert out[0, 1] == pytest.approx(-1.0)

    def test_drop_columns(self, with_nan):
        out = NaNHandler(strategy="drop_columns").fit_transform(with_nan)
        assert out.shape == (3, 2)

    def test_handles_infinities(self):
        X = np.array([[1.0, np.inf], [2.0, 3.0], [3.0, 5.0]])
        assert np.isfinite(NaNHandler().fit_transform(X)).all()

    def test_drops_mostly_missing_columns(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, 1.0]])
        out = NaNHandler(max_nan_fraction=0.5).fit_transform(X)
        assert out.shape == (3, 1)

    def test_all_nan_column_does_not_produce_nan(self):
        X = np.array([[1.0, np.nan], [2.0, np.nan]])
        assert np.isfinite(NaNHandler(max_nan_fraction=1.0).fit_transform(X)).all()

    def test_statistics_come_from_training_only(self):
        train = np.array([[1.0], [3.0]])
        handler = NaNHandler(strategy="median").fit(train)
        out = handler.transform(np.array([[np.nan]]))
        assert out[0, 0] == pytest.approx(2.0)

    def test_feature_names(self, with_nan):
        handler = NaNHandler(strategy="drop_columns").fit(with_nan)
        names = handler.get_feature_names_out(["a", "b", "c"])
        assert list(names) == ["a", "c"]

    def test_default_feature_names(self, with_nan):
        handler = NaNHandler().fit(with_nan)
        assert len(handler.get_feature_names_out()) == handler.transform(
            with_nan
        ).shape[1]

    def test_rejects_unknown_strategy(self, with_nan):
        with pytest.raises(ValueError, match="strategy must be"):
            NaNHandler(strategy="bogus").fit(with_nan)

    def test_rejects_bad_max_nan_fraction(self, with_nan):
        with pytest.raises(ValueError, match="max_nan_fraction must be"):
            NaNHandler(max_nan_fraction=2.0).fit(with_nan)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            NaNHandler().fit(np.zeros(5))

    def test_rejects_feature_mismatch(self, with_nan):
        handler = NaNHandler().fit(with_nan)
        with pytest.raises(ValueError, match="expected 3"):
            handler.transform(np.zeros((2, 7)))

    def test_requires_fitting(self, with_nan):
        with pytest.raises(ModelNotFittedError):
            NaNHandler().transform(with_nan)
        with pytest.raises(ModelNotFittedError):
            NaNHandler().get_feature_names_out()


class TestVarianceThresholdMol:
    def test_drops_constant_columns(self, with_nan):
        out = VarianceThresholdMol().fit_transform(with_nan)
        assert out.shape == (3, 2)

    def test_threshold_controls_strictness(self):
        X = np.array([[1.0, 1.0], [1.1, 5.0], [1.2, 9.0]])
        assert VarianceThresholdMol(threshold=1.0).fit_transform(X).shape == (3, 1)

    def test_get_support(self, with_nan):
        f = VarianceThresholdMol().fit(with_nan)
        assert f.get_support().tolist() == [True, True, False]
        assert f.get_support(indices=True).tolist() == [0, 1]

    def test_rejects_negative_threshold(self, with_nan):
        with pytest.raises(ValueError, match="non-negative"):
            VarianceThresholdMol(threshold=-1.0).fit(with_nan)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            VarianceThresholdMol().fit(np.zeros(5))

    def test_rejects_feature_mismatch(self, with_nan):
        f = VarianceThresholdMol().fit(with_nan)
        with pytest.raises(ValueError, match="expected 3"):
            f.transform(np.zeros((2, 9)))

    def test_requires_fitting(self, with_nan):
        with pytest.raises(ModelNotFittedError):
            VarianceThresholdMol().transform(with_nan)
        with pytest.raises(ModelNotFittedError):
            VarianceThresholdMol().get_support()


class TestDescriptorScaler:
    @pytest.fixture
    def X(self):
        return np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])

    def test_standard_scaling_centres_the_data(self, X):
        out = DescriptorScaler().fit_transform(X)
        assert np.allclose(out.mean(axis=0), 0.0)
        assert np.allclose(out.std(axis=0), 1.0)

    def test_minmax_scaling_bounds_the_data(self, X):
        out = DescriptorScaler(method="minmax").fit_transform(X)
        assert out.min() == pytest.approx(0.0) and out.max() == pytest.approx(1.0)

    def test_robust_scaling_runs(self, X):
        assert DescriptorScaler(method="robust").fit_transform(X).shape == X.shape

    def test_inverse_transform_roundtrips(self, X):
        scaler = DescriptorScaler().fit(X)
        assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)

    def test_clip_bounds_unseen_extremes(self, X):
        scaler = DescriptorScaler(clip=True).fit(X)
        extreme = scaler.transform(np.array([[1000.0, 1000.0]]))
        assert extreme.max() <= scaler.transform(X).max() + 1e-9

    def test_feature_names_pass_through(self, X):
        scaler = DescriptorScaler().fit(X)
        assert list(scaler.get_feature_names_out(["mw", "logp"])) == ["mw", "logp"]
        assert len(scaler.get_feature_names_out()) == 2

    def test_rejects_unknown_method(self, X):
        with pytest.raises(ValueError, match="method must be one of"):
            DescriptorScaler(method="bogus").fit(X)

    def test_requires_fitting(self, X):
        for call in ("transform", "inverse_transform", "get_feature_names_out"):
            with pytest.raises(ModelNotFittedError):
                getattr(DescriptorScaler(), call)(X)


class TestMakeQsarPipeline:
    def test_fits_and_predicts_from_smiles(self):
        from sklearn.ensemble import RandomForestRegressor

        from qsarkit.representation import MorganFingerprint

        pipe = make_qsar_pipeline(
            MorganFingerprint(n_bits=64),
            RandomForestRegressor(n_estimators=5, random_state=0),
            from_smiles=True,
        )
        pipe.fit(["CCO", "CCN", "c1ccccc1", "CCC"], [1.0, 2.0, 3.0, 4.0])
        assert pipe.predict(["CCO"]).shape == (1,)

    def test_fits_from_molecules(self, mols):
        from sklearn.ensemble import RandomForestRegressor

        from qsarkit.representation import MorganFingerprint

        pipe = make_qsar_pipeline(
            MorganFingerprint(n_bits=64),
            RandomForestRegressor(n_estimators=5, random_state=0),
        )
        pipe.fit(mols, [1.0, 2.0, 3.0])
        assert pipe.predict(mols).shape == (3,)

    def test_descriptor_pipeline_with_scaling(self):
        from sklearn.ensemble import RandomForestRegressor

        from qsarkit.representation import RDKitDescriptors

        pipe = make_qsar_pipeline(
            RDKitDescriptors(),
            RandomForestRegressor(n_estimators=5, random_state=0),
            scale=True, from_smiles=True,
        )
        pipe.fit(["CCO", "CCN", "c1ccccc1", "CCC"], [1.0, 2.0, 3.0, 4.0])
        assert np.isfinite(pipe.predict(["CCO"])).all()

    def test_step_composition_follows_the_flags(self):
        from sklearn.ensemble import RandomForestRegressor

        from qsarkit.representation import MorganFingerprint

        full = make_qsar_pipeline(
            MorganFingerprint(n_bits=8), RandomForestRegressor(),
            scale=True, handle_nan=True, from_smiles=True,
        )
        assert [name for name, _ in full.steps] == [
            "smiles_to_mol", "representation", "nan", "scale", "model"
        ]
        minimal = make_qsar_pipeline(
            MorganFingerprint(n_bits=8), RandomForestRegressor(),
            scale=False, handle_nan=False, from_smiles=False,
        )
        assert [name for name, _ in minimal.steps] == ["representation", "model"]

    def test_works_under_cross_validation(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        from qsarkit.representation import MorganFingerprint

        pipe = make_qsar_pipeline(
            MorganFingerprint(n_bits=64),
            RandomForestRegressor(n_estimators=5, random_state=0),
            from_smiles=True,
        )
        scores = cross_val_score(
            pipe,
            ["CCO", "CCN", "c1ccccc1", "CCC", "CCCC", "CO"],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            cv=2,
        )
        assert scores.shape == (2,)

    def test_pipeline_is_clonable(self):
        from sklearn.ensemble import RandomForestRegressor

        from qsarkit.representation import MorganFingerprint

        pipe = make_qsar_pipeline(
            MorganFingerprint(n_bits=8), RandomForestRegressor(n_estimators=3)
        )
        assert len(clone(pipe).steps) == len(pipe.steps)
