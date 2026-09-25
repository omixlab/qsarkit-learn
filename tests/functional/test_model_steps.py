"""Tests for the modelling half of the functional pipe API."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from sklearn.linear_model import Ridge

from qsarkit.functional import (
    FeatureSet,
    MoleculeSet,
    applicability_domain,
    collect,
    cross_validate,
    describe,
    desalt,
    drop_constant,
    drop_correlated,
    drop_invalid,
    featurize,
    fingerprint,
    fit,
    impute,
    molecules,
    pipeline,
    resample,
    scale,
    select_features,
    split,
)
from qsarkit.representation import MorganFingerprint, PhysicochemicalDescriptors

SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1Cl",
    "c1ccncc1C", "c1ccncc1CC", "c1ccncc1Cl", "c1ccncc1Br",
    "CCO", "CCN", "CCC", "CCCl",
]
Y = np.arange(len(SMILES), dtype=float)


@pytest.fixture
def demo():
    return molecules(SMILES, Y)


class TestFeaturize:
    def test_produces_a_feature_set(self, demo):
        fs = demo >> fingerprint(n_bits=64)
        assert isinstance(fs, FeatureSet)
        assert fs.shape == (len(SMILES), 64)

    def test_unpacks_as_X_y(self, demo):
        X, y = demo >> fingerprint(n_bits=32)
        assert X.shape == (len(SMILES), 32)
        assert y.tolist() == Y.tolist()

    def test_keeps_molecules_for_downstream_steps(self, demo):
        fs = demo >> fingerprint(n_bits=32)
        assert fs.mols is not None and len(fs.mols) == len(SMILES)

    def test_can_discard_molecules(self, demo):
        assert (demo >> fingerprint(n_bits=32, keep_mols=False)).mols is None

    def test_accepts_any_transformer(self, demo):
        assert (demo >> featurize(MorganFingerprint(n_bits=16))).shape[1] == 16
        assert (demo >> featurize(PhysicochemicalDescriptors())).shape[0] == len(SMILES)

    def test_propagates_feature_names(self, demo):
        fs = demo >> describe("lipinski")
        assert fs.feature_names is not None
        assert len(fs.feature_names) == fs.shape[1]

    def test_history_records_the_transition(self, demo):
        fs = demo >> desalt() >> fingerprint(n_bits=16)
        assert any("featurize" in entry for entry in fs.history)

    def test_refuses_invalid_molecules_rather_than_emitting_junk_rows(self):
        # A None row would silently break the X/y alignment the pipe
        # maintains everywhere else, so this must fail loudly.
        bad = molecules(["CCO", "not-a-molecule"], [1.0, 2.0])
        with pytest.raises(ValueError, match="drop_invalid"):
            bad >> fingerprint(n_bits=16)

    def test_drop_invalid_makes_it_work(self):
        fs = molecules(["CCO", "!!bad!!"], [1.0, 2.0]) >> drop_invalid() >> fingerprint(n_bits=16)
        assert fs.shape == (1, 16)
        assert fs.y.tolist() == [1.0]

    def test_rejects_unknown_names(self):
        with pytest.raises(ValueError, match="Unknown fingerprint"):
            fingerprint("nonesuch")
        with pytest.raises(ValueError, match="Unknown descriptor set"):
            describe("nonesuch")

    def test_cannot_featurize_twice(self, demo):
        fs = demo >> fingerprint(n_bits=16)
        with pytest.raises(TypeError, match="already been featurized"):
            fs >> fingerprint(n_bits=16)


class TestFeatureSteps:
    def test_scale_standardizes(self, demo):
        fs = demo >> describe() >> scale()
        assert abs(float(fs.X.mean())) < 1e-9

    def test_scale_accepts_a_positional_method(self, demo):
        # `scale("robust")` must configure the step, not be mistaken for data.
        fs = demo >> describe() >> scale("robust")
        assert fs.shape[0] == len(SMILES)

    def test_scale_none_is_a_noop(self, demo):
        before = demo >> describe()
        after = before >> scale("none")
        assert np.array_equal(before.X, after.X)

    def test_scale_rejects_unknown_method(self, demo):
        with pytest.raises(ValueError, match="Unknown scaling method"):
            demo >> describe() >> scale("nonesuch")

    def test_impute_fills_non_finite_values(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        filled, _, _ = impute(X, strategy="median")
        assert np.isfinite(filled).all()

    def test_impute_drop_keeps_labels_aligned(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
        kept, y, _ = impute(X, np.array([1.0, 2.0, 3.0]), strategy="drop")
        assert kept.shape == (2, 2)
        assert y.tolist() == [2.0, 3.0]

    def test_impute_handles_an_all_nan_column(self):
        # SimpleImputer drops such a column; the matrix must keep its width
        # or the feature names stop lining up.
        X = np.array([[1.0, np.nan], [3.0, np.nan]])
        filled, _, _ = impute(X, strategy="mean")
        assert filled.shape == (2, 2)
        assert np.isfinite(filled).all()

    def test_impute_passes_finite_matrices_through(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        out, _, _ = impute(X)
        assert np.array_equal(out, X)

    def test_drop_constant_removes_invariant_columns(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        reduced, _, _ = drop_constant(X)
        assert reduced.shape == (3, 1)

    def test_drop_correlated_removes_one_of_each_pair(self):
        X = np.array([[1.0, 2.0, 9.0], [2.0, 4.0, 1.0], [3.0, 6.0, 5.0]])
        reduced, _, _ = drop_correlated(X, threshold=0.99)
        assert reduced.shape == (3, 2)

    def test_select_features_keeps_k(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 10))
        y = X[:, 0] * 2 + rng.normal(scale=0.1, size=30)
        reduced, _, _ = select_features(X, y, k=3)
        assert reduced.shape == (30, 3)

    def test_select_features_needs_labels_for_supervised_methods(self, demo):
        unlabelled = molecules(SMILES) >> fingerprint(n_bits=32)
        with pytest.raises(ValueError, match="needs labels"):
            unlabelled >> select_features(k=4)

    def test_select_features_variance_works_unsupervised(self):
        unlabelled = molecules(SMILES) >> fingerprint(n_bits=64)
        assert (unlabelled >> select_features(method="variance")).shape[0] == len(SMILES)

    def test_feature_names_are_dropped_when_width_changes(self, demo):
        fs = demo >> describe() >> drop_constant()
        assert fs.feature_names is None or len(fs.feature_names) == fs.shape[1]

    def test_feature_step_rejects_molecules(self, demo):
        with pytest.raises(TypeError, match="featurize"):
            demo >> scale()


class TestTerminals:
    def test_split_returns_two_feature_sets(self, demo):
        train, test = demo >> fingerprint(n_bits=32) >> split(test_size=0.25)
        assert isinstance(train, FeatureSet) and isinstance(test, FeatureSet)
        assert len(train) + len(test) == len(SMILES)

    def test_split_keeps_scaffolds_apart(self, demo):
        train, test = demo >> fingerprint(n_bits=32) >> split("scaffold", test_size=0.3)
        from qsarkit.chemspace import bemis_murcko_smiles

        train_scaffolds = {bemis_murcko_smiles(m) for m in train.mols}
        test_scaffolds = {bemis_murcko_smiles(m) for m in test.mols}
        assert not (train_scaffolds & test_scaffolds)

    def test_split_works_without_molecules_for_matrix_methods(self, demo):
        fs = demo >> fingerprint(n_bits=32, keep_mols=False)
        train, test = fs >> split("random", test_size=0.25, random_state=0)
        assert len(train) + len(test) == len(SMILES)

    def test_scaffold_split_needs_molecules(self, demo):
        fs = demo >> fingerprint(n_bits=32, keep_mols=False)
        with pytest.raises(ValueError, match="needs the molecules"):
            fs >> split("scaffold")

    def test_split_rejects_unknown_method(self, demo):
        with pytest.raises(ValueError, match="Unknown split method"):
            demo >> fingerprint(n_bits=16) >> split("nonesuch")

    def test_fit_returns_a_fitted_model(self, demo):
        model = demo >> fingerprint(n_bits=32) >> fit("rf", random_state=0)
        assert hasattr(model, "predict")
        assert model.predict(np.zeros((1, 32))).shape == (1,)

    def test_fit_accepts_an_estimator_instance(self, demo):
        model = demo >> fingerprint(n_bits=32) >> fit(Ridge())
        assert type(model).__name__ == "Ridge"

    def test_fit_infers_classification_from_integer_labels(self):
        labels = np.array([0, 1] * 6)
        model = molecules(SMILES, labels) >> fingerprint(n_bits=32) >> fit("rf")
        assert hasattr(model, "predict_proba")

    def test_fit_infers_regression_from_continuous_labels(self, demo):
        model = demo >> fingerprint(n_bits=32) >> fit("rf", random_state=0)
        assert not hasattr(model, "predict_proba")

    def test_fit_needs_labels(self):
        with pytest.raises(ValueError, match="needs labels"):
            molecules(SMILES) >> fingerprint(n_bits=16) >> fit("rf")

    def test_cross_validate_returns_a_report(self, demo):
        report = demo >> fingerprint(n_bits=32) >> cross_validate(
            "rf", n_splits=3, random_state=0
        )
        assert "q2" in report

    def test_applicability_domain_returns_a_fitted_domain(self, demo):
        domain = demo >> fingerprint(n_bits=32) >> applicability_domain(
            "tanimoto", threshold=0.2
        )
        assert domain.predict(np.ones((1, 32))).shape == (1,)

    def test_applicability_domain_rejects_unknown_method(self, demo):
        with pytest.raises(ValueError, match="Unknown applicability domain"):
            demo >> fingerprint(n_bits=16) >> applicability_domain("nonesuch")

    def test_collect_returns_the_set(self, demo):
        assert isinstance(demo >> desalt() >> collect(), MoleculeSet)

    def test_collect_as_frame(self, demo):
        frame = demo >> desalt() >> collect(as_frame=True)
        assert list(frame.columns) == ["smiles", "y"]
        assert len(frame) == len(SMILES)

    def test_terminal_rejects_molecules(self, demo):
        with pytest.raises(TypeError, match="featurize"):
            demo >> fit("rf")


class TestMixedPipelines:
    def test_a_whole_workflow_composes(self, demo):
        model = (
            demo
            >> desalt()
            >> drop_invalid()
            >> fingerprint(n_bits=64)
            >> drop_constant()
            >> fit("rf", random_state=0)
        )
        assert hasattr(model, "predict")

    def test_a_mixed_pipeline_is_reusable(self):
        prep = desalt() >> drop_invalid() >> fingerprint(n_bits=32) >> scale()
        first = molecules(SMILES[:6], Y[:6]) >> prep
        second = molecules(SMILES[6:], Y[6:]) >> prep
        assert first.shape[1] == second.shape[1] == 32

    def test_pipeline_helper_accepts_mixed_steps(self, demo):
        prep = pipeline(desalt(), fingerprint(n_bits=16), scale())
        assert (demo >> prep).shape == (len(SMILES), 16)

    def test_molecule_step_after_featurize_is_rejected(self, demo):
        with pytest.raises(TypeError, match="must come before"):
            demo >> fingerprint(n_bits=16) >> desalt()


class TestResample:
    """Class rebalancing in feature space, where synthesis is meaningful."""

    @pytest.fixture
    def imbalanced(self):
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        return molecules(SMILES, labels) >> fingerprint(n_bits=64)

    def test_undersampling_equalizes_the_classes(self, imbalanced):
        balanced = imbalanced >> resample(random_state=0)
        _, counts = np.unique(balanced.y, return_counts=True)
        assert len(set(counts)) == 1

    def test_oversampling_equalizes_upward(self, imbalanced):
        balanced = imbalanced >> resample("oversample", random_state=0)
        _, counts = np.unique(balanced.y, return_counts=True)
        assert len(set(counts)) == 1
        assert len(balanced) > len(imbalanced) / 2

    def test_features_and_labels_stay_aligned(self, imbalanced):
        balanced = imbalanced >> resample(random_state=0)
        assert balanced.X.shape[0] == balanced.y.shape[0]

    def test_molecules_follow_the_selected_rows(self, imbalanced):
        balanced = imbalanced >> resample(random_state=0)
        assert balanced.mols is not None
        assert len(balanced.mols) == len(balanced)

    def test_a_selecting_sampler_keeps_the_molecules(self, imbalanced):
        class _Selector:
            def fit_resample(self, X, y):
                self.sample_indices_ = np.array([0, 8])
                return X[[0, 8]], y[[0, 8]]

        balanced = imbalanced >> resample(_Selector())
        assert balanced.mols is not None and len(balanced.mols) == 2

    def test_a_synthesizing_sampler_drops_the_molecules_and_warns(
        self, imbalanced
    ):
        """A synthesized row corresponds to no molecule.

        Returning a mismatched list would be worse than returning none, so
        they are dropped — loudly, because downstream steps that need them
        will now fail.
        """
        class _Synthesizer:
            def fit_resample(self, X, y):
                return np.vstack([X, X[:1]]), np.append(y, 1)

        with pytest.warns(UserWarning, match="synthesized new rows"):
            balanced = imbalanced >> resample(_Synthesizer())
        assert balanced.mols is None
        assert len(balanced) == len(imbalanced) + 1

    def test_needs_labels(self):
        unlabelled = molecules(SMILES) >> fingerprint(n_bits=32)
        with pytest.raises(ValueError, match="needs labels"):
            unlabelled >> resample()

    def test_rejects_an_unknown_strategy(self, imbalanced):
        with pytest.raises(ValueError, match="sampler must be"):
            imbalanced >> resample("magic")

    def test_rejects_an_object_that_is_not_a_sampler(self, imbalanced):
        with pytest.raises(ValueError, match="fit_resample"):
            imbalanced >> resample(object())

    def test_is_reproducible(self, imbalanced):
        first = imbalanced >> resample(random_state=7)
        second = imbalanced >> resample(random_state=7)
        assert np.array_equal(first.X, second.X)

    def test_an_already_balanced_set_is_unchanged(self):
        labels = np.array([0, 1] * 6)
        balanced_input = molecules(SMILES, labels) >> fingerprint(n_bits=32)
        result = balanced_input >> resample(random_state=0)
        assert len(result) == len(balanced_input)
