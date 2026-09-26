from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge

from qsarkit.base import ModelNotFittedError
from qsarkit.models import QSARClassifier, QSARRegressor
from qsarkit.explainability import (
    AtomicContributionMap,
    CounterfactualExplainer,
    FragmentContributionAnalyzer,
    LIMEExplainer,
    PartialDependence,
    PermutationImportance,
    SHAPExplainer,
)
from qsarkit.representation import MorganFingerprint

# A dataset with an obvious rule the model must learn: aromatics are
# active (~8), aliphatics are not (~1). Every explanation below is checked
# against that known ground truth.
SMILES = [
    "CCO", "CCN", "CCC", "CCCCO", "CC(=O)O", "CCOC",       # aliphatic, low
    "c1ccccc1", "c1ccccc1O", "c1ccccc1N", "c1ccccc1C",      # aromatic, high
]
ACTIVITY = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 8.0, 8.1, 8.2, 8.3])


@pytest.fixture(scope="module")
def setup():
    fp = MorganFingerprint(n_bits=256)
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    X = fp.transform(mols)
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, ACTIVITY)
    return fp, mols, X, model


@pytest.fixture(scope="module")
def numeric():
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=80, n_features=6, n_informative=2, random_state=0
    )
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    return X, y, model


class TestPermutationImportance:
    def test_finds_the_informative_features(self, numeric):
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=120, n_features=6, n_informative=2,
            coef=False, random_state=1, shuffle=False,
        )
        model = RandomForestRegressor(n_estimators=30, random_state=0).fit(X, y)
        imp = PermutationImportance(random_state=0).fit(model, X, y)
        # make_regression with shuffle=False puts informative features first
        top_two = set(imp.to_dataframe(2)["feature"])
        assert top_two <= {"x0", "x1"}

    def test_shapes_and_names(self, numeric):
        X, y, model = numeric
        imp = PermutationImportance(n_repeats=3, random_state=0).fit(model, X, y)
        assert imp.importances_mean_.shape == (6,)
        assert imp.importances_std_.shape == (6,)

    def test_custom_feature_names(self, numeric):
        X, y, model = numeric
        names = [f"desc{i}" for i in range(6)]
        imp = PermutationImportance(n_repeats=2, random_state=0).fit(
            model, X, y, feature_names=names
        )
        assert set(imp.to_dataframe()["feature"]) == set(names)

    def test_dataframe_is_sorted_descending(self, numeric):
        X, y, model = numeric
        frame = PermutationImportance(n_repeats=3, random_state=0).fit(
            model, X, y
        ).to_dataframe()
        assert list(frame.columns) == ["feature", "importance", "std"]
        assert frame["importance"].is_monotonic_decreasing

    def test_top_n_truncates(self, numeric):
        X, y, model = numeric
        imp = PermutationImportance(n_repeats=2, random_state=0).fit(model, X, y)
        assert len(imp.to_dataframe(top_n=3)) == 3

    def test_plot_returns_a_figure(self, numeric):
        import plotly.graph_objects as go

        X, y, model = numeric
        imp = PermutationImportance(n_repeats=2, random_state=0).fit(model, X, y)
        assert isinstance(imp.plot(top_n=3), go.Figure)

    def test_requires_fitting(self):
        with pytest.raises(ModelNotFittedError):
            PermutationImportance().to_dataframe()


class TestAtomicContributionMap:
    def test_one_weight_per_atom(self, setup):
        fp, mols, _, model = setup
        result = AtomicContributionMap(model, fp).explain(mols[0])
        assert result.weights.shape == (mols[0].GetNumAtoms(),)

    def test_prediction_matches_the_model(self, setup):
        fp, mols, _, model = setup
        result = AtomicContributionMap(model, fp).explain(mols[6])
        expected = model.predict(fp.transform([mols[6]]))[0]
        assert result.prediction == pytest.approx(expected)

    def test_aromatic_atoms_drive_the_prediction(self, setup):
        # the model learned "aromatic = active", so masking a ring atom of
        # a substituted aromatic must lower the prediction substantially
        fp, mols, _, model = setup
        result = AtomicContributionMap(model, fp).explain(
            Chem.MolFromSmiles("c1ccccc1O")
        )
        assert result.weights.max() > 0.5

    def test_symmetric_molecule_spreads_the_credit_evenly(self, setup):
        # Masking one carbon of benzene leaves five aromatic carbons still
        # setting the same bits, so no single atom is responsible. A flat
        # weight vector here means redundancy, not irrelevance.
        fp, mols, _, model = setup
        weights = AtomicContributionMap(model, fp).explain(mols[6]).weights
        assert np.ptp(weights) == pytest.approx(0.0, abs=1e-9)

    def test_aliphatic_and_aromatic_predictions_differ(self, setup):
        fp, mols, _, model = setup
        explainer = AtomicContributionMap(model, fp)
        assert explainer.explain(mols[6]).prediction > explainer.explain(
            mols[0]
        ).prediction

    def test_smiles_is_recorded(self, setup):
        fp, mols, _, model = setup
        assert AtomicContributionMap(model, fp).explain(mols[0]).smiles == "CCO"

    def test_most_positive_and_negative_indices(self, setup):
        fp, mols, _, model = setup
        result = AtomicContributionMap(model, fp).explain(mols[7])
        assert 0 <= result.most_positive < len(result.weights)
        assert 0 <= result.most_negative < len(result.weights)

    def test_none_molecule_raises(self, setup):
        fp, _, _, model = setup
        with pytest.raises(ValueError, match="None molecule"):
            AtomicContributionMap(model, fp).explain(None)

    def test_similarity_map_weights_length(self, setup):
        fp, mols, _, model = setup
        weights = AtomicContributionMap(model, fp).to_similarity_map_weights(
            mols[6]
        )
        assert len(weights) == mols[6].GetNumAtoms()
        assert all(isinstance(w, float) for w in weights)

    def test_batch_transform_skips_none(self, setup):
        fp, mols, _, model = setup
        results = AtomicContributionMap(model, fp).transform(
            [mols[0], None, mols[1]]
        )
        assert len(results) == 2

    def test_classifier_with_probabilities(self, setup):
        fp, mols, X, _ = setup
        labels = (ACTIVITY > 4).astype(int)
        clf = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, labels)
        result = AtomicContributionMap(clf, fp, use_proba=True).explain(mols[6])
        assert 0.0 <= result.prediction <= 1.0

    def test_plot_returns_a_figure(self, setup):
        import plotly.graph_objects as go

        fp, mols, _, model = setup
        assert isinstance(AtomicContributionMap(model, fp).plot(mols[0]), go.Figure)

    def test_accepts_a_plain_callable_featurizer(self, setup):
        fp, mols, _, model = setup
        explainer = AtomicContributionMap(model, lambda ms: fp.transform(ms))
        assert explainer.explain(mols[0]).weights.shape == (3,)


class TestFragmentContributionAnalyzer:
    def test_named_fragments_are_attributed(self, setup):
        fp, mols, _, model = setup
        analyzer = FragmentContributionAnalyzer(
            model, fp, fragments={"hydroxyl": "[OX2H]", "benzene": "c1ccccc1"}
        )
        frame = analyzer.analyze(Chem.MolFromSmiles("c1ccccc1O"))
        assert set(frame["fragment"]) == {"hydroxyl", "benzene"}
        assert list(frame.columns) == [
            "fragment", "n_atoms", "contribution", "mean_contribution"
        ]

    def test_the_aromatic_ring_carries_the_signal(self, setup):
        fp, mols, _, model = setup
        analyzer = FragmentContributionAnalyzer(
            model, fp, fragments={"hydroxyl": "[OX2H]", "benzene": "c1ccccc1"}
        )
        frame = analyzer.analyze(Chem.MolFromSmiles("c1ccccc1O")).set_index(
            "fragment"
        )
        assert frame.loc["benzene", "contribution"] > frame.loc[
            "hydroxyl", "contribution"
        ]

    def test_absent_fragments_are_omitted(self, setup):
        fp, mols, _, model = setup
        analyzer = FragmentContributionAnalyzer(
            model, fp, fragments={"nitro": "[N+](=O)[O-]"}
        )
        assert analyzer.analyze(Chem.MolFromSmiles("CCO")).empty

    def test_invalid_smarts_is_skipped(self, setup):
        fp, mols, _, model = setup
        analyzer = FragmentContributionAnalyzer(
            model, fp, fragments={"bad": "!!!", "hydroxyl": "[OX2H]"}
        )
        frame = analyzer.analyze(Chem.MolFromSmiles("CCO"))
        assert set(frame["fragment"]) == {"hydroxyl"}

    def test_brics_mode_finds_fragments(self, setup):
        fp, mols, _, model = setup
        frame = FragmentContributionAnalyzer(model, fp).analyze(
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        )
        assert len(frame) >= 1

    def test_brics_mode_on_an_unfragmentable_molecule(self, setup):
        fp, mols, _, model = setup
        frame = FragmentContributionAnalyzer(model, fp).analyze(
            Chem.MolFromSmiles("CCO")
        )
        assert len(frame) == 1

    def test_sorted_by_absolute_contribution(self, setup):
        fp, mols, _, model = setup
        frame = FragmentContributionAnalyzer(model, fp).analyze(
            Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
        )
        magnitudes = frame["contribution"].abs().tolist()
        assert magnitudes == sorted(magnitudes, reverse=True)


class TestCounterfactualExplainer:
    def test_finds_a_molecule_that_flips_the_prediction(self, setup):
        fp, mols, _, model = setup
        explainer = CounterfactualExplainer(
            model, fp, delta=2.0, min_similarity=0.0
        )
        result = explainer.explain(mols[0], mols)
        assert result is not None
        assert abs(result["delta"]) >= 2.0

    def test_counterfactual_is_aromatic_for_an_aliphatic_query(self, setup):
        # the only way to move the prediction is to become aromatic
        fp, mols, _, model = setup
        result = CounterfactualExplainer(
            model, fp, delta=2.0, min_similarity=0.0
        ).explain(mols[0], mols)
        assert any(a.GetIsAromatic() for a in result["counterfactual"].GetAtoms())

    def test_result_keys(self, setup):
        fp, mols, _, model = setup
        result = CounterfactualExplainer(
            model, fp, delta=2.0, min_similarity=0.0
        ).explain(mols[0], mols)
        assert set(result) == {
            "counterfactual", "smiles", "similarity", "prediction",
            "original_prediction", "delta", "transformation",
        }

    def test_returns_none_when_delta_is_unreachable(self, setup):
        fp, mols, _, model = setup
        assert CounterfactualExplainer(
            model, fp, delta=1000.0, min_similarity=0.0
        ).explain(mols[0], mols) is None

    def test_returns_none_when_similarity_is_too_strict(self, setup):
        fp, mols, _, model = setup
        assert CounterfactualExplainer(
            model, fp, delta=2.0, min_similarity=0.99
        ).explain(mols[0], mols) is None

    def test_empty_candidate_set(self, setup):
        fp, mols, _, model = setup
        assert CounterfactualExplainer(model, fp).explain(mols[0], []) is None

    def test_none_query_raises(self, setup):
        fp, mols, _, model = setup
        with pytest.raises(ValueError, match="None molecule"):
            CounterfactualExplainer(model, fp).explain(None, mols)

    def test_reports_a_matched_pair_transformation(self, setup):
        # phenol vs aniline differ by one atom, so they form an MMP
        fp, _, _, model = setup
        pair = [Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("c1ccccc1N")]
        explainer = CounterfactualExplainer(
            model, fp, delta=0.0, min_similarity=0.0
        )
        result = explainer.explain(pair[0], [pair[1]])
        assert result is not None

    def test_none_candidates_are_skipped(self, setup):
        fp, mols, _, model = setup
        result = CounterfactualExplainer(
            model, fp, delta=2.0, min_similarity=0.0
        ).explain(mols[0], [None, mols[6]])
        assert result is not None


class TestPartialDependence:
    def test_curve_shapes_agree(self, numeric):
        X, _, model = numeric
        grid, average = PartialDependence(model, grid_resolution=15).compute(
            X, feature=0
        )
        assert grid.shape == average.shape == (15,)

    def test_grid_spans_the_feature_range(self, numeric):
        X, _, model = numeric
        grid, _ = PartialDependence(model, grid_resolution=10).compute(X, 2)
        assert grid[0] == pytest.approx(X[:, 2].min())
        assert grid[-1] == pytest.approx(X[:, 2].max())

    def test_informative_feature_moves_the_prediction(self):
        from sklearn.datasets import make_regression

        X, y = make_regression(
            n_samples=100, n_features=4, n_informative=1,
            random_state=0, shuffle=False,
        )
        model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
        _, informative = PartialDependence(model, grid_resolution=20).compute(X, 0)
        _, noise = PartialDependence(model, grid_resolution=20).compute(X, 3)
        assert np.ptp(informative) > np.ptp(noise)

    def test_rejects_out_of_range_feature(self, numeric):
        X, _, model = numeric
        with pytest.raises(ValueError, match="out of range"):
            PartialDependence(model).compute(X, feature=99)

    def test_rejects_tiny_grid(self, numeric):
        X, _, model = numeric
        with pytest.raises(ValueError, match="grid_resolution must be"):
            PartialDependence(model, grid_resolution=1).compute(X, 0)

    def test_plot_returns_a_figure(self, numeric):
        import plotly.graph_objects as go

        X, _, model = numeric
        fig = PartialDependence(model, grid_resolution=8).plot(X, 0, "logP")
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2  # curve plus the data rug


class TestSHAPExplainer:
    def test_auto_selects_tree_for_a_forest(self, numeric):
        _, _, model = numeric
        assert SHAPExplainer(model)._resolve_type() == "tree"

    def test_auto_selects_linear_for_ridge(self):
        assert SHAPExplainer(Ridge())._resolve_type() == "linear"

    def test_auto_falls_back_to_kernel(self):
        from sklearn.neighbors import KNeighborsRegressor

        assert SHAPExplainer(KNeighborsRegressor())._resolve_type() == "kernel"

    def test_explicit_type_wins(self, numeric):
        _, _, model = numeric
        assert SHAPExplainer(model, explainer_type="kernel")._resolve_type() == (
            "kernel"
        )

    def test_background_is_required_for_kernel(self, numeric):
        _, _, model = numeric
        with pytest.raises(ValueError, match="needs a `background`"):
            SHAPExplainer(model, explainer_type="kernel")._sample_background()

    def test_background_is_subsampled(self, numeric):
        X, _, model = numeric
        explainer = SHAPExplainer(
            model, explainer_type="kernel", background=X,
            n_background=10, random_state=0,
        )
        assert len(explainer._sample_background()) == 10

    def test_small_background_is_used_whole(self, numeric):
        X, _, model = numeric
        explainer = SHAPExplainer(
            model, explainer_type="kernel", background=X[:5], n_background=100
        )
        assert len(explainer._sample_background()) == 5

    def test_tree_explainer_values(self, numeric):
        pytest.importorskip("shap")

        X, _, model = numeric
        explainer = SHAPExplainer(model)
        values = explainer.shap_values(X[:5])
        assert values.shape == (5, X.shape[1])

    def test_global_importance_is_ranked(self, numeric):
        pytest.importorskip("shap")

        X, _, model = numeric
        frame = SHAPExplainer(model).global_importance(X[:20])
        assert list(frame.columns) == ["feature", "importance"]
        assert frame["importance"].is_monotonic_decreasing

    def test_explain_one(self, numeric):
        pytest.importorskip("shap")

        X, _, model = numeric
        frame = SHAPExplainer(model).explain_one(X[0], top_n=3)
        assert len(frame) == 3
        assert list(frame.columns) == ["feature", "value", "shap_value"]

    def test_plot_importance(self, numeric):
        pytest.importorskip("shap")
        import plotly.graph_objects as go

        X, _, model = numeric
        assert isinstance(
            SHAPExplainer(model).plot_importance(X[:20], top_n=3), go.Figure
        )

    def test_rejects_unknown_explainer_type(self, numeric):
        # _build() imports shap before it validates, so without the extra the
        # (correct) OptionalDependencyError arrives first.
        pytest.importorskip("shap")
        _, _, model = numeric
        with pytest.raises(ValueError, match="explainer_type must be"):
            SHAPExplainer(model, explainer_type="bogus")._build()


class TestLIMEExplainer:
    def test_explains_a_single_prediction(self, numeric):
        pytest.importorskip("lime")

        X, _, model = numeric
        explainer = LIMEExplainer(model, X, n_samples=200, random_state=0)
        frame = explainer.explain_one(X[0], top_n=3)
        assert len(frame) == 3
        assert list(frame.columns) == ["feature", "weight"]

    def test_classification_mode(self):
        pytest.importorskip("lime")
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=60, n_features=5, random_state=0)
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        explainer = LIMEExplainer(
            model, X, mode="classification", n_samples=200, random_state=0
        )
        assert len(explainer.explain_one(X[0], top_n=2)) == 2


class TestSHAPUnwrapsQsarkitFacades:
    """SHAP must work on the package's own estimators.

    ``QSARRegressor`` and ``QSARClassifier`` hold the fitted backend in
    ``estimator_`` and delegate to it. Passing the facade straight to SHAP
    failed two ways: ``shap.TreeExplainer`` raised ``InvalidModelError`` on a
    ``QSARClassifier``, and ``explainer_type="auto"`` classified the facade by
    its own class name and silently chose the kernel explainer, which is far
    slower and needs a background set.
    """

    @pytest.fixture(scope="class")
    def binary(self):
        rng = np.random.default_rng(0)
        X = (rng.random((80, 12)) > 0.6).astype(float)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)
        return X, y

    def test_auto_sees_through_the_classifier_facade(self, binary):
        X, y = binary
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 10}
        ).fit(X, y)
        assert SHAPExplainer(model)._resolve_type() == "tree"

    def test_auto_sees_through_the_regressor_facade(self, binary):
        X, y = binary
        model = QSARRegressor("ridge").fit(X, y.astype(float))
        assert SHAPExplainer(model)._resolve_type() == "linear"

    def test_a_facade_wrapped_forest_explains(self, binary):
        pytest.importorskip("shap")
        X, y = binary
        model = QSARClassifier(
            "rf", random_state=0, model_params={"n_estimators": 10}
        ).fit(X, y)
        values = np.asarray(SHAPExplainer(model).shap_values(X[:5]))
        # Either (n, features) or (n, features, classes) depending on the
        # shap version; both must cover every feature.
        assert values.shape[0] == 5
        assert values.shape[1] == X.shape[1]

    def test_a_bare_estimator_still_works(self, binary):
        """Unwrapping must not break the case that already worked."""
        from sklearn.ensemble import RandomForestClassifier

        X, y = binary
        model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert SHAPExplainer(model)._resolve_type() == "tree"
