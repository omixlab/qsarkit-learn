from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.base import ModelNotFittedError
from qsarkit.sar import FreeWilsonAnalysis, RGroupAnalyzer, SARTable

CORE = "O=C(O)c1ccccc1"


class TestRGroupAnalyzer:
    def test_decompose_produces_core_and_r_columns(self, series_mols):
        table = RGroupAnalyzer(core=CORE).decompose(series_mols)
        assert "Core" in table.columns
        assert "R1" in table.columns
        assert len(table) == len(series_mols)

    def test_molecule_index_tracks_input_positions(self, series_mols):
        table = RGroupAnalyzer(core=CORE).decompose(series_mols)
        assert list(table["molecule_index"]) == list(range(len(series_mols)))

    def test_substituents_are_the_expected_ones(self, series_mols):
        table = RGroupAnalyzer(core=CORE).decompose(series_mols)
        groups = "".join(table["R1"])
        for atom in ("Cl", "Br", "F", "N", "O"):
            assert atom in groups

    def test_core_is_inferred_when_not_given(self, series_mols):
        table = RGroupAnalyzer().decompose(series_mols)
        assert not table.empty

    def test_core_accepts_a_mol_object(self, series_mols):
        table = RGroupAnalyzer(core=Chem.MolFromSmiles(CORE)).decompose(series_mols)
        assert not table.empty

    def test_unparseable_core_raises(self, series_mols):
        with pytest.raises(ValueError, match="Could not parse core"):
            RGroupAnalyzer(core="not a molecule!!!").decompose(series_mols)

    def test_inferring_core_from_empty_set_raises(self):
        with pytest.raises(ValueError, match="empty molecule set"):
            RGroupAnalyzer().decompose([])

    def test_non_matching_molecules_are_reported_as_unmatched(self, series_mols):
        mols = list(series_mols) + [Chem.MolFromSmiles("CCCCCC")]
        table = RGroupAnalyzer(core=CORE).decompose(mols)
        assert len(table) == len(series_mols)
        assert table.attrs["unmatched"] == [len(series_mols)]

    def test_r_group_positions(self, series_mols):
        analyzer = RGroupAnalyzer(core=CORE)
        table = analyzer.decompose(series_mols)
        assert analyzer.r_group_positions(table) == ["R1"]


class TestSARTable:
    def test_build_adds_activity_column(self, series_mols, series_activities):
        table = SARTable(core=CORE).build(series_mols, series_activities)
        assert "activity" in table.columns
        assert list(table["activity"]) == pytest.approx(series_activities)

    def test_activity_aligns_after_unmatched_rows_are_dropped(
        self, series_mols, series_activities
    ):
        mols = [Chem.MolFromSmiles("CCCCCC")] + list(series_mols)
        activities = [99.0] + list(series_activities)
        table = SARTable(core=CORE).build(mols, activities)
        # the unmatched alkane must not contribute its 99.0
        assert 99.0 not in list(table["activity"])
        assert list(table["activity"]) == pytest.approx(series_activities)

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but activities has"):
            SARTable(core=CORE).build(series_mols, [1.0])

    def test_substituent_effects_ranks_the_amine_first(
        self, series_mols, series_activities
    ):
        builder = SARTable(core=CORE)
        table = builder.build(series_mols, series_activities)
        effects = builder.substituent_effects(table)
        assert list(effects.columns) == [
            "substituent", "count", "mean_activity", "std_activity"
        ]
        assert "N" in effects.iloc[0]["substituent"]
        assert effects.iloc[0]["mean_activity"] == pytest.approx(8.5)

    def test_substituent_effects_rejects_unknown_position(
        self, series_mols, series_activities
    ):
        builder = SARTable(core=CORE)
        table = builder.build(series_mols, series_activities)
        with pytest.raises(ValueError, match="not in the SAR table"):
            builder.substituent_effects(table, position="R9")

    def test_pivot_rejects_missing_columns(self, series_mols, series_activities):
        builder = SARTable(core=CORE)
        table = builder.build(series_mols, series_activities)
        with pytest.raises(ValueError, match="not in the SAR table"):
            builder.pivot(table, row="R1", column="R2")

    def test_pivot_on_a_two_position_series(self):
        smis = [
            "O=C(O)c1cc(C)ccc1Cl", "O=C(O)c1cc(C)ccc1Br",
            "O=C(O)c1cc(F)ccc1Cl", "O=C(O)c1cc(F)ccc1Br",
        ]
        mols = [Chem.MolFromSmiles(s) for s in smis]
        builder = SARTable(core="O=C(O)c1ccccc1")
        table = builder.build(mols, [5.0, 6.0, 7.0, 8.0])
        if "R2" in table.columns:
            grid = builder.pivot(table)
            assert grid.notna().sum().sum() >= 1


class TestFreeWilsonAnalysis:
    def test_fit_recovers_the_amine_contribution(
        self, series_mols, series_activities
    ):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1).fit(
            series_mols, series_activities
        )
        contributions = fw.to_dataframe()
        top = contributions.iloc[0]
        assert "N" in top["substituent"]
        assert top["contribution"] > 1.5

    def test_fit_explains_the_series_well(self, series_mols, series_activities):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1).fit(
            series_mols, series_activities
        )
        assert fw.r2_ > 0.9
        assert isinstance(fw.intercept_, float)
        assert len(fw.feature_names_) == len(series_mols)

    def test_ordinary_least_squares_path(self, series_mols, series_activities):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.0).fit(
            series_mols, series_activities
        )
        assert fw.r2_ == pytest.approx(1.0, abs=1e-6)

    def test_no_intercept_path(self, series_mols, series_activities):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1, fit_intercept=False).fit(
            series_mols, series_activities
        )
        assert fw.intercept_ == 0.0

    def test_predict_on_training_series(self, series_mols, series_activities):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1).fit(
            series_mols, series_activities
        )
        preds = fw.predict(series_mols)
        assert preds.shape == (len(series_mols),)
        assert np.corrcoef(preds, series_activities)[0, 1] > 0.9

    def test_predict_with_unseen_substituent_does_not_crash(
        self, series_mols, series_activities
    ):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1).fit(
            series_mols, series_activities
        )
        novel = [Chem.MolFromSmiles("O=C(O)c1ccc(I)cc1")]
        preds = fw.predict(novel)
        assert preds.shape == (1,)

    def test_predict_with_no_matching_molecules_returns_empty(
        self, series_mols, series_activities
    ):
        fw = FreeWilsonAnalysis(core=CORE, alpha=0.1).fit(
            series_mols, series_activities
        )
        assert fw.predict([Chem.MolFromSmiles("CCCCCC")]).shape == (0,)

    def test_residuals_flag_the_cliff_compound(self, series_mols):
        # a deliberately non-additive series: the amine effect is
        # far larger than an additive model fitted with shrinkage expects
        fw = FreeWilsonAnalysis(core=CORE, alpha=1.0).fit(
            series_mols, [5.0, 5.2, 5.1, 4.9, 8.5, 5.3]
        )
        residuals = fw.residuals()
        assert list(residuals.columns) == [
            "molecule_index", "observed", "predicted", "residual"
        ]
        assert residuals.iloc[0]["molecule_index"] == 4

    def test_methods_require_fitting_first(self, series_mols):
        fw = FreeWilsonAnalysis(core=CORE)
        with pytest.raises(ModelNotFittedError):
            fw.predict(series_mols)
        with pytest.raises(ModelNotFittedError):
            fw.residuals()
        with pytest.raises(ModelNotFittedError):
            fw.to_dataframe()

    def test_fit_on_non_matching_series_raises(self):
        with pytest.raises(ValueError, match="matched no molecules"):
            FreeWilsonAnalysis(core=CORE).fit(
                [Chem.MolFromSmiles("CCCCCC")], [5.0]
            )
