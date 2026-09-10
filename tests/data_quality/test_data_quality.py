from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.data_quality import (
    ActivityOutlierDetector,
    CurationReport,
    DataCurationPipeline,
    DuplicateDetector,
    StructureValidator,
    check_activity_units,
    merge_replicates,
)


def mol(smiles):
    return Chem.MolFromSmiles(smiles)


class TestDuplicateDetector:
    def test_finds_duplicates_across_smiles_spellings(self):
        mols = [mol("CCO"), mol("OCC"), mol("c1ccccc1")]
        groups = DuplicateDetector().find_duplicates(mols)
        assert len(groups) == 1
        assert groups[0].indices == [0, 1]

    def test_flags_disagreeing_replicates(self):
        mols = [mol("CCO"), mol("CCO")]
        group = DuplicateDetector(activity_tolerance=1.0).find_duplicates(
            mols, [4.0, 9.5]
        )[0]
        assert group.spread == pytest.approx(5.5)
        assert not group.consistent

    def test_accepts_agreeing_replicates(self):
        mols = [mol("CCO"), mol("CCO")]
        group = DuplicateDetector(activity_tolerance=1.0).find_duplicates(
            mols, [5.0, 5.2]
        )[0]
        assert group.consistent

    def test_connectivity_level_ignores_stereochemistry(self):
        mols = [mol("C[C@H](N)C(=O)O"), mol("C[C@@H](N)C(=O)O")]
        assert DuplicateDetector(level="inchikey").find_duplicates(mols) == []
        assert len(DuplicateDetector(level="connectivity").find_duplicates(mols)) == 1

    def test_scaffold_level_collapses_a_series(self):
        mols = [mol("c1ccccc1C"), mol("c1ccccc1CC"), mol("CCO")]
        groups = DuplicateDetector(level="scaffold").find_duplicates(mols)
        assert any(g.indices == [0, 1] for g in groups)

    def test_smiles_level(self):
        mols = [mol("CCO"), mol("OCC")]
        assert len(DuplicateDetector(level="smiles").find_duplicates(mols)) == 1

    def test_none_molecules_are_skipped(self):
        groups = DuplicateDetector().find_duplicates([mol("CCO"), None, mol("CCO")])
        assert groups[0].indices == [0, 2]

    def test_singletons_are_not_reported(self):
        assert DuplicateDetector().find_duplicates([mol("CCO"), mol("CCN")]) == []

    def test_report_counts(self):
        mols = [mol("CCO"), mol("OCC"), mol("CCN"), mol("CCN")]
        report = DuplicateDetector().report(mols, [5.0, 5.1, 4.0, 9.5])
        assert report["n_records"] == 4
        assert report["n_duplicate_groups"] == 2
        assert report["n_duplicate_records"] == 2
        assert report["n_unique"] == 2
        assert report["duplicate_fraction"] == pytest.approx(0.5)
        assert report["n_inconsistent"] == 1

    def test_group_length(self):
        group = DuplicateDetector().find_duplicates([mol("CCO"), mol("CCO")])[0]
        assert len(group) == 2

    def test_to_dataframe(self):
        detector = DuplicateDetector()
        groups = detector.find_duplicates([mol("CCO"), mol("CCO")], [5.0, 5.1])
        df = detector.to_dataframe(groups)
        assert list(df.columns) == [
            "key", "n_records", "indices", "activities", "spread", "consistent"
        ]

    def test_rejects_mismatched_activities(self):
        with pytest.raises(ValueError, match="activities has length"):
            DuplicateDetector().find_duplicates([mol("CCO")], [1.0, 2.0])

    def test_rejects_negative_tolerance(self):
        with pytest.raises(ValueError, match="non-negative"):
            DuplicateDetector(activity_tolerance=-1).find_duplicates([mol("CCO")])

    def test_rejects_unknown_level(self):
        with pytest.raises(ValueError, match="level must be"):
            DuplicateDetector(level="bogus").find_duplicates(  # type: ignore[arg-type]
                [mol("CCO"), mol("CCO")]
            )


class TestMergeReplicates:
    def test_merges_agreeing_replicates(self):
        mols = [mol("CCO"), mol("OCC"), mol("c1ccccc1")]
        merged, y, report = merge_replicates(mols, [5.0, 5.4, 7.0])
        assert len(merged) == 2
        assert report["n_merged"] == 1
        assert y[0] == pytest.approx(5.2)

    def test_discards_disagreeing_replicates(self):
        mols = [mol("CCO"), mol("CCO"), mol("c1ccccc1")]
        merged, y, report = merge_replicates(mols, [4.0, 9.5, 7.0], max_spread=1.0)
        assert report["n_discarded"] == 1
        assert len(merged) == 1
        assert y.tolist() == [7.0]

    def test_max_spread_none_keeps_everything(self):
        mols = [mol("CCO"), mol("CCO")]
        merged, _, report = merge_replicates(mols, [4.0, 9.5], max_spread=None)
        assert len(merged) == 1 and report["n_discarded"] == 0

    @pytest.mark.parametrize(
        "method, expected",
        [("mean", 6.0), ("median", 6.0), ("min", 4.0), ("max", 8.0)],
    )
    def test_merge_methods(self, method, expected):
        mols = [mol("CCO"), mol("CCO"), mol("CCO")]
        _, y, _ = merge_replicates(
            mols, [4.0, 6.0, 8.0], method=method, max_spread=None
        )
        assert y[0] == pytest.approx(expected)

    def test_geometric_mean_on_linear_concentrations(self):
        mols = [mol("CCO"), mol("CCO")]
        _, y, _ = merge_replicates(
            mols, [10.0, 1000.0], method="geometric_mean",
            log_scale=False, max_spread=None,
        )
        assert y[0] == pytest.approx(100.0)

    def test_geometric_mean_refused_on_log_scale(self):
        with pytest.raises(ValueError, match="already is the geometric mean"):
            merge_replicates([mol("CCO")], [5.0], method="geometric_mean")

    def test_geometric_mean_refused_on_non_positive(self):
        with pytest.raises(ValueError, match="strictly positive"):
            merge_replicates(
                [mol("CCO")], [0.0], method="geometric_mean", log_scale=False
            )

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be one of"):
            merge_replicates([mol("CCO")], [1.0], method="mode")

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="activities has length"):
            merge_replicates([mol("CCO")], [1.0, 2.0])

    def test_preserves_original_order(self):
        mols = [mol("c1ccccc1"), mol("CCO"), mol("CCO")]
        merged, y, _ = merge_replicates(mols, [7.0, 5.0, 5.2])
        assert Chem.MolToSmiles(merged[0]) == "c1ccccc1"

    def test_none_molecules_survive_unmerged(self):
        merged, y, _ = merge_replicates([mol("CCO"), None], [5.0, 9.0])
        assert len(merged) == 2


class TestStructureValidator:
    def test_flags_the_expected_problems(self):
        mols = [mol("CCO"), mol("[Na+].[Cl-]"), mol("O")]
        codes = {(i.index, i.code) for i in StructureValidator().validate(mols)}
        assert (1, "mixture") in codes
        assert (1, "inorganic") in codes
        assert (2, "no_carbon") in codes
        assert (2, "too_small") in codes

    def test_clean_molecule_has_no_issues(self):
        assert StructureValidator().validate([mol("CC(=O)Oc1ccccc1C(=O)O")]) == []

    def test_none_is_unparseable(self):
        issues = StructureValidator().validate([None])
        assert issues[0].code == "unparseable" and issues[0].fatal

    def test_valid_mask(self):
        mols = [mol("CCO"), mol("[Na+].[Cl-]"), mol("c1ccccc1")]
        assert StructureValidator().valid_mask(mols).tolist() == [True, False, True]

    def test_allow_mixtures(self):
        codes = {
            i.code
            for i in StructureValidator(allow_mixtures=True).validate(
                [mol("CCO.CCN")]
            )
        }
        assert "mixture" not in codes

    def test_allow_inorganic(self):
        codes = {
            i.code
            for i in StructureValidator(allow_inorganic=True).validate(
                [mol("[Fe]CCCCC")]
            )
        }
        assert "inorganic" not in codes

    def test_isotopes_flagged_by_default(self):
        codes = {i.code for i in StructureValidator().validate([mol("[13CH3]C")])}
        assert "isotope" in codes

    def test_allow_isotopes(self):
        codes = {
            i.code
            for i in StructureValidator(allow_isotopes=True).validate([mol("[13CH3]C")])
        }
        assert "isotope" not in codes

    def test_size_limits(self):
        long_chain = mol("C" * 60)
        codes = {i.code for i in StructureValidator(max_heavy_atoms=10).validate([long_chain])}
        assert "too_large" in codes

    def test_charge_is_a_warning_not_fatal(self):
        issues = [
            i for i in StructureValidator().validate([mol("CC(=O)[O-]")])
            if i.code == "charged"
        ]
        assert issues and not issues[0].fatal

    def test_require_carbon_can_be_disabled(self):
        codes = {
            i.code
            for i in StructureValidator(require_carbon=False, min_heavy_atoms=1).validate(
                [mol("O")]
            )
        }
        assert "no_carbon" not in codes

    def test_to_dataframe(self):
        validator = StructureValidator()
        df = validator.to_dataframe(validator.validate([mol("O")]))
        assert list(df.columns) == ["index", "code", "message", "fatal"]


class TestActivityOutlierDetector:
    @pytest.fixture
    def spiked(self):
        return np.concatenate([np.full(20, 5.0) + np.arange(20) * 0.01, [50.0]])

    @pytest.mark.parametrize(
        "method", ["zscore", "modified_zscore", "iqr"]
    )
    def test_detects_an_obvious_outlier(self, spiked, method):
        threshold = {"zscore": 3.0, "modified_zscore": 3.5, "iqr": 1.5}[method]
        flags = ActivityOutlierDetector(method=method, threshold=threshold).detect(
            spiked
        )
        assert flags[-1]
        assert flags[:-1].sum() <= 2

    def test_clean_data_has_no_outliers(self):
        y = np.linspace(5.0, 6.0, 30)
        assert not ActivityOutlierDetector().detect(y).any()

    def test_modified_zscore_resists_masking(self):
        # Two extreme values inflate a plain z-score's standard deviation
        # enough to hide themselves; the MAD-based score does not.
        y = np.concatenate([np.full(30, 5.0) + np.arange(30) * 0.01, [80.0, 90.0]])
        plain = ActivityOutlierDetector(method="zscore", threshold=3.0).detect(y)
        robust = ActivityOutlierDetector(
            method="modified_zscore", threshold=3.5
        ).detect(y)
        assert robust[-2:].all()
        assert robust[-2:].sum() >= plain[-2:].sum()

    def test_neighbor_method_finds_a_structural_outlier(self):
        mols = [mol(s) for s in ("CCO", "CCCO", "CCCCO", "CCCCCO", "c1ccccc1")]
        y = np.array([5.0, 5.1, 5.2, 5.3, 5.15])
        scores = ActivityOutlierDetector(
            method="neighbor", n_neighbors=2
        ).scores(y, mols)
        assert scores.shape == (5,)
        assert np.all(np.isfinite(scores))

    def test_neighbor_method_requires_molecules(self):
        with pytest.raises(ValueError, match="needs molecules"):
            ActivityOutlierDetector(method="neighbor").scores([1.0, 2.0])

    def test_neighbor_method_rejects_length_mismatch(self):
        with pytest.raises(ValueError, match="activities has length"):
            ActivityOutlierDetector(method="neighbor").scores(
                [1.0, 2.0], [mol("CCO")]
            )

    def test_rejects_unknown_method(self):
        with pytest.raises(ValueError, match="method must be"):
            ActivityOutlierDetector(method="bogus").scores([1.0, 2.0])

    def test_constant_data_scores_zero(self):
        for method in ("zscore", "modified_zscore", "iqr"):
            scores = ActivityOutlierDetector(method=method).scores(np.full(10, 5.0))
            assert np.allclose(scores, 0.0)

    def test_empty_input(self):
        assert ActivityOutlierDetector().scores([]).shape == (0,)


class TestCheckActivityUnits:
    def test_flags_a_raw_concentration_column(self):
        report = check_activity_units([1.0, 10.0, 1000.0, 1e6], unit="nM")
        assert not report["looks_logarithmic"]
        assert report["log_range"] == pytest.approx(6.0)
        assert any("orders of magnitude" in w for w in report["warnings"])

    def test_accepts_a_log_scale_column(self):
        report = check_activity_units([5.0, 6.0, 7.0, 8.0])
        assert report["looks_logarithmic"]

    def test_flags_non_positive_values(self):
        report = check_activity_units([0.0, 1.0, 10.0, 1e5])
        assert report["n_non_positive"] == 1
        assert any("non-positive" in w for w in report["warnings"])

    def test_flags_missing_values(self):
        report = check_activity_units([1.0, np.nan, 3.0])
        assert report["n_missing"] == 1

    def test_flags_a_constant_column(self):
        report = check_activity_units([5.0, 5.0, 5.0])
        assert any("identical" in w for w in report["warnings"])

    def test_flags_a_stale_unit_label(self):
        report = check_activity_units([5.0, 6.0, 7.0], unit="nM")
        assert any("stale" in w for w in report["warnings"])

    def test_all_missing(self):
        report = check_activity_units([np.nan, np.nan])
        assert report["warnings"] and "No usable values." in report["warnings"]
        assert np.isnan(report["min"])

    def test_reports_summary_statistics(self):
        report = check_activity_units([5.0, 6.0, 7.0])
        assert report["min"] == 5.0 and report["max"] == 7.0
        assert report["median"] == 6.0
        assert report["n"] == 3


class TestDataCurationPipeline:
    @pytest.fixture
    def messy(self):
        smiles = [
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",  # salt -> desalted, then valid
            "CCO", "OCC",                       # agreeing duplicates
            "[Na+].[Cl-]",                      # inorganic mixture
            "c1ccccc1",
            "O",                                # too small, no carbon
            "CCN", "CCN",                       # disagreeing duplicates
        ]
        return [mol(s) for s in smiles], [5.0, 6.0, 6.2, 1.0, 7.0, 2.0, 4.0, 9.5]

    def test_removes_the_expected_records(self, messy):
        mols, y = messy
        curated, y_out, report = DataCurationPipeline().run(mols, y)
        assert report.n_input == 8
        assert report.n_output == len(curated) == len(y_out)
        # NaCl and water are invalid; the disagreeing CCN pair is discarded
        assert report.removed[3].startswith("invalid structure")
        assert report.removed[5].startswith("invalid structure")

    def test_desalting_happens_before_validation(self, messy):
        mols, y = messy
        curated, _, report = DataCurationPipeline().run(mols, y)
        # aspirin sodium salt is a mixture until standardization strips it;
        # running validation first would have discarded it
        assert 0 not in report.removed
        smiles = {Chem.MolToSmiles(m) for m in curated}
        assert any("CC(=O)Oc1ccccc1C(=O)O" == s for s in smiles)

    def test_merges_agreeing_duplicates(self, messy):
        mols, y = messy
        curated, y_out, _ = DataCurationPipeline().run(mols, y)
        smiles = [Chem.MolToSmiles(m) for m in curated]
        assert smiles.count("CCO") == 1
        assert y_out[smiles.index("CCO")] == pytest.approx(6.1)

    def test_discards_disagreeing_duplicates(self, messy):
        mols, y = messy
        curated, _, report = DataCurationPipeline().run(mols, y)
        assert "CCN" not in [Chem.MolToSmiles(m) for m in curated]
        assert any("disagreeing" in w for w in report.warnings)

    def test_report_stages_are_recorded_in_order(self, messy):
        mols, y = messy
        _, _, report = DataCurationPipeline().run(mols, y)
        assert [s["stage"] for s in report.stages] == [
            "standardize", "validate", "deduplicate"
        ]

    def test_report_arithmetic_is_consistent(self, messy):
        mols, y = messy
        _, _, report = DataCurationPipeline().run(mols, y)
        assert report.n_removed == report.n_input - report.n_output
        assert 0.0 <= report.retention <= 1.0
        for stage in report.stages:
            assert stage["n_after"] <= stage["n_before"]

    def test_report_dataframe_and_summary(self, messy):
        mols, y = messy
        _, _, report = DataCurationPipeline().run(mols, y)
        df = report.to_dataframe()
        assert list(df.columns) == ["stage", "n_before", "n_after", "n_removed"]
        assert "Curation:" in report.summary()
        assert "CurationReport" in repr(report)

    def test_activities_stay_aligned_with_molecules(self, messy):
        mols, y = messy
        curated, y_out, _ = DataCurationPipeline().run(mols, y)
        assert len(curated) == len(y_out)
        smiles = [Chem.MolToSmiles(m) for m in curated]
        assert y_out[smiles.index("c1ccccc1")] == pytest.approx(7.0)

    def test_runs_without_activities(self, messy):
        mols, _ = messy
        curated, y_out, report = DataCurationPipeline().run(mols)
        assert y_out is None
        assert report.n_output == len(curated)

    def test_stages_can_be_disabled(self, messy):
        mols, y = messy
        curated, _, report = DataCurationPipeline(
            standardize=False, validate_structures=False, remove_duplicates=False
        ).run(mols, y)
        assert report.stages == []
        assert len(curated) == len(mols)

    def test_outlier_removal_is_opt_in(self, messy):
        mols, y = messy
        _, _, off = DataCurationPipeline().run(mols, y)
        _, _, on = DataCurationPipeline(remove_outliers=True).run(mols, y)
        assert "remove_outliers" not in [s["stage"] for s in off.stages]
        assert "remove_outliers" in [s["stage"] for s in on.stages]

    def test_warns_when_most_of_the_dataset_is_removed(self, messy):
        mols, y = messy
        _, _, report = DataCurationPipeline().run(mols, y)
        if report.retention < 0.5:
            assert any("removed" in w for w in report.warnings)

    def test_activity_check_is_included(self, messy):
        mols, y = messy
        _, _, report = DataCurationPipeline().run(mols, y, unit="pIC50")
        assert report.activity_check["n"] == 8

    def test_rejects_mismatched_activities(self, messy):
        mols, _ = messy
        with pytest.raises(ValueError, match="activities has length"):
            DataCurationPipeline().run(mols, [1.0])

    def test_empty_report_defaults(self):
        report = CurationReport()
        assert report.retention == 0.0 and report.n_removed == 0
