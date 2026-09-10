from __future__ import annotations

import pytest
from rdkit import Chem

from qsarkit.sar import MatchedMolecularPairs, MMPAnalyzer


class TestMatchedMolecularPairs:
    def test_finds_single_atom_substituent_swap(self):
        mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1Cl", "c1ccccc1Br")]
        pairs = MatchedMolecularPairs().find_pairs(mols)
        assert len(pairs) >= 1
        transformations = {p.transformation for p in pairs}
        assert any("Cl" in t and "Br" in t for t in transformations)

    def test_full_series_is_a_complete_pair_network(
        self, series_mols, series_activities
    ):
        pairs = MatchedMolecularPairs().find_pairs(series_mols, series_activities)
        n = len(series_mols)
        # every compound pairs with every other via the para position
        assert len(pairs) == n * (n - 1) // 2

    def test_pairs_share_the_expected_core(self, series_mols):
        pairs = MatchedMolecularPairs().find_pairs(series_mols)
        assert all("C(=O)O" in p.core for p in pairs)

    def test_delta_activity_sign_and_value(self, series_mols, series_activities):
        pairs = MatchedMolecularPairs().find_pairs(series_mols, series_activities)
        for p in pairs:
            expected = series_activities[p.index_b] - series_activities[p.index_a]
            assert p.delta_activity == pytest.approx(expected)

    def test_activities_omitted_leaves_delta_none(self, series_mols):
        pairs = MatchedMolecularPairs().find_pairs(series_mols)
        assert all(p.delta_activity is None for p in pairs)

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but mols has"):
            MatchedMolecularPairs().find_pairs(series_mols, [1.0])

    def test_identical_molecules_yield_no_pair(self):
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
        assert MatchedMolecularPairs().find_pairs(mols) == []

    def test_unrelated_molecules_yield_no_pair(self):
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccncc1")]
        assert MatchedMolecularPairs().find_pairs(mols) == []

    def test_none_entries_are_skipped(self):
        mols = [Chem.MolFromSmiles("c1ccccc1Cl"), None, Chem.MolFromSmiles("c1ccccc1Br")]
        pairs = MatchedMolecularPairs().find_pairs(mols)
        assert all(p.index_a != 1 and p.index_b != 1 for p in pairs)

    def test_fragment_size_filter_excludes_large_fragments(self):
        mols = [
            Chem.MolFromSmiles("c1ccccc1CCCCCCCCCCCC"),
            Chem.MolFromSmiles("c1ccccc1C"),
        ]
        permissive = MatchedMolecularPairs(max_fragment_heavy_atoms=None).find_pairs(mols)
        strict = MatchedMolecularPairs(max_fragment_heavy_atoms=2).find_pairs(mols)
        assert len(permissive) >= len(strict)

    def test_max_cuts_two_finds_more_contexts(self, series_mols):
        one = MatchedMolecularPairs(max_cuts=1).find_pairs(series_mols)
        two = MatchedMolecularPairs(max_cuts=2).find_pairs(series_mols)
        assert len(two) >= len(one)

    def test_repr_is_informative(self, series_mols, series_activities):
        pair = MatchedMolecularPairs().find_pairs(series_mols, series_activities)[0]
        assert "MatchedPair" in repr(pair)
        assert ">>" in repr(pair)


class TestMMPAnalyzer:
    def test_transformation_summary_columns(self, series_mols, series_activities):
        analyzer = MMPAnalyzer()
        pairs = analyzer.find_pairs(series_mols, series_activities)
        summary = analyzer.transformation_summary(pairs)
        assert list(summary.columns) == [
            "transformation", "count", "mean_delta", "median_delta", "std_delta"
        ]
        assert len(summary) > 0

    def test_amine_transformations_have_the_largest_effect(
        self, series_mols, series_activities
    ):
        analyzer = MMPAnalyzer()
        pairs = analyzer.find_pairs(series_mols, series_activities)
        summary = analyzer.transformation_summary(pairs)
        top = summary.sort_values("mean_delta", ascending=False).iloc[0]
        assert "N" in top["transformation"]
        assert top["mean_delta"] > 3.0

    def test_summary_of_pairs_without_activities_is_empty(self, series_mols):
        analyzer = MMPAnalyzer()
        summary = analyzer.transformation_summary(analyzer.find_pairs(series_mols))
        assert summary.empty
        assert list(summary.columns) == [
            "transformation", "count", "mean_delta", "median_delta", "std_delta"
        ]

    def test_std_is_zero_for_single_observation(self, series_mols, series_activities):
        analyzer = MMPAnalyzer()
        summary = analyzer.transformation_summary(
            analyzer.find_pairs(series_mols, series_activities)
        )
        singles = summary[summary["count"] == 1]
        assert (singles["std_delta"] == 0.0).all()

    def test_to_dataframe(self, series_mols, series_activities):
        analyzer = MMPAnalyzer()
        pairs = analyzer.find_pairs(series_mols, series_activities)
        df = analyzer.to_dataframe(pairs)
        assert len(df) == len(pairs)
        assert list(df.columns) == [
            "smiles_a", "smiles_b", "core", "transformation",
            "delta_activity", "index_a", "index_b",
        ]

    def test_to_dataframe_of_empty_list(self):
        df = MMPAnalyzer().to_dataframe([])
        assert df.empty
        assert "transformation" in df.columns
