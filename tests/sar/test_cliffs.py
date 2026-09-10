from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.sar import (
    ActivityCliffDetector,
    ActivityLandscapePlotter,
    SALIAnalyzer,
    SARIAnalyzer,
    activity_cliff_report,
)
from tests.sar.conftest import CLIFF_INDEX


class TestActivityCliffDetector:
    def test_finds_the_designed_cliff(self, series_mols, series_activities):
        detector = ActivityCliffDetector(
            similarity_threshold=0.5, activity_threshold=2.0
        )
        cliffs = detector.detect(series_mols, series_activities)
        assert len(cliffs) == 5
        # every cliff must involve the 4-amino compound
        assert all(
            CLIFF_INDEX in (c.index_a, c.index_b) for c in cliffs
        )

    def test_sorted_by_descending_sali(self, series_mols, series_activities):
        cliffs = ActivityCliffDetector(similarity_threshold=0.5).detect(
            series_mols, series_activities
        )
        salis = [c.sali for c in cliffs]
        assert salis == sorted(salis, reverse=True)

    def test_no_cliffs_when_activities_are_flat(self, series_mols):
        cliffs = ActivityCliffDetector(similarity_threshold=0.5).detect(
            series_mols, [5.0] * len(series_mols)
        )
        assert cliffs == []

    def test_high_similarity_threshold_finds_nothing(
        self, series_mols, series_activities
    ):
        detector = ActivityCliffDetector(similarity_threshold=0.99)
        assert detector.detect(series_mols, series_activities) == []

    def test_delta_and_similarity_are_consistent(self, series_mols, series_activities):
        cliffs = ActivityCliffDetector(similarity_threshold=0.5).detect(
            series_mols, series_activities
        )
        for c in cliffs:
            assert c.delta == pytest.approx(abs(c.activity_a - c.activity_b))
            assert c.delta >= 2.0
            assert 0.5 <= c.similarity <= 1.0
            assert c.sali == pytest.approx(c.delta / (1 - c.similarity))

    def test_mmp_method(self, series_mols, series_activities):
        detector = ActivityCliffDetector(
            method="mmp", similarity_threshold=1.0, activity_threshold=2.0
        )
        cliffs = detector.detect(series_mols, series_activities)
        assert len(cliffs) == 5
        assert all(CLIFF_INDEX in (c.index_a, c.index_b) for c in cliffs)

    def test_scaffold_method(self, series_mols, series_activities):
        detector = ActivityCliffDetector(
            method="scaffold", similarity_threshold=1.0, activity_threshold=2.0
        )
        # all share the benzene scaffold, so every big-delta pair is a cliff
        cliffs = detector.detect(series_mols, series_activities)
        assert len(cliffs) == 5

    def test_unknown_method_raises(self, series_mols, series_activities):
        with pytest.raises(ValueError, match="method must be"):
            ActivityCliffDetector(method="bogus").detect(  # type: ignore[arg-type]
                series_mols, series_activities
            )

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but activities has"):
            ActivityCliffDetector().detect(series_mols, [1.0])

    def test_fewer_than_two_molecules(self, series_mols):
        assert ActivityCliffDetector().detect(series_mols[:1], [5.0]) == []
        assert ActivityCliffDetector().detect([], []) == []

    def test_to_dataframe(self, series_mols, series_activities):
        detector = ActivityCliffDetector(similarity_threshold=0.5)
        cliffs = detector.detect(series_mols, series_activities)
        df = detector.to_dataframe(cliffs)
        assert len(df) == len(cliffs)
        assert list(df.columns) == [
            "index_a", "index_b", "smiles_a", "smiles_b", "similarity",
            "activity_a", "activity_b", "delta", "sali",
        ]

    def test_repr_is_informative(self, series_mols, series_activities):
        cliff = ActivityCliffDetector(similarity_threshold=0.5).detect(
            series_mols, series_activities
        )[0]
        assert "ActivityCliff" in repr(cliff)
        assert "SALI" in repr(cliff)


class TestSALIAnalyzer:
    def test_sali_matches_hand_calculation(self, series_mols, series_activities):
        analyzer = SALIAnalyzer()
        detector = ActivityCliffDetector()
        sim = detector.similarity_matrix(series_mols)
        sali = analyzer.sali_matrix(series_mols, series_activities)
        expected = abs(series_activities[0] - series_activities[4]) / (1 - sim[0, 4])
        assert sali[0, 4] == pytest.approx(expected)

    def test_matrix_is_symmetric_with_zero_diagonal(
        self, series_mols, series_activities
    ):
        sali = SALIAnalyzer().sali_matrix(series_mols, series_activities)
        assert np.allclose(sali, sali.T)
        assert np.allclose(np.diag(sali), 0.0)

    def test_identical_structures_give_infinite_sali(self):
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
        sali = SALIAnalyzer().sali_matrix(mols, [5.0, 9.0])
        assert np.isinf(sali[0, 1])

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but activities has"):
            SALIAnalyzer().sali_matrix(series_mols, [1.0])

    def test_network_nodes_and_edges(self, series_mols, series_activities):
        graph = SALIAnalyzer().sali_network(
            series_mols, series_activities, percentile=80
        )
        assert graph.number_of_nodes() == len(series_mols)
        assert graph.number_of_edges() > 0
        assert all("sali" in d for _, _, d in graph.edges(data=True))
        assert all("activity" in d for _, d in graph.nodes(data=True))

    def test_network_percentile_controls_density(
        self, series_mols, series_activities
    ):
        sparse = SALIAnalyzer().sali_network(
            series_mols, series_activities, percentile=95
        )
        dense = SALIAnalyzer().sali_network(
            series_mols, series_activities, percentile=10
        )
        assert dense.number_of_edges() >= sparse.number_of_edges()

    def test_network_rejects_bad_percentile(self, series_mols, series_activities):
        with pytest.raises(ValueError, match="percentile must be"):
            SALIAnalyzer().sali_network(series_mols, series_activities, percentile=150)

    def test_sali_auc_rewards_a_good_model(self, series_mols, series_activities):
        analyzer = SALIAnalyzer()
        good = [5.1, 5.0, 5.2, 4.8, 8.0, 5.4]
        bad = list(reversed(series_activities))
        assert analyzer.sali_auc(series_mols, series_activities, good) > analyzer.sali_auc(
            series_mols, series_activities, bad
        )

    def test_perfect_prediction_gives_auc_one(self, series_mols, series_activities):
        auc = SALIAnalyzer().sali_auc(
            series_mols, series_activities, series_activities
        )
        assert auc == pytest.approx(1.0)

    def test_curve_shapes_agree(self, series_mols, series_activities):
        x, y = SALIAnalyzer().sali_curve(
            series_mols, series_activities, series_activities
        )
        assert x.shape == y.shape
        assert np.all((x > 0) & (x <= 1))
        assert np.all((y >= 0) & (y <= 1))

    def test_curve_length_mismatch_raises(self, series_mols, series_activities):
        with pytest.raises(ValueError, match="same length"):
            SALIAnalyzer().sali_curve(series_mols, series_activities, [1.0])

    def test_curve_with_all_infinite_sali_is_empty(self):
        mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CCO")]
        x, y = SALIAnalyzer().sali_curve(mols, [5.0, 9.0], [5.0, 9.0])
        assert x.size == 0 and y.size == 0
        assert SALIAnalyzer().sali_auc(mols, [5.0, 9.0], [5.0, 9.0]) == 0.0


class TestSARIAnalyzer:
    def test_returns_three_scores_in_range(self, series_mols, series_activities):
        scores = SARIAnalyzer().analyze(series_mols, series_activities)
        assert set(scores) == {"continuity", "discontinuity", "sari"}
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_cliff_series_is_more_discontinuous_than_flat_series(self, series_mols):
        cliffy = SARIAnalyzer(similarity_threshold=0.5).analyze(
            series_mols, [5.0, 5.2, 5.1, 4.9, 8.5, 5.3]
        )
        flat = SARIAnalyzer(similarity_threshold=0.5).analyze(
            series_mols, [5.0, 5.1, 5.0, 5.1, 5.0, 5.1]
        )
        assert cliffy["discontinuity"] > flat["discontinuity"]

    def test_degenerate_inputs(self, series_mols):
        assert SARIAnalyzer().analyze(series_mols[:1], [5.0])["sari"] == 0.0

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but activities has"):
            SARIAnalyzer().analyze(series_mols, [1.0])


class TestActivityLandscapePlotter:
    def test_sas_data_columns_and_length(self, series_mols, series_activities):
        df = ActivityLandscapePlotter().sas_data(series_mols, series_activities)
        n = len(series_mols)
        assert len(df) == n * (n - 1) // 2
        assert set(df.columns) == {
            "index_a", "index_b", "structure_similarity",
            "activity_similarity", "delta_activity", "quadrant",
        }

    def test_quadrants_are_from_the_known_set(self, series_mols, series_activities):
        df = ActivityLandscapePlotter().sas_data(series_mols, series_activities)
        assert set(df["quadrant"]) <= {
            "smooth SAR", "activity cliff", "scaffold hop", "nondescript"
        }

    def test_cliff_pairs_land_in_the_cliff_quadrant(
        self, series_mols, series_activities
    ):
        plotter = ActivityLandscapePlotter(
            similarity_threshold=0.5, activity_threshold=0.5
        )
        df = plotter.sas_data(series_mols, series_activities)
        cliff_rows = df[
            (df["index_a"] == CLIFF_INDEX) | (df["index_b"] == CLIFF_INDEX)
        ]
        assert (cliff_rows["quadrant"] == "activity cliff").any()

    def test_plot_returns_plotly_figure(self, series_mols, series_activities):
        import plotly.graph_objects as go

        fig = ActivityLandscapePlotter().plot(series_mols, series_activities)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_length_mismatch_raises(self, series_mols):
        with pytest.raises(ValueError, match="but activities has"):
            ActivityLandscapePlotter().sas_data(series_mols, [1.0])


class TestActivityCliffReport:
    def test_report_structure_and_values(self, series_mols, series_activities):
        report = activity_cliff_report(
            series_mols, series_activities, similarity_threshold=0.5
        )
        assert report["n_compounds"] == 6
        assert report["n_pairs"] == 15
        assert report["n_cliffs"] == 5
        assert report["cliff_ratio"] == pytest.approx(5 / 15)
        # 5 cliffs all involve compound 4, touching all 6 compounds
        assert report["cliff_compound_fraction"] == pytest.approx(1.0)
        assert report["max_sali"] > 0

    def test_top_transformations_all_introduce_the_amine(
        self, series_mols, series_activities
    ):
        report = activity_cliff_report(
            series_mols, series_activities, similarity_threshold=0.5
        )
        assert report["top_transformations"]
        assert all("N" in t for t in report["top_transformations"])

    def test_top_n_is_respected(self, series_mols, series_activities):
        report = activity_cliff_report(
            series_mols, series_activities, similarity_threshold=0.5, top_n=2
        )
        assert len(report["top_cliffs"]) == 2
        assert len(report["top_scaffolds"]) <= 2

    def test_flat_series_reports_no_cliffs(self, series_mols):
        report = activity_cliff_report(series_mols, [5.0] * len(series_mols))
        assert report["n_cliffs"] == 0
        assert report["cliff_ratio"] == 0.0
        assert report["max_sali"] == 0.0
        assert report["top_scaffolds"] == {}
