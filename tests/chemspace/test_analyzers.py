"""Tests for the chemical-space analyzers."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.chemspace import (
    ChemicalSpaceAnalyzer,
    ChemicalSpaceCoverage,
    ClusterAnalyzer,
    DiversityAnalyzer,
    NearestNeighborAnalyzer,
    ScaffoldAnalyzer,
)

SMILES = [
    "c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1Cl",
    "c1ccncc1C", "c1ccncc1CC", "c1ccncc1Cl",
    "CCO", "CCN", "CCC",
]


@pytest.fixture(scope="module")
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


class TestChemicalSpaceAnalyzer:
    def test_pca_projects_to_two_dimensions(self, mols):
        coords = ChemicalSpaceAnalyzer(random_state=0).fit_transform(mols)
        assert coords.shape == (len(mols), 2)

    def test_n_components_is_respected(self, mols):
        coords = ChemicalSpaceAnalyzer(n_components=3, random_state=0).fit_transform(mols)
        assert coords.shape == (len(mols), 3)

    def test_tsne(self, mols):
        analyzer = ChemicalSpaceAnalyzer(method="tsne", random_state=0, perplexity=3)
        assert analyzer.fit_transform(mols).shape == (len(mols), 2)

    def test_mds(self, mols):
        analyzer = ChemicalSpaceAnalyzer(method="mds", random_state=0)
        assert analyzer.fit_transform(mols).shape == (len(mols), 2)

    def test_rejects_unknown_method(self, mols):
        with pytest.raises(ValueError, match="method must be"):
            ChemicalSpaceAnalyzer(method="nonesuch").fit_transform(mols)

    def test_fit_returns_self(self, mols):
        analyzer = ChemicalSpaceAnalyzer(random_state=0)
        assert analyzer.fit(mols) is analyzer

    def test_plot_returns_a_figure(self, mols):
        analyzer = ChemicalSpaceAnalyzer(random_state=0).fit(mols)
        assert analyzer.plot().__class__.__name__ == "Figure"

    def test_plot_accepts_colouring(self, mols):
        analyzer = ChemicalSpaceAnalyzer(random_state=0).fit(mols)
        figure = analyzer.plot(color=np.arange(len(mols)), title="Space")
        assert figure.layout.title.text == "Space"


class TestDiversityAnalyzer:
    def test_reports_the_standard_measures(self, mols):
        report = DiversityAnalyzer().analyze(mols)
        assert {
            "n_molecules", "mean_pairwise_distance", "internal_diversity",
            "n_scaffolds", "scaffold_diversity", "bit_entropy",
        } <= set(report)

    def test_a_duplicated_library_is_less_diverse_than_a_varied_one(self, mols):
        analyzer = DiversityAnalyzer()
        identical = [Chem.MolFromSmiles("CCO")] * 5
        assert (
            analyzer.analyze(identical)["mean_pairwise_distance"]
            < analyzer.analyze(mols)["mean_pairwise_distance"]
        )

    def test_scaffold_count_agrees_with_scaffold_analyzer(self, mols):
        # Both exclude acyclic molecules, which have no framework at all;
        # counting the empty scaffold would make a set of straight chains
        # look scaffold-diverse.
        assert (
            DiversityAnalyzer().analyze(mols)["n_scaffolds"]
            == ScaffoldAnalyzer().fit(mols).n_scaffolds
        )

    def test_an_all_acyclic_library_has_no_scaffolds(self):
        acyclic = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "CCC")]
        report = DiversityAnalyzer().analyze(acyclic)
        assert report["n_scaffolds"] == 0.0
        assert report["acyclic_fraction"] == 1.0
        assert report["scaffold_entropy"] == 0.0

    def test_compare_returns_a_row_per_library(self, mols):
        frame = DiversityAnalyzer().compare(
            {"aromatics": mols[:7], "aliphatics": mols[7:]}
        )
        assert len(frame) == 2


class TestClusterAnalyzer:
    def test_butina_assigns_every_molecule(self, mols):
        labels = ClusterAnalyzer(cutoff=0.4).fit_predict(mols)
        assert len(labels) == len(mols)

    def test_summary_counts_clusters(self, mols):
        analyzer = ClusterAnalyzer(cutoff=0.4).fit(mols)
        summary = analyzer.summary()
        assert summary["n_clusters"] >= 1

    def test_cluster_members_partition_the_input(self, mols):
        analyzer = ClusterAnalyzer(cutoff=0.4).fit(mols)
        members = analyzer.cluster_members()
        assert sorted(i for group in members.values() for i in group) == list(
            range(len(mols))
        )

    def test_kmeans(self, mols):
        labels = ClusterAnalyzer(
            method="kmeans", n_clusters=3, random_state=0
        ).fit_predict(mols)
        assert len(set(labels)) == 3

    def test_rejects_unknown_method(self, mols):
        with pytest.raises(ValueError):
            ClusterAnalyzer(method="nonesuch").fit_predict(mols)


class TestNearestNeighborAnalyzer:
    def test_a_molecule_is_its_own_nearest_neighbour(self, mols):
        analyzer = NearestNeighborAnalyzer().fit(mols)
        assert float(analyzer.nearest_similarity([mols[0]])[0]) == pytest.approx(1.0)

    def test_exclude_self_finds_the_next_closest(self, mols):
        analyzer = NearestNeighborAnalyzer().fit(mols)
        assert float(analyzer.nearest_similarity(mols, exclude_self=True)[0]) < 1.0

    def test_novelty_is_the_complement_of_similarity(self, mols):
        analyzer = NearestNeighborAnalyzer().fit(mols)
        query = [Chem.MolFromSmiles("CCO")]
        assert float(analyzer.novelty(query)[0]) == pytest.approx(
            1.0 - float(analyzer.nearest_similarity(query)[0])
        )

    def test_a_varied_library_is_not_redundant_against_itself(self, mols):
        # `redundancy` compares against the *other* members, so a library
        # of distinct analogues scores 0 rather than 1.
        analyzer = NearestNeighborAnalyzer().fit(mols)
        assert analyzer.redundancy(mols, threshold=0.9) == pytest.approx(0.0)

    def test_duplicated_molecules_are_redundant(self, mols):
        library = mols + [mols[0]]
        analyzer = NearestNeighborAnalyzer().fit(library)
        assert analyzer.redundancy(library, threshold=0.9) > 0.0


class TestScaffoldAnalyzer:
    def test_counts_distinct_scaffolds(self, mols):
        analyzer = ScaffoldAnalyzer().fit(mols)
        # benzene and pyridine; the three acyclic molecules have none.
        assert analyzer.n_scaffolds == 2

    def test_acyclic_molecules_are_reported_separately(self, mols):
        summary = ScaffoldAnalyzer().fit(mols).summary()
        assert summary["acyclic_fraction"] == pytest.approx(3 / len(mols))

    def test_most_common_is_ordered(self, mols):
        common = ScaffoldAnalyzer().fit(mols).most_common(2)
        assert common[0][1] >= common[1][1]

    def test_groups_index_back_into_the_input(self, mols):
        groups = ScaffoldAnalyzer().fit(mols).groups()
        assert all(0 <= i < len(mols) for group in groups.values() for i in group)

    def test_generic_scaffolds_merge_heteroatom_variants(self, mols):
        specific = ScaffoldAnalyzer().fit(mols).n_scaffolds
        generic = ScaffoldAnalyzer(generic=True).fit(mols).n_scaffolds
        assert generic < specific

    def test_to_dataframe(self, mols):
        frame = ScaffoldAnalyzer().fit(mols).to_dataframe()
        assert len(frame) >= 1


class TestChemicalSpaceCoverage:
    def test_a_library_covers_itself(self, mols):
        report = ChemicalSpaceCoverage(threshold=0.7).compare(mols, mols)
        assert report["coverage_of_reference"] == pytest.approx(1.0)
        assert report["novel_fraction"] == pytest.approx(0.0)

    def test_disjoint_libraries_have_low_coverage(self, mols):
        # Aromatics against aliphatics: nothing in the query is close to
        # the reference, so all of it is novel.
        report = ChemicalSpaceCoverage(threshold=0.9).compare(mols[:4], mols[7:])
        assert report["coverage_of_reference"] < 1.0
        assert report["novel_fraction"] > 0.0

    def test_reports_the_documented_keys(self, mols):
        report = ChemicalSpaceCoverage().compare(mols[:4], mols[4:])
        assert {
            "coverage_of_reference", "mean_nearest_similarity",
            "n_novel_in_query", "novel_fraction",
        } <= set(report)
