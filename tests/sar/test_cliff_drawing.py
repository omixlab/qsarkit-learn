"""Pairwise activity-cliff rendering.

A SAS map says how many cliffs a dataset has; only the pair picture says what
changed. The cases below pin the behaviour that matters chemically: the
difference is highlighted, and a pair a topological representation cannot
distinguish highlights nothing, which is the finding rather than a failure.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.sar import (
    ActivityCliffDetector,
    cliff_difference_atoms,
    draw_activity_cliff,
    draw_activity_cliffs,
)

TRANS_STILBENE = r"C(=C/c1ccccc1)\c1ccccc1"
CIS_STILBENE = r"C(=C\c1ccccc1)\c1ccccc1"


class TestDifferenceAtoms:
    def test_a_single_substitution_is_isolated(self):
        a, b = Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("c1ccccc1N")
        in_a, in_b = cliff_difference_atoms(a, b)
        assert len(in_a) == 1 and len(in_b) == 1
        assert a.GetAtomWithIdx(in_a[0]).GetSymbol() == "O"
        assert b.GetAtomWithIdx(in_b[0]).GetSymbol() == "N"

    def test_stereoisomers_differ_nowhere_topologically(self):
        """Nothing to highlight: the difference is geometry, not connectivity."""
        a = Chem.MolFromSmiles(TRANS_STILBENE)
        b = Chem.MolFromSmiles(CIS_STILBENE)
        assert cliff_difference_atoms(a, b) == ([], [])

    def test_a_homologue_shows_the_extra_atoms(self):
        a = Chem.MolFromSmiles("CCCCCCOC(=O)c1ccccc1C(=O)OCCCCCC")
        b = Chem.MolFromSmiles("CCCCCCCCCCOC(=O)c1ccccc1C(=O)OCCCCCCCCCC")
        in_a, in_b = cliff_difference_atoms(a, b)
        assert in_a == []
        assert len(in_b) == 8


class TestDrawing:
    @pytest.fixture
    def pair(self):
        return Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("c1ccccc1N")

    def test_svg_is_returned_by_default(self, pair):
        a, b = pair
        svg = draw_activity_cliff(mol_a=a, mol_b=b, activity_a=8.1, activity_b=5.2)
        assert isinstance(svg, str)
        assert "svg" in svg[:400].lower()

    def test_png_bytes_are_returned_on_request(self, pair):
        a, b = pair
        png = draw_activity_cliff(mol_a=a, mol_b=b, fmt="png")
        assert isinstance(png, (bytes, bytearray))
        assert len(png) > 0

    def test_the_activities_reach_the_legend(self, pair):
        a, b = pair
        svg = draw_activity_cliff(
            mol_a=a, mol_b=b, activity_a=8.1, activity_b=5.2, similarity=0.86
        )
        assert "8.1" in svg and "5.2" in svg

    def test_a_cliff_object_supplies_everything(self):
        mols = [Chem.MolFromSmiles(s) for s in ("c1ccccc1O", "c1ccccc1N")]
        cliffs = ActivityCliffDetector(
            similarity_threshold=0.3, activity_threshold=1.0
        ).detect(mols, [8.0, 5.0])
        assert cliffs, "the fixture should produce at least one cliff"
        svg = draw_activity_cliff(cliff=cliffs[0])
        assert isinstance(svg, str) and len(svg) > 0

    def test_neither_a_cliff_nor_molecules_is_refused(self):
        with pytest.raises(ValueError, match="Pass either"):
            draw_activity_cliff()

    def test_a_batch_is_capped(self):
        smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl"]
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        cliffs = ActivityCliffDetector(
            similarity_threshold=0.3, activity_threshold=1.0
        ).detect(mols, [8.0, 5.0, 8.2, 5.1])
        drawings = draw_activity_cliffs(cliffs, n=2)
        assert len(drawings) <= 2
        assert all(isinstance(d, str) for d in drawings)
