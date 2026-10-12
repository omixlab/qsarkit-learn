"""Passing a feature matrix where molecules are needed says so.

A pipeline's result carries both the matrix and the molecules, and ``.X`` is
the more obvious attribute than ``.mols``. Handed the matrix, RDKit raised a
Boost ``ArgumentError`` quoting a C++ signature::

    ArgumentError: Python argument types in
        FingerprintGenerator64.GetFingerprint(FingerprintGenerator64, numpy.ndarray)
    did not match C++ signature: ...

which says nothing about what to pass instead, and arrived from a different
frame depending on whether the analyzer fingerprinted or took scaffolds first.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.base import InvalidMoleculeError
from qsarkit.chemspace import (
    ChemicalSpaceAnalyzer,
    ChemicalSpaceCoverage,
    ClusterAnalyzer,
    DiversityAnalyzer,
    NearestNeighborAnalyzer,
    ScaffoldAnalyzer,
    bemis_murcko_smiles,
    compute_fingerprints,
)

SMILES = ["CCO", "c1ccccc1", "CCCCCC", "c1ccccc1O", "CCN", "c1ccc2ccccc2c1"]


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


@pytest.fixture
def X(mols):
    """A feature matrix of the right length, as a pipeline would hand over."""
    return np.asarray(compute_fingerprints(mols), dtype=np.float64)


def _entry_points():
    return [
        ("DiversityAnalyzer", lambda arg: DiversityAnalyzer().analyze(arg)),
        ("ScaffoldAnalyzer", lambda arg: ScaffoldAnalyzer().fit(arg)),
        ("ClusterAnalyzer", lambda arg: ClusterAnalyzer().fit(arg)),
        ("NearestNeighborAnalyzer", lambda arg: NearestNeighborAnalyzer().fit(arg)),
        ("ChemicalSpaceCoverage", lambda arg: ChemicalSpaceCoverage().compare(arg, arg)),
        ("compute_fingerprints", lambda arg: compute_fingerprints(arg)),
    ]


@pytest.mark.parametrize("name, call", _entry_points(), ids=lambda v: v if isinstance(v, str) else "")
class TestAFeatureMatrixIsReported:
    def test_it_names_the_shape_and_the_attribute(self, X, name, call):
        with pytest.raises(InvalidMoleculeError) as excinfo:
            call(X)
        message = str(excinfo.value)
        assert f"{X.shape[0]}x{X.shape[1]} array" in message
        assert ".mols" in message and ".X" in message

    def test_unparsed_smiles_says_to_parse(self, name, call):
        with pytest.raises(InvalidMoleculeError, match="MolFromSmiles"):
            call(SMILES)

    def test_a_none_entry_says_to_drop_it(self, mols, name, call):
        with pytest.raises(InvalidMoleculeError, match="failed to parse"):
            call(mols[:-1] + [None])

    def test_molecules_are_accepted(self, mols, name, call):
        call(mols)


class TestTheDualInputAnalyzerKeepsWorking:
    """`ChemicalSpaceAnalyzer` documents accepting either, so it must."""

    def test_from_molecules(self, mols):
        assert ChemicalSpaceAnalyzer().fit_transform(mols).shape == (len(mols), 2)

    def test_from_a_matrix(self, mols, X):
        assert ChemicalSpaceAnalyzer().fit_transform(X).shape == (len(mols), 2)


class TestTheScaffoldHelper:
    def test_a_matrix_row_is_reported(self, X):
        with pytest.raises(InvalidMoleculeError, match="feature matrix"):
            bemis_murcko_smiles(X[0])

    def test_a_smiles_string_is_reported(self):
        with pytest.raises(InvalidMoleculeError, match="MolFromSmiles"):
            bemis_murcko_smiles("c1ccccc1")

    def test_a_molecule_still_works(self):
        assert bemis_murcko_smiles(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")) == "c1ccccc1"


class TestSarTakesTheSamePath:
    """`sar` rolls its own fingerprint helper, so it needed its own guard."""

    def test_a_feature_matrix_is_reported(self, X):
        from qsarkit.sar import ActivityCliffDetector

        y = np.arange(X.shape[0], dtype=float)
        with pytest.raises(InvalidMoleculeError, match="feature matrix"):
            ActivityCliffDetector().detect(X, y)

    def test_molecules_still_work(self, mols):
        from qsarkit.sar import ActivityCliffDetector

        y = np.arange(len(mols), dtype=float)
        assert isinstance(ActivityCliffDetector().detect(mols, y), list)
