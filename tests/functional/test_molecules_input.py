"""Input handling for :func:`qsarkit.functional.molecules`.

``molecules()`` is the entry point of the whole functional API, and it
accepts RDKit molecules, SMILES and InChI in any mixture. Getting the
input contract wrong is the failure that propagates furthest, so it is
tested on its own.
"""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.functional import drop_invalid, molecules

ETHANOL_INCHI = "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3"
BENZENE_INCHI = "InChI=1S/C6H6/c1-2-4-6-5-3-1/h1-6H"


def smis(mols):
    return [None if m is None else Chem.MolToSmiles(m) for m in mols]


class TestAcceptedInputs:
    def test_smiles(self):
        assert molecules(["CCO", "c1ccccc1"]).smiles == ["CCO", "c1ccccc1"]

    def test_rdkit_molecules(self):
        mols = [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1")]
        assert molecules(mols).smiles == ["CCO", "c1ccccc1"]

    def test_inchi(self):
        assert molecules([ETHANOL_INCHI, BENZENE_INCHI]).smiles == ["CCO", "c1ccccc1"]

    def test_a_mixture_of_all_three(self):
        ms = molecules([Chem.MolFromSmiles("c1ccccc1"), "CCN", ETHANOL_INCHI])
        assert ms.smiles == ["c1ccccc1", "CCN", "CCO"]

    def test_labels_are_carried(self):
        ms = molecules(["CCO", "CCN"], [1.0, 2.0])
        assert ms.y.tolist() == [1.0, 2.0]

    def test_empty_input(self):
        ms = molecules([])
        assert len(ms) == 0 and ms.y is None


class TestFormatSelection:
    def test_auto_detects_inchi_by_its_prefix(self):
        assert molecules([ETHANOL_INCHI]).smiles == ["CCO"]

    def test_explicit_inchi(self):
        assert molecules([ETHANOL_INCHI], fmt="inchi").smiles == ["CCO"]

    def test_explicit_smiles_does_not_reinterpret_inchi(self):
        # Read as SMILES, an InChI string is simply invalid -- which is the
        # point of naming the format: it fails instead of guessing.
        assert molecules([ETHANOL_INCHI], fmt="smiles").mols == [None]

    def test_mol_format_rejects_strings(self):
        with pytest.raises(ValueError, match="fmt='mol' was requested"):
            molecules(["CCO"], fmt="mol")

    def test_mol_format_accepts_molecules(self):
        mols = [Chem.MolFromSmiles("CCO")]
        assert molecules(mols, fmt="mol").smiles == ["CCO"]

    def test_rejects_unknown_format(self):
        with pytest.raises(ValueError, match="Unknown fmt"):
            molecules(["CCO"], fmt="mol2")

    def test_whitespace_is_tolerated(self):
        assert molecules(["  CCO  ", f"  {ETHANOL_INCHI} "]).smiles == ["CCO", "CCO"]


class TestInvalidInput:
    def test_unparseable_entries_become_none(self):
        assert molecules(["CCO", "not-a-molecule"]).mols[1] is None

    def test_unparseable_entries_keep_labels_aligned(self):
        ms = molecules(["CCO", "!!bad!!", "CCN"], [1.0, 2.0, 3.0])
        mols, y = ms >> drop_invalid()
        assert smis(mols) == ["CCO", "CCN"]
        assert y.tolist() == [1.0, 3.0]

    def test_malformed_inchi_becomes_none(self):
        assert molecules(["InChI=1S/nonsense"]).mols == [None]

    def test_none_passes_through(self):
        assert molecules([None, "CCO"]).mols[0] is None

    def test_rejects_a_type_that_is_neither_mol_nor_string(self):
        with pytest.raises(ValueError, match="neither an rdkit.Chem.Mol nor a string"):
            molecules([42])

    def test_mismatched_label_length_is_rejected(self):
        with pytest.raises(ValueError, match="y has length"):
            molecules(["CCO", "CCN"], [1.0])

    def test_parsing_failures_do_not_print_rdkit_warnings(self, capfd):
        molecules(["!!bad!!"] * 3)
        captured = capfd.readouterr()
        assert "SMILES Parse Error" not in captured.err
