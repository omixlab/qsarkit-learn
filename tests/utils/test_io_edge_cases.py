"""Error and edge-case paths of the I/O helpers.

The happy paths are covered in ``test_utils.py``. These are the branches
that only run when the input is wrong, which is most of the time with real
activity data: blank lines, comments, missing columns, unparseable SMILES,
absent files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from qsarkit.base import InvalidMoleculeError
from qsarkit.utils import (
    dataframe_to_mols,
    mols_to_dataframe,
    read_csv_mols,
    read_sdf,
    read_smiles,
    write_sdf,
    write_smiles,
)


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(s) for s in ("CCO", "c1ccccc1", "CCN")]


class TestReadSmiles:
    def test_reads_from_an_iterable_of_lines(self):
        assert len(read_smiles(["CCO", "CCN"])) == 2

    def test_skips_blank_lines_and_comments(self):
        lines = ["# a comment", "", "   ", "CCO", "CCN"]
        assert len(read_smiles(lines)) == 2

    def test_header_is_dropped_when_declared(self):
        lines = ["smiles name", "CCO ethanol"]
        mols = read_smiles(lines, has_header=True)
        assert len(mols) == 1
        assert mols[0].GetProp("_Name") == "ethanol"

    def test_a_name_column_is_read(self):
        mols = read_smiles(["CCO ethanol"])
        assert mols[0].GetProp("_Name") == "ethanol"

    def test_a_missing_name_column_is_tolerated(self):
        """A single-field line parses; no name is attached from a column."""
        mol = read_smiles(["CCO"])[0]
        assert Chem.MolToSmiles(mol) == "CCO"
        assert not mol.GetProp("_Name") if mol.HasProp("_Name") else True

    def test_an_explicit_delimiter_is_honoured(self):
        mols = read_smiles(["CCO,ethanol"], delimiter=",")
        assert mols[0].GetProp("_Name") == "ethanol"

    def test_a_non_default_smiles_column(self):
        mols = read_smiles(["ethanol CCO"], smiles_column=1, name_column=0)
        assert Chem.MolToSmiles(mols[0]) == "CCO"

    def test_skip_drops_unparseable_records(self):
        assert len(read_smiles(["CCO", "not-a-molecule"])) == 1

    def test_none_keeps_position_for_unparseable_records(self):
        mols = read_smiles(["CCO", "not-a-molecule"], on_error="none")
        assert len(mols) == 2 and mols[1] is None

    def test_raise_reports_the_line_number(self):
        with pytest.raises(InvalidMoleculeError, match="Line 2"):
            read_smiles(["CCO", "not-a-molecule"], on_error="raise")

    def test_a_short_line_is_skipped(self):
        assert len(read_smiles(["CCO", ""], smiles_column=3)) == 0

    def test_a_short_line_can_become_none(self):
        mols = read_smiles(["CCO"], smiles_column=3, on_error="none")
        assert mols == [None]

    def test_a_short_line_can_raise(self):
        with pytest.raises(InvalidMoleculeError, match="no field at index"):
            read_smiles(["CCO"], smiles_column=3, on_error="raise")

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No such SMILES file"):
            read_smiles(tmp_path / "absent.smi")

    def test_rejects_an_unknown_on_error(self):
        with pytest.raises(ValueError, match="on_error"):
            read_smiles(["CCO"], on_error="explode")

    def test_sanitize_can_be_turned_off(self):
        # A nitro group written with pentavalent nitrogen fails sanitization.
        assert read_smiles(["C1=CC=CC=C1N(=O)=O"], sanitize=False)

    def test_round_trips_through_a_file(self, mols, tmp_path):
        path = tmp_path / "out.smi"
        assert write_smiles(mols, path) == 3
        assert len(read_smiles(path)) == 3


class TestWriteSmiles:
    def test_returns_the_count_written(self, mols, tmp_path):
        assert write_smiles(mols, tmp_path / "a.smi") == 3

    def test_names_are_written_when_supplied(self, mols, tmp_path):
        path = tmp_path / "a.smi"
        write_smiles(mols, path, names=["a", "b", "c"])
        assert "a" in path.read_text().splitlines()[0]

    def test_none_entries_are_skipped(self, mols, tmp_path):
        assert write_smiles([*mols, None], tmp_path / "a.smi") == 3

    def test_a_custom_delimiter_is_used(self, mols, tmp_path):
        path = tmp_path / "a.smi"
        write_smiles(mols, path, names=["a", "b", "c"], delimiter=",")
        assert "," in path.read_text().splitlines()[0]

    def test_non_isomeric_output_drops_stereo(self, tmp_path):
        chiral = [Chem.MolFromSmiles("C[C@H](N)C(=O)O")]
        path = tmp_path / "a.smi"
        write_smiles(chiral, path, isomeric=False)
        assert "@" not in path.read_text()


class TestSdf:
    def test_round_trip(self, mols, tmp_path):
        path = tmp_path / "a.sdf"
        assert write_sdf(mols, path) == 3
        assert len(read_sdf(path)) == 3

    def test_properties_are_written_and_read_back(self, mols, tmp_path):
        path = tmp_path / "a.sdf"
        write_sdf(mols, path, properties={"pIC50": [5.0, 6.0, 7.0]})
        restored = read_sdf(path)
        assert restored[0].GetProp("pIC50") == "5.0"

    def test_none_entries_are_skipped_on_write(self, mols, tmp_path):
        assert write_sdf([*mols, None], tmp_path / "a.sdf") == 3

    def test_a_missing_file_is_reported(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_sdf(tmp_path / "absent.sdf")

    def test_rejects_an_unknown_on_error(self, mols, tmp_path):
        path = tmp_path / "a.sdf"
        write_sdf(mols, path)
        with pytest.raises(ValueError, match="on_error"):
            read_sdf(path, on_error="explode")

    def test_kekulize_can_be_turned_off(self, mols, tmp_path):
        assert write_sdf(mols, tmp_path / "a.sdf", kekulize=False) == 3

    def test_remove_hs_can_be_turned_off(self, mols, tmp_path):
        path = tmp_path / "a.sdf"
        write_sdf([Chem.AddHs(mols[0])], path)
        with_hs = read_sdf(path, remove_hs=False)
        assert with_hs[0].GetNumAtoms() > 3


class TestCsv:
    def test_reads_smiles_and_activity(self, tmp_path):
        """Returns ``(mols, y, dataframe)``."""
        path = tmp_path / "a.csv"
        path.write_text("smiles,pIC50\nCCO,5.0\nCCN,6.0\n")
        mols, y, frame = read_csv_mols(path, activity_column="pIC50")
        assert len(mols) == 2
        assert y.tolist() == [5.0, 6.0]
        assert len(frame) == 2

    def test_activity_is_none_when_no_column_is_named(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("smiles\nCCO\n")
        mols, y, _ = read_csv_mols(path)
        assert len(mols) == 1 and y is None

    def test_a_missing_smiles_column_is_reported(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("structure,pIC50\nCCO,5.0\n")
        with pytest.raises(KeyError, match="smiles"):
            read_csv_mols(path)

    def test_a_missing_activity_column_is_reported(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("smiles,pIC50\nCCO,5.0\n")
        with pytest.raises(KeyError):
            read_csv_mols(path, activity_column="absent")

    def test_unparseable_rows_become_none(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("smiles\nCCO\nnot-a-molecule\n")
        mols, _, _ = read_csv_mols(path, on_error="none")
        assert len(mols) == 2 and mols[1] is None

    def test_unparseable_rows_can_be_dropped(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("smiles\nCCO\nnot-a-molecule\n")
        mols, _, _ = read_csv_mols(path, on_error="skip")
        assert len(mols) == 1

    def test_skip_is_refused_when_an_activity_column_is_named(self, tmp_path):
        """Dropping rows would silently misalign them with the activities.

        The reader refuses rather than returning a shifted pair, which is
        the failure that produces a model trained on the wrong labels.
        """
        path = tmp_path / "a.csv"
        path.write_text("smiles,pIC50\nCCO,5.0\nnot-a-molecule,6.0\nCCN,7.0\n")
        with pytest.raises(ValueError, match="breaks alignment"):
            read_csv_mols(path, activity_column="pIC50", on_error="skip")

    def test_none_keeps_activities_aligned(self, tmp_path):
        """The alignment guarantee the rest of the package relies on."""
        path = tmp_path / "a.csv"
        path.write_text("smiles,pIC50\nCCO,5.0\nnot-a-molecule,6.0\nCCN,7.0\n")
        mols, y, _ = read_csv_mols(path, activity_column="pIC50", on_error="none")
        assert len(mols) == len(y) == 3
        assert mols[1] is None
        assert y.tolist() == [5.0, 6.0, 7.0]

    def test_unparseable_rows_can_raise(self, tmp_path):
        path = tmp_path / "a.csv"
        path.write_text("smiles\nnot-a-molecule\n")
        with pytest.raises(InvalidMoleculeError):
            read_csv_mols(path, on_error="raise")

    def test_extra_kwargs_reach_pandas(self, tmp_path):
        path = tmp_path / "a.tsv"
        path.write_text("smiles\tpIC50\nCCO\t5.0\n")
        mols, _, _ = read_csv_mols(path, sep="\t")
        assert len(mols) == 1


class TestDataFrameConversion:
    def test_mols_to_dataframe_includes_smiles(self, mols):
        frame = mols_to_dataframe(mols)
        assert "smiles" in frame.columns and len(frame) == 3

    def test_smiles_can_be_omitted(self, mols):
        assert "smiles" not in mols_to_dataframe(mols, include_smiles=False).columns

    def test_the_smiles_column_can_be_renamed(self, mols):
        assert "structure" in mols_to_dataframe(
            mols, smiles_column="structure"
        ).columns

    def test_molecule_properties_are_included(self, mols):
        mols[0].SetProp("source", "chembl")
        assert "source" in mols_to_dataframe(mols).columns

    def test_properties_can_be_omitted(self, mols):
        mols[0].SetProp("source", "chembl")
        assert "source" not in mols_to_dataframe(
            mols, include_properties=False
        ).columns

    def test_extra_columns_are_attached(self, mols):
        frame = mols_to_dataframe(mols, extra={"pIC50": [5.0, 6.0, 7.0]})
        assert frame["pIC50"].tolist() == [5.0, 6.0, 7.0]

    def test_a_mismatched_extra_column_is_reported(self, mols):
        with pytest.raises(ValueError):
            mols_to_dataframe(mols, extra={"pIC50": [5.0]})

    def test_none_entries_are_handled(self, mols):
        frame = mols_to_dataframe([*mols, None])
        assert len(frame) == 4

    def test_dataframe_to_mols_round_trips(self, mols):
        frame = mols_to_dataframe(mols)
        assert len(dataframe_to_mols(frame)) == 3

    def test_a_missing_column_is_reported(self):
        with pytest.raises(KeyError):
            dataframe_to_mols(pd.DataFrame({"structure": ["CCO"]}))

    def test_names_are_set_from_a_column(self):
        frame = pd.DataFrame({"smiles": ["CCO"], "name": ["ethanol"]})
        assert dataframe_to_mols(frame, name_column="name")[0].GetProp("_Name") == (
            "ethanol"
        )

    def test_property_columns_are_attached(self):
        frame = pd.DataFrame({"smiles": ["CCO"], "pIC50": [5.0]})
        mol = dataframe_to_mols(frame, property_columns=["pIC50"])[0]
        assert mol.GetProp("pIC50") == "5.0"

    def test_unparseable_rows_become_none(self):
        frame = pd.DataFrame({"smiles": ["CCO", "not-a-molecule"]})
        assert dataframe_to_mols(frame, on_error="none")[1] is None

    def test_unparseable_rows_can_be_skipped(self):
        frame = pd.DataFrame({"smiles": ["CCO", "not-a-molecule"]})
        assert len(dataframe_to_mols(frame, on_error="skip")) == 1

    def test_unparseable_rows_can_raise(self):
        frame = pd.DataFrame({"smiles": ["not-a-molecule"]})
        with pytest.raises(InvalidMoleculeError):
            dataframe_to_mols(frame, on_error="raise")

    def test_nan_smiles_are_handled(self):
        frame = pd.DataFrame({"smiles": ["CCO", np.nan]})
        assert dataframe_to_mols(frame, on_error="none")[1] is None
