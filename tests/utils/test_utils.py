from __future__ import annotations

import logging

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.utils import (
    CONCENTRATION_TO_MOLAR,
    GAS_CONSTANT_KCAL,
    GAS_CONSTANT_KJ,
    LIPINSKI_THRESHOLDS,
    PACTIVITY_ENDPOINTS,
    check_mols,
    check_X_y_mols,
    configure_logging,
    convert_concentration,
    dataframe_to_mols,
    from_pactivity,
    get_logger,
    mols_to_dataframe,
    nm_to_molar,
    pactivity_to_delta_g,
    read_csv_mols,
    read_sdf,
    read_smiles,
    to_pactivity,
    write_sdf,
    write_smiles,
)

SMILES = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O"]


@pytest.fixture
def mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


class TestUnitConversion:
    def test_nm_to_molar(self):
        assert nm_to_molar(1.0) == pytest.approx(1e-9)

    def test_nm_to_molar_is_vectorized(self):
        assert nm_to_molar([1.0, 1000.0]).tolist() == pytest.approx([1e-9, 1e-6])

    @pytest.mark.parametrize(
        "value, unit, expected",
        [(1.0, "M", 1.0), (1.0, "mM", 1e-3), (1.0, "uM", 1e-6),
         (1.0, "nM", 1e-9), (1.0, "pM", 1e-12)],
    )
    def test_convert_to_molar(self, value, unit, expected):
        assert convert_concentration(value, unit) == pytest.approx(expected)

    def test_convert_between_units(self):
        assert convert_concentration(1.0, "uM", "nM") == pytest.approx(1000.0)

    def test_convert_rejects_unknown_unit(self):
        with pytest.raises(ValueError):
            convert_concentration(1.0, "furlongs")

    @pytest.mark.parametrize(
        "value, unit, expected",
        [(1.0, "nM", 9.0), (1.0, "uM", 6.0), (1.0, "mM", 3.0), (1.0, "M", 0.0)],
    )
    def test_to_pactivity(self, value, unit, expected):
        assert to_pactivity(value, unit) == pytest.approx(expected)

    def test_pactivity_roundtrip(self):
        original = np.array([1.0, 10.0, 1000.0])
        recovered = from_pactivity(to_pactivity(original, "nM"), "nM")
        assert recovered == pytest.approx(original)

    def test_pactivity_of_non_positive_is_nan(self):
        assert np.isnan(to_pactivity([0.0, -1.0], "nM")).all()

    def test_delta_g_is_negative_for_a_potent_compound(self):
        # a 1 nM binder (pKd 9) has a strongly favourable binding free energy
        assert pactivity_to_delta_g(9.0) < -10.0

    def test_delta_g_units(self):
        kcal = pactivity_to_delta_g(9.0, units="kcal")
        kj = pactivity_to_delta_g(9.0, units="kj")
        assert kj == pytest.approx(kcal * 4.184, rel=1e-3)

    def test_stronger_binding_gives_more_negative_delta_g(self):
        assert pactivity_to_delta_g(10.0) < pactivity_to_delta_g(6.0)


class TestValidationHelpers:
    def test_check_mols_passes_valid_input(self, mols):
        assert len(check_mols(mols)) == 3

    def test_check_mols_rejects_none_by_default(self, mols):
        with pytest.raises(Exception):
            check_mols([*mols, None])

    def test_check_mols_allows_none_when_asked(self, mols):
        assert len(check_mols([*mols, None], allow_none=True)) == 4

    def test_check_mols_rejects_an_empty_set(self):
        with pytest.raises(Exception):
            check_mols([])

    def test_check_X_y_returns_aligned_arrays(self, mols):
        checked, y = check_X_y_mols(mols, [1.0, 2.0, 3.0])
        assert len(checked) == 3
        assert y.tolist() == [1.0, 2.0, 3.0]

    def test_check_X_y_without_labels(self, mols):
        checked, y = check_X_y_mols(mols)
        assert len(checked) == 3 and y is None

    def test_check_X_y_rejects_length_mismatch(self, mols):
        with pytest.raises(Exception):
            check_X_y_mols(mols, [1.0])


class TestIO:
    def test_smiles_roundtrip(self, mols, tmp_path):
        path = tmp_path / "out.smi"
        assert write_smiles(mols, path) == 3
        recovered = read_smiles(path)
        assert [Chem.MolToSmiles(m) for m in recovered] == [
            Chem.MolToSmiles(m) for m in mols
        ]

    def test_write_smiles_with_names(self, mols, tmp_path):
        path = tmp_path / "named.smi"
        write_smiles(mols, path, names=["a", "b", "c"])
        assert "a" in path.read_text()

    def test_read_smiles_from_an_iterable(self):
        recovered = read_smiles(SMILES)
        assert len(recovered) == 3

    def test_read_smiles_skips_bad_entries(self):
        assert len(read_smiles(["CCO", "!!bad!!"], on_error="skip")) == 1

    def test_sdf_roundtrip(self, mols, tmp_path):
        path = tmp_path / "out.sdf"
        assert write_sdf(mols, path) == 3
        recovered = read_sdf(path)
        assert len(recovered) == 3
        assert Chem.MolToSmiles(recovered[0]) == "CCO"

    def test_sdf_carries_properties(self, mols, tmp_path):
        path = tmp_path / "props.sdf"
        write_sdf(mols, path, properties={"activity": [1.0, 2.0, 3.0]})
        recovered = read_sdf(path)
        assert recovered[0].GetProp("activity") == "1.0"

    def test_mols_to_dataframe(self, mols):
        frame = mols_to_dataframe(mols)
        assert "smiles" in frame.columns and len(frame) == 3

    def test_mols_to_dataframe_with_extra_columns(self, mols):
        frame = mols_to_dataframe(mols, extra={"activity": [1.0, 2.0, 3.0]})
        assert frame["activity"].tolist() == [1.0, 2.0, 3.0]

    def test_dataframe_roundtrip(self, mols):
        frame = mols_to_dataframe(mols)
        recovered = dataframe_to_mols(frame)
        assert [Chem.MolToSmiles(m) for m in recovered] == [
            Chem.MolToSmiles(m) for m in mols
        ]

    def test_read_csv_mols(self, tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("smiles,activity\nCCO,1.0\nc1ccccc1,2.0\n")
        mols, activities, frame = read_csv_mols(path, activity_column="activity")
        assert len(mols) == 2 and len(frame) == 2
        assert activities.tolist() == [1.0, 2.0]


class TestLogging:
    def test_get_logger_returns_a_logger(self):
        assert isinstance(get_logger("qsarkit.test"), logging.Logger)

    def test_configure_logging_sets_the_level(self):
        logger = configure_logging(level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_configure_logging_is_idempotent(self):
        first = configure_logging()
        second = configure_logging()
        assert len(second.handlers) == len(first.handlers)


class TestConstants:
    def test_concentration_table_is_consistent(self):
        assert CONCENTRATION_TO_MOLAR["M"] == 1.0
        assert CONCENTRATION_TO_MOLAR["nM"] == pytest.approx(1e-9)

    def test_lipinski_thresholds(self):
        assert LIPINSKI_THRESHOLDS["mw_max"] == 500.0
        assert LIPINSKI_THRESHOLDS["logp_max"] == 5.0

    def test_pactivity_endpoints_are_recognised(self):
        assert "IC50" in PACTIVITY_ENDPOINTS
        assert "Ki" in PACTIVITY_ENDPOINTS

    def test_gas_constants_agree_after_unit_conversion(self):
        # 1 kcal = 4.184 kJ by definition of the thermochemical calorie.
        assert GAS_CONSTANT_KJ / GAS_CONSTANT_KCAL == pytest.approx(4.184, rel=1e-6)
