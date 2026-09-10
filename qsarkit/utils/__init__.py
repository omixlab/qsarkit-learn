"""Shared helpers: I/O, logging, validation and unit conversion.

Examples
--------
>>> from qsarkit.utils import to_pactivity
>>> float(to_pactivity(1.0, unit="nM"))
9.0
"""

from qsarkit.utils.constants import (
    AVOGADRO,
    CONCENTRATION_TO_MOLAR,
    DEFAULT_RANDOM_STATE,
    GAS_CONSTANT_KCAL,
    GAS_CONSTANT_KJ,
    LIPINSKI_THRESHOLDS,
    PACTIVITY_ENDPOINTS,
    ROOM_TEMPERATURE_K,
)
from qsarkit.utils.io import (
    dataframe_to_mols,
    mols_to_dataframe,
    read_csv_mols,
    read_sdf,
    read_smiles,
    write_sdf,
    write_smiles,
)
from qsarkit.utils.logging import configure_logging, get_logger
from qsarkit.utils.validation import (
    check_mols,
    check_X_y_mols,
    convert_concentration,
    from_pactivity,
    nm_to_molar,
    pactivity_to_delta_g,
    to_pactivity,
)

__all__ = [
    "read_smiles", "write_smiles", "read_sdf", "write_sdf",
    "mols_to_dataframe", "dataframe_to_mols", "read_csv_mols",
    "get_logger", "configure_logging",
    "check_mols", "check_X_y_mols", "nm_to_molar", "convert_concentration",
    "to_pactivity", "from_pactivity", "pactivity_to_delta_g",
    "GAS_CONSTANT_KCAL", "GAS_CONSTANT_KJ", "AVOGADRO", "ROOM_TEMPERATURE_K",
    "CONCENTRATION_TO_MOLAR", "PACTIVITY_ENDPOINTS", "LIPINSKI_THRESHOLDS",
    "DEFAULT_RANDOM_STATE",
]
