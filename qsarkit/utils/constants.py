"""Package-wide physical, chemical and formatting constants.

Values are taken from CODATA 2018 (physical constants) and from the IUPAC
recommendations for the reporting of biochemical activity data. Keeping them in
a single module avoids magic numbers scattered across ``qsarkit``.

References
----------
- Tiesinga et al. (2021). "CODATA Recommended Values of the Fundamental
  Physical Constants: 2018." Rev. Mod. Phys., 93, 025010.
  https://doi.org/10.1103/RevModPhys.93.025010
- Lipinski et al. (2001). "Experimental and computational approaches to
  estimate solubility and permeability in drug discovery and development
  settings." Adv. Drug Deliv. Rev., 46(1-3), 3-26.
  https://doi.org/10.1016/S0169-409X(00)00129-0
"""

from __future__ import annotations

from typing import Dict, Final, Tuple

#: Molar gas constant R, in kcal mol^-1 K^-1 (CODATA 2018: 8.314462618 J/mol/K).
GAS_CONSTANT_KCAL: Final[float] = 1.987204258640832e-3

#: Molar gas constant R, in kJ mol^-1 K^-1.
GAS_CONSTANT_KJ: Final[float] = 8.314462618e-3

#: Avogadro constant, mol^-1 (CODATA 2018, exact).
AVOGADRO: Final[float] = 6.02214076e23

#: Standard room temperature used for free-energy conversions, in kelvin.
ROOM_TEMPERATURE_K: Final[float] = 298.15

#: Multiplicative factors converting a concentration unit to molar (mol/L).
CONCENTRATION_TO_MOLAR: Final[Dict[str, float]] = {
    "M": 1.0,
    "mol/L": 1.0,
    "molar": 1.0,
    "mM": 1e-3,
    "uM": 1e-6,
    "µM": 1e-6,
    "microM": 1e-6,
    "nM": 1e-9,
    "pM": 1e-12,
    "fM": 1e-15,
}

#: Activity endpoints that are conventionally reported on a p-scale.
PACTIVITY_ENDPOINTS: Final[Tuple[str, ...]] = (
    "IC50",
    "EC50",
    "Ki",
    "Kd",
    "AC50",
    "MIC",
    "GI50",
    "LD50",
    "LC50",
)

#: Lipinski "rule of five" thresholds (Lipinski et al., 2001).
LIPINSKI_THRESHOLDS: Final[Dict[str, float]] = {
    "mw_max": 500.0,
    "logp_max": 5.0,
    "hbd_max": 5.0,
    "hba_max": 10.0,
}

#: Default random seed used when a component needs determinism but the user
#: did not supply ``random_state``.
DEFAULT_RANDOM_STATE: Final[int] = 0

__all__ = [
    "GAS_CONSTANT_KCAL",
    "GAS_CONSTANT_KJ",
    "AVOGADRO",
    "ROOM_TEMPERATURE_K",
    "CONCENTRATION_TO_MOLAR",
    "PACTIVITY_ENDPOINTS",
    "LIPINSKI_THRESHOLDS",
    "DEFAULT_RANDOM_STATE",
]
