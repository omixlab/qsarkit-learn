"""Input validation helpers and activity-unit conversions.

The molecule validators mirror ``sklearn.utils.validation`` semantics for the
``Iterable[Mol]`` inputs used throughout ``qsarkit``; the activity converters
implement the standard p-scale transformation used when curating bioactivity
data from ChEMBL/PubChem for QSAR modelling.

References
----------
- Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
  JMLR, 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html
- Kalliokoski et al. (2013). "Comparability of Mixed IC50 Data - A Statistical
  Analysis." PLoS ONE, 8(4), e61007.
  https://doi.org/10.1371/journal.pone.0061007
- Bento et al. (2014). "The ChEMBL bioactivity database: an update."
  Nucleic Acids Res., 42, D1083-D1090. https://doi.org/10.1093/nar/gkt1031
"""

from __future__ import annotations

import math
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from qsarkit.base import InvalidMoleculeError, ensure_mol_list
from qsarkit.utils.constants import CONCENTRATION_TO_MOLAR


def check_mols(
    mols: Iterable[Any],
    allow_none: bool = False,
    min_size: int = 1,
) -> List[Any]:
    """Validate and materialize an ``Iterable[Mol]``.

    Parameters
    ----------
    mols : iterable of rdkit.Chem.Mol
        Input molecules.
    allow_none : bool, default False
        If ``False`` a ``None`` entry (a molecule that failed an earlier
        parsing/curation step) raises :class:`InvalidMoleculeError`. If
        ``True`` ``None`` entries are preserved so downstream code can keep
        positional alignment.
    min_size : int, default 1
        Minimum acceptable number of molecules.

    Returns
    -------
    list of rdkit.Chem.Mol
        The materialized list.

    Raises
    ------
    InvalidMoleculeError
        If an element is not an RDKit ``Mol`` (or is ``None`` while
        ``allow_none`` is ``False``), or if fewer than ``min_size`` molecules
        were supplied.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.utils import check_mols
    >>> len(check_mols([Chem.MolFromSmiles("CCO")]))
    1

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
      JMLR, 12, 2825-2830.
    """
    mol_list = ensure_mol_list(mols)
    if len(mol_list) < min_size:
        raise InvalidMoleculeError(
            f"Expected at least {min_size} molecule(s), got {len(mol_list)}."
        )
    if not allow_none:
        bad = [i for i, m in enumerate(mol_list) if m is None]
        if bad:
            raise InvalidMoleculeError(
                f"Input contains {len(bad)} None molecule(s) at indices "
                f"{bad[:10]}{'...' if len(bad) > 10 else ''}. Pass "
                "allow_none=True to keep them, or filter them out first."
            )
    return mol_list


def check_X_y_mols(
    mols: Iterable[Any],
    y: Optional[Sequence[Any]] = None,
    allow_none: bool = False,
    min_size: int = 1,
    dtype: Any = float,
) -> Tuple[List[Any], Optional[np.ndarray]]:
    """Validate a ``(mols, y)`` pair for a supervised QSAR estimator.

    Parameters
    ----------
    mols : iterable of rdkit.Chem.Mol
        Input molecules.
    y : array-like, optional
        Target values. If ``None`` only the molecules are validated and the
        returned target is ``None`` (mirrors ``fit(X, y=None)``).
    allow_none : bool, default False
        Forwarded to :func:`check_mols`.
    min_size : int, default 1
        Minimum acceptable number of molecules.
    dtype : type, default ``float``
        NumPy dtype the target is cast to. Pass ``None`` to keep the original
        dtype (useful for string class labels).

    Returns
    -------
    mols : list of rdkit.Chem.Mol
        The validated molecules.
    y : numpy.ndarray or None
        The target as a 1-D array, or ``None``.

    Raises
    ------
    InvalidMoleculeError
        If the molecules are invalid.
    ValueError
        If ``len(y) != len(mols)`` or ``y`` is not 1-D.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.utils import check_X_y_mols
    >>> mols, y = check_X_y_mols([Chem.MolFromSmiles("CCO")], [1.0])
    >>> float(y[0])
    1.0

    References
    ----------
    - Pedregosa et al. (2011). "Scikit-learn: Machine Learning in Python."
      JMLR, 12, 2825-2830.
    """
    mol_list = check_mols(mols, allow_none=allow_none, min_size=min_size)
    if y is None:
        return mol_list, None
    y_arr = np.asarray(y) if dtype is None else np.asarray(y, dtype=dtype)
    if y_arr.ndim != 1:
        raise ValueError(f"y must be 1-dimensional, got shape {y_arr.shape}.")
    if y_arr.shape[0] != len(mol_list):
        raise ValueError(
            f"Length mismatch: {len(mol_list)} molecules but {y_arr.shape[0]} targets."
        )
    return mol_list, y_arr


def nm_to_molar(value: Any) -> np.ndarray:
    """Convert nanomolar concentrations to molar.

    Parameters
    ----------
    value : float or array-like
        Concentration(s) in nM.

    Returns
    -------
    numpy.ndarray
        The concentration(s) in mol/L.

    Examples
    --------
    >>> from qsarkit.utils import nm_to_molar
    >>> round(float(nm_to_molar(1000.0)), 12)
    1e-06

    References
    ----------
    - Bento et al. (2014). "The ChEMBL bioactivity database: an update."
      Nucleic Acids Res., 42, D1083-D1090.
      https://doi.org/10.1093/nar/gkt1031
    """
    return np.asarray(value, dtype=float) * 1e-9


def convert_concentration(value: Any, from_unit: str, to_unit: str = "M") -> np.ndarray:
    """Convert a concentration between the units in ``CONCENTRATION_TO_MOLAR``.

    Parameters
    ----------
    value : float or array-like
        Concentration value(s).
    from_unit, to_unit : str
        Source and destination units. Supported keys: ``M``, ``mM``, ``uM``,
        ``nM``, ``pM``, ``fM`` (plus the aliases in
        :data:`qsarkit.utils.constants.CONCENTRATION_TO_MOLAR`).

    Returns
    -------
    numpy.ndarray
        The converted concentration(s).

    Raises
    ------
    ValueError
        If either unit is unknown.

    Examples
    --------
    >>> from qsarkit.utils import convert_concentration
    >>> round(float(convert_concentration(1.0, "uM", "nM")), 6)
    1000.0

    References
    ----------
    - Bento et al. (2014). "The ChEMBL bioactivity database: an update."
      Nucleic Acids Res., 42, D1083-D1090.
      https://doi.org/10.1093/nar/gkt1031
    """
    for unit in (from_unit, to_unit):
        if unit not in CONCENTRATION_TO_MOLAR:
            raise ValueError(
                f"Unknown concentration unit {unit!r}. Supported: "
                f"{sorted(CONCENTRATION_TO_MOLAR)}"
            )
    factor = CONCENTRATION_TO_MOLAR[from_unit] / CONCENTRATION_TO_MOLAR[to_unit]
    return np.asarray(value, dtype=float) * factor


def to_pactivity(value: Any, unit: str = "nM") -> np.ndarray:
    """Convert a concentration-based activity to its p-scale value.

    The p-scale (pIC50, pKi, pEC50, ...) is the negative decadic logarithm of
    the molar concentration::

        pX = -log10(C / 1 M)

    Working on the p-scale is standard practice in QSAR because it linearises
    the relationship with free energy of binding and makes the error structure
    approximately homoscedastic.

    Parameters
    ----------
    value : float or array-like
        Activity concentration(s), strictly positive.
    unit : str, default ``"nM"``
        Unit of ``value``; any key of
        :data:`qsarkit.utils.constants.CONCENTRATION_TO_MOLAR`.

    Returns
    -------
    numpy.ndarray
        The p-scale activity value(s). Non-positive or non-finite inputs map
        to ``nan``.

    Examples
    --------
    >>> from qsarkit.utils import to_pactivity
    >>> float(to_pactivity(1.0, "nM"))
    9.0
    >>> float(to_pactivity(1000.0, "nM"))
    6.0

    References
    ----------
    - Kalliokoski et al. (2013). "Comparability of Mixed IC50 Data - A
      Statistical Analysis." PLoS ONE, 8(4), e61007.
      https://doi.org/10.1371/journal.pone.0061007
    - Bento et al. (2014). "The ChEMBL bioactivity database: an update."
      Nucleic Acids Res., 42, D1083-D1090.
      https://doi.org/10.1093/nar/gkt1031
    """
    molar = convert_concentration(value, unit, "M")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.log10(np.where(molar > 0, molar, np.nan))
    return np.asarray(out, dtype=float)


def from_pactivity(pvalue: Any, unit: str = "nM") -> np.ndarray:
    """Invert :func:`to_pactivity`, returning a concentration.

    Parameters
    ----------
    pvalue : float or array-like
        p-scale activity value(s), e.g. pIC50.
    unit : str, default ``"nM"``
        Unit of the returned concentration.

    Returns
    -------
    numpy.ndarray
        Concentration(s) expressed in ``unit``.

    Examples
    --------
    >>> from qsarkit.utils import from_pactivity
    >>> round(float(from_pactivity(9.0, "nM")), 6)
    1.0

    References
    ----------
    - Kalliokoski et al. (2013). "Comparability of Mixed IC50 Data - A
      Statistical Analysis." PLoS ONE, 8(4), e61007.
      https://doi.org/10.1371/journal.pone.0061007
    """
    molar = np.power(10.0, -np.asarray(pvalue, dtype=float))
    return convert_concentration(molar, "M", unit)


def pactivity_to_delta_g(
    pvalue: Any,
    temperature: float = 298.15,
    units: str = "kcal",
) -> np.ndarray:
    """Convert a p-scale affinity to a binding free energy.

    Uses ``dG = -RT ln(K) = -2.303 R T * pX`` with ``K`` the association
    constant, valid when the reported p-value is a ``pKd``/``pKi`` (an
    equilibrium dissociation constant). Applying it to a ``pIC50`` is an
    approximation that ignores the Cheng-Prusoff correction.

    Parameters
    ----------
    pvalue : float or array-like
        p-scale affinity (pKd or pKi).
    temperature : float, default 298.15
        Absolute temperature in kelvin.
    units : {"kcal", "kj"}, default ``"kcal"``
        Energy units of the result (per mole).

    Returns
    -------
    numpy.ndarray
        Binding free energy, negative for favourable binding.

    Examples
    --------
    >>> from qsarkit.utils import pactivity_to_delta_g
    >>> round(float(pactivity_to_delta_g(9.0)), 2)
    -12.28

    References
    ----------
    - Cheng & Prusoff (1973). "Relationship between the inhibition constant
      (Ki) and the concentration of inhibitor which causes 50 per cent
      inhibition (I50) of an enzymatic reaction." Biochem. Pharmacol., 22(23),
      3099-3108. https://doi.org/10.1016/0006-2952(73)90196-2
    """
    from qsarkit.utils.constants import GAS_CONSTANT_KCAL, GAS_CONSTANT_KJ

    if units not in ("kcal", "kj"):
        raise ValueError(f"units must be 'kcal' or 'kj', got {units!r}.")
    r = GAS_CONSTANT_KCAL if units == "kcal" else GAS_CONSTANT_KJ
    return -math.log(10.0) * r * temperature * np.asarray(pvalue, dtype=float)


__all__ = [
    "check_mols",
    "check_X_y_mols",
    "nm_to_molar",
    "convert_concentration",
    "to_pactivity",
    "from_pactivity",
    "pactivity_to_delta_g",
]
