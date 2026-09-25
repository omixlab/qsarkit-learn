"""Molecule I/O helpers: SMILES, SDF and CSV <-> ``list[Mol]``.

Parsing molecular file formats is deliberately outside the scope of the
modelling API (every estimator consumes ``Iterable[Mol]``); these helpers are
the sanctioned boundary where text/tabular formats are turned into RDKit
molecules and back.

References
----------
- Weininger (1988). "SMILES, a chemical language and information system. 1.
  Introduction to methodology and encoding rules." J. Chem. Inf. Comput. Sci.,
  28(1), 31-36. https://doi.org/10.1021/ci00057a005
- Dalby et al. (1992). "Description of several chemical structure file formats
  used by computer programs developed at Molecular Design Limited."
  J. Chem. Inf. Comput. Sci., 32(3), 244-255.
  https://doi.org/10.1021/ci00007a012
- RDKit documentation: https://www.rdkit.org/docs/GettingStartedInPython.html
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from qsarkit.base import InvalidMoleculeError, ensure_mol_list
from qsarkit.utils.logging import get_logger

PathLike = Union[str, "os.PathLike[str]"]

_LOG = get_logger(__name__)


def _mol_from_smiles(smiles: str, sanitize: bool = True) -> Optional[Any]:
    """Parse one SMILES, returning ``None`` rather than logging a failure.

    RDKit writes a five-line parse error to stderr for every unparseable
    record. A curation workflow is expected to receive bad input -- that is
    what ``on_error`` is for -- so a file with a thousand bad rows would
    bury its own report. ``BlockLogs`` restores whatever logging state the
    caller had, rather than switching it back on underneath them.
    """
    from rdkit import Chem, rdBase

    blocker = rdBase.BlockLogs()
    try:
        return Chem.MolFromSmiles(smiles, sanitize=sanitize)
    finally:
        del blocker


def read_smiles(
    source: Union[PathLike, Iterable[str]],
    delimiter: Optional[str] = None,
    smiles_column: int = 0,
    name_column: Optional[int] = 1,
    has_header: bool = False,
    sanitize: bool = True,
    on_error: str = "skip",
) -> List[Any]:
    """Read SMILES from a file path or an iterable of strings into RDKit Mols.

    Parameters
    ----------
    source : path-like or iterable of str
        Either a path to a ``.smi``/``.txt`` file, or an already-materialized
        iterable of SMILES strings (or whitespace/delimiter separated lines).
    delimiter : str, optional
        Field delimiter for multi-column lines. ``None`` splits on arbitrary
        whitespace (the classic ``.smi`` convention).
    smiles_column : int, default 0
        Index of the SMILES field within each split line.
    name_column : int, optional, default 1
        Index of an optional molecule-name field; stored on the molecule as
        the ``_Name`` property when present. Pass ``None`` to ignore names.
    has_header : bool, default False
        Skip the first line when reading from a file/iterable.
    sanitize : bool, default True
        Run RDKit sanitization on parsing.
    on_error : {"skip", "none", "raise"}, default ``"skip"``
        What to do with unparsable records: drop them, insert ``None`` in
        their position (preserving alignment with the source), or raise
        :class:`~qsarkit.base.InvalidMoleculeError`.

    Returns
    -------
    list of rdkit.Chem.Mol
        The parsed molecules.

    Raises
    ------
    InvalidMoleculeError
        If ``on_error="raise"`` and a record cannot be parsed.
    ValueError
        If ``on_error`` is not one of the accepted values.

    Examples
    --------
    >>> from qsarkit.utils import read_smiles
    >>> mols = read_smiles(["CCO ethanol", "c1ccccc1 benzene"])
    >>> [m.GetProp("_Name") for m in mols]
    ['ethanol', 'benzene']

    References
    ----------
    - Weininger (1988). "SMILES, a chemical language and information system. 1."
      J. Chem. Inf. Comput. Sci., 28(1), 31-36.
      https://doi.org/10.1021/ci00057a005
    - RDKit documentation:
      https://www.rdkit.org/docs/GettingStartedInPython.html
    """
    if on_error not in ("skip", "none", "raise"):
        raise ValueError(
            f"on_error must be 'skip', 'none' or 'raise', got {on_error!r}."
        )

    if isinstance(source, (str, os.PathLike)) and os.path.exists(source):
        with open(source, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    elif isinstance(source, (str, os.PathLike)):
        raise FileNotFoundError(f"No such SMILES file: {source!r}")
    else:
        lines = [str(line) for line in source]

    if has_header and lines:
        lines = lines[1:]

    mols: List[Any] = []
    for lineno, raw in enumerate(lines, start=2 if has_header else 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(delimiter) if delimiter else line.split()
        if smiles_column >= len(fields):
            if on_error == "raise":
                raise InvalidMoleculeError(
                    f"Line {lineno}: no field at index {smiles_column} in {line!r}."
                )
            if on_error == "none":
                mols.append(None)
            continue
        smiles = fields[smiles_column]
        mol = _mol_from_smiles(smiles, sanitize=sanitize)
        if mol is None:
            if on_error == "raise":
                raise InvalidMoleculeError(
                    f"Line {lineno}: RDKit could not parse SMILES {smiles!r}."
                )
            _LOG.debug("Skipping unparsable SMILES on line %d: %r", lineno, smiles)
            if on_error == "none":
                mols.append(None)
            continue
        if name_column is not None and name_column < len(fields):
            mol.SetProp("_Name", fields[name_column])
        mols.append(mol)
    return mols


def write_smiles(
    mols: Iterable[Any],
    path: PathLike,
    names: Optional[Sequence[str]] = None,
    isomeric: bool = True,
    delimiter: str = " ",
) -> int:
    """Write molecules to a ``.smi`` file.

    Parameters
    ----------
    mols : iterable of rdkit.Chem.Mol
        Molecules to serialize. ``None`` entries are skipped.
    path : path-like
        Destination file.
    names : sequence of str, optional
        Per-molecule names. Defaults to the molecule's ``_Name`` property when
        set, else no name column is emitted.
    isomeric : bool, default True
        Emit isomeric SMILES (retain stereochemistry).
    delimiter : str, default ``" "``
        Field separator.

    Returns
    -------
    int
        Number of molecules written.

    Examples
    --------
    >>> import tempfile, os
    >>> from rdkit import Chem
    >>> from qsarkit.utils import write_smiles, read_smiles
    >>> p = os.path.join(tempfile.mkdtemp(), "m.smi")
    >>> write_smiles([Chem.MolFromSmiles("CCO")], p)
    1
    >>> len(read_smiles(p))
    1

    References
    ----------
    - Weininger (1988). "SMILES, a chemical language and information system. 1."
      J. Chem. Inf. Comput. Sci., 28(1), 31-36.
      https://doi.org/10.1021/ci00057a005
    """
    from rdkit import Chem

    mol_list = ensure_mol_list(mols)
    written = 0
    with open(path, "w", encoding="utf-8") as fh:
        for i, mol in enumerate(mol_list):
            if mol is None:
                continue
            smi = Chem.MolToSmiles(mol, isomericSmiles=isomeric)
            if names is not None and i < len(names):
                name: Optional[str] = str(names[i])
            elif mol.HasProp("_Name"):
                name = mol.GetProp("_Name")
            else:
                name = None
            fh.write(smi if name is None else f"{smi}{delimiter}{name}")
            fh.write("\n")
            written += 1
    return written


def read_sdf(
    path: PathLike,
    sanitize: bool = True,
    remove_hs: bool = True,
    on_error: str = "skip",
) -> List[Any]:
    """Read an MDL SD file into a list of RDKit molecules.

    SD-file data fields are preserved as RDKit molecule properties, so they can
    be recovered with :func:`mols_to_dataframe`.

    Parameters
    ----------
    path : path-like
        Path to the ``.sdf`` / ``.sd`` file.
    sanitize : bool, default True
        Run RDKit sanitization on each record.
    remove_hs : bool, default True
        Remove explicit hydrogens (RDKit's implicit-H convention).
    on_error : {"skip", "none", "raise"}, default ``"skip"``
        Handling of records RDKit fails to parse.

    Returns
    -------
    list of rdkit.Chem.Mol
        Parsed molecules.

    Raises
    ------
    InvalidMoleculeError
        If ``on_error="raise"`` and a record fails to parse.
    FileNotFoundError
        If ``path`` does not exist.

    Examples
    --------
    >>> import tempfile, os
    >>> from rdkit import Chem
    >>> from qsarkit.utils import write_sdf, read_sdf
    >>> p = os.path.join(tempfile.mkdtemp(), "m.sdf")
    >>> _ = write_sdf([Chem.MolFromSmiles("CCO")], p)
    >>> len(read_sdf(p))
    1

    References
    ----------
    - Dalby et al. (1992). "Description of several chemical structure file
      formats used by computer programs developed at Molecular Design Limited."
      J. Chem. Inf. Comput. Sci., 32(3), 244-255.
      https://doi.org/10.1021/ci00007a012
    - RDKit documentation:
      https://www.rdkit.org/docs/GettingStartedInPython.html
    """
    from rdkit import Chem

    if on_error not in ("skip", "none", "raise"):
        raise ValueError(
            f"on_error must be 'skip', 'none' or 'raise', got {on_error!r}."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"No such SD file: {path!r}")

    supplier = Chem.SDMolSupplier(str(path), sanitize=sanitize, removeHs=remove_hs)
    mols: List[Any] = []
    for i, mol in enumerate(supplier):
        if mol is None:
            if on_error == "raise":
                raise InvalidMoleculeError(f"SD record {i} could not be parsed.")
            _LOG.debug("Skipping unparsable SD record %d", i)
            if on_error == "none":
                mols.append(None)
            continue
        mols.append(mol)
    return mols


def write_sdf(
    mols: Iterable[Any],
    path: PathLike,
    properties: Optional[Dict[str, Sequence[Any]]] = None,
    kekulize: bool = True,
) -> int:
    """Write molecules to an MDL SD file, optionally attaching data fields.

    Parameters
    ----------
    mols : iterable of rdkit.Chem.Mol
        Molecules to write. ``None`` entries are skipped.
    path : path-like
        Destination file.
    properties : dict of str -> sequence, optional
        Extra SD data fields, one sequence per field aligned with ``mols``
        (e.g. ``{"pIC50": [7.2, 6.4]}``). Values are written with ``str()``.
    kekulize : bool, default True
        Kekulize aromatic rings before writing (standard for MDL formats).

    Returns
    -------
    int
        Number of records written.

    Raises
    ------
    ValueError
        If a property sequence is shorter than the molecule list.

    Examples
    --------
    >>> import tempfile, os
    >>> from rdkit import Chem
    >>> from qsarkit.utils import write_sdf
    >>> p = os.path.join(tempfile.mkdtemp(), "m.sdf")
    >>> write_sdf([Chem.MolFromSmiles("CCO")], p, {"pIC50": [7.1]})
    1

    References
    ----------
    - Dalby et al. (1992). "Description of several chemical structure file
      formats used by computer programs developed at Molecular Design Limited."
      J. Chem. Inf. Comput. Sci., 32(3), 244-255.
      https://doi.org/10.1021/ci00007a012
    """
    from rdkit import Chem

    mol_list = ensure_mol_list(mols)
    if properties:
        for key, values in properties.items():
            if len(values) != len(mol_list):
                raise ValueError(
                    f"Property {key!r} has {len(values)} values but "
                    f"{len(mol_list)} molecules were given."
                )

    writer = Chem.SDWriter(str(path))
    writer.SetKekulize(kekulize)
    written = 0
    try:
        for i, mol in enumerate(mol_list):
            if mol is None:
                continue
            if properties:
                mol = Chem.Mol(mol)
                for key, values in properties.items():
                    mol.SetProp(str(key), str(values[i]))
            writer.write(mol)
            written += 1
    finally:
        writer.close()
    return written


def mols_to_dataframe(
    mols: Iterable[Any],
    include_smiles: bool = True,
    include_properties: bool = True,
    smiles_column: str = "smiles",
    extra: Optional[Dict[str, Sequence[Any]]] = None,
) -> Any:
    """Flatten molecules and their RDKit properties into a ``pandas.DataFrame``.

    Parameters
    ----------
    mols : iterable of rdkit.Chem.Mol
        Molecules. ``None`` entries produce a row of missing values.
    include_smiles : bool, default True
        Add a canonical isomeric SMILES column.
    include_properties : bool, default True
        Add one column per RDKit molecule property found across the input
        (the union of all property names; missing values become ``None``).
    smiles_column : str, default ``"smiles"``
        Name of the SMILES column.
    extra : dict of str -> sequence, optional
        Additional aligned columns, e.g. measured activities.

    Returns
    -------
    pandas.DataFrame
        One row per input molecule, in input order.

    Raises
    ------
    ValueError
        If an ``extra`` sequence length does not match the molecule count.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.utils import mols_to_dataframe
    >>> df = mols_to_dataframe([Chem.MolFromSmiles("CCO")], extra={"y": [1.0]})
    >>> list(df.columns)
    ['smiles', 'y']

    References
    ----------
    - McKinney (2010). "Data Structures for Statistical Computing in Python."
      Proc. 9th Python in Science Conf., 56-61.
      https://doi.org/10.25080/Majora-92bf1922-00a
    - RDKit documentation: https://www.rdkit.org/docs/
    """
    import pandas as pd
    from rdkit import Chem

    mol_list = ensure_mol_list(mols)
    if extra:
        for key, values in extra.items():
            if len(values) != len(mol_list):
                raise ValueError(
                    f"extra[{key!r}] has {len(values)} values but "
                    f"{len(mol_list)} molecules were given."
                )

    data: Dict[str, List[Any]] = {}
    if include_smiles:
        data[smiles_column] = [
            None if m is None else Chem.MolToSmiles(m) for m in mol_list
        ]
    if include_properties:
        prop_names: List[str] = []
        for m in mol_list:
            if m is None:
                continue
            for name in m.GetPropNames():
                if name not in prop_names:
                    prop_names.append(name)
        for name in prop_names:
            data[name] = [
                m.GetProp(name) if (m is not None and m.HasProp(name)) else None
                for m in mol_list
            ]
    if extra:
        for key, values in extra.items():
            data[str(key)] = list(values)
    return pd.DataFrame(data, index=range(len(mol_list)))


def dataframe_to_mols(
    df: Any,
    smiles_column: str = "smiles",
    name_column: Optional[str] = None,
    property_columns: Optional[Sequence[str]] = None,
    sanitize: bool = True,
    on_error: str = "none",
) -> List[Any]:
    """Build RDKit molecules from a SMILES column of a ``pandas.DataFrame``.

    Parameters
    ----------
    df : pandas.DataFrame
        Source table.
    smiles_column : str, default ``"smiles"``
        Column holding the SMILES strings.
    name_column : str, optional
        Column copied onto each molecule as its ``_Name`` property.
    property_columns : sequence of str, optional
        Columns copied onto each molecule as RDKit properties (stringified).
    sanitize : bool, default True
        Run RDKit sanitization on parsing.
    on_error : {"skip", "none", "raise"}, default ``"none"``
        Handling of unparsable SMILES. ``"none"`` (the default here) keeps
        positional alignment with the DataFrame rows.

    Returns
    -------
    list of rdkit.Chem.Mol
        The parsed molecules.

    Raises
    ------
    KeyError
        If ``smiles_column`` is not a column of ``df``.
    InvalidMoleculeError
        If ``on_error="raise"`` and a SMILES cannot be parsed.

    Examples
    --------
    >>> import pandas as pd
    >>> from qsarkit.utils import dataframe_to_mols
    >>> df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
    >>> len(dataframe_to_mols(df))
    2

    References
    ----------
    - Weininger (1988). "SMILES, a chemical language and information system. 1."
      J. Chem. Inf. Comput. Sci., 28(1), 31-36.
      https://doi.org/10.1021/ci00057a005
    - RDKit documentation: https://www.rdkit.org/docs/
    """
    if on_error not in ("skip", "none", "raise"):
        raise ValueError(
            f"on_error must be 'skip', 'none' or 'raise', got {on_error!r}."
        )
    if smiles_column not in df.columns:
        raise KeyError(
            f"Column {smiles_column!r} not found; available: {list(df.columns)}"
        )

    mols: List[Any] = []
    for idx, row in df.iterrows():
        smiles = row[smiles_column]
        mol = None if smiles is None else _mol_from_smiles(str(smiles), sanitize)
        if mol is None:
            if on_error == "raise":
                raise InvalidMoleculeError(
                    f"Row {idx!r}: RDKit could not parse SMILES {smiles!r}."
                )
            if on_error == "none":
                mols.append(None)
            continue
        if name_column is not None and name_column in df.columns:
            mol.SetProp("_Name", str(row[name_column]))
        for col in property_columns or ():
            if col in df.columns:
                mol.SetProp(str(col), str(row[col]))
        mols.append(mol)
    return mols


def read_csv_mols(
    path: PathLike,
    smiles_column: str = "smiles",
    activity_column: Optional[str] = None,
    sanitize: bool = True,
    on_error: str = "none",
    **read_csv_kwargs: Any,
) -> Any:
    """Read a CSV of structures, returning ``(mols, y, dataframe)``.

    Parameters
    ----------
    path : path-like
        CSV file path.
    smiles_column : str, default ``"smiles"``
        Column holding SMILES strings.
    activity_column : str, optional
        Column holding the target values. When ``None`` the returned ``y`` is
        ``None``.
    sanitize : bool, default True
        Run RDKit sanitization on parsing.
    on_error : {"skip", "none", "raise"}, default ``"none"``
        Handling of unparsable SMILES. ``"none"`` keeps alignment between
        ``mols``, ``y`` and the DataFrame rows.
    **read_csv_kwargs
        Forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    mols : list of rdkit.Chem.Mol
        Parsed molecules.
    y : numpy.ndarray or None
        Target values when ``activity_column`` is given.
    df : pandas.DataFrame
        The raw table as read.

    Examples
    --------
    >>> import tempfile, os, pandas as pd
    >>> from qsarkit.utils import read_csv_mols
    >>> p = os.path.join(tempfile.mkdtemp(), "d.csv")
    >>> pd.DataFrame({"smiles": ["CCO"], "y": [1.0]}).to_csv(p, index=False)
    >>> mols, y, df = read_csv_mols(p, activity_column="y")
    >>> len(mols), float(y[0])
    (1, 1.0)

    References
    ----------
    - McKinney (2010). "Data Structures for Statistical Computing in Python."
      Proc. 9th Python in Science Conf., 56-61.
      https://doi.org/10.25080/Majora-92bf1922-00a
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path, **read_csv_kwargs)
    mols = dataframe_to_mols(
        df, smiles_column=smiles_column, sanitize=sanitize, on_error=on_error
    )
    y = None
    if activity_column is not None:
        if activity_column not in df.columns:
            raise KeyError(
                f"Column {activity_column!r} not found; available: {list(df.columns)}"
            )
        y = np.asarray(df[activity_column].to_numpy(), dtype=float)
        if on_error == "skip" and len(y) != len(mols):
            raise ValueError(
                "on_error='skip' drops rows and breaks alignment with the "
                "activity column; use on_error='none' instead."
            )
    return mols, y, df


__all__ = [
    "read_smiles",
    "write_smiles",
    "read_sdf",
    "write_sdf",
    "mols_to_dataframe",
    "dataframe_to_mols",
    "read_csv_mols",
]
