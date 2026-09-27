"""Draw an activity cliff: the two structures, with what differs highlighted.

A SAS map says how many cliffs a dataset contains and where they sit; it
cannot say *what changed*. That question is answered by putting the two
structures side by side with their common core greyed out and the differing
atoms picked out, which is the form a medicinal chemist reads directly.

The common core is the maximum common substructure, so the highlight is the
part of each molecule the other one lacks. For a pair the fingerprint cannot
tell apart at all -- stereoisomers, or homologues differing only in chain
length -- the MCS covers everything and nothing is highlighted, which is
itself the finding: the difference is invisible to a topological
representation, and no model reading one can separate the pair.

References
----------
- Stumpfe, D. & Bajorath, J. (2012). "Exploring Activity Cliffs in Medicinal
  Chemistry." J. Med. Chem., 55(7), 2932-2942.
  https://doi.org/10.1021/jm201706b
- Wassermann, A. M., Wawer, M. & Bajorath, J. (2010). "Activity Landscape
  Representations for Structure-Activity Relationship Analysis."
  J. Med. Chem., 53(23), 8209-8223. https://doi.org/10.1021/jm100933w
"""

from __future__ import annotations

from typing import Any, List, Literal, Optional, Sequence, Tuple

import numpy as np

__all__ = ["cliff_difference_atoms", "draw_activity_cliff", "draw_activity_cliffs"]

_CORE_COLOUR = (0.85, 0.85, 0.85)
_GAINED_COLOUR = (0.30, 0.69, 0.49)
_LOST_COLOUR = (0.90, 0.35, 0.35)


def cliff_difference_atoms(
    mol_a: Any, mol_b: Any, timeout: int = 5
) -> Tuple[List[int], List[int]]:
    """Atoms of each molecule that lie outside their common core.

    Parameters
    ----------
    mol_a, mol_b : Mol
    timeout : int, default 5
        Seconds allowed for the maximum-common-substructure search. The MCS
        problem is NP-hard, so a pathological pair is cut off rather than
        allowed to hang; on a timeout every atom is reported as differing,
        which is honest about having failed to find the core.

    Returns
    -------
    (list of int, list of int)
        Indices into ``mol_a`` and ``mol_b``. Both are empty when the two
        molecules are identical under the MCS -- the case where a topological
        representation cannot distinguish them.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.sar import cliff_difference_atoms
    >>> a = Chem.MolFromSmiles("c1ccccc1O")
    >>> b = Chem.MolFromSmiles("c1ccccc1N")
    >>> in_a, in_b = cliff_difference_atoms(a, b)
    >>> len(in_a), len(in_b)
    (1, 1)
    """
    from rdkit import Chem
    from rdkit.Chem import rdFMCS

    result = rdFMCS.FindMCS(
        [mol_a, mol_b],
        timeout=timeout,
        ringMatchesRingOnly=True,
        completeRingsOnly=False,
    )
    if not result.smartsString:
        return list(range(mol_a.GetNumAtoms())), list(range(mol_b.GetNumAtoms()))

    core = Chem.MolFromSmarts(result.smartsString)
    if core is None:  # pragma: no cover - malformed SMARTS from RDKit
        return list(range(mol_a.GetNumAtoms())), list(range(mol_b.GetNumAtoms()))

    def _outside(mol: Any) -> List[int]:
        match = mol.GetSubstructMatch(core)
        matched = set(match)
        return [i for i in range(mol.GetNumAtoms()) if i not in matched]

    return _outside(mol_a), _outside(mol_b)


def _draw_pair(
    mol_a: Any,
    mol_b: Any,
    legends: Sequence[str],
    size: Tuple[int, int],
    fmt: str,
    timeout: int,
) -> Any:
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem.Draw import rdMolDraw2D

    diff_a, diff_b = cliff_difference_atoms(mol_a, mol_b, timeout=timeout)

    copies = []
    highlights: List[List[int]] = []
    colours: List[dict] = []
    for mol, differing, colour in (
        (mol_a, diff_a, _LOST_COLOUR),
        (mol_b, diff_b, _GAINED_COLOUR),
    ):
        copy = Chem.Mol(mol)
        Chem.rdDepictor.Compute2DCoords(copy)
        copies.append(copy)
        highlights.append(list(differing))
        colours.append({int(i): colour for i in differing})

    width, height = size
    drawer = (
        rdMolDraw2D.MolDraw2DSVG(width * 2, height, width, height)
        if fmt == "svg"
        else rdMolDraw2D.MolDraw2DCairo(width * 2, height, width, height)
    )
    # The bundled RDKit stubs type drawOptions() too loosely to accept these
    # assignments; see the note on typings/ in the README.
    options: Any = drawer.drawOptions()
    options.legendFontSize = 15
    options.highlightBondWidthMultiplier = 16

    drawer.DrawMolecules(
        copies,
        legends=list(legends),
        highlightAtoms=highlights,
        highlightAtomColors=colours,
    )
    drawer.FinishDrawing()
    text = drawer.GetDrawingText()
    return text if fmt == "svg" else text


def draw_activity_cliff(
    cliff: Any = None,
    mol_a: Any = None,
    mol_b: Any = None,
    activity_a: Optional[float] = None,
    activity_b: Optional[float] = None,
    similarity: Optional[float] = None,
    size: Tuple[int, int] = (360, 300),
    fmt: Literal["svg", "png"] = "svg",
    timeout: int = 5,
) -> Any:
    """Render one activity cliff as a labelled pair of structures.

    Pass either an :class:`~qsarkit.sar.ActivityCliff` as ``cliff``, or the
    two molecules and their activities directly.

    Parameters
    ----------
    cliff : ActivityCliff, optional
        As returned by :class:`~qsarkit.sar.ActivityCliffDetector`.
    mol_a, mol_b : Mol, optional
        Used when ``cliff`` is not given.
    activity_a, activity_b : float, optional
        Shown in the legends.
    similarity : float, optional
        Shown in the legend of the second structure.
    size : tuple of int, default (360, 300)
        Size of *each* panel; the image is twice this wide.
    fmt : {"svg", "png"}, default "svg"
    timeout : int, default 5
        Seconds for the MCS search.

    Returns
    -------
    str or bytes
        SVG text, or PNG bytes.

    Raises
    ------
    ValueError
        If neither a cliff nor a pair of molecules is given.

    Examples
    --------
    >>> from rdkit import Chem
    >>> from qsarkit.sar import draw_activity_cliff
    >>> a, b = Chem.MolFromSmiles("c1ccccc1O"), Chem.MolFromSmiles("c1ccccc1N")
    >>> svg = draw_activity_cliff(
    ...     mol_a=a, mol_b=b, activity_a=8.1, activity_b=5.2, similarity=0.86)
    >>> svg.startswith("<?xml") or svg.lstrip().startswith("<svg")
    True
    """
    if cliff is not None:
        mol_a = getattr(cliff, "mol_a", mol_a)
        mol_b = getattr(cliff, "mol_b", mol_b)
        activity_a = getattr(cliff, "activity_a", activity_a)
        activity_b = getattr(cliff, "activity_b", activity_b)
        similarity = getattr(cliff, "similarity", similarity)
    if mol_a is None or mol_b is None:
        raise ValueError(
            "Pass either `cliff=` an ActivityCliff, or both `mol_a=` and "
            "`mol_b=`."
        )

    left = "A" if activity_a is None else f"A: activity {activity_a:g}"
    right = "B" if activity_b is None else f"B: activity {activity_b:g}"
    if activity_a is not None and activity_b is not None:
        # "delta", not the Greek letter: RDKit's drawing font has no glyph for
        # it and silently renders a blank.
        right += f"  (delta {abs(float(activity_b) - float(activity_a)):g})"
    if similarity is not None:
        right += f"  sim {float(similarity):.2f}"

    return _draw_pair(mol_a, mol_b, (left, right), size, fmt, timeout)


def draw_activity_cliffs(
    cliffs: Sequence[Any],
    n: int = 4,
    size: Tuple[int, int] = (320, 260),
    fmt: Literal["svg", "png"] = "svg",
    timeout: int = 5,
) -> List[Any]:
    """Render the first ``n`` cliffs, each as its own image.

    Returned separately rather than tiled into one grid, so a notebook can
    display them inline and a report can lay them out as it likes.

    Parameters
    ----------
    cliffs : sequence of ActivityCliff
    n : int, default 4
    size : tuple of int, default (320, 260)
    fmt : {"svg", "png"}, default "svg"
    timeout : int, default 5

    Returns
    -------
    list
        One drawing per cliff, at most ``n``.

    Examples
    --------
    >>> import numpy as np
    >>> from rdkit import Chem
    >>> from qsarkit.sar import ActivityCliffDetector, draw_activity_cliffs
    >>> smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl"]
    >>> mols = [Chem.MolFromSmiles(s) for s in smiles]
    >>> cliffs = ActivityCliffDetector(
    ...     similarity_threshold=0.3, activity_threshold=1.0
    ... ).detect(mols, [8.0, 5.0, 8.2, 5.1])
    >>> drawings = draw_activity_cliffs(cliffs, n=2)
    >>> len(drawings) <= 2
    True
    """
    return [
        draw_activity_cliff(cliff=c, size=size, fmt=fmt, timeout=timeout)
        for c in list(cliffs)[:n]
    ]
