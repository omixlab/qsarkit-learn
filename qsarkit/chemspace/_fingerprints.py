"""Shared fingerprint / similarity primitives used across the analysis modules.

This module is deliberately small and dependency-light: it wraps RDKit's
:mod:`rdkit.Chem.rdFingerprintGenerator` so that every chemical-space,
applicability-domain and SAR component computes similarities the same way
(same default ECFP4 parameters, same Tanimoto definition). Higher-level
packages (``qsarkit.sar``, ``qsarkit.applicability``) import from here
rather than each rolling their own fingerprint call.

References
----------
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  J. Chem. Inf. Model., 50(5), 742-754. https://doi.org/10.1021/ci100050t
- Tanimoto, T. T. (1958). "An Elementary Mathematical Theory of
  Classification and Prediction." IBM Internal Report.
- RDKit ``rdFingerprintGenerator`` documentation:
  https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

#: Default ECFP4 settings (Morgan radius 2 folded to 2048 bits), the de-facto
#: standard for activity-cliff and matched-pair analyses.
DEFAULT_RADIUS = 2
DEFAULT_N_BITS = 2048


def morgan_generator(radius: int = DEFAULT_RADIUS, n_bits: int = DEFAULT_N_BITS) -> Any:
    """Return a configured RDKit Morgan fingerprint generator.

    Parameters
    ----------
    radius : int, default 2
        Morgan radius. ``radius=2`` corresponds to ECFP4.
    n_bits : int, default 2048
        Folded bit-vector length.

    Returns
    -------
    rdkit.Chem.rdFingerprintGenerator.FingerprintGenerator64
        A generator object exposing ``GetFingerprint``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> gen = morgan_generator()
    >>> fp = gen.GetFingerprint(Chem.MolFromSmiles("c1ccccc1"))
    >>> fp.GetNumBits()
    2048

    References
    ----------
    - Rogers, D. & Hahn, M. (2010). J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    """
    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


def compute_fingerprints(
    mols: Sequence[Any],
    radius: int = DEFAULT_RADIUS,
    n_bits: int = DEFAULT_N_BITS,
) -> List[Any]:
    """Compute ECFP bit vectors for a sequence of molecules.

    Parameters
    ----------
    mols : sequence of rdkit.Chem.Mol
        Input molecules. ``None`` entries are not allowed.
    radius : int, default 2
        Morgan radius.
    n_bits : int, default 2048
        Folded bit-vector length.

    Returns
    -------
    list of rdkit.DataStructs.ExplicitBitVect
        One fingerprint per input molecule, positionally aligned.

    Examples
    --------
    >>> from rdkit import Chem
    >>> fps = compute_fingerprints([Chem.MolFromSmiles("CCO")])
    >>> len(fps)
    1

    References
    ----------
    - Rogers, D. & Hahn, M. (2010). J. Chem. Inf. Model., 50(5), 742-754.
      https://doi.org/10.1021/ci100050t
    """
    gen = morgan_generator(radius=radius, n_bits=n_bits)
    return [gen.GetFingerprint(m) for m in mols]


def fingerprints_to_array(fps: Sequence[Any]) -> np.ndarray:
    """Convert a list of RDKit bit vectors into a dense ``(n, n_bits)`` array.

    Parameters
    ----------
    fps : sequence of ExplicitBitVect
        Fingerprints to densify.

    Returns
    -------
    numpy.ndarray of shape (n_molecules, n_bits), dtype uint8
        The dense binary matrix.

    Examples
    --------
    >>> from rdkit import Chem
    >>> arr = fingerprints_to_array(compute_fingerprints([Chem.MolFromSmiles("CCO")]))
    >>> arr.shape
    (1, 2048)

    References
    ----------
    - RDKit ``DataStructs.ConvertToNumpyArray`` documentation:
      https://www.rdkit.org/docs/source/rdkit.DataStructs.cDataStructs.html
    """
    from rdkit import DataStructs

    if not fps:
        return np.zeros((0, 0), dtype=np.uint8)
    out = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for i, fp in enumerate(fps):
        row = np.zeros((fps[0].GetNumBits(),), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        out[i] = row
    return out


def tanimoto_matrix(fps: Sequence[Any], other: Sequence[Any] | None = None) -> np.ndarray:
    """Full pairwise Tanimoto similarity matrix.

    Parameters
    ----------
    fps : sequence of ExplicitBitVect
        Query fingerprints (rows of the output).
    other : sequence of ExplicitBitVect, optional
        Reference fingerprints (columns). Defaults to ``fps`` itself, in
        which case the returned matrix is symmetric with a unit diagonal.

    Returns
    -------
    numpy.ndarray of shape (len(fps), len(other))
        Tanimoto (Jaccard) similarities in ``[0, 1]``.

    Examples
    --------
    >>> from rdkit import Chem
    >>> fps = compute_fingerprints([Chem.MolFromSmiles(s) for s in ("CCO", "CCO")])
    >>> float(tanimoto_matrix(fps)[0, 1])
    1.0

    References
    ----------
    - Tanimoto, T. T. (1958). IBM Internal Report.
    - Bajusz, D. et al. (2015). "Why is Tanimoto index an appropriate
      choice for fingerprint-based similarity calculations?"
      J. Cheminform., 7, 20. https://doi.org/10.1186/s13321-015-0069-3
    """
    from rdkit import DataStructs

    ref = list(fps) if other is None else list(other)
    if not fps or not ref:
        return np.zeros((len(fps), len(ref)), dtype=float)
    return np.asarray(
        [DataStructs.BulkTanimotoSimilarity(fp, ref) for fp in fps], dtype=float
    )


def bemis_murcko_smiles(mol: Any, generic: bool = False) -> str:
    """Canonical SMILES of a molecule's Bemis-Murcko scaffold.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Input molecule.
    generic : bool, default False
        If ``True``, strip element and bond-order information to obtain the
        cyclic skeleton ("graph framework").

    Returns
    -------
    str
        Canonical scaffold SMILES (empty string for acyclic molecules).

    Examples
    --------
    >>> from rdkit import Chem
    >>> bemis_murcko_smiles(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    'c1ccccc1'

    References
    ----------
    - Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs.
      1. Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
      https://doi.org/10.1021/jm9602928
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if generic:
        scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        Chem.SanitizeMol(scaffold)
    return Chem.MolToSmiles(scaffold)
