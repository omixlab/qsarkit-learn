"""A small, deterministic dataset shared by every example in the docs.

Every code example in this documentation is a doctest that is executed by
the test suite (``pytest tests/docs``) and by ``make doctest``. They all
start from the objects defined here, so the examples stay short and, more
importantly, cannot drift away from the API they document.

The dataset is 24 compounds in four substituent series with synthetic pIC50
values, resolving to three Bemis-Murcko scaffolds (the benzoic acids and the
anilides share a benzene framework once their acyclic side chains are
stripped -- which is exactly the behaviour a scaffold split relies on). It is
deliberately tiny: examples should run in milliseconds, and every number
printed in the docs is reproducible.
"""

from __future__ import annotations

from typing import List

import numpy as np
from rdkit import Chem, RDLogger

# Doc examples parse SMILES on purpose (including one invalid string, to
# demonstrate curation); RDKit's warnings would otherwise pollute the output.
RDLogger.DisableLog("rdApp.*")

#: Four series x six substituent variations. Enough structure for a
#: scaffold split, a Butina clustering and a k-NN applicability domain,
#: small enough that every example runs instantly.
DEMO_SMILES: List[str] = [
    # benzoic acids
    "OC(=O)c1ccccc1",
    "OC(=O)c1ccc(C)cc1",
    "OC(=O)c1ccc(Cl)cc1",
    "OC(=O)c1ccc(Br)cc1",
    "OC(=O)c1ccc(OC)cc1",
    "OC(=O)c1ccc(N)cc1",
    # anilides
    "CC(=O)Nc1ccccc1",
    "CC(=O)Nc1ccc(C)cc1",
    "CC(=O)Nc1ccc(Cl)cc1",
    "CC(=O)Nc1ccc(F)cc1",
    "CC(=O)Nc1ccc(OC)cc1",
    "CC(=O)Nc1ccc(O)cc1",
    # pyridine carboxamides
    "NC(=O)c1ccncc1",
    "NC(=O)c1ccc(C)nc1",
    "NC(=O)c1ccc(Cl)nc1",
    "NC(=O)c1ccc(OC)nc1",
    "CNC(=O)c1ccncc1",
    "CCNC(=O)c1ccncc1",
    # benzimidazoles
    "c1ccc2[nH]cnc2c1",
    "Cc1ccc2[nH]cnc2c1",
    "Clc1ccc2[nH]cnc2c1",
    "COc1ccc2[nH]cnc2c1",
    "Cn1cnc2ccccc21",
    "CCn1cnc2ccccc21",
]

#: Synthetic pIC50 values, one per entry of :data:`DEMO_SMILES`.
#:
#: Constructed so that the four scaffold families sit at different potency
#: levels and one pair (indices 2 and 3, the 4-Cl and 4-Br benzoic acids)
#: forms a deliberate activity cliff: near-identical structures, 2.4 log
#: units apart.
DEMO_Y: "np.ndarray" = np.array(
    [
        5.10, 5.35, 7.80, 5.40, 5.05, 4.90,   # benzoic acids
        6.20, 6.45, 6.70, 6.55, 6.10, 6.05,   # anilides
        7.10, 7.35, 7.55, 7.20, 7.05, 6.95,   # pyridine carboxamides
        8.00, 8.25, 8.45, 8.10, 7.90, 7.85,   # benzimidazoles
    ]
)

#: Binary labels derived from :data:`DEMO_Y` at a pIC50 cutoff of 7.
DEMO_LABELS: "np.ndarray" = (DEMO_Y >= 7.0).astype(int)

#: The parsed molecules, in the same order as :data:`DEMO_SMILES`.
demo_mols = [Chem.MolFromSmiles(smi) for smi in DEMO_SMILES]

#: A handful of well-known molecules referenced by name in the examples.
NAMED_SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "aspirin_sodium": "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",
    "benzene": "c1ccccc1",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ethanol": "CCO",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "quercetin_3_glucoside": (
        "OC[C@H]1O[C@@H](Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    ),
}

#: Parsed counterparts of :data:`NAMED_SMILES`.
named_mols = {name: Chem.MolFromSmiles(smi) for name, smi in NAMED_SMILES.items()}


def demo_fingerprints(n_bits: int = 512) -> "np.ndarray":
    """Morgan fingerprints of :data:`demo_mols` as a dense ``(24, n_bits)`` array.

    Parameters
    ----------
    n_bits : int, default 512
        Fingerprint length. Kept small so examples stay fast.

    Returns
    -------
    numpy.ndarray
        Binary feature matrix of shape ``(24, n_bits)``.
    """
    from qsarkit.representation import MorganFingerprint

    return MorganFingerprint(radius=2, n_bits=n_bits).transform(demo_mols)


def demo_descriptors() -> "np.ndarray":
    """Physicochemical descriptors of :data:`demo_mols` as a dense array.

    Returns
    -------
    numpy.ndarray
        Continuous feature matrix with one row per demo molecule.
    """
    from qsarkit.representation import PhysicochemicalDescriptors

    return PhysicochemicalDescriptors().transform(demo_mols)


__all__ = [
    "DEMO_SMILES",
    "DEMO_Y",
    "DEMO_LABELS",
    "demo_mols",
    "NAMED_SMILES",
    "named_mols",
    "demo_fingerprints",
    "demo_descriptors",
]
