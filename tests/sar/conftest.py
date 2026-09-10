"""Fixtures for SAR tests: a congeneric series with one designed activity cliff."""

from __future__ import annotations

import pytest

# Para-substituted benzoic acids. Every member is a one-atom substituent
# change from every other, so the whole set is a matched-pair network.
# The 4-amino analogue is deliberately ~3.5 log units more potent than
# the rest, making it the single activity-cliff-forming compound.
SERIES_SMILES = [
    "O=C(O)c1ccc(C)cc1",   # 0  4-methyl
    "O=C(O)c1ccc(Cl)cc1",  # 1  4-chloro
    "O=C(O)c1ccc(Br)cc1",  # 2  4-bromo
    "O=C(O)c1ccc(F)cc1",   # 3  4-fluoro
    "O=C(O)c1ccc(N)cc1",   # 4  4-amino   <- the cliff
    "O=C(O)c1ccc(O)cc1",   # 5  4-hydroxy
]
SERIES_ACTIVITIES = [5.0, 5.2, 5.1, 4.9, 8.5, 5.3]
CLIFF_INDEX = 4


@pytest.fixture(scope="module")
def series_smiles():
    return list(SERIES_SMILES)


@pytest.fixture(scope="module")
def series_mols():
    from rdkit import Chem

    return [Chem.MolFromSmiles(s) for s in SERIES_SMILES]


@pytest.fixture(scope="module")
def series_activities():
    return list(SERIES_ACTIVITIES)
