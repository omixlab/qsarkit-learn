"""Shared pytest fixtures and markers for the qsarkit test suite."""

from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (deselected by default)")


# Small, well-known molecules reused across the suite.
SMILES = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "benzene": "c1ccccc1",
    "ethanol": "CCO",
    "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "ibuprofen": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "quercetin": "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
    "quercetin_3_glucoside": (
        "OC[C@H]1O[C@@H](Oc2c(-c3ccc(O)c(O)c3)oc3cc(O)cc(O)c3c2=O)"
        "[C@H](O)[C@@H](O)[C@@H]1O"
    ),
    "glucose": "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    "phenyl_glucoside": "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O",
    "tetrahydropyran": "C1CCOCC1",
    "cyclohexane": "C1CCCCC1",
    "boc_benzylamine": "CC(C)(C)OC(=O)NCc1ccccc1",
    "ethyl_acetate": "CC(=O)OCC",
    "anisole": "COc1ccccc1",
}


@pytest.fixture(scope="session")
def smiles():
    """Mapping of common-name -> SMILES for the shared reference molecules."""
    return dict(SMILES)


@pytest.fixture(scope="session")
def mols(smiles):
    """Mapping of common-name -> RDKit Mol for the shared reference molecules."""
    from rdkit import Chem

    return {name: Chem.MolFromSmiles(smi) for name, smi in smiles.items()}
