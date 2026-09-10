from __future__ import annotations

import pytest
from rdkit import Chem

from qsarkit.base import InvalidMoleculeError
from qsarkit.chemistry import MolecularStandardizer


def canon(mol):
    return Chem.MolToSmiles(mol)


def test_removes_salt_counterion():
    std = MolecularStandardizer()
    mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
    out = std.transform([mol])[0]
    assert canon(out) == canon(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))


def test_neutralizes_charges():
    std = MolecularStandardizer(remove_salts=False, normalize_tautomers=False)
    out = std.transform([Chem.MolFromSmiles("CC(=O)[O-]")])[0]
    assert Chem.GetFormalCharge(out) == 0


def test_permanent_charge_is_preserved():
    std = MolecularStandardizer()
    out = std.transform([Chem.MolFromSmiles("C[N+](C)(C)C.[Cl-]")])[0]
    assert Chem.GetFormalCharge(out) == 1


def test_stereochemistry_removed_when_requested():
    mol = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
    kept = MolecularStandardizer(handle_stereochemistry="retain").transform([mol])[0]
    dropped = MolecularStandardizer(handle_stereochemistry="remove").transform([mol])[0]
    assert "@" in canon(kept)
    assert "@" not in canon(dropped)


def test_batch_preserves_positions_and_none_entries():
    std = MolecularStandardizer()
    out = std.transform([Chem.MolFromSmiles("CCO"), None, Chem.MolFromSmiles("c1ccccc1")])
    assert len(out) == 3
    assert out[1] is None
    assert canon(out[0]) == "CCO"


def test_on_error_raise_propagates():
    bad = Chem.MolFromSmiles("c1ccccc1", sanitize=False)
    bad.GetAtomWithIdx(0).SetNumExplicitHs(5)
    with pytest.raises(InvalidMoleculeError):
        MolecularStandardizer(on_error="raise").transform([bad])


def test_on_error_none_yields_none():
    bad = Chem.MolFromSmiles("c1ccccc1", sanitize=False)
    bad.GetAtomWithIdx(0).SetNumExplicitHs(5)
    assert MolecularStandardizer(on_error="none").transform([bad])[0] is None


def test_invalid_input_type_rejected():
    with pytest.raises(InvalidMoleculeError):
        MolecularStandardizer().transform(["CCO"])


def test_sklearn_params_roundtrip():
    from sklearn.base import clone

    std = MolecularStandardizer(remove_salts=False, on_error="raise")
    assert clone(std).get_params() == std.get_params()


def test_idempotent():
    std = MolecularStandardizer()
    once = std.transform([Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")])[0]
    twice = std.transform([once])[0]
    assert canon(once) == canon(twice)
