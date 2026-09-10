from __future__ import annotations

import pytest
from rdkit import Chem

from qsarkit.chemistry import GlycanDescriptors, GlycanDetector, GlycanRemover


def canon(mol):
    return Chem.MolToSmiles(mol)


def test_detects_single_sugar_in_glycoside(mols):
    det = GlycanDetector().detect(mols["phenyl_glucoside"])
    assert det["num_sugar_residues"] == 1
    assert len(det["glycan_atoms"]) > 5
    assert 0.0 < det["glycan_mw_fraction"] < 1.0


def test_free_glucose_is_one_sugar(mols):
    assert GlycanDetector().detect(mols["glucose"])["num_sugar_residues"] == 1


@pytest.mark.parametrize("name", ["benzene", "cyclohexane", "tetrahydropyran", "aspirin"])
def test_non_sugars_are_not_flagged(mols, name):
    det = GlycanDetector().detect(mols[name])
    assert det["num_sugar_residues"] == 0
    assert det["glycan_atoms"] == []
    assert det["glycan_mw_fraction"] == 0.0


def test_disaccharide_detects_two_rings():
    sucrose = Chem.MolFromSmiles(
        "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)"
        "[C@@H](O)[C@@H]1O"
    )
    assert GlycanDetector().detect(sucrose)["num_sugar_residues"] == 2


def test_removes_glucose_to_give_quercetin(mols):
    result = GlycanRemover().remove(mols["quercetin_3_glucoside"])
    assert canon(result["aglycone"]) == canon(
        Chem.MolFromSmiles("O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12")
    )
    assert len(result["removed_fragments"]) == 1
    assert "O" in canon(result["removed_fragments"][0])


def test_remover_returns_contract_keys(mols):
    result = GlycanRemover().remove(mols["phenyl_glucoside"])
    assert set(result) == {"original", "aglycone", "removed_fragments"}
    assert result["original"] is mols["phenyl_glucoside"]
    assert canon(result["aglycone"]) == "c1ccccc1"


def test_remover_is_noop_on_non_glycoside(mols):
    result = GlycanRemover().remove(mols["aspirin"])
    assert result["removed_fragments"] == []
    assert canon(result["aglycone"]) == canon(mols["aspirin"])


def test_descriptors_columns_and_values(mols):
    df = GlycanDescriptors().transform(
        [mols["quercetin_3_glucoside"], mols["benzene"]]
    )
    assert list(df.columns) == ["sugar_count", "glycan_fraction", "glycosylation_pattern"]
    assert df.loc[0, "sugar_count"] == 1
    assert df.loc[1, "sugar_count"] == 0
    assert df.loc[0, "glycan_fraction"] > df.loc[1, "glycan_fraction"]
    assert "6-ring" in df.loc[0, "glycosylation_pattern"]


def test_batch_handles_none(mols):
    assert GlycanDetector().transform([mols["glucose"], None])[1] is None
    assert GlycanRemover().transform([mols["glucose"], None])[1] is None
