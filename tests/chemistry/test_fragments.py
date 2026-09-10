from __future__ import annotations

from rdkit import Chem

from qsarkit.chemistry import CoreExtractor, FragmentRemover


def canon(mol):
    return Chem.MolToSmiles(mol)


def test_removes_boc_protecting_group(mols):
    out = FragmentRemover().transform([mols["boc_benzylamine"]])[0]
    assert canon(out) == canon(Chem.MolFromSmiles("NCc1ccccc1"))


def test_removes_cbz_protecting_group():
    cbz = Chem.MolFromSmiles("O=C(OCc1ccccc1)NCCO")
    out = FragmentRemover().transform([cbz])[0]
    assert "NCCO" in canon(out)


def test_removes_two_boc_groups_iteratively():
    di_boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)NCCNC(=O)OC(C)(C)C")
    out = FragmentRemover().transform([di_boc])[0]
    assert canon(out) == canon(Chem.MolFromSmiles("NCCN"))


def test_leaves_unprotected_molecule_untouched(mols):
    out = FragmentRemover().transform([mols["benzene"]])[0]
    assert canon(out) == "c1ccccc1"


def test_custom_group_table():
    remover = FragmentRemover(groups={"nitro": "[N+](=O)[O-]"})
    out = remover.transform([Chem.MolFromSmiles("O=[N+]([O-])c1ccccc1")])[0]
    assert canon(out) == "c1ccccc1"


def test_bemis_murcko_scaffold_of_aspirin(mols):
    assert canon(CoreExtractor().bemis_murcko(mols["aspirin"])) == "c1ccccc1"


def test_bemis_murcko_keeps_linker_between_rings():
    biphenyl_ether = Chem.MolFromSmiles("c1ccccc1OCc1ccccc1")
    scaffold = CoreExtractor().bemis_murcko(biphenyl_ether)
    assert scaffold.GetNumAtoms() == 14


def test_generic_scaffold_strips_elements(mols):
    generic = CoreExtractor().bemis_murcko(mols["caffeine"], generic=True)
    assert {a.GetSymbol() for a in generic.GetAtoms()} == {"C"}


def test_mcs_finds_shared_core():
    mols_ = [Chem.MolFromSmiles(s) for s in ("c1ccccc1CCO", "c1ccccc1CCN")]
    mcs = CoreExtractor().mcs(mols_)
    assert mcs is not None
    assert all(m.HasSubstructMatch(mcs) for m in mols_)
    assert mcs.GetNumAtoms() == 8


def test_mcs_returns_none_for_disjoint_molecules():
    disjoint = [Chem.MolFromSmiles("[Na+]"), Chem.MolFromSmiles("[Cl-]")]
    assert CoreExtractor().mcs(disjoint) is None


def test_medchem_core_drops_exocyclic_stubs(mols):
    core = CoreExtractor().medchem_core(mols["ibuprofen"])
    assert all(a.IsInRing() or a.GetDegree() > 1 for a in core.GetAtoms())


def test_transform_batch_handles_none(mols):
    out = CoreExtractor().transform([mols["aspirin"], None])
    assert out[1] is None
    assert canon(out[0]) == "c1ccccc1"
