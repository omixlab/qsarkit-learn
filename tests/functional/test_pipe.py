from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from qsarkit.functional import (
    PipeStep,
    MoleculeSet,
    Step,
    apply,
    balance,
    canonicalize_tautomers,
    deglycate,
    desalt,
    drop_if,
    drop_invalid,
    filter_by_property,
    keep_if,
    molecules,
    neutralize,
    pipeline,
    remove_duplicates,
    remove_protecting_groups,
    sample,
    shuffle,
    standardize,
    step,
    to_pactivity,
)

PHENYL_GLUCOSIDE = "OC[C@H]1O[C@@H](Oc2ccccc2)[C@H](O)[C@@H](O)[C@@H]1O"


def smis(mols):
    return [Chem.MolToSmiles(m) for m in mols]


class TestMoleculeSet:
    def test_parses_smiles_and_keeps_labels(self):
        ms = molecules(["CCO", "c1ccccc1"], [1.0, 2.0])
        assert len(ms) == 2
        assert ms.y.tolist() == [1.0, 2.0]

    def test_accepts_mol_objects(self):
        ms = molecules([Chem.MolFromSmiles("CCO")])
        assert ms.mols[0] is not None

    def test_bad_smiles_becomes_none_and_stays_aligned(self):
        ms = molecules(["CCO", "!!bad!!"], [1.0, 2.0])
        assert ms.mols[1] is None
        assert len(ms.y) == 2

    def test_unpacks_as_x_y(self):
        mols, y = molecules(["CCO"], [1.0])
        assert len(mols) == 1
        assert y.tolist() == [1.0]

    def test_unpacks_with_none_labels(self):
        mols, y = molecules(["CCO"])
        assert y is None

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="but there are"):
            MoleculeSet([Chem.MolFromSmiles("CCO")], [1.0, 2.0])

    def test_iter_mols_iterates_molecules(self):
        ms = molecules(["CCO", "CCN"])
        assert len(list(ms.iter_mols())) == 2

    def test_smiles_property_handles_none(self):
        assert molecules(["CCO", "!!bad!!"]).smiles == ["CCO", None]

    def test_to_frame(self):
        df = molecules(["CCO"], [1.0]).to_frame()
        assert list(df.columns) == ["smiles", "y"]
        assert df.loc[0, "smiles"] == "CCO"

    def test_to_frame_without_labels(self):
        assert list(molecules(["CCO"]).to_frame().columns) == ["smiles"]

    def test_repr_is_informative(self):
        text = repr(molecules(["CCO", "!!bad!!"], [1.0, 2.0]) >> desalt())
        assert "MoleculeSet" in text and "invalid" in text

    def test_history_records_each_step(self):
        ms = molecules(["CCO"]) >> desalt() >> drop_invalid()
        assert ms.history == ["molecules(n=1)", "desalt()", "drop_invalid()"]

    def test_history_records_parameters(self):
        ms = molecules(["CCO"]) >> sample(n=1, random_state=0)
        assert "n=1" in ms.history[-1]

    def test_steps_do_not_mutate_the_input(self):
        original = molecules(["CCO", "!!bad!!"])
        result = original >> drop_invalid()
        assert len(original) == 2
        assert len(result) == 1


class TestPipeOperators:
    def test_gt_is_rejected_with_an_explanation(self):
        with pytest.raises(TypeError, match="chained comparison"):
            molecules(["CCO"]) > desalt()

    def test_gt_on_a_step_is_also_rejected(self):
        with pytest.raises(TypeError, match="chained comparison|pipe operator"):
            desalt() > drop_invalid()

    def test_chained_gt_would_have_lost_data(self):
        # This is the whole reason `>` is refused: Python evaluates
        # `a > b > c` as `(a > b) and (b > c)`, so the pipeline's output
        # would be the comparison of the last two steps, not the data.
        class Probe:
            def __init__(self, name):
                self.name = name

            def __gt__(self, other):
                return Probe(f"({self.name}>{other.name})")

            def __bool__(self):
                return True

        assert (Probe("a") > Probe("b") > Probe("c")).name == "(b>c)"

    def test_rshift_chains_left_to_right(self):
        mols, _ = molecules(["CCO.[Na+]", "!!bad!!"]) >> desalt() >> drop_invalid()
        assert smis(mols) == ["CCO"]

    def test_or_is_an_alias_for_rshift(self):
        via_rshift = molecules(["CCO.[Na+]"]) >> desalt()
        via_or = molecules(["CCO.[Na+]"]) | desalt()
        assert via_rshift.smiles == via_or.smiles

    def test_piping_into_a_non_step_raises(self):
        with pytest.raises(TypeError, match="Can only pipe into a Step"):
            molecules(["CCO"]) >> "not a step"

    def test_forgetting_to_call_the_step_is_explained(self):
        with pytest.raises(TypeError, match="forget to call the step"):
            molecules(["CCO"]) >> desalt


class TestStepComposition:
    def test_steps_compose_into_a_reusable_pipeline(self):
        curate = desalt() >> drop_invalid()
        # A composed pipeline is a PipeStep, not a Step: composition may
        # legitimately mix molecule steps with feature steps, so the
        # common base is the only honest type for the result.
        assert isinstance(curate, PipeStep)
        mols, _ = molecules(["CCO.[Na+]", "!!bad!!"]) >> curate
        assert smis(mols) == ["CCO"]

    def test_composition_flattens(self):
        curate = desalt() >> drop_invalid() >> remove_duplicates()
        assert "desalt() >> drop_invalid() >> remove_duplicates()" in repr(curate)

    def test_pipeline_helper_matches_operator_form(self):
        a = molecules(["CCO.[Na+]"]) >> pipeline(desalt(), drop_invalid())
        b = molecules(["CCO.[Na+]"]) >> (desalt() >> drop_invalid())
        assert a.smiles == b.smiles

    def test_pipeline_requires_steps(self):
        with pytest.raises(ValueError, match="at least one step"):
            pipeline()

    def test_pipeline_rejects_non_steps(self):
        with pytest.raises(TypeError, match="takes Step objects"):
            pipeline(desalt(), "nope")

    def test_composing_with_a_non_step_raises(self):
        with pytest.raises(TypeError, match="Can only compose a Step"):
            desalt() >> "nope"

    def test_a_pipeline_can_be_reused_on_several_datasets(self):
        curate = desalt() >> drop_invalid()
        first, _ = molecules(["CCO.[Na+]"]) >> curate
        second, _ = molecules(["CCN.[Cl-]"]) >> curate
        assert smis(first) == ["CCO"]
        assert smis(second) == ["CCN"]

    def test_step_repr(self):
        assert "desalt" in repr(desalt())


class TestDualMode:
    def test_deferred_when_called_without_molecules(self):
        assert isinstance(desalt(), Step)
        assert isinstance(remove_duplicates(agg="mean"), Step)

    def test_immediate_when_called_with_molecules(self):
        mols, y = desalt([Chem.MolFromSmiles("CC(=O)[O-].[Na+]")], [1.0])
        assert smis(mols) == ["CC(=O)[O-]"]
        assert y.tolist() == [1.0]

    def test_applying_a_step_to_a_molecule_set_directly(self):
        result = desalt(molecules(["CCO.[Na+]"]))
        assert isinstance(result, MoleculeSet)

    def test_custom_step_decorator(self):
        @step
        def keep_first(X, y=None, n=1):
            return X[:n], (None if y is None else y[:n])

        mols, y = molecules(["CCO", "CCN", "CCC"], [1.0, 2.0, 3.0]) >> keep_first(n=2)
        assert len(mols) == 2
        assert y.tolist() == [1.0, 2.0]


class TestCurationSteps:
    def test_desalt_keeps_the_largest_fragment(self):
        mols, _ = molecules(["CC(=O)[O-].[Na+]"]) >> desalt()
        assert smis(mols) == ["CC(=O)[O-]"]

    def test_desalt_passes_none_through(self):
        mols, _ = molecules(["!!bad!!"]) >> desalt()
        assert mols == [None]

    def test_neutralize(self):
        mols, _ = molecules(["CC(=O)[O-]"]) >> neutralize()
        assert smis(mols) == ["CC(=O)O"]

    def test_neutralize_passes_none_through(self):
        assert (molecules(["!!bad!!"]) >> neutralize()).mols == [None]

    def test_standardize_strips_salt_and_charge(self):
        mols, _ = molecules(["CC(=O)Oc1ccccc1C(=O)[O-].[Na+]"]) >> standardize()
        assert smis(mols) == ["CC(=O)Oc1ccccc1C(=O)O"]

    def test_standardize_keeps_stereo_by_default(self):
        mols, _ = molecules(["C[C@H](N)C(=O)O"]) >> standardize()
        assert "@" in smis(mols)[0]

    def test_canonicalize_tautomers_runs(self):
        mols, _ = molecules(["Oc1ccccn1"]) >> canonicalize_tautomers()
        assert mols[0] is not None

    def test_canonicalize_tautomers_passes_none_through(self):
        assert (molecules(["!!bad!!"]) >> canonicalize_tautomers()).mols == [None]

    def test_deglycate_returns_the_aglycone(self):
        mols, _ = molecules([PHENYL_GLUCOSIDE]) >> deglycate()
        assert smis(mols) == ["c1ccccc1"]

    def test_deglycate_keep_originals_duplicates_the_label(self):
        mols, y = molecules([PHENYL_GLUCOSIDE], [7.0]) >> deglycate(
            keep_originals=True
        )
        assert len(mols) == 2
        assert y.tolist() == [7.0, 7.0]

    def test_deglycate_leaves_non_glycosides_alone(self):
        mols, y = molecules(["c1ccccc1"], [1.0]) >> deglycate(keep_originals=True)
        assert len(mols) == 1

    def test_remove_protecting_groups(self):
        mols, _ = molecules(["CC(C)(C)OC(=O)NCc1ccccc1"]) >> remove_protecting_groups()
        assert smis(mols) == ["NCc1ccccc1"]

    def test_drop_invalid_removes_labels_too(self):
        mols, y = molecules(["CCO", "!!bad!!"], [1.0, 2.0]) >> drop_invalid()
        assert len(mols) == 1
        assert y.tolist() == [1.0]


class TestRemoveDuplicates:
    def test_averages_duplicate_labels(self):
        mols, y = molecules(["CCO", "CCO", "c1ccccc1"], [1.0, 3.0, 5.0]) >> (
            remove_duplicates(agg="mean")
        )
        assert len(mols) == 2
        assert sorted(y.tolist()) == [2.0, 5.0]

    @pytest.mark.parametrize(
        "agg, expected", [("min", 1.0), ("max", 3.0), ("median", 2.0), ("first", 1.0)]
    )
    def test_aggregators(self, agg, expected):
        _, y = molecules(["CCO", "CCO"], [1.0, 3.0]) >> remove_duplicates(agg=agg)
        assert y.tolist() == [expected]

    def test_preserves_first_occurrence_order(self):
        mols, _ = molecules(["c1ccccc1", "CCO", "CCO"]) >> remove_duplicates()
        assert smis(mols) == ["c1ccccc1", "CCO"]

    def test_max_spread_discards_disagreeing_replicates(self):
        mols, y = molecules(["CCO", "CCO", "CCN"], [1.0, 9.0, 5.0]) >> (
            remove_duplicates(max_spread=1.0)
        )
        assert smis(mols) == ["CCN"]
        assert y.tolist() == [5.0]

    def test_max_spread_keeps_consistent_replicates(self):
        mols, y = molecules(["CCO", "CCO"], [1.0, 1.5]) >> remove_duplicates(
            max_spread=1.0
        )
        assert len(mols) == 1
        assert y.tolist() == pytest.approx([1.25])

    def test_tautomers_deduplicate_after_canonicalization(self):
        pipe = canonicalize_tautomers() >> remove_duplicates(on="smiles")
        mols, _ = molecules(["Oc1ccccn1", "O=c1cccc[nH]1"]) >> pipe
        assert len(mols) == 1

    def test_dedupe_on_smiles(self):
        mols, _ = molecules(["CCO", "OCC"]) >> remove_duplicates(on="smiles")
        assert len(mols) == 1

    def test_dedupe_on_scaffold(self):
        mols, _ = molecules(["c1ccccc1C", "c1ccccc1CC"]) >> remove_duplicates(
            on="scaffold"
        )
        assert len(mols) == 1

    def test_unlabelled_dedupe(self):
        mols, y = molecules(["CCO", "CCO"]) >> remove_duplicates()
        assert len(mols) == 1 and y is None

    def test_invalid_molecules_survive_for_drop_invalid(self):
        mols, _ = molecules(["CCO", "!!bad!!"]) >> remove_duplicates()
        assert None in mols

    def test_invalid_on_raises(self):
        with pytest.raises(ValueError, match="on must be"):
            molecules(["CCO"]) >> remove_duplicates(on="bogus")

    def test_invalid_agg_raises(self):
        with pytest.raises(ValueError, match="agg must be"):
            molecules(["CCO"]) >> remove_duplicates(agg="bogus")


class TestBalance:
    def test_undersamples_to_the_minority_class(self):
        mols, y = molecules(["CCO", "CCN", "CCC", "c1ccccc1"], [0, 0, 0, 1]) >> (
            balance(random_state=0)
        )
        assert sorted(y.tolist()) == [0, 1]
        assert len(mols) == 2

    def test_oversamples_to_the_majority_class(self):
        _, y = molecules(["CCO", "CCN", "CCC", "c1ccccc1"], [0, 0, 0, 1]) >> (
            balance(method="oversample", random_state=0)
        )
        counts = np.unique(y, return_counts=True)[1]
        assert counts.tolist() == [3, 3]

    def test_already_balanced_set_is_unchanged(self):
        mols, _ = molecules(["CCO", "CCN"], [0, 1]) >> balance(random_state=0)
        assert len(mols) == 2

    def test_is_deterministic_under_a_seed(self):
        a = molecules(["CCO", "CCN", "CCC", "c1ccccc1"], [0, 0, 0, 1]) >> balance(
            random_state=42
        )
        b = molecules(["CCO", "CCN", "CCC", "c1ccccc1"], [0, 0, 0, 1]) >> balance(
            random_state=42
        )
        assert a.smiles == b.smiles

    def test_unlabelled_raises(self):
        with pytest.raises(ValueError, match="needs labels"):
            molecules(["CCO"]) >> balance()

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="method must be"):
            molecules(["CCO", "CCN"], [0, 1]) >> balance(method="smote")


class TestFilteringSteps:
    def test_keep_if(self):
        mols, _ = molecules(["CCO", "c1ccccc1"]) >> keep_if(
            predicate=lambda m: m.GetNumAtoms() > 3
        )
        assert smis(mols) == ["c1ccccc1"]

    def test_keep_if_drops_invalid(self):
        mols, _ = molecules(["!!bad!!"]) >> keep_if(predicate=lambda m: True)
        assert mols == []

    def test_keep_if_requires_a_predicate(self):
        with pytest.raises(ValueError, match="requires a `predicate`"):
            molecules(["CCO"]) >> keep_if()

    def test_drop_if(self):
        mols, _ = molecules(["CCO", "c1ccccc1"]) >> drop_if(
            predicate=lambda m: m.GetNumAtoms() > 3
        )
        assert smis(mols) == ["CCO"]

    def test_drop_if_keeps_invalid_for_drop_invalid(self):
        mols, _ = molecules(["!!bad!!"]) >> drop_if(predicate=lambda m: True)
        assert mols == [None]

    def test_drop_if_requires_a_predicate(self):
        with pytest.raises(ValueError, match="requires a `predicate`"):
            molecules(["CCO"]) >> drop_if()

    def test_filter_by_molecular_weight(self):
        mols, _ = molecules(["CCO", "CCCCCCCCCCCCCCCCCC"]) >> filter_by_property(
            mw=(0, 100)
        )
        assert smis(mols) == ["CCO"]

    def test_filter_by_several_properties(self):
        mols, _ = molecules(["CCO", "c1ccccc1"]) >> filter_by_property(
            mw=(0, 200), heavy_atoms=(1, 3), logp=(-5, 5), rotatable_bonds=(0, 5)
        )
        assert smis(mols) == ["CCO"]

    def test_filter_with_no_bounds_keeps_everything_valid(self):
        mols, _ = molecules(["CCO", "!!bad!!"]) >> filter_by_property()
        assert len(mols) == 1


class TestLabelSteps:
    def test_pactivity_of_one_nanomolar(self):
        _, y = molecules(["CCO"], [1.0]) >> to_pactivity(unit="nM")
        assert float(y[0]) == pytest.approx(9.0)

    @pytest.mark.parametrize(
        "unit, value, expected",
        [("M", 1.0, 0.0), ("mM", 1.0, 3.0), ("uM", 1.0, 6.0), ("pM", 1.0, 12.0)],
    )
    def test_pactivity_units(self, unit, value, expected):
        _, y = molecules(["CCO"], [value]) >> to_pactivity(unit=unit)
        assert float(y[0]) == pytest.approx(expected)

    def test_non_positive_concentration_becomes_nan(self):
        _, y = molecules(["CCO", "CCN"], [0.0, -1.0]) >> to_pactivity()
        assert np.isnan(y).all()

    def test_unlabelled_raises(self):
        with pytest.raises(ValueError, match="needs labels"):
            molecules(["CCO"]) >> to_pactivity()

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="unit must be"):
            molecules(["CCO"], [1.0]) >> to_pactivity(unit="ng/mL")


class TestSamplingSteps:
    def test_sample_n(self):
        mols, _ = molecules(["CCO", "CCN", "CCC"]) >> sample(n=2, random_state=0)
        assert len(mols) == 2

    def test_sample_fraction(self):
        mols, _ = molecules(["CCO", "CCN", "CCC", "CCCC"]) >> sample(
            fraction=0.5, random_state=0
        )
        assert len(mols) == 2

    def test_sample_needs_exactly_one_of_n_or_fraction(self):
        with pytest.raises(ValueError, match="exactly one of"):
            molecules(["CCO"]) >> sample()
        with pytest.raises(ValueError, match="exactly one of"):
            molecules(["CCO"]) >> sample(n=1, fraction=0.5)

    def test_sample_rejects_bad_fraction(self):
        with pytest.raises(ValueError, match="fraction must be"):
            molecules(["CCO"]) >> sample(fraction=1.5)

    def test_sample_larger_than_the_set_raises(self):
        with pytest.raises(ValueError, match="Cannot sample"):
            molecules(["CCO"]) >> sample(n=5)

    def test_shuffle_keeps_pairs_aligned(self):
        smiles = ["CCO", "CCN", "CCC", "c1ccccc1"]
        pairs = dict(zip(smiles, [1.0, 2.0, 3.0, 4.0]))
        mols, y = molecules(smiles, [1.0, 2.0, 3.0, 4.0]) >> shuffle(random_state=1)
        for mol, label in zip(mols, y):
            assert pairs[Chem.MolToSmiles(mol)] == label

    def test_shuffle_is_deterministic(self):
        a = molecules(["CCO", "CCN", "CCC"]) >> shuffle(random_state=3)
        b = molecules(["CCO", "CCN", "CCC"]) >> shuffle(random_state=3)
        assert a.smiles == b.smiles


class TestApply:
    def test_applies_a_function(self):
        mols, _ = molecules(["CCO"]) >> apply(func=Chem.AddHs)
        assert mols[0].GetNumAtoms() == 9

    def test_passes_none_through(self):
        mols, _ = molecules(["!!bad!!"]) >> apply(func=Chem.AddHs)
        assert mols == [None]

    def test_requires_a_function(self):
        with pytest.raises(ValueError, match="requires a `func`"):
            molecules(["CCO"]) >> apply()


class TestEndToEnd:
    def test_the_full_workflow_from_the_design(self):
        smiles = [
            "CC(=O)Oc1ccccc1C(=O)[O-].[Na+]",  # salt
            PHENYL_GLUCOSIDE,                  # glycoside
            "CCO", "CCO",                      # duplicates
            "c1ccccc1",
            "!!bad!!",                         # unparseable
            "CCN",
        ]
        y = [1.0, 1.0, 2.0, 4.0, 5.0, 9.9, 5.0]

        mols, labels = (
            molecules(smiles, y)
            >> desalt()
            >> deglycate(keep_originals=True)
            >> drop_invalid()
            >> remove_duplicates(agg="mean")
        )

        result = dict(zip(smis(mols), labels))
        # the salt lost its counter-ion
        assert "CC(=O)Oc1ccccc1C(=O)[O-]" in result
        # keep_originals kept the glycoside itself alongside its aglycone
        assert Chem.MolToSmiles(Chem.MolFromSmiles(PHENYL_GLUCOSIDE)) in result
        # duplicate ethanol averaged: (2 + 4) / 2
        assert result["CCO"] == pytest.approx(3.0)
        # benzene appears twice - once given, once from the aglycone - so its
        # label is the mean of 5.0 and the glycoside's 1.0
        assert result["c1ccccc1"] == pytest.approx(3.0)
        # the unparseable record is gone, taking its label with it
        assert 9.9 not in labels.tolist()

    def test_labels_stay_aligned_through_every_step(self):
        smiles = ["CCO.[Na+]", "!!bad!!", "c1ccccc1", "CCO"]
        y = [10.0, 20.0, 30.0, 40.0]
        mols, labels = (
            molecules(smiles, y) >> desalt() >> drop_invalid() >> shuffle(random_state=0)
        )
        assert len(mols) == len(labels)
        expected = {"CCO": {10.0, 40.0}, "c1ccccc1": {30.0}}
        for mol, label in zip(mols, labels):
            assert label in expected[Chem.MolToSmiles(mol)]

    def test_unlabelled_pipeline_runs_end_to_end(self):
        mols, y = (
            molecules(["CCO.[Na+]", "!!bad!!", "CCO"])
            >> desalt()
            >> drop_invalid()
            >> remove_duplicates()
        )
        assert smis(mols) == ["CCO"]
        assert y is None
