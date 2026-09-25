Curation
========

Garbage structures produce garbage models, and most public activity data
arrives with salts, duplicates, mixtures and inconsistent tautomers. This
stage is not optional housekeeping — Fourches, Muratov and Tropsha showed
that curation changes model performance more than the choice of algorithm.

Standardization
---------------

:class:`~qsarkit.chemistry.MolecularStandardizer` runs the whole
normalization pipeline: sanitize, strip salts and solvates, neutralize
charges, canonicalize tautomers, handle stereochemistry, normalize
hydrogens.

.. doctest::

   >>> from rdkit import Chem
   >>> from qsarkit.chemistry import MolecularStandardizer
   >>> standardizer = MolecularStandardizer(
   ...     remove_salts=True,
   ...     neutralize=True,
   ...     normalize_tautomers=True,
   ...     handle_stereochemistry="retain",
   ...     normalize_hydrogens=True,
   ... )
   >>> salt = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
   >>> Chem.MolToSmiles(standardizer.transform([salt])[0])
   'CC(=O)Oc1ccccc1C(=O)O'

Two records of the same compound that differ only in salt form, protonation
or tautomer are the same compound. A model that sees them as different is
learning the registration system rather than the chemistry.

**Failures keep their position.** A molecule that cannot be standardized
becomes ``None`` in place, so a parallel array of activities stays aligned:

.. doctest::

   >>> [m is None for m in standardizer.transform([salt, None])]
   [False, True]

.. warning::

   RDKit's tautomer canonicalizer strips sp3 stereochemistry at atoms it
   considers tautomeric, which silently destroys genuine stereocentres —
   the alpha carbon of an amino acid, for instance. qsarkit disables that
   unless you explicitly ask for ``handle_stereochemistry="remove"``:

   .. doctest::

      >>> alanine = Chem.MolFromSmiles("C[C@H](N)C(=O)O")
      >>> Chem.MolToSmiles(standardizer.transform([alanine])[0])   # stereo kept
      'C[C@H](N)C(=O)O'
      >>> flat = MolecularStandardizer(handle_stereochemistry="remove")
      >>> Chem.MolToSmiles(flat.transform([alanine])[0])
      'CC(N)C(=O)O'

Glycans
-------

Natural-product datasets are full of glycosides whose activity belongs to
the aglycone. The sugar carries no activity of its own but dominates the
fingerprint, so two glycosides of unrelated aglycones score as more similar
to each other than either does to its own aglycone.

.. doctest::

   >>> from qsarkit.chemistry import GlycanRemover
   >>> q3g = named_mols["quercetin_3_glucoside"]
   >>> result = GlycanRemover().remove(q3g)
   >>> Chem.MolToSmiles(result["aglycone"])
   'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'

The result keeps the input alongside what was cut away, so a curation step
can record why a structure changed:

.. doctest::

   >>> sorted(result)
   ['aglycone', 'original', 'removed_fragments']
   >>> len(result["removed_fragments"])
   1

Sugar detection follows the Sugar Removal Utility: a 5- or 6-membered ring
with one ring oxygen, sp3 carbons, and enough exocyclic oxygens to match a
real aldose. The hydroxylation requirement is what separates a sugar from a
look-alike ring — tetrahydropyran has the right ring and is correctly
rejected:

.. doctest::

   >>> from qsarkit.chemistry import GlycanDetector
   >>> detector = GlycanDetector()
   >>> detector.detect(q3g)["num_sugar_residues"]
   1
   >>> detector.detect(Chem.MolFromSmiles("C1CCOCC1"))["num_sugar_residues"]
   0

Protecting groups and cores
---------------------------

A protecting group is a synthesis artefact, not a pharmacophore. Leaving it
on makes every intermediate in a series look like a distinct chemotype and
lets a model key on the tag:

.. doctest::

   >>> from qsarkit.chemistry import CoreExtractor, FragmentRemover
   >>> boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)NCc1ccccc1")
   >>> Chem.MolToSmiles(FragmentRemover().transform([boc])[0])
   'NCc1ccccc1'

:class:`~qsarkit.chemistry.CoreExtractor` offers three definitions of
"core", answering different questions:

.. doctest::

   >>> extractor = CoreExtractor()
   >>> anilide = Chem.MolFromSmiles("CC(=O)Nc1ccc(Cl)cc1")
   >>> Chem.MolToSmiles(extractor.bemis_murcko(anilide))
   'c1ccccc1'
   >>> Chem.MolToSmiles(extractor.bemis_murcko(anilide, generic=True))
   'C1CCCCC1'
   >>> pair = [Chem.MolFromSmiles(s) for s in
   ...         ("CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1")]
   >>> Chem.MolToSmarts(extractor.mcs(pair))
   '[#6]-[#6](=[#8])-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

``generic=True`` discards element identity and bond order, so pyridine and
benzene analogues collapse onto one skeleton — the right granularity for
"how many ring systems are in this library", the wrong one for "which
chemotype is this".

Dataset-level curation
----------------------

:mod:`qsarkit.data_quality` handles what standardization cannot: duplicate
structures with disagreeing activities, activity outliers, and records that
are not really molecules at all.

.. doctest::

   >>> from qsarkit.data_quality import DataCurationPipeline
   >>> mols_in = demo_mols[:6] + [demo_mols[0]]        # one deliberate duplicate
   >>> activities = list(DEMO_Y[:6]) + [5.1]
   >>> mols, y, report = DataCurationPipeline().run(mols_in, activities)
   >>> len(mols), len(y)
   (6, 6)

``run`` returns ``(mols, y, report)``. The report is the audit trail OECD
principle 2 asks for — every removal, attributed to the stage that made it:

.. doctest::

   >>> print(report.summary())
   Curation: 7 -> 6 records (85.7% retained)
     standardize                 7 -> 7      (0 removed)
     validate                    7 -> 7      (0 removed)
     deduplicate                 7 -> 6      (1 removed)

Duplicates are found by structural identity, not string equality. With
activities attached, a group also reports whether its measurements agree —
a duplicate pair three log units apart is not a duplicate to be merged but
a data problem to be investigated:

.. doctest::

   >>> from qsarkit.data_quality import DuplicateDetector
   >>> groups = DuplicateDetector(activity_tolerance=0.5).find_duplicates(
   ...     [demo_mols[0], demo_mols[0]], [5.1, 8.4])
   >>> round(groups[0].spread, 2), groups[0].consistent
   (3.3, False)

The default outlier detector uses the modified z-score — median and MAD
rather than mean and standard deviation. That matters here: a single extreme
value inflates the standard deviation enough to mask itself, so a plain
z-score is least reliable exactly when you need it:

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.data_quality import ActivityOutlierDetector
   >>> ActivityOutlierDetector().detect(np.array([5.0, 5.1, 5.2, 9.9])).tolist()
   [False, False, False, True]

Checking the activity column
----------------------------

The single most damaging silent error in QSAR is a column that mixes molar
and p-scale values. Nothing in the numbers announces it:

.. doctest::

   >>> from qsarkit.data_quality import check_activity_units
   >>> check_activity_units(DEMO_Y, endpoint="IC50")["looks_logarithmic"]
   True
   >>> raw = check_activity_units([1.0, 10.0, 1000.0, 1e6], unit="nM")
   >>> raw["looks_logarithmic"], round(raw["log_range"], 1)
   (False, 6.0)

Converting to a p-scale makes the unit explicit, puts the values on the
scale every model here assumes, and makes the errors roughly normal — which
is what the regression metrics assume in turn:

.. doctest::

   >>> from qsarkit.utils import to_pactivity
   >>> float(to_pactivity(1.0, unit="nM")), float(to_pactivity(1000.0, unit="nM"))
   (9.0, 6.0)

The same thing as a pipe
------------------------

Every step above has a functional equivalent, and they keep ``y`` aligned
with the molecules automatically — which is the bookkeeping hand-written
curation scripts get subtly wrong:

.. doctest::

   >>> from qsarkit.functional import (
   ...     drop_invalid, molecules, remove_duplicates, standardize)
   >>> mols, y = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> standardize()
   ...     >> drop_invalid()
   ...     >> remove_duplicates(agg="mean")
   ... )
   >>> len(mols), len(y)
   (24, 24)

Unparseable input becomes ``None`` rather than raising, and
:func:`~qsarkit.functional.drop_invalid` removes it together with its label:

.. doctest::

   >>> ms = molecules(["CCO", "not-a-molecule", "CCN"], [1.0, 2.0, 3.0])
   >>> [m is None for m in ms.mols]
   [False, True, False]
   >>> mols, y = ms >> drop_invalid()
   >>> y.tolist()
   [1.0, 3.0]

See :doc:`functional` for the rest of the pipe API.

References
----------

- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and QSAR
  Modeling Research." *J. Chem. Inf. Model.*, 50(7), 1189-1204.
  :doi:`10.1021/ci100176x`
- Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify II."
  *J. Chem. Inf. Model.*, 56(7), 1243-1252.
  :doi:`10.1021/acs.jcim.6b00129`
- Fischer, J. et al. (2020). "The Sugar Removal Utility." *Molecules*,
  25(8), 1988. :doi:`10.3390/molecules25081988`
- Kalliokoski, T. et al. (2013). "Comparability of Mixed IC50 Data — A
  Statistical Analysis." *PLoS ONE*, 8(4), e61007.
  :doi:`10.1371/journal.pone.0061007`
