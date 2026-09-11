Curation
========

Garbage structures produce garbage models, and most public activity data
arrives with salts, duplicates, mixtures and inconsistent tautomers. This
stage is not optional housekeeping — Fourches, Muratov and Tropsha showed
that curation changes model performance more than the choice of algorithm.

Standardization
---------------

:class:`~qsarkit.chemistry.MolecularStandardizer` runs the whole
normalization pipeline:

.. code-block:: python

   from qsarkit.chemistry import MolecularStandardizer

   standardizer = MolecularStandardizer(
       remove_salts=True,
       neutralize=True,
       normalize_tautomers=True,
       handle_stereochemistry="retain",
       normalize_hydrogens=True,
   )
   clean = standardizer.transform(mols)

.. warning::

   RDKit's tautomer canonicalizer strips sp3 stereochemistry at atoms it
   considers tautomeric, which silently destroys genuine stereocentres —
   the alpha carbon of an amino acid, for instance. qsarkit disables that
   unless you explicitly ask for ``handle_stereochemistry="remove"``.

Glycans
-------

Natural-product datasets are full of glycosides whose activity belongs to
the aglycone. :class:`~qsarkit.chemistry.GlycanRemover` cleaves the
glycosidic bonds and returns the core:

.. code-block:: python

   from qsarkit.chemistry import GlycanRemover

   result = GlycanRemover().remove(quercetin_3_glucoside)
   Chem.MolToSmiles(result["aglycone"])
   # 'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'
   result["removed_fragments"]   # the released sugar

Sugar detection follows the Sugar Removal Utility: a 5- or 6-membered ring
with one ring oxygen, sp3 carbons, and enough exocyclic oxygens to match a
real aldose — which correctly rejects cyclohexane and tetrahydropyran.

Protecting groups and cores
---------------------------

.. code-block:: python

   from qsarkit.chemistry import CoreExtractor, FragmentRemover

   FragmentRemover().transform([boc_protected])   # strips Boc, Cbz, Fmoc, ...
   CoreExtractor().bemis_murcko(mol)              # the scaffold
   CoreExtractor().mcs([mol_a, mol_b, mol_c])     # maximum common substructure

Dataset-level curation
----------------------

:mod:`qsarkit.data_quality` handles what standardization cannot: duplicate
structures with disagreeing activities, activity outliers, and records that
are not really molecules at all.

.. code-block:: python

   from qsarkit.data_quality import DataCurationPipeline

   pipeline = DataCurationPipeline()
   curated, report = pipeline.run(mols, activities)
   report   # what was removed, and why

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
