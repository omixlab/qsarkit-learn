Chemistry
=========

.. currentmodule:: qsarkit.chemistry

Chemical graph manipulation performed *before* modelling: standardization,
glycan handling, protecting-group removal, core extraction and graph
conversion. Every component accepts ``Iterable[rdkit.Chem.Mol]`` and
follows the scikit-learn transformer protocol.

Standardization
---------------

The first step of any QSAR workflow. Two records of the same compound
that differ only in salt form, protonation or tautomer are the same
compound, and a model that sees them as different is learning the
registration system rather than the chemistry.

.. doctest::

   >>> from rdkit import Chem
   >>> from qsarkit.chemistry import MolecularStandardizer
   >>> standardizer = MolecularStandardizer()
   >>> mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
   >>> Chem.MolToSmiles(standardizer.transform([mol])[0])
   'CC(=O)Oc1ccccc1C(=O)O'

Failures keep their position, so a parallel array of activities stays
aligned:

.. doctest::

   >>> [m is None for m in standardizer.transform([mol, None])]
   [False, True]

Glycans
-------

Natural-product datasets are full of glycosides. The sugar usually
carries no activity of its own but dominates the fingerprint, so two
glycosides of unrelated aglycones score as more similar to each other
than either does to its own aglycone.

.. doctest::

   >>> from qsarkit.chemistry import GlycanDetector, GlycanRemover
   >>> q3g = named_mols["quercetin_3_glucoside"]
   >>> GlycanDetector().detect(q3g)["num_sugar_residues"]
   1
   >>> Chem.MolToSmiles(GlycanRemover().remove(q3g)["aglycone"])
   'O=c1cc(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12'

Detection is not just "a ring with an oxygen in it" — the exocyclic
hydroxylation pattern is what separates a real sugar from a look-alike:

.. doctest::

   >>> detector = GlycanDetector()
   >>> detector.detect(Chem.MolFromSmiles("C1CCOCC1"))["num_sugar_residues"]
   0

:class:`GlycanDescriptors` turns that into features. It returns a
DataFrame rather than an array, because one column is a string:

.. doctest::

   >>> from qsarkit.chemistry import GlycanDescriptors
   >>> frame = GlycanDescriptors().transform([q3g, named_mols["aspirin"]])
   >>> frame["sugar_count"].tolist()
   [1, 0]

Protecting groups and cores
---------------------------

.. doctest::

   >>> from qsarkit.chemistry import CoreExtractor, FragmentRemover
   >>> boc = Chem.MolFromSmiles("CC(C)(C)OC(=O)NCc1ccccc1")
   >>> Chem.MolToSmiles(FragmentRemover().transform([boc])[0])
   'NCc1ccccc1'

A protecting group is a synthesis artefact, not a pharmacophore. Leaving
it on makes every intermediate look like a distinct chemotype and lets a
model key on the tag.

.. doctest::

   >>> extractor = CoreExtractor()
   >>> Chem.MolToSmiles(extractor.bemis_murcko(demo_mols[8]))
   'c1ccccc1'
   >>> pair = [demo_mols[8], demo_mols[9]]
   >>> Chem.MolToSmarts(extractor.mcs(pair))
   '[#6]-[#6](=[#8])-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1'

Graphs
------

.. doctest::

   >>> from qsarkit.chemistry import MolecularGraph
   >>> graph = MolecularGraph()
   >>> G = graph.to_networkx(Chem.MolFromSmiles("CCO"))
   >>> G.number_of_nodes(), G.number_of_edges()
   (3, 2)
   >>> d = graph.descriptors(Chem.MolFromSmiles("c1ccccc1"))
   >>> d["num_rings"], d["diameter"], round(d["wiener_index"], 1)
   (1, 3, 27.0)

API
---

.. automodule:: qsarkit.chemistry
   :members:
   :show-inheritance:

References
----------

- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify."
  J. Chem. Inf. Model., 50(7), 1189-1204. :doi:`10.1021/ci100176x`
- Fischer, J. et al. (2020). "The Sugar Removal Utility." Molecules,
  25(8), 1988. :doi:`10.3390/molecules25081988`
- Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs. 1.
  Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
  :doi:`10.1021/jm9602928`
