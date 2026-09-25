Representation
==============

.. currentmodule:: qsarkit.representation

Turning molecules into numbers. Fingerprints, descriptor blocks and
learned embeddings, all as scikit-learn transformers accepting
``Iterable[rdkit.Chem.Mol]`` and returning a NumPy array.

Choosing a representation matters more than choosing a model. A random
forest on good features beats a tuned neural network on bad ones, and no
amount of hyperparameter search recovers information the representation
threw away.

Fingerprints
------------

.. doctest::

   >>> from qsarkit.representation import MorganFingerprint
   >>> X = MorganFingerprint(radius=2, n_bits=1024).transform(demo_mols)
   >>> X.shape
   (24, 1024)

Morgan (ECFP) fingerprints are the default for good reason: they encode
circular atom environments, are cheap, and work well with tree ensembles
and Tanimoto-kernel methods. The radius is the substantive choice —
radius 2 (ECFP4) captures functional groups, radius 3 (ECFP6) captures
larger motifs at the cost of sparsity.

Fingerprints are sparse. Most bits never fire on a small dataset:

.. doctest::

   >>> int((X.sum(axis=0) > 0).sum())
   133

The other families answer different questions:

.. doctest::

   >>> from qsarkit.representation import (
   ...     AtomPairFingerprint, MACCSKeysFingerprint, RDKitFingerprint)
   >>> MACCSKeysFingerprint().transform(demo_mols).shape
   (24, 167)
   >>> RDKitFingerprint(n_bits=512).transform(demo_mols).shape
   (24, 512)
   >>> AtomPairFingerprint(n_bits=512).transform(demo_mols).shape
   (24, 512)

MACCS keys are 166 hand-curated substructure questions — interpretable
and fixed-length, but far less expressive than a hashed fingerprint. Atom
pairs and topological torsions are *count* vectors by design, encoding
how often a feature occurs rather than merely whether it does.

Combining them needs no new machinery:

.. doctest::

   >>> from qsarkit.representation import FingerprintCombiner
   >>> combined = FingerprintCombiner([
   ...     ("morgan", MorganFingerprint(n_bits=64)),
   ...     ("maccs", MACCSKeysFingerprint()),
   ... ])
   >>> combined.transform(demo_mols).shape
   (24, 231)

Descriptors
-----------

Where fingerprints answer "what substructures are present", descriptors
answer "what is this molecule like".

.. doctest::

   >>> from qsarkit.representation import PhysicochemicalDescriptors
   >>> block = PhysicochemicalDescriptors()
   >>> list(block.get_feature_names_out()[:4])
   ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors']
   >>> block.transform(demo_mols).shape
   (24, 9)

.. doctest::

   >>> from qsarkit.representation import LipinskiDescriptors, RDKitDescriptors
   >>> RDKitDescriptors().transform(demo_mols[:1]).shape
   (1, 217)
   >>> LipinskiDescriptors().transform(demo_mols).shape
   (24, 9)

.. warning::

   Descriptors are continuous and on wildly different scales — molecular
   weight in the hundreds, logP in single digits. Any distance-based or
   regularized model needs them scaled first (see
   :class:`~qsarkit.transform.DescriptorScaler`). Fingerprints, being
   binary, do not.

Learned embeddings
------------------

:class:`Mol2VecTransformer` and :class:`ChemBERTaTransformer` need the
``embeddings`` and ``nlp`` extras respectively. They are lazily imported,
so ``import qsarkit`` stays cheap::

   from qsarkit.representation import ChemBERTaTransformer

   X = ChemBERTaTransformer().transform(mols)     # needs qsarkit-learn[nlp]

API
---

.. automodule:: qsarkit.representation
   :members:
   :show-inheritance:

References
----------

- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  J. Chem. Inf. Model., 50(5), 742-754. :doi:`10.1021/ci100050t`
- Durant, J. L. et al. (2002). "Reoptimization of MDL Keys for Use in
  Drug Discovery." J. Chem. Inf. Comput. Sci., 42(6), 1273-1280.
  :doi:`10.1021/ci010132r`
- Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
  Chemoinformatics." Wiley. :doi:`10.1002/9783527628766`
- Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec." J. Chem. Inf.
  Model., 58(1), 27-35. :doi:`10.1021/acs.jcim.7b00616`
- Chithrananda, S., Grand, G. & Ramsundar, B. (2020). "ChemBERTa."
  :arxiv:`2010.09885`
