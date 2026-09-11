Representation
==============

Every representation is a scikit-learn transformer taking
``Iterable[rdkit.Chem.Mol]`` and returning a NumPy array, with
``get_feature_names_out()`` for traceability.

Fingerprints
------------

.. code-block:: python

   from qsarkit.representation.fingerprints import (
       MorganFingerprint, MACCSKeysFingerprint, FingerprintCombiner,
   )

   MorganFingerprint(radius=2, n_bits=2048).transform(mols)   # ECFP4
   MorganFingerprint(radius=2, use_features=True)             # FCFP4
   MorganFingerprint(radius=2, use_counts=True)               # count vector
   FingerprintCombiner([MorganFingerprint(), MACCSKeysFingerprint()])

ECFP4 (``radius=2``) is the default worth reaching for first. Use counts
when substructure *frequency* matters; binary otherwise.

Descriptors
-----------

.. code-block:: python

   from qsarkit.representation.descriptors import (
       RDKitDescriptors, PhysicochemicalDescriptors, Descriptors3D,
   )

   RDKitDescriptors().transform(mols)              # all ~210, NaN-safe
   PhysicochemicalDescriptors().transform(mols)    # MW, logP, TPSA, QED, ...
   Descriptors3D().transform(mols)                 # embeds conformers first

Descriptors are interpretable in a way fingerprints are not — which matters
for OECD principle 5. They also need scaling; see
:class:`~qsarkit.transform.DescriptorScaler`.

Learned embeddings
------------------

.. code-block:: python

   from qsarkit.representation.embeddings import ChemBERTaTransformer
   from qsarkit.representation.mol2vec import Mol2VecTransformer

   Mol2VecTransformer().fit(corpus_mols).transform(mols)
   ChemBERTaTransformer(pooling="mean").transform(mols)

Requires the ``embeddings`` and ``nlp`` extras respectively.

.. note::

   Graph neural network representations (MPNN, D-MPNN, GCN, GAT, SchNet)
   are **not** included. They are a different engineering problem —
   training loops, GPU management, batching — and shipping a half-hearted
   version would be worse than pointing at the projects that do it
   properly. Use `Chemprop <https://github.com/chemprop/chemprop>`_ or
   `DeepChem <https://deepchem.io>`_, and bring the learned embeddings
   back here as a plain feature matrix: everything downstream in qsarkit
   takes an array.

Composing
---------

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from qsarkit.transform import MoleculeFeatureUnion

   Pipeline([
       ("features", MoleculeFeatureUnion([
           MorganFingerprint(),
           PhysicochemicalDescriptors(),
       ])),
       ("model", QSARRegressor("rf")),
   ])

References
----------

- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  *J. Chem. Inf. Model.*, 50(5), 742-754. :doi:`10.1021/ci100050t`
- Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec." *J. Chem. Inf.
  Model.*, 58(1), 27-35. :doi:`10.1021/acs.jcim.7b00616`
- Chithrananda, S., Grand, G. & Ramsundar, B. (2020). "ChemBERTa."
  :arxiv:`2010.09885`
- Yang, K. et al. (2019). "Analyzing Learned Molecular Representations for
  Property Prediction." *J. Chem. Inf. Model.*, 59(8), 3370-3388.
  :doi:`10.1021/acs.jcim.9b00237`
- Gilmer, J. et al. (2017). "Neural Message Passing for Quantum Chemistry."
  :arxiv:`1704.01212`
