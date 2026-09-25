Representation
==============

Every representation is a scikit-learn transformer taking
``Iterable[rdkit.Chem.Mol]`` and returning a NumPy array, with
``get_feature_names_out()`` for traceability where the features have names.

Choosing a representation matters more than choosing a model. A random
forest on good features beats a tuned network on bad ones, and no amount of
hyperparameter search recovers information the representation discarded.

Fingerprints
------------

.. doctest::

   >>> from qsarkit.representation import MorganFingerprint
   >>> MorganFingerprint(radius=2, n_bits=2048).transform(demo_mols).shape
   (24, 2048)

ECFP4 (``radius=2``) is the default worth reaching for first. The radius is
the substantive choice: 2 captures functional groups, 3 captures larger
motifs at the cost of sparsity.

.. doctest::

   >>> ecfp6 = MorganFingerprint(radius=3, n_bits=2048).transform(demo_mols)
   >>> fcfp4 = MorganFingerprint(radius=2, use_features=True).transform(demo_mols)
   >>> counts = MorganFingerprint(radius=2, use_counts=True).transform(demo_mols)
   >>> ecfp6.shape, fcfp4.shape, counts.shape
   ((24, 2048), (24, 2048), (24, 2048))

Use counts when substructure *frequency* matters; binary otherwise. A count
vector holds values above 1 where a binary one saturates:

.. doctest::

   >>> import numpy as np
   >>> binary = MorganFingerprint(radius=2).transform(demo_mols)
   >>> float(binary.max()), bool(counts.max() > 1)
   (1.0, True)

Fingerprints are sparse. Most bits never fire on a small dataset — that is
what makes the representation general, but it does mean a variance filter
is nearly free:

.. doctest::

   >>> int((binary.sum(axis=0) > 0).sum())
   138

The other families answer different questions:

.. doctest::

   >>> from qsarkit.representation import (
   ...     AtomPairFingerprint, MACCSKeysFingerprint, RDKitFingerprint,
   ...     TopologicalTorsionFingerprint)
   >>> MACCSKeysFingerprint().transform(demo_mols).shape
   (24, 167)
   >>> RDKitFingerprint(n_bits=512).transform(demo_mols).shape
   (24, 512)
   >>> AtomPairFingerprint(n_bits=512).transform(demo_mols).shape
   (24, 512)
   >>> TopologicalTorsionFingerprint(n_bits=512).transform(demo_mols).shape
   (24, 512)

MACCS keys are 166 hand-curated substructure questions: interpretable and
fixed-length, but far less expressive than a hashed fingerprint. Atom pairs
and topological torsions are *count* vectors by design.

Combining fingerprints takes ``(name, transformer)`` pairs, so each block
stays identifiable in the output:

.. doctest::

   >>> from qsarkit.representation import FingerprintCombiner
   >>> combined = FingerprintCombiner([
   ...     ("morgan", MorganFingerprint(n_bits=256)),
   ...     ("maccs", MACCSKeysFingerprint()),
   ... ])
   >>> combined.transform(demo_mols).shape
   (24, 423)

Descriptors
-----------

Where fingerprints answer "what substructures are present", descriptors
answer "what is this molecule like".

.. doctest::

   >>> from qsarkit.representation import (
   ...     Descriptors3D, PhysicochemicalDescriptors, RDKitDescriptors)
   >>> RDKitDescriptors().transform(demo_mols).shape        # all of them, NaN-safe
   (24, 217)
   >>> block = PhysicochemicalDescriptors()
   >>> list(block.get_feature_names_out()[:4])
   ['MolWt', 'MolLogP', 'TPSA', 'NumHDonors']
   >>> Descriptors3D().transform(demo_mols[:2]).shape       # embeds conformers first
   (2, 10)

Descriptors are interpretable in a way fingerprints are not — which matters
for OECD principle 5.

.. warning::

   They are also on wildly different scales. Molecular weight is in the
   hundreds and logP in single digits, so any distance-based or regularized
   model needs them scaled:

   .. doctest::

      >>> features = block.transform(demo_mols)
      >>> features.std(axis=0).round(1)[:3]        # MolWt, MolLogP, TPSA
      array([17.3,  0.6, 12.8])
      >>> from qsarkit.transform import DescriptorScaler
      >>> scaled = DescriptorScaler(method="standard").fit_transform(features)
      >>> bool(abs(scaled.mean()) < 1e-9), scaled.std().round(3)
      (True, np.float64(1.0))

   Fingerprints, being binary and already on one scale, do not.

Learned embeddings
------------------

:class:`~qsarkit.representation.Mol2VecTransformer` and
:class:`~qsarkit.representation.ChemBERTaTransformer` need the
``embeddings`` and ``nlp`` extras respectively. They are lazily imported,
so ``import qsarkit`` stays cheap:

.. code-block:: python

   from qsarkit.representation import ChemBERTaTransformer, Mol2VecTransformer

   Mol2VecTransformer().fit(corpus_mols).transform(mols)   # needs [embeddings]
   ChemBERTaTransformer(pooling="mean").transform(mols)    # needs [nlp]

A missing extra raises an error naming it rather than an ``ImportError`` you
have to interpret.

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

:class:`~qsarkit.transform.MoleculeFeatureUnion` concatenates blocks of
different kinds — a fingerprint beside a descriptor set:

.. doctest::

   >>> from qsarkit.transform import MoleculeFeatureUnion
   >>> union = MoleculeFeatureUnion([
   ...     ("maccs", MACCSKeysFingerprint()),
   ...     ("physchem", PhysicochemicalDescriptors()),
   ... ])
   >>> union.fit_transform(demo_mols).shape
   (24, 176)

Unlike the individual transformers it requires ``fit`` before
``transform``, because it has to learn each branch's width to know where
the blocks join.

The whole thing composes into an ordinary pipeline:

.. doctest::

   >>> from sklearn.pipeline import Pipeline
   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.transform import NaNHandler
   >>> pipeline = Pipeline([
   ...     ("features", union),
   ...     ("nan", NaNHandler(strategy="median")),
   ...     ("model", QSARRegressor("rf", random_state=0)),
   ... ])
   >>> pipeline.fit(demo_mols, DEMO_Y).predict(demo_mols).shape
   (24,)

``NaNHandler`` belongs before any estimator in a descriptor pipeline: RDKit
emits NaN for undefined quantities — a 3D descriptor with no conformer, a
ratio with a zero denominator — and most estimators refuse to fit on them.

:func:`~qsarkit.transform.make_qsar_pipeline` assembles the common case,
and can take SMILES directly so structure parsing happens *inside* the
cross-validation fold:

.. doctest::

   >>> from qsarkit.transform import make_qsar_pipeline
   >>> pipeline = make_qsar_pipeline(
   ...     MorganFingerprint(n_bits=512),
   ...     QSARRegressor("rf", random_state=0),
   ...     from_smiles=True,
   ... )
   >>> [name for name, _ in pipeline.steps]
   ['smiles_to_mol', 'representation', 'nan', 'model']
   >>> pipeline.fit(DEMO_SMILES, DEMO_Y).predict(DEMO_SMILES[:3]).shape
   (3,)

Or in the pipe notation, where :func:`~qsarkit.functional.featurize` accepts
any of the transformers above:

.. doctest::

   >>> from qsarkit.functional import describe, featurize, fingerprint, molecules
   >>> (molecules(DEMO_SMILES) >> fingerprint("morgan", n_bits=512)).shape
   (24, 512)
   >>> (molecules(DEMO_SMILES) >> describe("lipinski")).shape
   (24, 9)
   >>> (molecules(DEMO_SMILES) >> featurize(union)).shape
   (24, 176)

References
----------

- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  *J. Chem. Inf. Model.*, 50(5), 742-754. :doi:`10.1021/ci100050t`
- Durant, J. L. et al. (2002). "Reoptimization of MDL Keys for Use in Drug
  Discovery." *J. Chem. Inf. Comput. Sci.*, 42(6), 1273-1280.
  :doi:`10.1021/ci010132r`
- Todeschini, R. & Consonni, V. (2009). *Molecular Descriptors for
  Chemoinformatics.* Wiley. :doi:`10.1002/9783527628766`
- Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec." *J. Chem. Inf.
  Model.*, 58(1), 27-35. :doi:`10.1021/acs.jcim.7b00616`
- Chithrananda, S., Grand, G. & Ramsundar, B. (2020). "ChemBERTa."
  :arxiv:`2010.09885`
- Yang, K. et al. (2019). "Analyzing Learned Molecular Representations for
  Property Prediction." *J. Chem. Inf. Model.*, 59(8), 3370-3388.
  :doi:`10.1021/acs.jcim.9b00237`
- Gilmer, J. et al. (2017). "Neural Message Passing for Quantum Chemistry."
  :arxiv:`1704.01212`
