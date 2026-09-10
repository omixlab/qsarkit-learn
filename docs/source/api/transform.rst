Transform
=========

.. currentmodule:: qsarkit.transform

The scikit-learn glue: SMILES↔Mol conversion, feature unions, NaN
handling, scaling, and a pipeline builder for the common case.

Everything here exists so that a QSAR workflow is an ordinary
:class:`sklearn.pipeline.Pipeline` — which means ``GridSearchCV``,
``cross_val_score`` and ``clone`` all work without special cases.

A whole workflow in one object
------------------------------

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.representation import MorganFingerprint
   >>> from qsarkit.transform import make_qsar_pipeline
   >>> pipeline = make_qsar_pipeline(
   ...     MorganFingerprint(n_bits=64),
   ...     QSARRegressor("rf", random_state=0),
   ...     from_smiles=True,
   ... )
   >>> [name for name, _ in pipeline.steps]
   ['smiles_to_mol', 'representation', 'nan', 'model']
   >>> pipeline.fit(DEMO_SMILES, DEMO_Y).predict(DEMO_SMILES[:3]).shape
   (3,)

Taking SMILES directly matters more than it looks: it puts structure
parsing *inside* the cross-validation fold, so no preprocessing happens
outside the loop where it could leak.

Conversion
----------

.. doctest::

   >>> from qsarkit.transform import MolToSmiles, SmilesToMol
   >>> mols = SmilesToMol().transform(["CCO", "c1ccccc1"])
   >>> MolToSmiles().transform(mols)
   ['CCO', 'c1ccccc1']

Combining representations
-------------------------

.. doctest::

   >>> from qsarkit.representation import MACCSKeysFingerprint, PhysicochemicalDescriptors
   >>> from qsarkit.transform import MoleculeFeatureUnion
   >>> union = MoleculeFeatureUnion([
   ...     ("maccs", MACCSKeysFingerprint()),
   ...     ("physchem", PhysicochemicalDescriptors()),
   ... ])
   >>> union.fit_transform(demo_mols).shape
   (24, 176)

Unlike the individual transformers, the union requires ``fit`` before
``transform`` — it has to learn each branch's output width to know where
the blocks join.

Scaling and missing values
--------------------------

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.transform import DescriptorScaler, NaNHandler
   >>> X = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, 6.0]])
   >>> np.isfinite(NaNHandler(strategy="median").fit_transform(X)).all()
   np.True_
   >>> scaled = DescriptorScaler(method="robust").fit_transform(X[1:])
   >>> scaled.shape
   (2, 2)

``NaNHandler`` belongs before any estimator in a descriptor pipeline:
RDKit emits NaN for undefined quantities (a 3D descriptor with no
conformer, a ratio with a zero denominator), and most estimators refuse
to fit on them.

.. warning::

   Scale inside the pipeline, never on the full dataset beforehand.
   Fitting a scaler on all the data before splitting leaks the test set's
   mean and variance into the transform.

API
---

.. automodule:: qsarkit.transform
   :members:
   :show-inheritance:

References
----------

- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
  Python." J. Mach. Learn. Res., 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
- Buitinck, L. et al. (2013). "API Design for Machine Learning Software."
  ECML PKDD Workshop. :arxiv:`1309.0238`
