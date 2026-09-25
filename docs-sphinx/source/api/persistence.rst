Persistence
===========

.. currentmodule:: qsarkit.persistence

Saving a model so it still works next year, and so someone else can load
it safely.

Why not pickle
--------------

``pickle`` is the obvious choice and the wrong one for a model you intend
to keep:

* It embeds the exact class layout of every object, so a file written
  under one scikit-learn or NumPy release can fail to load — or load into
  a subtly different object — under the next.
* Loading one executes arbitrary code, which makes a shared model file a
  security problem rather than a data file.

qsarkit writes a **directory bundle** instead:

.. code-block:: text

    model.qsar/
      manifest.json      what is in here, and what wrote it
      metadata.json      endpoint, task, feature names, provenance
      estimator.skops    the fitted estimator, in skops' inspectable format
      pipeline.skops     the preprocessing pipeline, when there is one

Everything except the estimator is plain JSON, readable without importing
qsarkit at all. The estimator uses `skops
<https://skops.readthedocs.io>`_, the format scikit-learn recommends: it
stores parameters as data rather than as a pickled object graph, and
refuses to reconstruct types that were not explicitly trusted.

Saving and loading
------------------

.. doctest::

   >>> import os, tempfile
   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.persistence import ModelMetadata, load_model, save_model
   >>> X, y = demo_fingerprints(256), DEMO_Y
   >>> model = QSARRegressor("rf", random_state=0).fit(X, y)
   >>> path = save_model(
   ...     model,
   ...     os.path.join(tempfile.mkdtemp(), "demo"),
   ...     metadata=ModelMetadata(
   ...         name="benzoic acid pIC50",
   ...         endpoint="pIC50 (-log10 M)",
   ...         task="regression",
   ...         n_training_samples=len(y),
   ...     ),
   ... )
   >>> path.endswith(".qsar")
   True

The reloaded model predicts identically:

.. doctest::

   >>> import numpy as np
   >>> bundle = load_model(path)
   >>> bool(np.allclose(bundle.predict(X), model.predict(X)))
   True
   >>> bundle.metadata.endpoint
   'pIC50 (-log10 M)'

Predicting from molecules
-------------------------

Save the transformer alongside the estimator and the bundle can go
straight from structures to predictions — which also removes the
commonest way to misuse a saved model, feeding it features from a
different fingerprint:

.. doctest::

   >>> from qsarkit.representation import MorganFingerprint
   >>> fingerprint = MorganFingerprint(radius=2, n_bits=256)
   >>> path = save_model(
   ...     model,
   ...     os.path.join(tempfile.mkdtemp(), "with_pipeline"),
   ...     pipeline=fingerprint,
   ...     metadata=ModelMetadata(name="demo", endpoint="pIC50 (-log10 M)"),
   ... )
   >>> bundle = load_model(path)
   >>> bool(np.allclose(bundle.predict_mols(demo_mols), model.predict(X)))
   True

Without a pipeline that is refused rather than guessed at:

.. doctest::

   >>> bare = load_model(save_model(model, os.path.join(tempfile.mkdtemp(), "bare")))
   >>> bare.predict_mols(demo_mols)
   Traceback (most recent call last):
       ...
   ValueError: This bundle has no pipeline, so molecules cannot be featurized...

Guardrails
----------

A feature-width mismatch is the failure that otherwise produces confident
nonsense, so it is checked:

.. doctest::

   >>> bundle.predict(np.zeros((2, 64)))
   Traceback (most recent call last):
       ...
   ValueError: This model expects 256 features but was given 64...

And a model loaded under different package versions says so, because it
is not guaranteed to reproduce its original predictions:

.. doctest::

   >>> import warnings
   >>> bundle.metadata.environment["scikit-learn"] = "0.1"
   >>> from qsarkit.persistence import save_model as _save
   >>> stale = _save(bundle, os.path.join(tempfile.mkdtemp(), "stale"))
   >>> with warnings.catch_warnings(record=True) as caught:
   ...     warnings.simplefilter("always")
   ...     _ = load_model(stale)
   >>> "scikit-learn" in str(caught[0].message)
   True

Inspecting before loading
-------------------------

The safe first step with a model from someone else. It reports what the
bundle contains without reconstructing anything:

.. doctest::

   >>> from qsarkit.persistence import inspect_bundle
   >>> report = inspect_bundle(path)
   >>> report["manifest"]["estimator_class"]
   'qsarkit.models._facades.QSARRegressor'
   >>> report["untrusted"]
   []

``untrusted`` lists the third-party types skops would need permission to
build. qsarkit's own classes are trusted automatically — that is no more
dangerous than ``import qsarkit``, since the class comes from the
installed package and the file supplies only attribute values. Anything
else you must recognize and name:

.. code-block:: python

   bundle = load_model(path, trusted=["mypackage.MyTransformer"])

.. warning::

   ``allow_pickle_fallback=True`` exists for estimators skops cannot
   represent, and reintroduces exactly the version fragility and
   arbitrary-code-execution risk this module avoids. It warns when used.
   Prefer wrapping the object in a scikit-learn-compatible estimator.

Provenance
----------

.. doctest::

   >>> from qsarkit.persistence import environment_summary
   >>> summary = environment_summary()
   >>> sorted(summary)[:3]
   ['numpy', 'platform', 'python']

Recording this is what OECD principle 2 — an unambiguous algorithm — asks
for in practice: a year later, the metadata is how you establish what the
model was and what its numbers meant.

API
---

.. automodule:: qsarkit.persistence
   :members:
   :show-inheritance:

References
----------

- skops documentation, "Secure persistence with skops":
  https://skops.readthedocs.io/en/stable/persistence.html
- scikit-learn, "Model persistence":
  https://scikit-learn.org/stable/model_persistence.html
- OECD (2007). *Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models*, ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
