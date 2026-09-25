Saving and sharing a model
==========================

A model that exists only in a notebook session is not a result anyone can
use. This page covers getting one out of memory in a form that still works
next year, and that a colleague can load without taking your word for it.

Why not pickle
--------------

``pickle`` — and ``joblib``, which wraps it — is the obvious choice and the
wrong one for a model you intend to keep.

**It is tied to the exact class layout of every object it stores.** A file
written under one scikit-learn release can fail to load under the next, or,
worse, load into a subtly different object that predicts something else. The
failure surfaces months later, when the original environment is gone.

**Loading one executes arbitrary code.** That makes a shared model file a
security problem rather than a data file: you cannot inspect a pickle
without running it.

**It records nothing about the model.** A ``.pkl`` on a shared drive does
not say what it predicts, in what units, on which descriptors, or who
built it — which is exactly what OECD principle 2 asks you to be able to
state.

The bundle format
-----------------

:mod:`qsarkit.persistence` writes a directory instead:

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

Saving
------

.. doctest::

   >>> import os, tempfile
   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.persistence import ModelMetadata, save_model
   >>> from qsarkit.representation import MorganFingerprint
   >>> fingerprint = MorganFingerprint(radius=2, n_bits=512)
   >>> X, y = fingerprint.transform(demo_mols), DEMO_Y
   >>> model = QSARRegressor("rf", random_state=0).fit(X, y)
   >>> path = save_model(
   ...     model,
   ...     os.path.join(tempfile.mkdtemp(), "benzoic_acid_pIC50"),
   ...     pipeline=fingerprint,
   ...     metadata=ModelMetadata(
   ...         name="benzoic acid series pIC50",
   ...         endpoint="pIC50 (-log10 M)",
   ...         task="regression",
   ...         n_training_samples=len(y),
   ...         description="Random forest on Morgan ECFP4, 512 bits.",
   ...     ),
   ... )
   >>> sorted(p.name for p in __import__("pathlib").Path(path).iterdir())
   ['estimator.skops', 'manifest.json', 'metadata.json', 'pipeline.skops']

**Save the transformer with the estimator.** It is what makes
:meth:`~qsarkit.persistence.ModelBundle.predict_mols` possible, and it
removes the commonest way to misuse a saved model — feeding it features
from a differently-configured fingerprint:

.. doctest::

   >>> import numpy as np
   >>> from qsarkit.persistence import load_model
   >>> bundle = load_model(path)
   >>> bool(np.allclose(bundle.predict_mols(demo_mols), model.predict(X)))
   True

Without one, that is refused rather than guessed at:

.. doctest::

   >>> bare = load_model(save_model(model, os.path.join(tempfile.mkdtemp(), "bare")))
   >>> bare.predict_mols(demo_mols)
   Traceback (most recent call last):
       ...
   ValueError: This bundle has no pipeline, so molecules cannot be featurized...

What the metadata is for
------------------------

A year from now the metadata is how you establish what the model was. It
is plain JSON, so that does not depend on qsarkit still being installed:

.. doctest::

   >>> bundle.metadata.endpoint
   'pIC50 (-log10 M)'
   >>> bundle.metadata.n_features
   512
   >>> sorted(bundle.metadata.environment)[:4]
   ['numpy', 'platform', 'python', 'qsarkit']

Record the endpoint *with its units*. ``"pIC50 (-log10 M)"`` is a
statement; ``"activity"`` is not, and the difference is what makes a
prediction of 7.2 interpretable. Put whatever else matters in ``extra``:

.. doctest::

   >>> from qsarkit.persistence import ModelMetadata
   >>> meta = ModelMetadata(
   ...     name="demo",
   ...     endpoint="pIC50 (-log10 M)",
   ...     extra={
   ...         "dataset_doi": "10.1021/example",
   ...         "assay": "cell-based, 48 h",
   ...         "split": "scaffold, test_size=0.25",
   ...         "q2_cv": 0.694,
   ...     },
   ... )
   >>> sorted(meta.extra)
   ['assay', 'dataset_doi', 'q2_cv', 'split']

Guardrails
----------

Two failures are common enough to be checked rather than documented.

**A feature-width mismatch** otherwise produces confident nonsense — the
model happily predicts on the wrong representation:

.. doctest::

   >>> bundle.predict(np.zeros((2, 64)))
   Traceback (most recent call last):
       ...
   ValueError: This model expects 512 features but was given 64...

**A changed environment** means the model is not guaranteed to reproduce
its original predictions, so loading it says so:

.. doctest::

   >>> import warnings
   >>> stale = ModelMetadata(name="demo", endpoint="pIC50")
   >>> stale.environment["scikit-learn"] = "0.1"
   >>> stale_path = save_model(
   ...     model, os.path.join(tempfile.mkdtemp(), "stale"), metadata=stale)
   >>> with warnings.catch_warnings(record=True) as caught:
   ...     warnings.simplefilter("always")
   ...     _ = load_model(stale_path)
   >>> "scikit-learn" in str(caught[0].message)
   True

Pass ``warn_on_environment_change=False`` when you have already checked.

Loading someone else's model
----------------------------

This is where the format earns its keep. Inspect the bundle first — it
reports what is inside without reconstructing anything:

.. doctest::

   >>> from qsarkit.persistence import inspect_bundle
   >>> report = inspect_bundle(path)
   >>> report["manifest"]["estimator_class"]
   'qsarkit.models._facades.QSARRegressor'
   >>> report["metadata"]["endpoint"]
   'pIC50 (-log10 M)'
   >>> report["untrusted"]
   []

``untrusted`` lists the third-party types skops would need permission to
build. qsarkit's own classes are trusted automatically — that is no more
dangerous than ``import qsarkit``, since the class comes from the installed
package and the file supplies only attribute values. Anything else you must
recognize and name:

.. code-block:: python

   bundle = load_model(path, trusted=["mypackage.MyTransformer"])

If that list contains something you do not recognize, do not load the
bundle. That is the whole point of it being a list.

.. warning::

   ``allow_pickle_fallback=True`` exists for estimators skops cannot
   represent, and reintroduces exactly the version fragility and
   arbitrary-code-execution risk described at the top of this page. It
   warns when used, and the manifest records that the file is pickle-based.
   Prefer wrapping the object in a scikit-learn-compatible estimator.

What to ship alongside
----------------------

A bundle records the model. It does not, on its own, tell a reviewer
whether the model is any good — for that, generate a report next to it:

.. doctest::

   >>> from qsarkit.metrics import qsar_regression_report
   >>> from qsarkit.reporting import QSARReport
   >>> report_doc = (
   ...     QSARReport(title="Benzoic acid series — pIC50", endpoint="pIC50")
   ...     .add_dataset_section(n_compounds=len(demo_mols))
   ...     .add_model_section(model, descriptors="Morgan ECFP4, 512 bits")
   ...     .add_validation_section(qsar_regression_report(y, model.predict(X)))
   ... )
   >>> report_doc.to_markdown().splitlines()[0]
   '# Benzoic acid series — pIC50'

See :doc:`oecd` for the regulator-facing version, which tracks which of
the five principles you have and have not addressed.

References
----------

- skops documentation, "Secure persistence with skops":
  https://skops.readthedocs.io/en/stable/persistence.html
- scikit-learn, "Model persistence":
  https://scikit-learn.org/stable/model_persistence.html
- OECD (2007). *Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models*, ENV/JM/MONO(2007)2.
  :doi:`10.1787/9789264085442-en`
- Python documentation, ``pickle`` — "Warning: The pickle module is not
  secure": https://docs.python.org/3/library/pickle.html
