Quick start
===========

A complete QSAR workflow in one page, on the shared demo dataset: 24
compounds in four substituent families with synthetic pIC50 values,
available in every example here as ``DEMO_SMILES``, ``DEMO_Y`` and
``demo_mols``.

Curate the structures
---------------------

.. doctest::

   >>> from rdkit import Chem
   >>> from qsarkit.chemistry import MolecularStandardizer
   >>> standardizer = MolecularStandardizer()
   >>> salt = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
   >>> Chem.MolToSmiles(standardizer.transform([salt])[0])
   'CC(=O)Oc1ccccc1C(=O)O'

The standardizer sanitizes, removes salts and solvates, neutralizes
charges, canonicalizes tautomers, handles stereochemistry and normalizes
hydrogens. Molecules that fail become ``None`` in the output, so positions
stay aligned with your activity data:

.. doctest::

   >>> [m is None for m in standardizer.transform([salt, None])]
   [False, True]

Featurize
---------

.. doctest::

   >>> from qsarkit.representation import MorganFingerprint
   >>> X = MorganFingerprint(radius=2, n_bits=2048).transform(demo_mols)
   >>> X.shape
   (24, 2048)

Model, with a scaffold split
----------------------------

.. doctest::

   >>> from qsarkit.models import QSARRegressor
   >>> from qsarkit.model_selection import ScaffoldSplitter
   >>> y = DEMO_Y
   >>> train, test = next(ScaffoldSplitter(test_size=0.25).split_mols(demo_mols, y))
   >>> model = QSARRegressor("rf", random_state=0).fit(X[train], y[train])
   >>> y_pred = model.predict(X[test])
   >>> y_pred.shape
   (6,)

A random split flatters a QSAR model, because near-duplicate analogues end
up on both sides. A scaffold split keeps whole chemical series together and
measures what you actually care about: generalization to new chemistry.

The difference is not subtle:

.. doctest::

   >>> from qsarkit.metrics import q2_f1
   >>> from qsarkit.model_selection import RandomSplitter
   >>> r_train, r_test = next(
   ...     RandomSplitter(test_size=0.25, random_state=0).split(X, y))
   >>> random_model = QSARRegressor("rf", random_state=0).fit(X[r_train], y[r_train])
   >>> round(q2_f1(y[r_test], random_model.predict(X[r_test]), y[r_train]), 2)
   0.82
   >>> round(q2_f1(y[test], y_pred, y[train]), 2)
   -0.87

Both numbers are honestly computed. Only the second answers "will this work
on chemistry I have not seen".

Check the applicability domain
------------------------------

.. doctest::

   >>> from qsarkit.applicability import ADAnalyzer, KNNApplicabilityDomain
   >>> domain = KNNApplicabilityDomain(n_neighbors=3).fit(X[train])
   >>> domain.predict(X[test]).tolist()
   [False, False, False, False, False, False]

Every held-out compound is outside the domain — which is correct, because
the scaffold split held out an entire chemotype. An applicability domain
that marks everything in-domain is usually telling you about your split
rather than your model.

Interpret the SAR
-----------------

.. doctest::

   >>> from qsarkit.sar import activity_cliff_report
   >>> cliffs = activity_cliff_report(demo_mols, y, similarity_threshold=0.5)
   >>> cliffs["n_cliffs"], round(cliffs["cliff_ratio"], 4)
   (4, 0.0145)
   >>> sorted(cliffs["top_transformations"])
   ['[1*]C>>[1*]Cl', '[1*]Cl>>[1*]Br', '[1*]Cl>>[1*]N']

Activity cliffs are pairs of near-identical structures with very different
activity — the places a similarity-based model must be wrong. Running this
*before* modelling tells you whether a regression model can work at all.

Report it
---------

.. doctest::

   >>> from qsarkit.metrics import qsar_regression_report
   >>> from qsarkit.reporting import QSARReport
   >>> report = (
   ...     QSARReport(title="Quick-start model", endpoint="pIC50")
   ...     .add_dataset_section(n_compounds=24, n_train=len(train), n_test=len(test))
   ...     .add_model_section(model, descriptors="Morgan ECFP4, 2048 bits")
   ...     .add_validation_section(qsar_regression_report(y[test], y_pred))
   ... )
   >>> report.to_markdown().splitlines()[0]
   '# Quick-start model'

The same report renders as plain text, Markdown, HTML, JSON and PDF. See
:doc:`oecd` for the regulator-facing version.

The same thing, as a pipe
-------------------------

Every step above has a functional equivalent, and they compose:

.. doctest::

   >>> from qsarkit.functional import (
   ...     drop_invalid, fingerprint, fit, molecules, split, standardize)
   >>> train_set, test_set = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> standardize()
   ...     >> drop_invalid()
   ...     >> fingerprint("morgan", n_bits=2048)
   ...     >> split("scaffold", test_size=0.25)
   ... )
   >>> model = train_set >> fit("rf", random_state=0)
   >>> model.predict(test_set.X).shape
   (6,)

Next steps
----------

- :doc:`curation` — why the first stage decides the rest
- :doc:`modeling` — splitters, backends and honest evaluation
- :doc:`applicability` — where a model may be trusted
- :doc:`oecd` — validating and reporting for regulatory use
