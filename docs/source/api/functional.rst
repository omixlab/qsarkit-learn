Functional pipe API
===================

.. currentmodule:: qsarkit.functional

A left-to-right pipe for the whole QSAR workflow, in the spirit of R's
``%>%``. Chain stages with ``>>`` (or ``|``); ``>`` is rejected, because
Python parses ``a > b > c`` as the chained comparison ``(a > b) and
(b > c)`` and would silently discard your data.

.. note::

   Every example on this page is executed by the test suite. They all
   start from the shared demo dataset described in :doc:`../guide/index`:
   ``DEMO_SMILES`` (24 compounds in four substituent series), ``DEMO_Y``
   (synthetic pIC50 values) and ``demo_mols``.

Two domains, one notation
-------------------------

A pipeline moves through two value types. :class:`MoleculeSet` carries
molecules and labels; :class:`FeatureSet` carries a matrix and labels.
:func:`featurize` is the hinge between them.

.. code-block:: text

    molecules()          -> MoleculeSet     chemistry
      >> desalt()        -> MoleculeSet
      >> drop_invalid()  -> MoleculeSet
      >> featurize(...)  -> FeatureSet      <-- the transition
      >> scale()         -> FeatureSet      linear algebra
      >> fit(...)        -> fitted model    terminal

Steps check what they receive, so a mis-ordered chain names the offending
stage rather than failing somewhere deeper:

.. doctest::

   >>> from qsarkit.functional import desalt, fingerprint, molecules
   >>> molecules(["CCO"]) >> fingerprint(n_bits=16) >> desalt()
   Traceback (most recent call last):
       ...
   TypeError: Step 'desalt' works on molecules, but it received a FeatureSet...

Starting a pipeline
-------------------

:func:`molecules` accepts RDKit molecules, SMILES, InChI, or any mixture:

.. doctest::

   >>> from rdkit import Chem
   >>> from qsarkit.functional import molecules
   >>> ms = molecules([
   ...     Chem.MolFromSmiles("c1ccccc1"),
   ...     "CCN",
   ...     "InChI=1S/C2H6O/c1-2-3/h3H,2H2,1H3",
   ... ])
   >>> ms.smiles
   ['c1ccccc1', 'CCN', 'CCO']

Anything unparseable becomes ``None`` rather than raising, so it stays
aligned with ``y`` until you decide what to do with it:

.. doctest::

   >>> from qsarkit.functional import drop_invalid
   >>> ms = molecules(["CCO", "not-a-molecule", "CCN"], [1.0, 2.0, 3.0])
   >>> [m is None for m in ms.mols]
   [False, True, False]
   >>> mols, y = ms >> drop_invalid()
   >>> y.tolist()
   [1.0, 3.0]

That index alignment is the whole point of the API. It is the bookkeeping
hand-written curation scripts get subtly wrong: a molecule dropped
without its label shifts every subsequent activity by one, and the model
still trains.

Curating
--------

.. doctest::

   >>> from qsarkit.functional import remove_duplicates, standardize
   >>> ms = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> standardize()
   ...     >> drop_invalid()
   ...     >> remove_duplicates(agg="mean")
   ... )
   >>> len(ms)
   24

Each set carries its own provenance, which is what a QMRF report needs
under OECD Principle 2:

.. doctest::

   >>> ms.history
   ['molecules(n=24)', 'standardize()', 'drop_invalid()', "remove_duplicates(agg='mean')"]

Steps compose with each other, so a protocol is defined once and reused
on train and test alike:

.. doctest::

   >>> curate = standardize() >> drop_invalid() >> remove_duplicates()
   >>> train = molecules(DEMO_SMILES[:12], DEMO_Y[:12]) >> curate
   >>> test = molecules(DEMO_SMILES[12:], DEMO_Y[12:]) >> curate
   >>> len(train), len(test)
   (12, 12)

Featurizing
-----------

:func:`featurize` takes any transformer with a ``transform(mols)``
method, so the pipe inherits the whole of :mod:`qsarkit.representation`
without duplicating it:

.. doctest::

   >>> from qsarkit.functional import featurize
   >>> from qsarkit.representation import MorganFingerprint
   >>> fs = molecules(DEMO_SMILES, DEMO_Y) >> featurize(MorganFingerprint(n_bits=512))
   >>> fs.shape
   (24, 512)

:func:`fingerprint` and :func:`describe` are shorthands for the common
cases:

.. doctest::

   >>> from qsarkit.functional import describe, fingerprint
   >>> (molecules(DEMO_SMILES) >> fingerprint("maccs")).shape
   (24, 167)
   >>> fs = molecules(DEMO_SMILES) >> describe("lipinski")
   >>> fs.feature_names[:3]
   ['MolWt', 'MolLogP', 'NumHDonors']

Modelling
---------

.. doctest::

   >>> from qsarkit.functional import fit, scale, select_features, split
   >>> train, test = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> fingerprint(n_bits=512)
   ...     >> split("scaffold", test_size=0.25)
   ... )
   >>> len(train) + len(test)
   24

The default split is by scaffold, not at random. A random split of a QSAR
dataset measures interpolation: public sets are dense with near-duplicate
analogues, so random assignment scatters a congeneric series across both
sides and scores the model on compounds whose close relatives it has
memorized.

.. doctest::

   >>> model = train >> fit("rf", random_state=0)
   >>> model.predict(test.X).shape
   (6,)

:func:`fit` accepts any estimator, not just the built-in names:

.. doctest::

   >>> from sklearn.linear_model import Ridge
   >>> model = train >> fit(Ridge(alpha=1.0))
   >>> type(model).__name__
   'Ridge'

:func:`cross_validate` and :func:`applicability_domain` end a pipe the
same way:

.. doctest::

   >>> from qsarkit.functional import applicability_domain, cross_validate
   >>> report = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> fingerprint(n_bits=512)
   ...     >> cross_validate("rf", n_splits=3, random_state=0)
   ... )
   >>> sorted(report)[:3]
   ['mae_cv', 'method', 'n_splits']
   >>> domain = train >> applicability_domain("tanimoto", threshold=0.3)
   >>> domain.predict(test.X).shape
   (6,)

.. warning::

   :func:`scale` and :func:`select_features` fit on whatever flows
   through them. Placing them *before* :func:`split` leaks the test set's
   statistics — and, for selection, its labels — into the transform, and
   the held-out score comes out optimistic. Put them after the split, or
   inside a :class:`sklearn.pipeline.Pipeline` given to
   :func:`cross_validate`.

Drawing the pipeline
--------------------

A pipeline is a graph, and drawing it is the quickest way to confirm the
stages are in the order you meant:

.. doctest::

   >>> pipe = standardize() >> drop_invalid() >> fingerprint() >> scale() >> fit("rf")
   >>> figure = pipe.plot()                    # Plotly, no extra dependency
   >>> type(figure).__name__
   'Figure'
   >>> print(pipe.to_dot().splitlines()[0])    # Graphviz DOT source
   digraph qsarkit_pipeline {

:meth:`~PipeStep.render` writes PNG, PDF or SVG, using Graphviz when it
is installed and Plotly otherwise::

   pipe.render("workflow.pdf")

Nodes are coloured by domain — molecules, features, terminal — so the
point where the pipeline crosses from chemistry into a feature matrix is
visible at a glance.

Writing your own steps
----------------------

:func:`step` and :func:`feature_step` turn an ordinary function into a
dual-mode pipe stage: called without data it defers, called with data it
runs immediately.

.. doctest::

   >>> from qsarkit.functional import step
   >>> @step
   ... def heaviest(mols, y=None, n=5):
   ...     order = sorted(range(len(mols)), key=lambda i: -mols[i].GetNumHeavyAtoms())
   ...     keep = order[:n]
   ...     return [mols[i] for i in keep], (None if y is None else y[keep])
   >>> mols, y = molecules(DEMO_SMILES, DEMO_Y) >> heaviest(n=3)
   >>> len(mols), len(y)
   (3, 3)

Core types
----------

.. autoclass:: MoleculeSet
   :members:
   :show-inheritance:

.. autoclass:: FeatureSet
   :members:
   :show-inheritance:

.. autoclass:: PipeStep
   :members:
   :show-inheritance:

.. autoclass:: Step
   :members:
   :show-inheritance:

.. autoclass:: FeatureStep
   :members:
   :show-inheritance:

.. autofunction:: molecules
.. autofunction:: step
.. autofunction:: feature_step
.. autofunction:: pipeline

Curation steps
--------------

.. autofunction:: standardize
.. autofunction:: desalt
.. autofunction:: neutralize
.. autofunction:: canonicalize_tautomers
.. autofunction:: deglycate
.. autofunction:: remove_protecting_groups
.. autofunction:: drop_invalid
.. autofunction:: remove_duplicates
.. autofunction:: balance
.. autofunction:: keep_if
.. autofunction:: drop_if
.. autofunction:: filter_by_property
.. autofunction:: to_pactivity
.. autofunction:: sample
.. autofunction:: shuffle
.. autofunction:: apply

Representation steps
--------------------

.. autofunction:: featurize
.. autofunction:: fingerprint
.. autofunction:: describe

Feature steps
-------------

.. autofunction:: scale
.. autofunction:: impute
.. autofunction:: drop_constant
.. autofunction:: drop_correlated
.. autofunction:: select_features

Terminal steps
--------------

.. autofunction:: split
.. autofunction:: fit
.. autofunction:: cross_validate
.. autofunction:: applicability_domain
.. autofunction:: collect

Flowchart
---------

.. autoclass:: PipelineNode
   :members:

.. autofunction:: pipeline_nodes
.. autofunction:: to_dot
.. autofunction:: plot_pipeline
.. autofunction:: render_pipeline

References
----------

- Bache, S. M. & Wickham, H. (2014). "magrittr: A Forward-Pipe Operator
  for R." https://CRAN.R-project.org/package=magrittr
- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and
  QSAR Modeling Research." J. Chem. Inf. Model., 50(7), 1189-1204.
  :doi:`10.1021/ci100176x`
- Heller, S. R. et al. (2015). "InChI - the Worldwide Chemical Structure
  Identifier Standard." J. Cheminform., 7, 23.
  :doi:`10.1186/s13321-015-0068-4`
