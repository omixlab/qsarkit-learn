The functional pipe API
=======================

A second way to write the workflow, reading in the order the work
happens — in the spirit of R's ``%>%``. It reaches the whole package, not
just curation: :func:`~qsarkit.functional.featurize` takes any
transformer and :func:`~qsarkit.functional.fit` any estimator, so the
pipe is a way of *writing* the workflow rather than a reimplementation of
part of it.

:func:`~qsarkit.functional.molecules` accepts RDKit molecules, SMILES and
InChI in any mixture:

.. doctest::

   >>> from qsarkit.functional import (
   ...     drop_invalid, fingerprint, fit, molecules, remove_duplicates,
   ...     split, standardize)
   >>> mols, y = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> standardize()
   ...     >> drop_invalid()
   ...     >> remove_duplicates(agg="mean")
   ... )
   >>> len(mols), len(y)
   (24, 24)

A pipeline crosses from molecules into a feature matrix at
:func:`~qsarkit.functional.featurize`, and can run all the way to a
fitted model:

.. doctest::

   >>> train, test = (
   ...     molecules(DEMO_SMILES, DEMO_Y)
   ...     >> standardize()
   ...     >> drop_invalid()
   ...     >> fingerprint("morgan", n_bits=512)
   ...     >> split("scaffold", test_size=0.25)
   ... )
   >>> model = train >> fit("rf", random_state=0)
   >>> model.predict(test.X).shape
   (6,)

Drawing it is the quickest way to confirm the stages are in the order you
meant:

.. doctest::

   >>> pipe = standardize() >> drop_invalid() >> fingerprint() >> fit("rf")
   >>> type(pipe.plot()).__name__
   'Figure'
   >>> print(pipe.to_dot().splitlines()[0])
   digraph qsarkit_pipeline {

``pipe.render("workflow.pdf")`` writes PNG, PDF or SVG.

:func:`~qsarkit.functional.molecules` returns a
:class:`~qsarkit.functional.MoleculeSet` carrying the molecules, the
optional labels, and a provenance log. Each step consumes one and returns
a new one — nothing is mutated in place — and every step keeps ``y``
index-aligned with the molecules. Dropping a molecule drops its label with
it, which is the bookkeeping that makes most hand-written curation scripts
subtly wrong.

The result unpacks straight into ``X, y``.

.. warning::

   **Use** ``>>``\ **, not** ``>``.

   ``>`` cannot work as a pipe operator in Python. The language parses
   ``a > b > c`` as the *chained comparison* ``(a > b) and (b > c)``, so a
   ``>``-based pipe would evaluate the first stage, throw the result away,
   and hand back the comparison of the last two stages. There is no way to
   intercept that from ``__gt__``, and the failure is silent — you get a
   result, just not the one you asked for.

   ``>>`` and ``|`` are ordinary left-associative binary operators, so
   they chain correctly. Piping with ``>`` raises a ``TypeError``
   explaining this rather than misbehaving quietly.

Inspecting the pipeline
-----------------------

.. doctest::

   >>> from qsarkit.functional import desalt, drop_invalid, molecules
   >>> ms = molecules(DEMO_SMILES[:6], DEMO_Y[:6]) >> desalt() >> drop_invalid()
   >>> ms
   <MoleculeSet 6 molecules, y shape (6,), 3 steps>
   >>> ms.history
   ['molecules(n=6)', 'desalt()', 'drop_invalid()']
   >>> ms.smiles[0]                 # canonical SMILES, None where invalid
   'O=C(O)c1ccccc1'
   >>> list(ms.to_frame().columns)  # DataFrame with smiles + y
   ['smiles', 'y']
   >>> len(ms)
   6

``history`` records every step with its arguments, so a curated dataset
carries its own provenance — which is what OECD principle 1 asks you to
document.

Reusable pipelines
------------------

Steps compose with each other, so a curation protocol can be defined once
and applied to several datasets:

.. doctest::

   >>> from qsarkit.functional import remove_duplicates, standardize
   >>> curate = standardize() >> drop_invalid() >> remove_duplicates(max_spread=1.0)
   >>> train = molecules(DEMO_SMILES[:12], DEMO_Y[:12]) >> curate
   >>> test = molecules(DEMO_SMILES[12:], DEMO_Y[12:]) >> curate
   >>> len(train), len(test)
   (12, 12)

The composed protocol is itself a step, and prints as one:

.. doctest::

   >>> curate
   <Pipeline standardize() >> drop_invalid() >> remove_duplicates(max_spread=1.0)>

:func:`~qsarkit.functional.pipeline` does the same thing from a list, for
when the steps are assembled programmatically.

Dual-mode steps
---------------

Every step works two ways, so the same function serves the pipe API and
ordinary imperative code:

.. doctest::

   >>> from rdkit import Chem
   >>> mols, y = molecules(["CC(=O)[O-].[Na+]"]) >> desalt()      # deferred
   >>> Chem.MolToSmiles(mols[0])
   'CC(=O)[O-]'
   >>> mols, y = desalt([Chem.MolFromSmiles("CC(=O)[O-].[Na+]")])  # immediate
   >>> Chem.MolToSmiles(mols[0])
   'CC(=O)[O-]'

Calling a step with no molecules returns a deferred
:class:`~qsarkit.functional.Step`; calling it with molecules runs it now.

Available steps
---------------

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Step
     - Does
   * - :func:`~qsarkit.functional.standardize`
     - The full standardization pipeline
   * - :func:`~qsarkit.functional.desalt`
     - Keep the largest organic fragment
   * - :func:`~qsarkit.functional.neutralize`
     - Neutralize charges where a neutral form exists
   * - :func:`~qsarkit.functional.canonicalize_tautomers`
     - Map to the canonical tautomer
   * - :func:`~qsarkit.functional.deglycate`
     - Remove sugars, keeping the aglycone
   * - :func:`~qsarkit.functional.remove_protecting_groups`
     - Strip Boc, Cbz, Fmoc, linkers, tags
   * - :func:`~qsarkit.functional.drop_invalid`
     - Drop ``None`` entries and their labels
   * - :func:`~qsarkit.functional.remove_duplicates`
     - Collapse duplicates, aggregating labels
   * - :func:`~qsarkit.functional.balance`
     - Balance classes by under/oversampling
   * - :func:`~qsarkit.functional.keep_if` / :func:`~qsarkit.functional.drop_if`
     - Filter on an arbitrary predicate
   * - :func:`~qsarkit.functional.filter_by_property`
     - Filter on MW, logP, heavy atoms, rotatable bonds
   * - :func:`~qsarkit.functional.to_pactivity`
     - Convert concentrations to ``-log10`` molar
   * - :func:`~qsarkit.functional.sample` / :func:`~qsarkit.functional.shuffle`
     - Subsample or reorder
   * - :func:`~qsarkit.functional.apply`
     - Apply an arbitrary per-molecule function

Writing your own step
---------------------

Decorate a function with the ``(X, y=None, **params) -> (X, y)`` signature:

.. doctest::

   >>> from qsarkit.functional import step
   >>> @step
   ... def keep_heaviest(X, y=None, n=100):
   ...     from rdkit.Chem import Descriptors
   ...     order = sorted(range(len(X)), key=lambda i: -Descriptors.MolWt(X[i]))
   ...     keep = sorted(order[:n])
   ...     return [X[i] for i in keep], (None if y is None else y[keep])
   >>> mols, y = molecules(DEMO_SMILES, DEMO_Y) >> keep_heaviest(n=5)
   >>> len(mols), len(y)
   (5, 5)

The decorator gives it both modes and the alignment guarantee for free —
returning the subset of ``y`` alongside the molecules is the whole
contract. :func:`~qsarkit.functional.feature_step` does the same for a step
that operates on a feature matrix.

Relationship to the scikit-learn API
------------------------------------

The two APIs are alternatives, not layers — both call the same underlying
classes in :mod:`qsarkit.chemistry`. Use the pipe API for exploratory
curation, where reading order and provenance matter; use
:class:`sklearn.pipeline.Pipeline` when the preprocessing has to be fitted
on training data and reapplied at prediction time, since only that form
participates in cross-validation and model persistence.

References
----------

- Bache, S. M. & Wickham, H. (2014). "magrittr: A Forward-Pipe Operator
  for R." https://CRAN.R-project.org/package=magrittr
- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify."
  *J. Chem. Inf. Model.*, 50(7), 1189-1204. :doi:`10.1021/ci100176x`
