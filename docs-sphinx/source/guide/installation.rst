Installation
============

.. code-block:: bash

   pip install git+https://github.com/omixlab/qsarkit-learn

That installs the core: NumPy, SciPy, pandas, scikit-learn, RDKit,
NetworkX, requests and Plotly. Everything in :mod:`qsarkit.chemistry`,
:mod:`qsarkit.sar`, :mod:`qsarkit.applicability`, :mod:`qsarkit.neighbors`
and :mod:`qsarkit.cluster` works with just that.

Optional extras
---------------

Heavier features load their dependencies lazily, so you install only what
you use. A missing dependency raises
:class:`~qsarkit.base.OptionalDependencyError`, which names the extra to
install.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Enables
   * - ``embeddings``
     - Mol2Vec (``gensim``)
   * - ``nlp``
     - ChemBERTa embeddings (``transformers``, ``torch``)
   * - ``explainability``
     - SHAP and LIME attribution
   * - ``boosting``
     - XGBoost and LightGBM estimators
   * - ``embedding_viz``
     - UMAP chemical-space projections (``umap-learn``)
   * - ``reporting``
     - Static image export for reports (``kaleido``)

.. code-block:: bash

   pip install "qsarkit[explainability]"   # pick what you need
   pip install "qsarkit[all]"              # or take everything

A missing optional dependency raises an error naming the extra that
provides it, rather than an ``ImportError`` you have to interpret.

Development install
-------------------

.. code-block:: bash

   git clone https://github.com/omixlab/qsarkit-learn
   cd qsarkit-learn
   pip install -e ".[dev]"

   pytest                             # run the suite
   pytest -m slow                     # execute the example notebooks too
   pytest --cov=qsarkit --cov-branch  # with coverage
   mypy qsarkit                       # strict type check
   ruff check qsarkit                 # lint

Verifying a source distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The sdist ships the test suite, so a packager can verify a build without
the repository:

.. code-block:: bash

   tar xf qsarkit-0.3.0.tar.gz
   cd qsarkit-0.3.0
   pip install ".[dev]"
   pytest                             # the notebooks need `-m slow`

Building the documentation. The sources are in ``docs-sphinx/``; the built
site lives in ``docs/`` and is committed, because that is what GitHub Pages
serves:

.. code-block:: bash

   pip install -e ".[docs]"

   make docs        # rebuild and copy the site into docs/
   make preview     # build into docs-sphinx/build/html, leaving docs/ alone
   make doctest     # execute every example in this documentation
   make linkcheck   # verify external links resolve

``make docs`` builds from scratch with warnings treated as errors, so a
broken cross-reference cannot reach the published site and a page deleted
from the source cannot linger in it. The same targets are available from
inside ``docs-sphinx/``, where the one that writes to ``../docs`` is called
``make publish``.

.. note::

   ``typings/`` holds a local stub shadow for RDKit. The ``rdkit-stubs``
   bundled with the RDKit wheel contain an auto-generation bug — a C++ enum
   member named ``None``, which is an illegal annotation target in a
   ``.pyi`` — that otherwise aborts every mypy run and silently hides all
   real type errors. Keep it.
