qsarkit
=======

A focused, open-source Python library for QSAR modeling.

qsarkit covers the QSAR workflow proper — curating structures, turning
them into features, fitting and validating a model, defining where it
applies, and interpreting what it learned. It deliberately stops there:
it is not a literature-mining, database-retrieval, docking or de-novo
design toolkit, and does not pretend to be.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Getting started
      :link: guide/installation
      :link-type: doc

      Install qsarkit and run your first QSAR model.

   .. grid-item-card:: User guide
      :link: guide/index
      :link-type: doc

      Workflow-oriented walkthroughs of each stage.

   .. grid-item-card:: API reference
      :link: api/index
      :link-type: doc

      Every public class and function, with its scientific references.

   .. grid-item-card:: OECD validation
      :link: guide/oecd
      :link-type: doc

      Validating and reporting a model against the five OECD principles.

The workflow
------------

.. code-block:: text

    Structures + measured activities
              |
              v
    Curation                   qsarkit.chemistry, qsarkit.data_quality
              |
              v
    Representation             qsarkit.representation
              |
              v
    Modeling                   qsarkit.models, qsarkit.model_selection,
                               qsarkit.feature_selection
              |
              v
    Validation                 qsarkit.validation, qsarkit.metrics
              |
              v
    Applicability domain       qsarkit.applicability
              |
              v
    Uncertainty                qsarkit.uncertainty
              |
              v
    Interpretation             qsarkit.sar, qsarkit.explainability
              |
              v
    Reporting                  qsarkit.reporting

Design principles
-----------------

**Everything is a scikit-learn estimator.**
   Transformers accept ``Iterable[rdkit.Chem.Mol]`` and implement
   ``fit``/``transform``/``fit_transform``; models implement
   ``fit``/``predict``. They compose in :class:`sklearn.pipeline.Pipeline`,
   work with :class:`~sklearn.model_selection.GridSearchCV`, and survive
   :func:`sklearn.base.clone`.

**Every algorithm cites its source.**
   Each class documents the original publication with a DOI and, where it
   wraps one, the official implementation documentation. There are no
   undocumented algorithms in this package.

**Typed and checked.**
   The package is ``mypy --strict`` clean and ships a ``py.typed`` marker.

**Narrow on purpose.**
   Everything here earns its place in the QSAR workflow. Data
   acquisition, molecular generation and structure-based methods are
   deliberately out of scope — they are different disciplines with
   different failure modes, and bundling them makes a library that is
   broad rather than trustworthy.

A first example
---------------

.. code-block:: python

   from rdkit import Chem
   from qsarkit.chemistry import MolecularStandardizer
   from qsarkit.sar import activity_cliff_report

   # Curate: strip the salt, neutralize the charge
   standardizer = MolecularStandardizer()
   mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
   Chem.MolToSmiles(standardizer.transform([mol])[0])
   # 'CC(=O)Oc1ccccc1C(=O)O'

   # Diagnose the dataset before modeling it
   report = activity_cliff_report(mols, pIC50_values)
   report["cliff_ratio"]           # how much of the SAR is discontinuous
   report["top_transformations"]   # which R-group swaps cause the cliffs

.. toctree::
   :maxdepth: 2
   :caption: Contents
   :hidden:

   guide/index
   api/index
   references
   changelog

Citing qsarkit
--------------

If qsarkit contributes to work you publish, please cite the package along
with the primary reference for whichever algorithm you used — each class
docstring names it.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
