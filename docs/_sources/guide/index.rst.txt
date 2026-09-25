User guide
==========

Workflow-oriented walkthroughs. Each page covers one stage of the QSAR
workflow and links to the API reference for the details.

Every example in this documentation is executed by the test suite, so the
output shown is what the code actually produces — including the
unflattering numbers. They share one demo dataset, defined in
``docs-sphinx/source/demo_data.py`` and available in every example as
``DEMO_SMILES``, ``DEMO_Y`` and ``demo_mols``.

The demo set is 24 compounds in four substituent series with synthetic
pIC50 values, resolving to three Bemis-Murcko scaffolds. It is deliberately
small, so every example runs instantly, and deliberately *hard*: one
compound is a planted activity-cliff outlier and the scaffold families are
distinct enough that a scaffold split is genuinely difficult. Several pages
here show models scoring badly on it. That is the point — an example where
everything succeeds teaches nothing about the failure modes these tools
exist to detect.

For runnable end-to-end walkthroughs, see the five
`example notebooks <https://github.com/omixlab/qsarkit-learn/tree/main/notebooks>`_,
which together cover every public subpackage.

.. toctree::
   :maxdepth: 2

   installation
   quickstart
   functional
   curation
   representation
   modeling
   applicability
   sar
   oecd
   deployment
