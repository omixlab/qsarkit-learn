User guide
==========

Workflow-oriented walkthroughs. Each page covers one stage of the QSAR
workflow and links to the API reference for the details.

Every example in this documentation is executed by the test suite, so the
output shown is what the code actually produces. They share one demo
dataset — 24 compounds in four substituent families with synthetic pIC50
values, defined in ``docs/source/demo_data.py`` — available in every
example as ``DEMO_SMILES``, ``DEMO_Y`` and ``demo_mols``.

For runnable end-to-end walkthroughs, see the four
`example notebooks <https://github.com/fredericokremer/qsarkit-learn/tree/main/notebooks>`_,
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
