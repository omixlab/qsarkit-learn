Modeling
========

Estimators
----------

.. code-block:: python

   from qsarkit.models import QSARRegressor, QSARClassifier, ConsensusModel

   QSARRegressor("rf").fit(X, y)
   QSARRegressor("gp").fit(X, y)          # Tanimoto-kernel Gaussian process
   QSARRegressor("pls", n_components=5)   # with VIP scores
   ConsensusModel([QSARRegressor("rf"), QSARRegressor("svm")])

All follow the scikit-learn API, so they drop into
:class:`~sklearn.pipeline.Pipeline`,
:class:`~sklearn.model_selection.GridSearchCV` and
:func:`~sklearn.base.clone`.

Splitting
---------

.. code-block:: python

   from qsarkit.model_selection import ScaffoldSplitter, ButinaClusterSplitter

   train, test = next(ScaffoldSplitter(test_size=0.2).split_mols(mols))

.. important::

   A random split flatters a QSAR model. Public datasets are full of
   near-duplicate analogues, so a random split puts close relatives on both
   sides and measures interpolation, not generalization. Report a scaffold
   or cluster split unless you have a specific reason not to.

:class:`~qsarkit.model_selection.TimeSplitter` is the most honest of all
where you have dates: it reproduces the real prospective task.

Feature selection
-----------------

.. code-block:: python

   from qsarkit.feature_selection import BorutaSelector, CorrelationFilter

   CorrelationFilter(threshold=0.95).fit_transform(X, y)
   BorutaSelector().fit(X, y).get_support()

Metrics
-------

.. code-block:: python

   from qsarkit.metrics import qsar_regression_report, q2_f3, ccc, bedroc

   qsar_regression_report(y_true, y_pred)

Prefer Q²F3 to R² on external sets, and BEDROC to ROC-AUC for virtual
screening, where only the top of the ranked list matters.

References
----------

- Wold, S., Sjöström, M. & Eriksson, L. (2001). "PLS-Regression."
  *Chemom. Intell. Lab. Syst.*, 58(2), 109-130.
  :doi:`10.1016/S0169-7439(01)00155-1`
- Ralaivola, L. et al. (2005). "Graph Kernels for Chemical Informatics."
  *Neural Netw.*, 18(8), 1093-1110. :doi:`10.1016/j.neunet.2005.07.009`
- Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine
  Learning." *Chem. Sci.*, 9, 513-530. :doi:`10.1039/C7SC02664A`
- Truchon, J.-F. & Bayly, C. I. (2007). "Evaluating Virtual Screening
  Methods: Good and Bad Metrics for the 'Early Recognition' Problem."
  *J. Chem. Inf. Model.*, 47(2), 488-508. :doi:`10.1021/ci600426e`
