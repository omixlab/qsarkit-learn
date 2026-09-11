Base
====

.. currentmodule:: qsarkit.base

Shared estimator base classes, the exception hierarchy, and the
lazy-import helper that keeps heavy optional dependencies out of
``import qsarkit``.

Writing a transformer
---------------------

The contract is one method. Implement ``_transform`` on a list of
molecules, and inherit input validation, ``fit``, ``fit_transform`` and
full scikit-learn compatibility:

.. doctest::

   >>> import numpy as np
   >>> from rdkit import Chem
   >>> from qsarkit.base import MoleculeTransformer
   >>> class HeavyAtomCount(MoleculeTransformer):
   ...     def _transform(self, mols):
   ...         return np.array([[m.GetNumHeavyAtoms()] for m in mols], dtype=float)
   >>> HeavyAtomCount().fit_transform(demo_mols).shape
   (24, 1)

It is a real estimator, so it composes and clones:

.. doctest::

   >>> from sklearn.base import clone
   >>> clone(HeavyAtomCount())
   HeavyAtomCount()

Three bases, by what they return
--------------------------------

:class:`MoleculeTransformer`
   Molecules in, feature matrix out.
:class:`MoleculeToMoleculeTransformer`
   Molecules in, molecules out — standardization and curation. The
   separate type is what lets these chain with each other.
:class:`FittableMoleculeTransformer`
   Adds an ``is_fitted`` flag and a guard, so calling ``transform``
   before ``fit`` raises rather than producing meaningless features:

.. doctest::

   >>> from qsarkit.base import FittableMoleculeTransformer
   >>> class MeanCentredSize(FittableMoleculeTransformer):
   ...     def fit(self, mols, y=None):
   ...         self.mean_ = np.mean([m.GetNumHeavyAtoms() for m in mols])
   ...         self._is_fitted = True
   ...         return self
   ...     def _transform(self, mols):
   ...         self._check_is_fitted()
   ...         return np.array([m.GetNumHeavyAtoms() - self.mean_ for m in mols])
   >>> MeanCentredSize().transform(demo_mols)
   Traceback (most recent call last):
       ...
   qsarkit.base.exceptions.ModelNotFittedError: MeanCentredSize must be fitted...

Input validation
----------------

.. doctest::

   >>> from qsarkit.base import ensure_mol_list
   >>> ensure_mol_list([None])
   [None]

``None`` passes through deliberately: a molecule that failed an earlier
parsing step must keep its position, because dropping it here would
silently shift every downstream label by one. Anything that is neither a
molecule nor ``None`` is a programming error and is reported as one:

.. doctest::

   >>> ensure_mol_list(["CCO"])
   Traceback (most recent call last):
       ...
   qsarkit.base.exceptions.InvalidMoleculeError: Element 0 is not an rdkit.Chem.Mol...

Optional dependencies
---------------------

.. doctest::

   >>> from qsarkit.base import require
   >>> require("numpy").__name__
   'numpy'

A missing dependency raises an error naming the extra that provides it,
rather than an ``ImportError`` the user has to interpret:

.. doctest::

   >>> require("nonexistent_package_xyz")
   Traceback (most recent call last):
       ...
   qsarkit.base.exceptions.OptionalDependencyError: This feature requires...

API
---

.. automodule:: qsarkit.base
   :members:
   :show-inheritance:

References
----------

- Pedregosa, F. et al. (2011). "Scikit-learn: Machine Learning in
  Python." J. Mach. Learn. Res., 12, 2825-2830.
  https://jmlr.org/papers/v12/pedregosa11a.html
- Buitinck, L. et al. (2013). "API Design for Machine Learning Software:
  Experiences from the scikit-learn Project." ECML PKDD Workshop.
  :arxiv:`1309.0238`
