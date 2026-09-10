Utilities
=========

.. currentmodule:: qsarkit.utils

I/O, unit conversion, validation helpers, logging and package constants.

Activity units
--------------

The single most damaging silent error in QSAR is a mixed activity column.
Converting to a p-scale first makes the units explicit and puts the
values on the scale the models assume.

.. doctest::

   >>> from qsarkit.utils import from_pactivity, to_pactivity
   >>> float(to_pactivity(1.0, unit="nM"))
   9.0
   >>> float(to_pactivity(1000.0, unit="nM"))
   6.0
   >>> round(float(from_pactivity(9.0, "nM")), 6)
   1.0

pIC50 = −log10(IC50 in molar), so 1 nM is 9 and a thousand-fold weaker
compound is 6. Working on the p-scale also makes the errors
approximately normal, which is what every regression metric here assumes.

.. doctest::

   >>> from qsarkit.utils import convert_concentration, nm_to_molar
   >>> round(float(nm_to_molar(1000.0)), 12)
   1e-06
   >>> round(float(convert_concentration(1.0, "uM", "nM")), 6)
   1000.0

Binding free energy, for comparison with calorimetry or docking scores:

.. doctest::

   >>> from qsarkit.utils import pactivity_to_delta_g
   >>> round(float(pactivity_to_delta_g(9.0)), 2)
   -12.28

I/O
---

.. doctest::

   >>> from qsarkit.utils import mols_to_dataframe, read_smiles, write_smiles
   >>> frame = mols_to_dataframe(demo_mols[:3], extra={"pIC50": DEMO_Y[:3]})
   >>> list(frame.columns)
   ['smiles', 'pIC50']

.. doctest::

   >>> import tempfile, os
   >>> path = os.path.join(tempfile.mkdtemp(), "demo.smi")
   >>> write_smiles(demo_mols[:3], path)      # returns the count written
   3
   >>> len(read_smiles(path))
   3

Validation helpers
------------------

.. doctest::

   >>> from qsarkit.utils import check_X_y_mols, check_mols
   >>> len(check_mols(demo_mols))
   24
   >>> mols, y = check_X_y_mols(demo_mols, DEMO_Y)
   >>> len(mols) == len(y)
   True

Constants
---------

.. doctest::

   >>> from qsarkit.utils import CONCENTRATION_TO_MOLAR, LIPINSKI_THRESHOLDS
   >>> CONCENTRATION_TO_MOLAR["nM"]
   1e-09
   >>> LIPINSKI_THRESHOLDS["mw_max"]
   500.0

API
---

.. automodule:: qsarkit.utils
   :members:
   :show-inheritance:

References
----------

- Kalliokoski, T. et al. (2013). "Comparability of Mixed IC50 Data — A
  Statistical Analysis." PLoS ONE, 8(4), e61007.
  :doi:`10.1371/journal.pone.0061007`
- Bento, A. P. et al. (2014). "The ChEMBL Bioactivity Database: An
  Update." Nucleic Acids Res., 42, D1083-D1090.
  :doi:`10.1093/nar/gkt1031`
- Tiesinga, E. et al. (2021). "CODATA Recommended Values of the
  Fundamental Physical Constants: 2018." Rev. Mod. Phys., 93, 025010.
  :doi:`10.1103/RevModPhys.93.025010`
