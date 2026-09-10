"""Named, dense molecular-descriptor transformers.

Every class here is a stateless :class:`qsarkit.base.MoleculeTransformer`
mapping ``Iterable[rdkit.Chem.Mol]`` to a dense, named ``(n_molecules,
n_features)`` ``float64`` array, NaN-safe against descriptors that raise on
unusual inputs, and scikit-learn compatible (``fit`` / ``transform`` /
``fit_transform`` / ``get_params`` / ``clone``).

References
----------
- Todeschini, R. & Consonni, V. (2009). "Molecular Descriptors for
  Chemoinformatics." Wiley-VCH. https://doi.org/10.1002/9783527628766
- RDKit documentation, "List of Available Descriptors":
  https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors
"""

from __future__ import annotations

from qsarkit.representation.descriptors._3d import Descriptors3D
from qsarkit.representation.descriptors._base import BaseDescriptorTransformer
from qsarkit.representation.descriptors._calculator import DescriptorCalculator
from qsarkit.representation.descriptors._constitutional import ConstitutionalDescriptors
from qsarkit.representation.descriptors._fragments import FragmentDescriptors
from qsarkit.representation.descriptors._lipinski import LipinskiDescriptors
from qsarkit.representation.descriptors._physicochemical import PhysicochemicalDescriptors
from qsarkit.representation.descriptors._rdkit_descriptors import RDKitDescriptors

__all__ = [
    "BaseDescriptorTransformer",
    "RDKitDescriptors",
    "Descriptors3D",
    "ConstitutionalDescriptors",
    "PhysicochemicalDescriptors",
    "LipinskiDescriptors",
    "FragmentDescriptors",
    "DescriptorCalculator",
]
