"""Chemical structure curation performed before modeling.

Standardization is the step that makes a QSAR dataset comparable at all:
without it the same compound appears as several distinct structures and
duplicate detection, splitting and modeling all quietly go wrong. The
remaining components support that work -- glycan handling for
natural-product datasets, protecting-group removal, scaffold extraction
and graph descriptors.

All components accept ``Iterable[rdkit.Chem.Mol]``.

Examples
--------
>>> from rdkit import Chem
>>> from qsarkit.chemistry import MolecularStandardizer
>>> mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-].[Na+]")
>>> Chem.MolToSmiles(MolecularStandardizer().transform([mol])[0])
'CC(=O)Oc1ccccc1C(=O)O'

References
----------
- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and
  QSAR Modeling Research." J. Chem. Inf. Model., 50(7), 1189-1204.
  https://doi.org/10.1021/ci100176x
- Bemis, G. W. & Murcko, M. A. (1996). "The Properties of Known Drugs. 1.
  Molecular Frameworks." J. Med. Chem., 39(15), 2887-2893.
  https://doi.org/10.1021/jm9602928
- RDKit: Open-source cheminformatics. https://www.rdkit.org
"""

from qsarkit.chemistry.fragments import CoreExtractor, FragmentRemover
from qsarkit.chemistry.glycans import GlycanDescriptors, GlycanDetector, GlycanRemover
from qsarkit.chemistry.graph import MolecularGraph
from qsarkit.chemistry.standardization import MolecularStandardizer

__all__ = [
    "MolecularStandardizer",
    "GlycanDetector",
    "GlycanRemover",
    "GlycanDescriptors",
    "FragmentRemover",
    "CoreExtractor",
    "MolecularGraph",
]
