"""Molecular representations: fingerprints, descriptors and embeddings.

Every transformer accepts ``Iterable[rdkit.Chem.Mol]``, returns a NumPy
array, and exposes ``get_feature_names_out()``.

Examples
--------
>>> from rdkit import Chem
>>> from qsarkit.representation import MorganFingerprint
>>> X = MorganFingerprint(n_bits=256).transform([Chem.MolFromSmiles("CCO")])
>>> X.shape
(1, 256)

References
----------
- Rogers, D. & Hahn, M. (2010). "Extended-Connectivity Fingerprints."
  J. Chem. Inf. Model., 50(5), 742-754. https://doi.org/10.1021/ci100050t
- RDKit: Open-source cheminformatics. https://www.rdkit.org
"""

from qsarkit.representation.descriptors import (
    BaseDescriptorTransformer,
    ConstitutionalDescriptors,
    DescriptorCalculator,
    Descriptors3D,
    FragmentDescriptors,
    LipinskiDescriptors,
    PhysicochemicalDescriptors,
    RDKitDescriptors,
)
from qsarkit.representation.fingerprints import (
    AtomPairFingerprint,
    AvalonFingerprint,
    BaseFingerprintTransformer,
    FeatureMorganFingerprint,
    FingerprintCombiner,
    LayeredFingerprint,
    MACCSKeysFingerprint,
    MAP4Fingerprint,
    MHFPFingerprint,
    MorganFingerprint,
    PatternFingerprint,
    PharmacophoreFingerprint,
    RDKitFingerprint,
    TopologicalTorsionFingerprint,
    fold_on_bits,
)
from qsarkit.representation.embeddings import ChemBERTaTransformer
from qsarkit.representation.mol2vec import Mol2VecTransformer, mol_to_sentence

__all__ = [
    # fingerprints
    "BaseFingerprintTransformer", "fold_on_bits",
    "MorganFingerprint", "FeatureMorganFingerprint",
    "RDKitFingerprint", "PatternFingerprint", "LayeredFingerprint",
    "AtomPairFingerprint", "TopologicalTorsionFingerprint",
    "MACCSKeysFingerprint", "AvalonFingerprint", "PharmacophoreFingerprint",
    "MHFPFingerprint", "MAP4Fingerprint", "FingerprintCombiner",
    # descriptors
    "BaseDescriptorTransformer", "RDKitDescriptors", "Descriptors3D",
    "ConstitutionalDescriptors", "PhysicochemicalDescriptors",
    "LipinskiDescriptors", "FragmentDescriptors", "DescriptorCalculator",
    # embeddings
    "Mol2VecTransformer", "mol_to_sentence", "ChemBERTaTransformer",
]
