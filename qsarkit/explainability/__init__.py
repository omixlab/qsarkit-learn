"""Model interpretation -- OECD validation principle 5.

Two families, answering different questions:

- **Feature attribution** (:class:`SHAPExplainer`,
  :class:`PermutationImportance`, :class:`LIMEExplainer`,
  :class:`PartialDependence`) works on the descriptor matrix and says
  which columns the model uses.
- **Structural attribution** (:class:`AtomicContributionMap`,
  :class:`FragmentContributionAnalyzer`,
  :class:`CounterfactualExplainer`) maps the explanation back onto the
  molecule, which is what makes it actionable for a chemist.

For a fingerprint model the second family is the one that matters: 2048
anonymous bits are not an explanation, but "the nitro group costs you a
log unit" is.

Examples
--------
>>> from rdkit import Chem
>>> from sklearn.ensemble import RandomForestRegressor
>>> from qsarkit.explainability import AtomicContributionMap
>>> from qsarkit.representation import MorganFingerprint
>>> fp = MorganFingerprint(n_bits=64)
>>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "CCN", "c1ccccc1", "CCC")]
>>> model = RandomForestRegressor(n_estimators=5, random_state=0).fit(
...     fp.transform(mols), [1.0, 2.0, 3.0, 4.0]
... )
>>> AtomicContributionMap(model, fp).explain(mols[0]).weights.shape
(3,)

References
----------
- Lundberg, S. M. & Lee, S.-I. (2017). "A Unified Approach to
  Interpreting Model Predictions." NeurIPS 2017.
  https://arxiv.org/abs/1705.07874
- Riniker, S. & Landrum, G. A. (2013). "Similarity Maps - A
  Visualization Strategy for Molecular Fingerprints and Machine-Learning
  Methods." J. Cheminform., 5, 43.
  https://doi.org/10.1186/1758-2946-5-43
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust
  You?" KDD 2016, 1135-1144. https://doi.org/10.1145/2939672.2939778
- OECD (2007). Guidance Document No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
"""

from qsarkit.explainability._atom_maps import (
    AttributionAtomMapper,
    bit_atom_environments,
    bit_weights_to_atom_weights,
    draw_atom_weights,
)
from qsarkit.explainability._atomic import (
    AtomicContribution,
    AtomicContributionMap,
    CounterfactualExplainer,
    FragmentContributionAnalyzer,
)
from qsarkit.explainability._importance import (
    LIMEExplainer,
    PartialDependence,
    PermutationImportance,
    SHAPExplainer,
)

__all__ = [
    # per-atom projection of feature attributions
    "AttributionAtomMapper",
    "bit_atom_environments",
    "bit_weights_to_atom_weights",
    "draw_atom_weights",
    "PermutationImportance",
    "SHAPExplainer",
    "LIMEExplainer",
    "PartialDependence",
    "AtomicContribution",
    "AtomicContributionMap",
    "FragmentContributionAnalyzer",
    "CounterfactualExplainer",
]
