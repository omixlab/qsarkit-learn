"""Mol2vec unsupervised molecular embeddings.

References
----------
- Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec: Unsupervised Machine
  Learning Approach with Chemical Intuition." J. Chem. Inf. Model., 58(1),
  27-35. https://doi.org/10.1021/acs.jcim.7b00616
"""

from __future__ import annotations

from qsarkit.representation.mol2vec._mol2vec import Mol2VecTransformer, mol_to_sentence

__all__ = ["Mol2VecTransformer", "mol_to_sentence"]
