"""Learned molecular embeddings: ChemBERTa and Mol2Vec.

References
----------
- Chithrananda, S., Grand, G. & Ramsundar, B. (2020). "ChemBERTa:
  Large-Scale Self-Supervised Pretraining for Molecular Property
  Prediction." https://arxiv.org/abs/2010.09885
- Jaeger, S., Fulle, S. & Turk, S. (2018). "Mol2vec: Unsupervised Machine
  Learning Approach with Chemical Intuition." J. Chem. Inf. Model.,
  58(1), 27-35. https://doi.org/10.1021/acs.jcim.7b00616
"""

from qsarkit.representation.embeddings._chemberta import ChemBERTaTransformer
from qsarkit.representation.mol2vec import Mol2VecTransformer

__all__ = ["ChemBERTaTransformer", "Mol2VecTransformer"]
