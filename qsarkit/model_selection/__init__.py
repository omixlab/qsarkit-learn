"""QSAR-aware data splitting and hyperparameter search.

A random split flatters a QSAR model: molecular datasets are dense with
near-duplicate analogues, so random assignment puts close relatives on
both sides and measures interpolation rather than generalization to new
chemistry. The splitters here make the evaluation harder in specific,
defensible ways -- and the gap between a random and a scaffold split is
the size of the illusion.

Examples
--------
>>> from rdkit import Chem
>>> from qsarkit.model_selection import ScaffoldSplitter
>>> mols = [Chem.MolFromSmiles(s) for s in
...         ("c1ccccc1C", "c1ccccc1CC", "c1ccncc1C", "CCO", "CCN")]
>>> train, test = next(ScaffoldSplitter(test_size=0.4).split_mols(mols))
>>> set(train) & set(test)
set()

References
----------
- Wu, Z. et al. (2018). "MoleculeNet: A Benchmark for Molecular Machine
  Learning." Chem. Sci., 9, 513-530. https://doi.org/10.1039/C7SC02664A
- Sheridan, R. P. (2013). "Time-Split Cross-Validation." J. Chem. Inf.
  Model., 53(4), 783-790. https://doi.org/10.1021/ci400084k
"""

from qsarkit.model_selection._search import NestedCV, hyperparameter_search
from qsarkit.model_selection._splitters import (
    BaseSplitter,
    ButinaClusterSplitter,
    KennardStoneSplitter,
    MaxMinSplitter,
    PerimeterSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    SphereExclusionSplitter,
    StratifiedScaffoldSplitter,
    TimeSplitter,
)

__all__ = [
    "BaseSplitter",
    "RandomSplitter",
    "ScaffoldSplitter",
    "StratifiedScaffoldSplitter",
    "ButinaClusterSplitter",
    "SphereExclusionSplitter",
    "MaxMinSplitter",
    "TimeSplitter",
    "KennardStoneSplitter",
    "PerimeterSplitter",
    "NestedCV",
    "hyperparameter_search",
]
