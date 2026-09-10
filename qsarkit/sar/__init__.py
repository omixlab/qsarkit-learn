"""Structure-activity relationship analysis: MMPs, activity cliffs, SAR tables.

This module answers the interpretation questions that come after a QSAR
model is fitted: which structural changes drive activity, where the SAR
is smooth versus discontinuous, and which compound pairs form activity
cliffs that any similarity-based model will struggle with.

Examples
--------
>>> from rdkit import Chem
>>> from qsarkit.sar import ActivityCliffDetector, activity_cliff_report
>>> mols = [Chem.MolFromSmiles(s) for s in
...         ("CC(=O)Nc1ccc(Cl)cc1", "CC(=O)Nc1ccc(Br)cc1")]
>>> cliffs = ActivityCliffDetector(similarity_threshold=0.6).detect(mols, [9.0, 5.0])
>>> len(cliffs)
1
>>> activity_cliff_report(mols, [9.0, 5.0], similarity_threshold=0.6)["cliff_ratio"]
1.0
"""

from qsarkit.sar._cliffs import (
    ActivityCliff,
    ActivityCliffDetector,
    ActivityLandscapePlotter,
    SALIAnalyzer,
    SARIAnalyzer,
    activity_cliff_report,
)
from qsarkit.sar._mmp import MatchedMolecularPairs, MatchedPair, MMPAnalyzer
from qsarkit.sar._rgroup import FreeWilsonAnalysis, RGroupAnalyzer, SARTable

__all__ = [
    "MatchedPair",
    "MatchedMolecularPairs",
    "MMPAnalyzer",
    "ActivityCliff",
    "ActivityCliffDetector",
    "SALIAnalyzer",
    "SARIAnalyzer",
    "ActivityLandscapePlotter",
    "activity_cliff_report",
    "RGroupAnalyzer",
    "SARTable",
    "FreeWilsonAnalysis",
]
