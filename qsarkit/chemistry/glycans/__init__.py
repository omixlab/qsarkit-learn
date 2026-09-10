"""Carbohydrate (glycan) detection, removal and descriptor generation."""

from qsarkit.chemistry.glycans._descriptors import GlycanDescriptors
from qsarkit.chemistry.glycans._detector import GlycanDetector, GlycanMatch
from qsarkit.chemistry.glycans._remover import GlycanRemover

__all__ = ["GlycanDetector", "GlycanMatch", "GlycanRemover", "GlycanDescriptors"]
