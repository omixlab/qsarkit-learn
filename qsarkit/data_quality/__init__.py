"""Dataset curation: duplicates, structural validity and activity sanity.

Curation changes QSAR model performance more than the choice of
algorithm does. This module implements the protocol of Fourches, Muratov
and Tropsha, and records what it removed so the result is auditable.

Examples
--------
>>> from rdkit import Chem
>>> from qsarkit.data_quality import DataCurationPipeline
>>> mols = [Chem.MolFromSmiles(s) for s in ("CCO", "OCC", "[Na+].[Cl-]")]
>>> curated, y, report = DataCurationPipeline().run(mols, [5.0, 5.2, 1.0])
>>> report.n_output < report.n_input
True

References
----------
- Fourches, D., Muratov, E. & Tropsha, A. (2010). "Trust, But Verify: On
  the Importance of Chemical Structure Curation in Cheminformatics and
  QSAR Modeling Research." J. Chem. Inf. Model., 50(7), 1189-1204.
  https://doi.org/10.1021/ci100176x
- Fourches, D., Muratov, E. & Tropsha, A. (2016). "Trust, but Verify II."
  J. Chem. Inf. Model., 56(7), 1243-1252.
  https://doi.org/10.1021/acs.jcim.6b00129
- Tropsha, A. (2010). "Best Practices for QSAR Model Development,
  Validation, and Exploitation." Mol. Inform., 29(6-7), 476-488.
  https://doi.org/10.1002/minf.201000061
"""

from qsarkit.data_quality._duplicates import (
    DuplicateDetector,
    DuplicateGroup,
    merge_replicates,
)
from qsarkit.data_quality._pipeline import CurationReport, DataCurationPipeline
from qsarkit.data_quality._validators import (
    ActivityOutlierDetector,
    StructureValidator,
    ValidationIssue,
    check_activity_units,
)

__all__ = [
    "DuplicateDetector",
    "DuplicateGroup",
    "merge_replicates",
    "StructureValidator",
    "ValidationIssue",
    "ActivityOutlierDetector",
    "check_activity_units",
    "DataCurationPipeline",
    "CurationReport",
]
