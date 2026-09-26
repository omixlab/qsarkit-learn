"""Model validation against the OECD principles.

Every validator here takes a ``scoring`` argument. It defaults to
:math:`R^2`, which is the right default for a regression QSAR and wrong for
everything else -- a toxicity classifier has to be argued in ROC-AUC or
average precision. Name a metric from :func:`available_metrics`, pass a
``(y_true, y_pred)`` callable, or wrap one with :func:`make_scorer` when it
needs probabilities or is a loss. Pass several metrics and every score in the
result becomes an array in the order given, so one pass reports them all::

    CrossValidator(scoring=["roc_auc", "pr_auc", "mcc"]).evaluate(model, X, y)

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
  Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
- Golbraikh, A. & Tropsha, A. (2002). "Beware of q2!" J. Mol. Graph.
  Model., 20(4), 269-276. https://doi.org/10.1016/S1093-3263(01)00123-1
- Rucker, C., Rucker, G. & Meringer, M. (2007). "y-Randomization and Its
  Variants in QSPR/QSAR." J. Chem. Inf. Model., 47(6), 2345-2357.
  https://doi.org/10.1021/ci700157b
"""

from qsarkit.validation._cross_validation import CrossValidator
from qsarkit.validation._robustness import (
    BootstrapValidator,
    ExternalValidator,
    YScrambling,
)
from qsarkit.validation._scoring import Scorer, available_metrics, make_scorer

__all__ = [
    "CrossValidator",
    "YScrambling",
    "ExternalValidator",
    "BootstrapValidator",
    # Metric selection, shared by all four
    "Scorer",
    "make_scorer",
    "available_metrics",
]
