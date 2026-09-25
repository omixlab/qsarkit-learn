"""QSAR model reports in Markdown, HTML and JSON.

A model is reproducible only if the record says what data it was built
on, how that data was curated, which descriptors and algorithm were
used, how it was validated and where it applies.
:class:`OECDReportBuilder` structures that against the five OECD
validation principles, and marks explicitly any principle that has *not*
been addressed.

All plots return Plotly figures and embed directly in the HTML output.

Examples
--------
>>> from qsarkit.reporting import QSARReport
>>> report = QSARReport(title="EGFR pIC50 model", endpoint="pIC50")
>>> _ = report.add_section("Dataset", {"n_compounds": 1200})
>>> "EGFR" in report.to_markdown()
True

References
----------
- OECD (2007). "Guidance Document on the Validation of (Quantitative)
  Structure-Activity Relationship [(Q)SAR] Models." OECD Series on
  Testing and Assessment No. 69, ENV/JM/MONO(2007)2.
  https://doi.org/10.1787/9789264085442-en
- European Commission Joint Research Centre. "QSAR Model Reporting
  Format (QMRF)."
  https://joint-research-centre.ec.europa.eu/scientific-tools-databases/qsar-toolbox_en
- Tropsha, A. (2010). "Best Practices for QSAR Model Development,
  Validation, and Exploitation." Mol. Inform., 29(6-7), 476-488.
  https://doi.org/10.1002/minf.201000061
"""

from qsarkit.reporting._plots import (
    plot_atom_contributions,
    plot_calibration_curve,
    plot_precision_recall,
    plot_qq,
    plot_threshold_sweep,
    figure_to_html,
    plot_feature_importance,
    plot_learning_curve,
    plot_predicted_vs_observed,
    plot_residuals,
    plot_roc_curve,
    plot_williams,
)
from qsarkit.reporting._report import OECDReportBuilder, QSARReport, ReportSection

__all__ = [
    "plot_calibration_curve",
    "plot_qq",
    "plot_threshold_sweep",
    "plot_precision_recall",
    "plot_atom_contributions",
    "QSARReport",
    "ReportSection",
    "OECDReportBuilder",
    "plot_predicted_vs_observed",
    "plot_residuals",
    "plot_williams",
    "plot_roc_curve",
    "plot_learning_curve",
    "plot_feature_importance",
    "figure_to_html",
]
