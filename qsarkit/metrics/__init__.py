"""QSAR-specific regression and classification metrics.

Covers the classical statistics (RMSE, MAE, R^2, MCC, ...) plus the
QSAR-specific external-predictivity coefficients (Q^2_F1/F2/F3, CCC,
r_m^2, the Golbraikh-Tropsha criteria) and the early-recognition metrics
used in virtual screening (enrichment factor, RIE, BEDROC).
"""

from qsarkit.metrics._calibration import (
    calibration_curve,
    calibration_report,
    expected_calibration_error,
    maximum_calibration_error,
    qq_data,
    residual_normality,
)
from qsarkit.metrics._classification import (
    accuracy,
    balanced_accuracy,
    bedroc,
    brier_score,
    cohen_kappa,
    confusion_counts,
    enrichment_factor,
    f1_score,
    matthews_corrcoef,
    pr_auc,
    precision,
    recall,
    roc_auc,
    robust_initial_enhancement,
    sensitivity,
    specificity,
)
from qsarkit.metrics._reports import qsar_classification_report, qsar_regression_report
from qsarkit.metrics._thresholds import (
    optimal_threshold,
    threshold_report,
    threshold_sweep,
)
from qsarkit.metrics._regression import (
    adjusted_r2_score,
    average_r2m,
    bias,
    ccc,
    delta_r2m,
    golbraikh_tropsha_criteria,
    k_prime_slope,
    k_slope,
    mae,
    median_ae,
    mse,
    press,
    q2_f1,
    q2_f2,
    q2_f3,
    r0_prime_squared,
    r0_squared,
    r2_score,
    r2m,
    r2m_prime,
    rmse,
    rmsep,
    see,
)

__all__ = [
    # calibration and residual diagnostics
    "calibration_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "calibration_report",
    "qq_data",
    "residual_normality",
    # threshold selection
    "threshold_sweep",
    "optimal_threshold",
    "threshold_report",
    # regression
    "mse",
    "rmse",
    "rmsep",
    "mae",
    "median_ae",
    "bias",
    "press",
    "see",
    "r2_score",
    "adjusted_r2_score",
    "ccc",
    "q2_f1",
    "q2_f2",
    "q2_f3",
    "k_slope",
    "k_prime_slope",
    "r0_squared",
    "r0_prime_squared",
    "r2m",
    "r2m_prime",
    "average_r2m",
    "delta_r2m",
    "golbraikh_tropsha_criteria",
    # classification
    "confusion_counts",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "recall",
    "f1_score",
    "matthews_corrcoef",
    "cohen_kappa",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "enrichment_factor",
    "robust_initial_enhancement",
    "bedroc",
    # reports
    "qsar_regression_report",
    "qsar_classification_report",
]
