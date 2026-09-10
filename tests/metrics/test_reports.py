"""Tests for the aggregate QSAR regression/classification report builders."""

from __future__ import annotations

from qsarkit.metrics import (
    average_r2m,
    ccc,
    delta_r2m,
    mae,
    q2_f1,
    q2_f2,
    q2_f3,
    qsar_classification_report,
    qsar_regression_report,
    r2_score,
    rmse,
)

Y_TRUE = [1.0, 2.0, 3.0, 4.0]
Y_PRED = [1.1, 2.0, 2.9, 4.05]
Y_TRAIN = [0.0, 1.0, 2.0, 3.0, 4.0]


def test_qsar_regression_report_without_y_train() -> None:
    report = qsar_regression_report(Y_TRUE, Y_PRED)

    assert report["r2"] == r2_score(Y_TRUE, Y_PRED)
    assert report["rmse"] == rmse(Y_TRUE, Y_PRED)
    assert report["mae"] == mae(Y_TRUE, Y_PRED)
    assert report["ccc"] == ccc(Y_TRUE, Y_PRED)
    assert report["q2_f2"] == q2_f2(Y_TRUE, Y_PRED)
    assert report["average_r2m"] == average_r2m(Y_TRUE, Y_PRED)
    assert report["delta_r2m"] == delta_r2m(Y_TRUE, Y_PRED)
    assert "q2_f1" not in report
    assert "q2_f3" not in report
    assert report["golbraikh_tropsha"]["q2_available"] is False
    assert report["golbraikh_tropsha"]["criterion_1_q2"] is None


def test_qsar_regression_report_with_y_train_and_q2() -> None:
    report = qsar_regression_report(Y_TRUE, Y_PRED, y_train=Y_TRAIN, q2=0.9)

    assert report["q2_f1"] == q2_f1(Y_TRUE, Y_PRED, Y_TRAIN)
    assert report["q2_f3"] == q2_f3(Y_TRUE, Y_PRED, Y_TRAIN)
    assert report["golbraikh_tropsha"]["q2_available"] is True
    assert report["golbraikh_tropsha"]["passed"] is True


def test_qsar_classification_report_without_scores() -> None:
    report = qsar_classification_report([1, 1, 0, 0], [1, 0, 0, 0])

    assert report["confusion"] == {"tp": 1, "tn": 2, "fp": 0, "fn": 1}
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["balanced_accuracy"] <= 1.0
    assert "roc_auc" not in report
    assert "pr_auc" not in report


def test_qsar_classification_report_with_scores() -> None:
    report = qsar_classification_report(
        [1, 1, 0, 0], [1, 0, 0, 0], y_score=[0.9, 0.4, 0.2, 0.1]
    )

    assert report["mcc"] > 0
    assert 0.0 <= report["roc_auc"] <= 1.0
    assert 0.0 <= report["pr_auc"] <= 1.0
