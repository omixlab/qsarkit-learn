"""Hand-computed correctness tests for qsarkit.metrics classification functions."""

from __future__ import annotations

import pytest

from qsarkit.metrics import (
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
    robust_initial_enhancement,
    roc_auc,
    sensitivity,
    specificity,
)
from qsarkit.metrics._classification import _check_binary_labels, _check_scores


def test_confusion_counts_hand_computed() -> None:
    assert confusion_counts([1, 1, 0, 0], [1, 0, 0, 0]) == {
        "tp": 1,
        "tn": 2,
        "fp": 0,
        "fn": 1,
    }


def test_accuracy_hand_computed() -> None:
    assert accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)


def test_sensitivity_hand_computed() -> None:
    assert sensitivity([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)


def test_sensitivity_zero_when_no_positives() -> None:
    assert sensitivity([0, 0], [0, 0]) == 0.0


def test_specificity_hand_computed() -> None:
    assert specificity([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(1.0)


def test_specificity_zero_when_no_negatives() -> None:
    assert specificity([1, 1], [1, 1]) == 0.0


def test_recall_matches_sensitivity() -> None:
    assert recall([1, 1, 0, 0], [1, 0, 0, 0]) == sensitivity([1, 1, 0, 0], [1, 0, 0, 0])


def test_precision_hand_computed() -> None:
    assert precision([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(1.0)


def test_precision_zero_when_nothing_predicted_positive() -> None:
    assert precision([1, 0], [0, 0]) == 0.0


def test_f1_score_hand_computed() -> None:
    assert f1_score([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(2 / 3)


def test_f1_score_zero_when_precision_and_recall_zero() -> None:
    assert f1_score([1, 0], [0, 0]) == 0.0


def test_balanced_accuracy_hand_computed() -> None:
    assert balanced_accuracy([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.75)


def test_matthews_corrcoef_perfect() -> None:
    assert matthews_corrcoef([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_matthews_corrcoef_hand_computed() -> None:
    # tp=1,tn=2,fp=0,fn=1 -> num = 1*2-0*1=2; denom=sqrt(1*2*2*3)=sqrt(12)
    import math

    expected = 2 / math.sqrt(1 * 2 * 2 * 3)
    assert matthews_corrcoef([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(expected)


def test_matthews_corrcoef_zero_on_degenerate_marginal() -> None:
    assert matthews_corrcoef([1, 1], [1, 1]) == 0.0


def test_cohen_kappa_perfect_agreement() -> None:
    assert cohen_kappa([1, 1, 0, 0], [1, 1, 0, 0]) == pytest.approx(1.0)


def test_cohen_kappa_hand_computed() -> None:
    yt = [1, 1, 0, 0]
    yp = [1, 0, 0, 0]
    # p_o = 0.75; p_e = P(true=0)*P(pred=0) + P(true=1)*P(pred=1)
    #      = 0.5*0.75 + 0.5*0.25 = 0.5
    expected = (0.75 - 0.5) / (1 - 0.5)
    assert cohen_kappa(yt, yp) == pytest.approx(expected)


def test_cohen_kappa_degenerate_all_same_class() -> None:
    # p_e == 1 forces exact agreement -> kappa is 1.0
    assert cohen_kappa([0, 0, 0], [0, 0, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1], [1, 1]) == pytest.approx(1.0)


def test_roc_auc_perfect() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_roc_auc_raises_single_class() -> None:
    with pytest.raises(ValueError, match="single class"):
        roc_auc([1, 1, 1], [0.1, 0.2, 0.3])


def test_pr_auc_perfect() -> None:
    assert pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_pr_auc_raises_single_class() -> None:
    with pytest.raises(ValueError, match="single class"):
        pr_auc([0, 0, 0], [0.1, 0.2, 0.3])


def test_brier_score_perfect() -> None:
    assert brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_brier_score_hand_computed() -> None:
    assert brier_score([0, 1], [0.2, 0.6]) == pytest.approx((0.2**2 + 0.4**2) / 2)


def test_enrichment_factor_hand_computed() -> None:
    y_true = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y_score = [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    assert enrichment_factor(y_true, y_score, fraction=0.2) == pytest.approx(5.0)


def test_enrichment_factor_raises_on_bad_fraction() -> None:
    with pytest.raises(ValueError, match="fraction must be"):
        enrichment_factor([1, 0], [0.1, 0.2], fraction=1.5)


def test_enrichment_factor_raises_without_actives() -> None:
    with pytest.raises(ValueError, match="without actives"):
        enrichment_factor([0, 0, 0], [0.1, 0.2, 0.3])


def test_robust_initial_enhancement_greater_than_one_for_good_ranking() -> None:
    scores = [1.0, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    labels = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    assert robust_initial_enhancement(labels, scores, alpha=20.0) > 1.0


def test_robust_initial_enhancement_raises_on_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        robust_initial_enhancement([1, 0], [0.1, 0.2], alpha=0.0)


def test_robust_initial_enhancement_raises_without_actives() -> None:
    with pytest.raises(ValueError, match="without actives"):
        robust_initial_enhancement([0, 0], [0.1, 0.2])


def test_bedroc_near_one_for_perfect_ranking() -> None:
    labels = [1] * 5 + [0] * 95
    scores = list(range(100, 0, -1))
    assert bedroc(labels, scores, alpha=20.0) > 0.99


def test_bedroc_near_ra_for_random_ranking() -> None:
    # Actives spread uniformly through the ranking -> BEDROC close to Ra.
    n = 100
    n_active = 10
    labels = [1 if i % (n // n_active) == 0 else 0 for i in range(n)]
    scores = list(range(n, 0, -1))
    ra = sum(labels) / n
    assert bedroc(labels, scores, alpha=20.0) == pytest.approx(ra, abs=0.2)


def test_bedroc_raises_on_bad_alpha() -> None:
    with pytest.raises(ValueError, match="alpha must be positive"):
        bedroc([1, 0], [0.1, 0.2], alpha=0.0)


def test_bedroc_raises_without_actives() -> None:
    with pytest.raises(ValueError, match="without actives"):
        bedroc([0, 0], [0.1, 0.2])


def test_bedroc_raises_when_all_active() -> None:
    with pytest.raises(ValueError, match="every compound is active"):
        bedroc([1, 1], [0.1, 0.2])


def test_check_binary_labels_rejects_mismatched_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        _check_binary_labels([1, 0], [1, 0, 1])


def test_check_binary_labels_rejects_empty() -> None:
    with pytest.raises(ValueError, match="is empty"):
        _check_binary_labels([], [])


def test_check_binary_labels_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="integer class labels"):
        _check_binary_labels([0.5, 1.0], [0, 1])


def test_check_binary_labels_rejects_extra_labels() -> None:
    with pytest.raises(ValueError, match="must be binary"):
        _check_binary_labels([0, 1, 2], [0, 1, 0])


def test_check_scores_rejects_extra_labels() -> None:
    with pytest.raises(ValueError, match="must be binary"):
        _check_scores([0, 1, 2], [0.1, 0.2, 0.3])


def test_check_scores_rejects_mismatched_length() -> None:
    with pytest.raises(ValueError, match="same length"):
        _check_scores([0, 1], [0.1, 0.2, 0.3])


def test_check_scores_rejects_empty() -> None:
    with pytest.raises(ValueError, match="is empty"):
        _check_scores([], [])


def test_check_scores_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        _check_scores([0, 1], [0.1, float("inf")])
