"""Hand-computed correctness tests for qsarkit.metrics regression functions."""

from __future__ import annotations

import numpy as np
import pytest

from qsarkit.metrics import (
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


def test_mse_hand_computed() -> None:
    assert mse([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4.0 / 3.0)


def test_rmse_hand_computed() -> None:
    assert rmse([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_rmsep_matches_rmse() -> None:
    assert rmsep([1.0, 2.0], [1.5, 2.5]) == pytest.approx(0.5)


def test_mae_hand_computed() -> None:
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2.0 / 3.0)


def test_median_ae_hand_computed() -> None:
    assert median_ae([1.0, 2.0, 3.0], [1.0, 2.0, 30.0]) == pytest.approx(0.0)


def test_bias_hand_computed() -> None:
    assert bias([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) == pytest.approx(1.0)


def test_press_hand_computed() -> None:
    assert press([1.0, 2.0, 3.0], [1.0, 2.0, 5.0]) == pytest.approx(4.0)


def test_see_hand_computed() -> None:
    val = see([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0], n_parameters=1)
    assert val == pytest.approx(0.70710678, abs=1e-6)


def test_see_raises_on_non_positive_dof() -> None:
    with pytest.raises(ValueError, match="Degrees of freedom"):
        see([1.0, 2.0], [1.0, 2.0], n_parameters=1)


def test_r2_score_perfect() -> None:
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_score_hand_computed() -> None:
    # y_true mean = 2, TSS = 2; RSS = 0+0+1=1 -> R2 = 1 - 1/2 = 0.5
    assert r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(0.5)


def test_r2_score_raises_zero_variance() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        r2_score([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


def test_adjusted_r2_score_hand_computed() -> None:
    val = adjusted_r2_score([1.0, 2.0, 3.0, 4.5], [1.1, 2.0, 2.9, 4.4], 1)
    assert val == pytest.approx(0.993, abs=1e-3)


def test_adjusted_r2_score_raises_on_non_positive_dof() -> None:
    with pytest.raises(ValueError, match="Adjusted R\\^2"):
        adjusted_r2_score([1.0, 2.0], [1.0, 2.0], n_features=1)


def test_ccc_perfect_agreement() -> None:
    assert ccc([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_ccc_hand_computed() -> None:
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.array([1.2, 1.8, 3.3, 3.7])
    mt, mp = yt.mean(), yp.mean()
    vt = np.mean((yt - mt) ** 2)
    vp = np.mean((yp - mp) ** 2)
    cov = np.mean((yt - mt) * (yp - mp))
    expected = 2 * cov / (vt + vp + (mt - mp) ** 2)
    assert ccc(yt, yp) == pytest.approx(expected)


def test_ccc_degenerate_denominator_returns_one() -> None:
    # Both constant and equal -> denom is 0 -> defined as perfect agreement.
    assert ccc([2.0, 2.0], [2.0, 2.0]) == pytest.approx(1.0)


def test_q2_f1_hand_computed() -> None:
    assert q2_f1([2.0, 4.0], [2.0, 3.0], y_train=[0.0, 2.0, 4.0]) == pytest.approx(0.75)


def test_q2_f1_raises_on_zero_denominator() -> None:
    with pytest.raises(ValueError, match="Q\\^2_F1"):
        q2_f1([2.0, 2.0], [1.0, 3.0], y_train=[0.0, 2.0, 4.0])


def test_q2_f2_matches_r2_score() -> None:
    assert q2_f2([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(0.5)


def test_q2_f3_hand_computed() -> None:
    assert q2_f3([2.0, 4.0], [2.0, 3.0], y_train=[0.0, 2.0, 4.0]) == pytest.approx(0.8125)


def test_q2_f3_raises_on_zero_train_variance() -> None:
    with pytest.raises(ValueError, match="Q\\^2_F3"):
        q2_f3([2.0, 4.0], [2.0, 3.0], y_train=[2.0, 2.0, 2.0])


def test_k_slope_hand_computed() -> None:
    assert k_slope([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)
    # k = sum(y*yhat)/sum(y^2)
    assert k_slope([1.0, 2.0], [2.0, 2.0]) == pytest.approx((2 + 4) / 5.0)


def test_k_slope_raises_when_y_true_all_zero() -> None:
    with pytest.raises(ValueError, match="k is undefined"):
        k_slope([0.0, 0.0], [1.0, 2.0])


def test_k_prime_slope_hand_computed() -> None:
    assert k_prime_slope([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(0.5)


def test_k_prime_slope_raises_when_y_pred_all_zero() -> None:
    with pytest.raises(ValueError, match="k' is undefined"):
        k_prime_slope([1.0, 2.0], [0.0, 0.0])


def test_r0_squared_perfect() -> None:
    assert r0_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r0_squared_raises_when_y_pred_constant() -> None:
    with pytest.raises(ValueError, match="R0\\^2"):
        r0_squared([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])


def test_r0_prime_squared_perfect() -> None:
    assert r0_prime_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r0_prime_squared_raises_when_y_true_constant() -> None:
    with pytest.raises(ValueError, match="R0'\\^2"):
        r0_prime_squared([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])


def test_r2m_perfect() -> None:
    assert r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2m_raises_on_constant_input() -> None:
    with pytest.raises(ValueError, match="squared correlation"):
        r2m([1.0, 1.0], [1.0, 2.0])


def test_r2m_prime_perfect() -> None:
    assert r2m_prime([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_pearson_r2_requires_two_samples() -> None:
    with pytest.raises(ValueError, match="At least two samples"):
        r2m([1.0], [1.0])


def test_average_r2m_perfect() -> None:
    assert average_r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_delta_r2m_zero_for_symmetric_perfect_fit() -> None:
    assert delta_r2m([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_golbraikh_tropsha_passes_with_good_model() -> None:
    res = golbraikh_tropsha_criteria(
        [1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 2.9, 4.05], q2=0.9
    )
    assert res["passed"] is True
    assert res["q2_available"] is True
    assert res["criterion_1_q2"] is True


def test_golbraikh_tropsha_without_q2() -> None:
    res = golbraikh_tropsha_criteria([1.0, 2.0, 3.0, 4.0], [1.1, 2.0, 2.9, 4.05])
    assert res["q2"] is None
    assert res["criterion_1_q2"] is None
    assert res["q2_available"] is False
    # passed determined only by criteria 2-5
    assert res["passed"] == all(
        [res["criterion_2_r2"], res["criterion_3_r0"], res["criterion_4_slope"], res["criterion_5_delta_r0"]]
    )


def test_golbraikh_tropsha_fails_with_bad_model() -> None:
    # Weak, noisy relationship -> low r2 -> criterion 2 fails -> overall fail.
    res = golbraikh_tropsha_criteria([1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 4.0, 1.0], q2=0.9)
    assert res["passed"] is False
    assert res["criterion_2_r2"] is False
