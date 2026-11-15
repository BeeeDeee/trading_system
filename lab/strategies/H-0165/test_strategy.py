import numpy as np
import pytest

from strategy import target_weights, PARAMS
from strategy import trailing_valid_mean, fresh_values, residual_z, tilt_targets


def _inputs(T, seed=0):
    rng = np.random.default_rng(seed)
    pc = np.exp(np.log(0.65) + 0.15 * rng.standard_normal(T))
    pc = np.round(pc, 4) + 1e-6 * np.arange(T)          # no accidental repeats
    vix = np.exp(np.log(20.0) + np.cumsum(0.05 * rng.standard_normal(T)) * 0.3)
    lr = 0.01 * rng.standard_normal(T)
    return pc, vix, lr


def test_trailing_valid_mean_hand():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, np.nan, np.nan])
    out = trailing_valid_mean(x, 3, 2)
    # rows 0,1: window incomplete; row2: (1+2)/2; row3: (2+4)/2; row4: (4+5)/2;
    # row5: (4+5+6)/3; row6: (5+6)/2; row7: only one valid value
    exp = np.array([np.nan, np.nan, 1.5, 3.0, 4.5, 5.0, 5.5, np.nan])
    np.testing.assert_allclose(out, exp, equal_nan=True)


def test_fresh_values_marks_third_repeat_as_missing():
    pc = np.array([0.6, 0.6, 0.6, 0.6, 0.7, np.nan, 0.7, -1.0, 0.8])
    lp = fresh_values(pc)
    assert np.isfinite(lp[0]) and np.isfinite(lp[1])
    assert np.isnan(lp[2]) and np.isnan(lp[3])              # carried-forward copies
    assert np.isfinite(lp[4]) and np.isnan(lp[5]) and np.isfinite(lp[6])
    assert np.isnan(lp[7])                                   # non-positive ratio
    assert lp[8] == pytest.approx(np.log(0.8))


def test_z_warmup_first_row():
    pc, vix, lr = _inputs(400)
    z = residual_z(pc, vix, lr)
    # p valid from aligned row 4, x from row 5; full 252-row window first at row 251
    assert np.all(np.isnan(z[:251]))
    assert np.all(np.isfinite(z[251:]))


def test_z_needs_200_valid_rows():
    pc, vix, lr = _inputs(400)
    pc2 = pc.copy()
    pc2[10:70] = np.nan            # p NaN on rows 10..73 -> 64 invalid rows
    z = residual_z(pc2, vix, lr)
    # p needs 4 of 5 valid: p NaN on rows 10..72, valid on 5..9 and from 73.
    # Window t-251..t holds >= 200 valid rows first at t = 272 (rows 73..272).
    assert np.all(np.isnan(z[:272]))
    assert np.all(np.isfinite(z[272:]))
    # a gap at the decision row itself: no z
    pc3 = pc.copy()
    pc3[296:300] = np.nan
    z3 = residual_z(pc3, vix, lr)
    assert np.isnan(z3[299]) and np.isnan(z3[300])


def test_z_point_in_time_truncation_and_perturbation():
    pc, vix, lr = _inputs(420, seed=3)
    z = residual_z(pc, vix, lr)
    for cut in (260, 300, 377):
        zc = residual_z(pc[:cut], vix[:cut], lr[:cut])
        np.testing.assert_array_equal(zc, z[:cut])
        pcp, vixp, lrp = pc.copy(), vix.copy(), lr.copy()
        rng = np.random.default_rng(cut)
        pcp[cut:] = rng.uniform(0.3, 1.5, 420 - cut)
        vixp[cut:] = rng.uniform(10, 80, 420 - cut)
        lrp[cut:] = rng.standard_normal(420 - cut) * 0.05
        zp = residual_z(pcp, vixp, lrp)
        np.testing.assert_array_equal(zp[:cut], z[:cut])


def test_z_regression_removes_price_and_vix():
    # ln(PC) built exactly from x and v plus a small noise, then a put spike at the end
    T = 330
    rng = np.random.default_rng(7)
    lr = 0.01 * rng.standard_normal(T)
    vix = 20.0 * np.exp(0.2 * rng.standard_normal(T))
    pc = np.exp(-0.4 + 0.01 * rng.standard_normal(T)) + 1e-7 * np.arange(T)
    z = residual_z(pc, vix, lr)
    assert np.nanmax(np.abs(z[251:])) < 6
    spike = pc.copy()
    spike[320] = spike[320] * 3.0
    zs = residual_z(spike, vix, lr)
    np.testing.assert_array_equal(zs[:320], z[:320])
    assert zs[320] > 3.0                     # excess put demand -> fear


def test_constant_ratio_gives_no_signal():
    T = 300
    _, vix, lr = _inputs(T)
    pc = np.full(T, 0.7)
    z = residual_z(pc, vix, lr)
    assert np.all(np.isnan(z))               # carried-forward value is no data


def test_tilt_targets_hand():
    nan = np.nan
    z = np.array([nan, 2.0, nan, nan, nan, nan, nan])
    np.testing.assert_allclose(tilt_targets(z, 3, 1.5), [0.6, 0.8, 0.8, 0.8, 0.6, 0.6, 0.6])
    # latest signal wins
    z = np.array([2.0, nan, -2.0, nan, nan, nan, nan])
    np.testing.assert_allclose(tilt_targets(z, 3, 1.5), [0.8, 0.8, 0.4, 0.4, 0.4, 0.6, 0.6])
    # same-sign repeat extends
    z = np.array([2.0, nan, 1.6, nan, nan, nan, nan])
    np.testing.assert_allclose(tilt_targets(z, 3, 1.5), [0.8, 0.8, 0.8, 0.8, 0.8, 0.6, 0.6])
    # threshold is inclusive, and below it nothing happens
    z = np.array([1.5, 1.4, -1.5, nan, nan])
    np.testing.assert_allclose(tilt_targets(z, 1, 1.5), [0.8, 0.6, 0.4, 0.6, 0.6])


def test_params_change_output():
    nan = np.nan
    z = np.array([nan, 1.2, nan, nan, nan, nan, nan, nan, nan])
    a = tilt_targets(z, 5, 1.5)
    b = tilt_targets(z, 5, 1.0)
    assert np.all(a == 0.6) and b[1] == pytest.approx(0.8)
    c = tilt_targets(z, 7, 1.0)
    assert (c > 0.6).sum() == 7 and (b > 0.6).sum() == 5
    assert set(PARAMS) == {"hold_days", "z_threshold"}
