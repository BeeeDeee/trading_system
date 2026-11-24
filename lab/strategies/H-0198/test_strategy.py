import dataclasses

import numpy as np
import pytest
from lab.framework.data import synthetic

from strategy import (PARAMS, _book, _columns, _cs, _decision_rows, _entry_events, _event_flags,
                      _raw_inputs, _rsum, _score, target_weights)


def _weekdays(n):
    base = synthetic().dates[0]
    return np.busday_offset(base, np.arange(n), roll="forward")


def test_rsum_trailing_window():
    x = np.arange(1.0, 7.0)[:, None]
    out = _rsum(x, 3)[:, 0]
    assert out.tolist() == [1.0, 3.0, 6.0, 9.0, 12.0, 15.0]


def test_event_flags_pre_post_active():
    T, N, W = 400, 3, 10
    dates = _weekdays(T)
    ea = np.zeros((T, N), dtype=np.float32)
    ea[10, 0] = 1.0
    ea[10, 1] = 1.0
    ea[267, 1] = 1.0                    # actual release 3 sessions before the predicted one
    ea[:, 2] = np.nan                   # no EDGAR history
    uni = np.ones((T, N), dtype=bool)
    pre, post, active = _event_flags(ea, dates, uni, W, min_pred_members=2)
    rp = 10 + 260                       # 364 days = 52 weeks = 260 weekdays
    assert dates[rp].astype(np.int64) - dates[10].astype(np.int64) == 364
    assert np.nonzero(pre[:, 0])[0].tolist() == list(range(rp - W, rp))
    # col 1: suppressed from the row of the actual release on (it is in t-W+1..t)
    assert np.nonzero(pre[:, 1])[0].tolist() == list(range(rp - W, 267))
    assert not pre[:, 2].any() and not post[:, 2].any()
    assert np.nonzero(post[:, 0])[0].tolist() == [10, 11, 12, 13, 14]
    # active once two members have reached a predicted release date
    assert np.nonzero(active)[0][0] == rp
    assert active[rp:].all()
    # stricter threshold: never active
    _, _, act3 = _event_flags(ea, dates, uni, W, min_pred_members=3)
    assert not act3.any()
    # pre_window changes the flag
    pre5, _, _ = _event_flags(ea, dates, uni, 5, min_pred_members=2)
    assert np.nonzero(pre5[:, 0])[0].tolist() == list(range(rp - 5, rp))


def test_event_flags_truncation_is_point_in_time():
    T, N = 400, 4
    dates = _weekdays(T)
    rng = np.random.default_rng(1)
    ea = (rng.random((T, N)) < 0.02).astype(np.float32)
    uni = np.ones((T, N), dtype=bool)
    full = _event_flags(ea, dates, uni, 10, min_pred_members=2)
    for cut in (50, 265, 300, 399):
        part = _event_flags(ea[: cut + 1], dates[: cut + 1], uni[: cut + 1], 10, min_pred_members=2)
        for a, b in zip(full, part):
            assert np.array_equal(a[: cut + 1], b)


def test_raw_inputs_hand_values():
    T, N = 80, 4
    ret_co = np.zeros((T, N))
    ret_oc = np.zeros((T, N))
    ret_co[:, 0] = 0.01
    ret_co[:, 1] = np.where(np.arange(T) % 2 == 0, 0.02, -0.02)
    ret_co[:, 2] = -0.01
    listed = np.ones((T, N), dtype=bool)
    listed[:20, 3] = False              # late listing
    dv = np.ones((T, N))
    dv[75:, 0] = 2.0
    dv[30, 2] = np.nan
    uni = np.ones((T, N), dtype=bool)
    rev5, vol, vlm = _raw_inputs(ret_co, ret_oc, listed, dv, uni)
    r5 = np.array([1.01 ** 5 - 1, 1.02 ** 3 * 0.98 ** 2 - 1, 0.99 ** 5 - 1, 0.0])
    r5[1] = np.prod(1 + ret_co[75:80, 1]) - 1
    assert np.allclose(rev5[79], -(r5 - r5.mean()))
    assert np.isnan(rev5[4]).all() and np.isfinite(rev5[5, :3]).all()   # row 0 has no previous close
    assert np.isnan(rev5[24, 3]) and np.isfinite(rev5[25, 3])           # listed from row 20
    assert np.isclose(vol[79, 1], np.std(ret_co[17:80, 1], ddof=1))
    assert np.isclose(vol[79, 0], 0.0, atol=1e-9)
    assert np.isnan(vol[62, 0]) and np.isfinite(vol[63, 0])
    assert np.isclose(vlm[79, 0], np.log(2.0))
    assert np.isclose(vlm[79, 1], 0.0)
    assert np.isnan(vlm[66, 1]) and np.isfinite(vlm[67, 1])
    assert np.isnan(vlm[79, 2])                                          # missing dollar volume
    # non-members get nothing
    uni2 = uni.copy()
    uni2[:, 1] = False
    rev5b, volb, vlmb = _raw_inputs(ret_co, ret_oc, listed, dv, uni2)
    assert np.isnan(rev5b[:, 1]).all() and np.isnan(volb[:, 1]).all() and np.isnan(vlmb[:, 1]).all()


def test_cs_zero_dispersion_is_zero():
    x = np.array([[1.0, 2.0, 3.0, 5.0], [0.0, 0.0, 0.0, 9.0]])
    uni = np.ones((2, 4), dtype=bool)
    valid = np.array([[True, True, True, False], [True, True, True, False]])
    z = _cs(x, uni, valid)
    assert np.allclose(z[0, :3], [-1.0, 0.0, 1.0])      # framework helper: sample std
    assert z[1, :3].tolist() == [0.0, 0.0, 0.0]
    assert np.isnan(z[:, 3]).all()


def _scores_for_book():
    N = 150
    s = np.full((6, N), np.nan)
    j = np.arange(N)
    s[0] = -j
    r1 = np.empty(N)
    r1[15:35] = -(j[15:35] - 15)        # ranks 0..19
    r1[0:15] = -(20 + j[0:15])          # ranks 20..34
    r1[35:] = -j[35:]
    s[2] = r1
    s[3] = r1
    s[4, :99] = r1[:99]                 # 99 scored: flat
    s[5] = r1
    return s


def test_book_buffer_exit_and_flat():
    s = _scores_for_book()
    T, N = s.shape
    uni = np.ones((T, N), dtype=bool)
    uni[3, 3] = False                   # held name leaves the universe
    w = _book(s, uni, np.array([0, 2, 3, 4, 5]), 0.1)
    assert np.isnan(w[1]).all()         # not a decision row: keep positions
    assert np.nonzero(w[0])[0].tolist() == list(range(15))
    assert np.allclose(w[0][w[0] > 0], 1 / 15)
    # held at ranks 20..29 kept (buffer 30), the 5 best non-held added
    assert set(np.nonzero(w[2])[0]) == set(range(10)) | set(range(15, 20))
    # col 3 left the universe: sold; S=149 -> K=15, K2=30; 14 kept, best non-held (col 20) added
    assert set(np.nonzero(w[3])[0]) == {0, 1, 2, 4, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20}
    assert w[3, 3] == 0.0
    assert (w[4] == 0).all()            # fewer than 100 scored: flat
    assert set(np.nonzero(w[5])[0]) == set(range(15, 30))   # fresh book after a flat week
    rows = ~np.isnan(w).all(axis=1)
    assert (np.nansum(w[rows], axis=1) <= 1 + 1e-12).all() and (np.nan_to_num(w) >= 0).all()
    # top changes the book size
    w2 = _book(s, uni, np.array([0]), 0.2)
    assert np.count_nonzero(w2[0]) == 30


def test_entry_events_interaction_only():
    N = 150
    s_int = _scores_for_book()[[0, 2]]
    s_pl = s_int.copy()
    s_pl[1, 15:17] = -1000.0            # plain ridge does not rank cols 15, 16 in its top
    uni = np.ones((2, N), dtype=bool)
    trad = np.ones((2, N), dtype=bool)
    trad[1, 17] = False
    m = _entry_events(s_int, s_pl, uni, trad, np.array([0, 1]), 0.1)
    assert not m[0].any()               # interaction top == plain top at row 0
    # row 1: interaction top = cols 15..29; new vs row 0 (cols 0..14) = all of them;
    # plain drops 15, 16 from its top (its top now has 17..29 + 0, 1) -> events at 15, 16 only
    assert np.nonzero(m[1])[0].tolist() == [15, 16]


def _dense_view():
    d = synthetic()
    T, N = len(d.dates), len(d.instruments)
    rng = np.random.default_rng(7)
    ea = (rng.random((T, N)) < 0.016).astype(np.float32)
    ea[:300, 0] = np.nan
    uni = np.asarray(d.listed, dtype=bool).copy()
    return dataclasses.replace(d, universe=uni, extras={"ea_reaction_day": ea})


def test_end_to_end_mechanics():
    d = _dense_view()
    params = dict(PARAMS)
    inter, plain, uni, sc = _columns(d, params, min_pred_members=10, with_plain=True)
    assert len(inter) == 6 and len(plain) == 5
    for c in inter:
        assert c.shape == uni.shape
        assert np.array_equal(np.isfinite(c), sc)          # NaN exactly where not scored
    first_active = np.nonzero(sc.any(axis=1))[0][0]
    assert first_active > 250                              # about a year of 8-K history first
    score = _score(inter, d, uni)
    fin = np.nonzero(np.isfinite(score).any(axis=1))[0]
    assert fin.size and fin[0] > first_active             # first scores only after the first refit
    assert not np.isfinite(score[~uni]).any()
    rows = _decision_rows(d.dates)
    w = _book(score, uni, rows, params["top"], min_scored=10)
    dec = ~np.isnan(w).all(axis=1)
    assert set(np.nonzero(dec)[0]) == set(rows.tolist())
    assert (np.nansum(np.abs(w), axis=1) <= 1 + 1e-9).all()
    assert (np.nan_to_num(w) >= 0).all()
    assert (np.nan_to_num(w)[~uni] == 0).all()
    assert (np.nansum(w[dec], axis=1) > 0.99).sum() > 100
    # the real rule (100 scored members) on a 31-name market: always flat, never NaN on decisions
    w_real = target_weights(d, params)
    assert (w_real[dec] == 0).all()
    assert np.isnan(w_real[~dec]).all()


def test_deterministic():
    d = _dense_view()
    a = target_weights(d, PARAMS)
    b = target_weights(d, PARAMS)
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


def test_entry_events_on_model_scores():
    d = _dense_view()
    inter, plain, uni, _ = _columns(d, PARAMS, min_pred_members=10, with_plain=True)
    s_int = _score(inter, d, uni)
    s_pl = _score(plain, d, uni)
    rows = _decision_rows(d.dates)
    trad = np.asarray(d.tradable, dtype=bool)
    m = _entry_events(s_int, s_pl, uni, trad, rows, 0.1, min_scored=10)
    assert m.any()
    assert not m[~uni].any()
    off = np.ones(len(d.dates), dtype=bool)
    off[rows] = False
    assert not m[off].any()                                # events only on weekly decisions


def test_missing_input_fails_loudly():
    d = synthetic()
    with pytest.raises(Exception):
        target_weights(d, PARAMS)
