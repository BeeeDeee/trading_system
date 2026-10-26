import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

T0 = 70          # rows before the first possible event (listing age >= 63)


def make_view(T=100, N=6, sp=None, uni=None, trad=None, listed=None, close=None):
    dates = np.arange(10000, 10000 + T).astype("datetime64[D]")
    instruments = tuple("E%d" % (i + 1) for i in range(N))
    if sp is None:
        sp = np.zeros((T, N))
    if uni is None:
        uni = np.ones((T, N), dtype=bool)
    if trad is None:
        trad = np.ones((T, N), dtype=bool)
    if listed is None:
        listed = np.ones((T, N), dtype=bool)
    if close is None:
        close = np.full((T, N), 10.0)
    z = np.zeros((T, N))
    return DataView(
        dates=dates,
        instruments=instruments,
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=z.copy(),
        ret_oc=z.copy(),
        tradable=trad,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e8),
        universe=uni,
        extras={"sp500_member": sp.astype(np.float32), "liq_rank": np.ones((T, N), dtype=np.float32)},
        series={},
        cash_ret=np.zeros(T),
    )


def kept(w):
    """Effective targets after NaN rows / NaN elements are forward-filled (keep positions)."""
    out = np.zeros_like(w)
    cur = np.zeros(w.shape[1])
    for r in range(w.shape[0]):
        row = w[r]
        cur = np.where(np.isnan(row), cur, row)
        out[r] = cur
    return out


def test_params_match_card():
    assert PARAMS == {"hold_days": 5}


def test_deletion_long_hedged_and_hold_window():
    T, N = 100, 6
    sp = np.zeros((T, N))
    sp[:T0 + 5, 0] = 1.0                       # stock 0 deleted: first non-member session at row T0+5
    v = make_view(T, N, sp=sp)
    w = target_weights(v, {"hold_days": 5})
    e = kept(w)
    t = T0 + 5
    assert np.all(e[:t] == 0.0)
    for r in range(t, t + 5):                  # decision rows t..t+4 -> held over sessions t+1..t+5
        assert e[r, 0] == pytest.approx(0.5)
        assert np.allclose(e[r, 1:], -0.5 / 5)
        assert e[r].sum() == pytest.approx(0.0)
    assert np.all(e[t + 5:] == 0.0)            # closed at the open of t+6
    # recomputed only on entry and exit
    changed = np.where(~np.all(np.isnan(w), axis=1))[0]
    assert set(changed) - {0} == {t, t + 5}


def test_addition_short_side():
    T, N = 100, 6
    sp = np.zeros((T, N))
    sp[T0 + 2:, 3] = 1.0                       # stock 3 added at row T0+2
    w = kept(target_weights(make_view(T, N, sp=sp), PARAMS))
    t = T0 + 2
    assert w[t, 3] == pytest.approx(-0.5)
    assert np.allclose(np.delete(w[t], 3), 0.5 / 5)


def test_two_events_split_and_cancel():
    T, N = 100, 6
    sp = np.zeros((T, N))
    sp[:T0 + 1, 0] = 1.0                       # deletion of 0 at T0+1
    sp[T0 + 3:, 1] = 1.0                       # addition of 1 at T0+3
    w = kept(target_weights(make_view(T, N, sp=sp), {"hold_days": 5}))
    assert w[T0 + 1, 0] == pytest.approx(0.5)
    r = T0 + 3                                 # both active: +0.25 / -0.25, net 0 -> no hedge
    assert w[r, 0] == pytest.approx(0.25) and w[r, 1] == pytest.approx(-0.25)
    assert np.allclose(w[r, 2:], 0.0)
    r = T0 + 6                                 # deletion expired (rows T0+1..T0+5), addition alone
    assert w[r, 0] == pytest.approx(0.1) and w[r, 1] == pytest.approx(-0.5)
    assert np.nanmax(np.abs(kept(target_weights(make_view(T, N, sp=sp), PARAMS))).sum(axis=1)) <= 1.0 + 1e-12


def test_hold_days_changes_output():
    T, N = 100, 6
    sp = np.zeros((T, N))
    sp[:T0 + 5, 0] = 1.0
    v = make_view(T, N, sp=sp)
    w3 = kept(target_weights(v, {"hold_days": 3}))
    w7 = kept(target_weights(v, {"hold_days": 7}))
    t = T0 + 5
    assert w3[t + 2, 0] > 0 and w3[t + 3, 0] == 0
    assert w7[t + 6, 0] > 0 and w7[t + 7, 0] == 0


def test_eligibility_filters():
    T, N = 100, 6
    t = T0 + 5
    sp = np.zeros((T, N))
    sp[t:, :] = 1.0                            # every stock "added" at t
    uni = np.ones((T, N), dtype=bool)
    trad = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    close = np.full((T, N), 10.0)
    uni[t, 0] = False                          # 0: not in LIQ-500 on t -> no event
    trad[t - 1, 1] = False                     # 1: not tradable on t-1 -> no event
    listed[: t - 62, 2] = False                # 2: only 62 listed sessions before t -> no event
    close[t, 3] = np.nan                       # 3: no valid price on t -> no event
    uni[t - 1, 4] = False                      # 4: not in LIQ-500 on t-1 only -> still an event (LIQ-500 on t)
    v = make_view(T, N, sp=sp, uni=uni, trad=trad, listed=listed, close=close)
    w = kept(target_weights(v, PARAMS))
    assert np.all(w[:t] == 0.0)
    assert set(np.where(w[t] < 0)[0]) == {4, 5}        # only 4 and 5 are (short) events
    assert w[t, 4] == pytest.approx(-0.25) and w[t, 5] == pytest.approx(-0.25)
    assert w[t, 0] == 0.0                              # not a member on t: no hedge weight either
    assert np.allclose(w[t, 1:4], 0.5 / 3)             # hedge over the other tradable members
    listed2 = listed.copy()
    listed2[: t - 63, 2] = False
    listed2[t - 63:, 2] = True                 # exactly 63 listed sessions before t -> event
    w2 = kept(target_weights(make_view(T, N, sp=sp, uni=uni, trad=trad, listed=listed2, close=close), PARAMS))
    assert set(np.where(w2[t] < 0)[0]) == {2, 4, 5}


def test_no_events_in_first_63_rows_and_nan_membership():
    T, N = 100, 4
    sp = np.zeros((T, N))
    sp[30:, 0] = 1.0                           # too early: listing age < 63
    sp[:, 1] = np.nan
    sp[T0 + 1:, 1] = 1.0                       # NaN -> 1 is not an event
    w = target_weights(make_view(T, N, sp=sp), PARAMS)
    assert np.all(kept(w) == 0.0)


def test_event_stock_leaving_universe_and_hedge_leaving():
    T, N = 100, 6
    t = T0 + 5
    sp = np.zeros((T, N))
    sp[:t, 0] = 1.0                            # deletion of 0 at t
    uni = np.ones((T, N), dtype=bool)
    uni[t + 2:, 0] = False                     # event stock drops out of LIQ-500 at t+2 -> A empty -> flat
    v = make_view(T, N, sp=sp, uni=uni)
    w = kept(target_weights(v, PARAMS))
    assert w[t + 1, 0] == pytest.approx(0.5)
    assert np.all(w[t + 2] == 0.0)
    # a hedge stock leaving the universe is zeroed without re-equalising the others
    uni2 = np.ones((T, N), dtype=bool)
    uni2[t + 2:, 5] = False
    raw = target_weights(make_view(T, N, sp=sp, uni=uni2), PARAMS)
    w2 = kept(raw)
    assert w2[t + 2, 5] == 0.0
    assert np.allclose(w2[t + 2, 1:5], -0.1)
    assert np.isnan(raw[t + 2, 1:5]).all()
    assert np.all(w2[~uni2] == 0.0)


def test_missing_extra_fails_loudly():
    v = make_view()
    v.extras.pop("sp500_member")
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T, N = 200, 8
    sp = (rng.random((T, N)) < 0.5).astype(float)
    sp = np.repeat(sp[::20], 20, axis=0)[:T]
    uni = rng.random((T, N)) < 0.9
    v = make_view(T, N, sp=sp, uni=uni)
    full = target_weights(v, PARAMS)
    for cut in (80, 101, 150):
        vc = make_view(cut, N, sp=sp[:cut], uni=uni[:cut])
        part = target_weights(vc, PARAMS)
        assert np.array_equal(np.isnan(part), np.isnan(full[:cut]))
        assert np.allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))
    g = np.abs(np.nan_to_num(full)).sum(axis=1)
    assert g.max() <= 1.0 + 1e-12
    assert np.all(kept(full)[~uni] == 0.0)          # never a held weight outside the universe
