import math

import numpy as np
import pytest

from strategy import PARAMS, target_weights, month_end_rows, daily_returns, decision_book, events


# ---------------------------------------------------------------- helpers

def weekdays(n, start=12000):
    d = (np.arange(start, start + 2 * n + 10)).astype("datetime64[D]")
    d = d[np.is_busday(d)]
    return d[:n]


def make(T=420, N=12, seed=0, factor_vol=0.01, idio_vol=0.015):
    rng = np.random.default_rng(seed)
    f = rng.normal(0.0, factor_vol, T)
    load = np.linspace(0.5, 1.5, N)
    r = f[:, None] * load[None, :] + rng.normal(0.0, idio_vol, (T, N))
    close = 100.0 * np.cumprod(1.0 + r, axis=0)
    return dict(
        dates=weekdays(T),
        ret_co=np.zeros((T, N)),
        ret_oc=r,
        close=close,
        listed=np.ones((T, N), dtype=bool),
        tradable=np.ones((T, N), dtype=bool),
        universe=np.ones((T, N), dtype=bool),
        ea=np.zeros((T, N)),
    )


def book(m, q=0.2, k=3):
    return decision_book(m["dates"], m["ret_co"], m["ret_oc"], m["close"], m["listed"], m["tradable"],
                         m["universe"], m["ea"], q, k)


def ref_row(m, t, q, k):
    """Loop reference of the card's rules for one decision row."""
    r = (1 + m["ret_co"]) * (1 + m["ret_oc"]) - 1
    T, N = r.shape
    haspx = m["listed"] & np.isfinite(m["close"])
    valid = np.zeros((T, N), dtype=bool)
    valid[1:] = haspx[1:] & haspx[:-1]
    w = np.zeros(N)
    if t < 251:
        return w
    lo = t - 251
    cand = [j for j in range(N) if m["universe"][t, j] and valid[lo:t + 1, j].all()
            and np.std(r[lo:t + 1, j]) > 0]
    mkt = np.array([np.mean([r[s, j] for j in range(N) if m["universe"][s, j] and valid[s, j]])
                    for s in range(lo, t + 1)])
    cum = {j: np.prod(1 + r[t - 20:t + 1, j]) - 1 for j in cand}
    elig = [j for j in cand if m["tradable"][t, j] and np.all(m["ea"][t - 20:t + 1, j] == 0)]
    if len(cand) - 1 < k:
        return w
    R = {}
    for j in elig:
        others = [i for i in cand if i != j]
        cs = [np.corrcoef(r[lo:t + 1, j], r[lo:t + 1, i])[0, 1] for i in others]
        top = [others[i] for i in np.argsort(-np.array(cs), kind="stable")[:k]]
        R[j] = cum[j] - np.mean([cum[i] for i in top])
    names = sorted(R, key=lambda j: R[j])
    n = len(names)
    nl = int(math.floor(q * n + 1e-9))
    if nl < 1:
        return w
    longs, shorts = names[:nl], names[n - nl:]
    beta = {j: np.polyfit(mkt, r[lo:t + 1, j], 1)[0] for j in longs + shorts}
    bl = max(np.mean([beta[j] for j in longs]), 0.3)
    bs = max(np.mean([beta[j] for j in shorts]), 0.3)
    for j in longs:
        w[j] = bs / (bl + bs) / nl
    for j in shorts:
        w[j] = -bl / (bl + bs) / nl
    return w


class FakeData:
    def __init__(self, m, with_universe=True, with_ea=True):
        self.dates = m["dates"]
        self.ret_co, self.ret_oc, self.close = m["ret_co"], m["ret_oc"], m["close"]
        self.listed, self.tradable = m["listed"], m["tradable"]
        self.universe = m["universe"] if with_universe else None
        self.extras = {"ea_reaction_day": m["ea"]} if with_ea else {}
        self.instruments = tuple("S%02d" % j for j in range(m["close"].shape[1]))


# ---------------------------------------------------------------- calendar

def test_month_end_is_last_weekday_of_month():
    d = weekdays(300)
    f = month_end_rows(d)
    mon = d.astype("datetime64[M]")
    expected = np.zeros(len(d), dtype=bool)
    expected[:-1] = mon[1:] != mon[:-1]          # hindsight answer in a calendar with no holidays
    assert np.array_equal(f[:-1], expected[:-1])
    assert f.sum() >= 13


def test_month_end_holiday_falls_back_to_first_session():
    d = weekdays(300)
    mon = d.astype("datetime64[M]")
    last = np.flatnonzero(mon[1:] != mon[:-1])   # last weekday of each month
    hol = last[3]
    d2 = np.delete(d, hol)                       # last weekday of that month is an exchange holiday
    f = month_end_rows(d2)
    mon2 = d2.astype("datetime64[M]")
    month_rows = np.flatnonzero(mon2 == mon[hol])
    assert not f[month_rows].any()               # nothing knowable in that month
    first_next = month_rows[-1] + 1
    assert f[first_next]                         # decided one session late
    nxt_rows = np.flatnonzero(mon2 == mon2[first_next])
    assert f[nxt_rows].sum() == 2 and f[nxt_rows[-1]]


def test_month_end_is_point_in_time():
    d = weekdays(300)
    f = month_end_rows(d)
    for cut in (5, 40, 41, 120, 250):
        assert np.array_equal(month_end_rows(d[:cut + 1]), f[:cut + 1])


# ---------------------------------------------------------------- the book

def test_returns_compound_co_and_oc():
    co = np.array([[0.0], [0.01], [0.02]])
    oc = np.array([[0.0], [0.03], [-0.01]])
    close = np.array([[1.0], [1.0], [np.nan]])
    r, valid = daily_returns(co, oc, close, np.ones((3, 1), dtype=bool))
    assert valid.tolist() == [[False], [True], [False]]
    assert r[1, 0] == pytest.approx(1.01 * 1.03 - 1)
    assert np.isnan(r[2, 0])


def test_no_positions_before_252_sessions_and_only_on_decisions():
    m = make()
    W, dec = book(m)
    assert np.all(np.isnan(W[~dec]))
    early = np.flatnonzero(dec & (np.arange(len(dec)) < 251))
    assert early.size > 0 and np.all(W[early] == 0)
    late = np.flatnonzero(dec & (np.arange(len(dec)) >= 251))
    assert late.size >= 5
    assert all(np.abs(W[t]).sum() > 0 for t in late)


@pytest.mark.parametrize("q,k", [(0.2, 3), (0.1, 5), (0.3, 2), (0.25, 10)])
def test_book_matches_loop_reference(q, k):
    m = make(seed=1)
    W, dec = book(m, q, k)
    for t in np.flatnonzero(dec):
        np.testing.assert_allclose(W[t], ref_row(m, int(t), q, k), atol=1e-10)


def test_gross_signs_and_leg_sizes():
    m = make(seed=2, N=20)
    W, dec = book(m, 0.2, 3)
    for t in np.flatnonzero(dec & (np.arange(len(dec)) >= 251)):
        w = W[t]
        assert np.abs(w).sum() == pytest.approx(1.0)
        assert (w > 0).sum() == 4 and (w < 0).sum() == 4          # floor(0.2 * 20)
        assert np.allclose(w[w > 0], w[w > 0][0]) and np.allclose(w[w < 0], w[w < 0][0])


def test_beta_floor_binds_on_a_zero_beta_loser():
    m = make(seed=3, N=10)
    T = m["close"].shape[0]
    t = int(np.flatnonzero(month_end_rows(m["dates"]) & (np.arange(T) >= 300))[0])
    rng = np.random.default_rng(9)
    m["ret_oc"][:, 0] = rng.normal(0, 0.0005, T)                 # stock 0 has beta ~ 0 ...
    m["ret_oc"][t - 20:t + 1, 0] = -0.02                          # ... and is the biggest loser
    m["close"] = 100.0 * np.cumprod(1.0 + m["ret_oc"], axis=0)
    W, dec = book(m, 0.1, 3)                                      # one name per leg
    w = W[t]
    assert w[0] > 0
    np.testing.assert_allclose(w, ref_row(m, t, 0.1, 3), atol=1e-10)
    # stock 0's own beta is far below the floor, so b_L = 0.3 and w_L = b_S / (0.3 + b_S)
    r = m["ret_oc"]
    mkt = r[t - 251:t + 1].mean(axis=1)
    s = int(np.flatnonzero(w < 0)[0])
    b0 = np.polyfit(mkt, r[t - 251:t + 1, 0], 1)[0]
    bs = max(np.polyfit(mkt, r[t - 251:t + 1, s], 1)[0], 0.3)
    assert b0 < 0.3
    assert w[0] == pytest.approx(bs / (0.3 + bs))
    assert w[0] * 0.3 == pytest.approx(-w[s] * bs)                # w_L * b_L = w_S * b_S
    assert w[0] != pytest.approx(bs / (b0 + bs))                  # the unfloored sizing differs


def test_earnings_window_excludes_t_minus_20_but_not_t_minus_21():
    m = make(seed=4)
    W0, dec = book(m)
    t = int(np.flatnonzero(dec & (np.arange(len(dec)) >= 251))[0])
    held = np.flatnonzero(W0[t] != 0)
    j = int(held[0])
    m1 = make(seed=4)
    m1["ea"][t - 20, j] = 1.0
    W1, _ = book(m1)
    assert W1[t, j] == 0
    np.testing.assert_allclose(W1[t], ref_row(m1, t, 0.2, 3), atol=1e-10)
    m2 = make(seed=4)
    m2["ea"][t - 21, j] = 1.0
    W2, _ = book(m2)
    np.testing.assert_allclose(W2[t], W0[t])
    m3 = make(seed=4)
    m3["ea"][t - 3, j] = np.nan                                     # no EDGAR coverage
    W3, _ = book(m3)
    assert W3[t, j] == 0


def test_ineligible_names_still_serve_as_peers():
    m = make(seed=5)
    W0, dec = book(m)
    t = int(np.flatnonzero(dec & (np.arange(len(dec)) >= 251))[0])
    m["ea"][t, 0] = 1.0
    m["tradable"][t, 1] = False
    W, _ = book(m)
    assert W[t, 0] == 0 and W[t, 1] == 0
    np.testing.assert_allclose(W[t], ref_row(m, t, 0.2, 3), atol=1e-10)


def test_late_listing_and_missing_price():
    m = make(seed=6)
    T = m["close"].shape[0]
    m["listed"][:100, 0] = False
    m["close"][:100, 0] = np.nan
    W, dec = book(m)
    dec_rows = np.flatnonzero(dec)
    for t in dec_rows:
        if t < 100 + 252:                                           # first valid return at row 101
            assert W[t, 0] == 0
    np.testing.assert_allclose(W[dec_rows[-1]], ref_row(m, int(dec_rows[-1]), 0.2, 3), atol=1e-10)
    # a single missing price in the last 21 sessions makes a name ineligible
    m2 = make(seed=6)
    t = int(dec_rows[-1])
    m2["close"][t - 5, 3] = np.nan
    W2, _ = book(m2)
    assert W2[t, 3] == 0


def test_non_members_are_never_held_or_peers():
    m = make(seed=7)
    W0, dec = book(m)
    t = int(np.flatnonzero(dec & (np.arange(len(dec)) >= 251))[1])
    m["universe"][t, 2] = False
    W, _ = book(m)
    assert W[t, 2] == 0
    np.testing.assert_allclose(W[t], ref_row(m, t, 0.2, 3), atol=1e-10)


def test_parameters_change_output():
    m = make(seed=8, N=30)
    Wa, dec = book(m, 0.2, 3)
    Wb, _ = book(m, 0.1, 3)
    Wc, _ = book(m, 0.2, 10)
    rows = np.flatnonzero(dec & (np.arange(len(dec)) >= 251))
    assert all((Wa[t] != 0).sum() == 12 for t in rows)
    assert all((Wb[t] != 0).sum() == 6 for t in rows)
    assert any(not np.allclose(Wa[t], Wc[t]) for t in rows)


def test_too_few_peers_gives_flat_book():
    m = make(seed=8, N=6)
    W, dec = book(m, 0.2, 10)
    assert np.all(W[dec] == 0)


def test_no_reweighting_when_a_held_name_leaves_the_index():
    m = make(seed=9)
    W, dec = book(m)
    rows = np.flatnonzero(dec & (np.arange(len(dec)) >= 251))
    t = int(rows[0])
    j = int(np.flatnonzero(W[t] != 0)[0])
    m["universe"][t + 5:, j] = False
    data = FakeData(m)
    p = {"leg_quantile": 0.2, "peers_k": 3}
    out = target_weights(data, p)
    np.testing.assert_allclose(out[t], W[t])                        # decided while still a member
    assert np.all(np.isnan(out[t + 1:int(rows[1])]))                # the engine exits it; no replacement
    assert out[int(rows[1]), j] == 0                                # not a member: not bought again


def test_point_in_time_truncation_and_perturbation():
    m = make(seed=10)
    full, _ = book(m)
    T = full.shape[0]
    rng = np.random.default_rng(0)
    for cut in (260, 270, 290, 330, 371, T - 2):
        mt = {k: v[:cut + 1] for k, v in m.items()}
        wt, _ = book(mt)
        np.testing.assert_array_equal(np.isnan(wt), np.isnan(full[:cut + 1]))
        np.testing.assert_allclose(np.nan_to_num(wt), np.nan_to_num(full[:cut + 1]))
        mp = {k: v.copy() for k, v in m.items()}
        mp["ret_oc"][cut + 1:] = rng.normal(0, 0.05, mp["ret_oc"][cut + 1:].shape)
        mp["ea"][cut + 1:] = 1.0
        wp, _ = book(mp)
        np.testing.assert_allclose(np.nan_to_num(wp[:cut + 1]), np.nan_to_num(full[:cut + 1]))


def test_target_weights_and_diagnostic_agree():
    m = make(seed=11, N=15)
    data = FakeData(m)
    p = dict(PARAMS)
    p["peers_k"] = 3
    w = target_weights(data, p)
    assert np.all(np.nansum(np.abs(w), axis=1) <= 1 + 1e-9)
    mask, side = events(data, p)
    W, dec = book(m, p["leg_quantile"], 3)
    exp = dec[:, None] & (np.nan_to_num(W) != 0)
    assert np.array_equal(mask, exp)
    assert np.all(side[mask & (np.nan_to_num(W) > 0)] == 1)
    assert np.all(side[mask & (np.nan_to_num(W) < 0)] == -1)
    assert mask.sum() > 0


def test_missing_inputs_fail_loudly():
    m = make(seed=12)
    with pytest.raises(KeyError):
        target_weights(FakeData(m, with_ea=False), PARAMS)
    with pytest.raises(ValueError):
        target_weights(FakeData(m, with_universe=False), PARAMS)


def test_deterministic():
    m = make(seed=13)
    a, _ = book(m)
    b, _ = book(m)
    np.testing.assert_array_equal(np.nan_to_num(a, nan=7.0), np.nan_to_num(b, nan=7.0))
