import numpy as np
import pytest
from lab.framework.data import synthetic, DataView
from lab.framework.api import period_starts

from strategy import target_weights, PARAMS


def make(T=200, N=5, seed=0, vol=0.01):
    """Hand-made LIQ market: all names listed, members, tradable; ret_co = 0, ret_oc ~ N(0, vol)."""
    base = synthetic()
    rng = np.random.default_rng(seed)
    names = tuple("E" + str(1000 + j) for j in range(N))
    return {
        "dates": base.dates[:T],
        "instruments": names,
        "asset_class": tuple("us_equity" for _ in range(N)),
        "ret_co": np.zeros((T, N)),
        "ret_oc": rng.normal(0.0, vol, (T, N)),
        "tradable": np.ones((T, N), dtype=bool),
        "listed": np.ones((T, N), dtype=bool),
        "delisting": np.zeros((T, N), dtype=bool),
        "close": np.full((T, N), 50.0),
        "dollar_volume": np.full((T, N), 1e8),
        "universe": np.ones((T, N), dtype=bool),
        "extras": {
            "ea_reaction_day": np.zeros((T, N), dtype=np.float32),
            "liq_rank": np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1)),
        },
        "series": {},
        "cash_ret": None,
    }


def view(a):
    return DataView(**a)


def expected_slot(a, t, j, params=PARAMS):
    """Independent re-computation of rules (1)-(4) for one event."""
    r = (1 + a["ret_co"]) * (1 + a["ret_oc"]) - 1
    ok = a["listed"] & np.isfinite(a["close"])
    uni = a["universe"]
    T, N = r.shape
    abn = np.full((T, N), np.nan)
    for s in range(T):
        m = uni[s] & ok[s]
        if m.any():
            abn[s] = np.where(ok[s], r[s] - r[s, m].mean(), np.nan)

    def sig(col):
        x = abn[max(0, t - 63):t, col]
        x = x[np.isfinite(x)]
        return np.std(x, ddof=1) if len(x) >= 50 else np.nan

    sigmas = np.array([sig(c) if uni[t, c] else np.nan for c in range(N)])
    med = np.nanmedian(sigmas)
    z = abn[t, j] / sigmas[j]
    w = np.sign(z) * params["event_weight"] * np.clip(med / sigmas[j], 0.5, 2.0)
    return w, z


def with_event(a, t, j, move):
    a["ret_oc"][t, j] = move
    a["extras"]["ea_reaction_day"][t, j] = 1.0
    return a


# ---------------------------------------------------------------- rules (1)-(4)

def test_event_slot_weight_and_hold():
    a = with_event(make(), 100, 0, 0.10)
    w = target_weights(view(a), PARAMS)
    w0, z = expected_slot(a, 100, 0)
    assert z >= 2.0 and w0 > 0
    assert np.all(w[:100] == 0.0)                         # flat before any event
    assert np.allclose(w[100:105, 0], w0)                 # held on hold_days decision rows
    assert np.all(w[105:, 0] == 0.0)                      # closed at the open of t+1+hold_days
    assert np.allclose(w[100:105, 1:], -w0 / 4)           # hedge -N_t over the other members
    assert np.allclose(w[100:105].sum(axis=1), 0.0)
    assert np.all(w[105:] == 0.0)                          # flat when no slot is open


def test_negative_reaction_is_short_and_hedge_long():
    a = with_event(make(), 100, 2, -0.10)
    w = target_weights(view(a), PARAMS)
    w0, _ = expected_slot(a, 100, 2)
    assert w0 < 0
    assert np.allclose(w[100:105, 2], w0)
    others = [0, 1, 3, 4]
    assert np.allclose(w[100:105][:, others], -w0 / 4)


def test_risk_scaling_clip():
    a = make()
    a["ret_oc"][:, 0] *= 10.0                             # very volatile name -> ratio clipped at 0.5
    a = with_event(a, 100, 0, 1.0)
    w = target_weights(view(a), PARAMS)
    assert np.isclose(w[100, 0], 0.5 * PARAMS["event_weight"])
    b = make()
    b["ret_oc"][:, 1] *= 0.1                              # very quiet name -> ratio clipped at 2.0
    b = with_event(b, 100, 1, 0.05)
    w = target_weights(view(b), PARAMS)
    assert np.isclose(w[100, 1], 2.0 * PARAMS["event_weight"])


def test_small_move_or_no_flag_or_nan_flag_is_no_event():
    a = make()
    a["ret_oc"][100, 0] = 0.10                            # big move, no flag
    a["extras"]["ea_reaction_day"][:, 1] = np.nan         # NaN flags count as 0
    a["ret_oc"][100, 1] = 0.10
    assert np.all(target_weights(view(a), PARAMS) == 0.0)
    b = make()
    b["extras"]["ea_reaction_day"][100, 3] = 1.0          # flag, small abnormal move
    b["ret_oc"][100, :] = 0.0
    b["ret_oc"][100, 3] = 0.001
    assert np.all(target_weights(view(b), PARAMS) == 0.0)


def test_needs_50_valid_vol_values():
    a = with_event(make(), 45, 0, 0.10)                   # only 45 prior rows
    assert np.all(target_weights(view(a), PARAMS) == 0.0)
    a = with_event(make(), 50, 0, 0.10)                   # exactly 50 prior rows
    assert target_weights(view(a), PARAMS)[50, 0] > 0


def test_late_listing():
    a = make()
    a["listed"][:80, 4] = False
    a["close"][:80, 4] = np.nan
    a["ret_oc"][:80, 4] = 0.0
    a["universe"][:80, 4] = False
    a = with_event(a, 125, 4, 0.10)                       # 45 valid values since listing
    a = with_event(a, 140, 4, 0.10)                       # 60 valid values
    w = target_weights(view(a), PARAMS)
    assert np.all(w[125:140, 4] == 0.0)
    w0, _ = expected_slot(a, 140, 4)
    assert np.allclose(w[140:145, 4], w0)
    assert np.all(w[:80, 4] == 0.0)                       # never held while not a member


def test_reaction_session_excluded_from_sigma():
    a = with_event(make(), 100, 0, 0.10)
    b = with_event(make(), 100, 0, 0.30)
    wa = target_weights(view(a), PARAMS)[100, 0]
    wb = target_weights(view(b), PARAMS)[100, 0]
    assert np.isclose(wa, wb)                              # size does not depend on today's move


# ---------------------------------------------------------------- exits and replacement

def test_leaving_universe_closes_slot():
    a = with_event(make(), 100, 0, 0.10)
    a["universe"][102, 0] = False
    w = target_weights(view(a), PARAMS)
    assert w[101, 0] > 0
    assert np.all(w[102:, 0] == 0.0)
    assert np.all(w[102:] == 0.0)                          # no slot -> no hedge


def test_not_tradable_closes_slot():
    a = with_event(make(), 100, 0, 0.10)
    a["tradable"][103, 0] = False
    w = target_weights(view(a), PARAMS)
    assert w[102, 0] > 0
    assert np.all(w[103:, 0] == 0.0)


def test_new_event_replaces_slot():
    a = with_event(make(), 100, 0, 0.10)
    a = with_event(a, 103, 0, -0.10)
    w = target_weights(view(a), PARAMS)
    w2, _ = expected_slot(a, 103, 0)
    assert w[100, 0] > 0 and w2 < 0
    assert np.allclose(w[103:108, 0], w2)
    assert np.all(w[108:, 0] == 0.0)


# ---------------------------------------------------------------- parameters

def test_params_change_output():
    a = with_event(make(), 100, 0, 0.10)
    v = view(a)
    base = target_weights(v, PARAMS)
    p = dict(PARAMS, hold_days=3)
    w = target_weights(v, p)
    assert np.all(w[100:103, 0] > 0) and np.all(w[103:, 0] == 0.0)
    p = dict(PARAMS, hold_days=7)
    w = target_weights(v, p)
    assert np.all(w[100:107, 0] > 0) and np.all(w[107:, 0] == 0.0)
    p = dict(PARAMS, event_weight=0.035)
    w = target_weights(v, p)
    assert np.allclose(w[100], base[100] * 0.035 / 0.025)
    _, z = expected_slot(a, 100, 0)
    p = dict(PARAMS, z_threshold=float(z) + 0.01)
    assert np.all(target_weights(v, p) == 0.0)
    p = dict(PARAMS, z_threshold=float(z) - 0.01)
    assert np.allclose(target_weights(v, p), base)


# ---------------------------------------------------------------- gross cap

def test_gross_cap_scales_proportionally():
    a = make(T=120, N=80, seed=3)
    for j in range(30):
        a = with_event(a, 100, j, 0.15)
    for j in range(30, 60):
        a = with_event(a, 100, j, -0.15)
    v = view(a)
    small = target_weights(v, dict(PARAMS, event_weight=0.015))
    g_small = np.abs(small[100]).sum()
    assert g_small < 1.0
    big = target_weights(v, dict(PARAMS, event_weight=0.035))
    assert np.isclose(np.abs(big[100]).sum(), 1.0)
    assert np.allclose(big[100], small[100] / g_small)
    for p in (PARAMS, dict(PARAMS, event_weight=0.035)):
        assert np.all(np.abs(target_weights(v, p)).sum(axis=1) <= 1.0 + 1e-12)


# ---------------------------------------------------------------- hedge basket

def test_hedge_basket_top100_monthly_refresh():
    T, N = 200, 120
    a = make(T=T, N=N, seed=5)
    ms = np.flatnonzero(np.asarray(period_starts(a["dates"], "M")))
    p = int(ms[ms > 100][0])                              # a month start (old ranks used here)
    m = int(ms[ms > p][0])                                # the next month start
    assert m < T - 5
    rank_new = np.arange(N, 0, -1, dtype=np.float32)      # reversed liquidity order mid-month
    a["extras"]["liq_rank"][p + 3:] = rank_new
    e = m - 2
    a = with_event(a, e, 60, 0.15)
    w = target_weights(view(a), PARAMS)
    w60 = w[e, 60]
    assert w60 > 0
    # before the refresh: old basket = columns 0..99 (best old rank), minus the event name
    old = [j for j in range(100) if j != 60]
    assert np.allclose(w[m - 1, old], -w60 / 99)
    assert np.all(w[m - 1, 100:] == 0.0)
    # at the refresh: top 100 by the new rank among names without a slot = 19..119 minus 60
    new = [j for j in range(19, N) if j != 60]
    assert len(new) == 100
    assert np.allclose(w[m, new], -w60 / 100)
    assert np.all(w[m, :19] == 0.0)


def test_hedge_drops_universe_leavers():
    a = make(T=200, N=6)
    a = with_event(a, 100, 0, 0.10)
    a["universe"][101:, 5] = False
    w = target_weights(view(a), PARAMS)
    w0 = w[100, 0]
    assert np.allclose(w[100, 1:], -w0 / 5)
    assert np.allclose(w[101:105, 1:5], -w0 / 4)
    assert np.all(w[101:, 5] == 0.0)


# ---------------------------------------------------------------- point in time, diagnostic

def busy_market(seed=11):
    T, N = 260, 30
    a = make(T=T, N=N, seed=seed)
    rng = np.random.default_rng(seed + 1)
    flags = rng.random((T, N)) < 0.04
    moves = rng.choice([-0.08, 0.08, 0.01], size=(T, N))
    a["ret_oc"] = np.where(flags, moves, a["ret_oc"])
    a["extras"]["ea_reaction_day"] = flags.astype(np.float32)
    a["extras"]["liq_rank"] = rng.permutation(np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1)).T).T
    a["tradable"][rng.random((T, N)) < 0.01] = False
    return a


def cut(a, k):
    b = dict(a)
    for key in ("dates", "ret_co", "ret_oc", "tradable", "listed", "delisting", "close",
                "dollar_volume", "universe"):
        b[key] = a[key][:k]
    b["extras"] = {key: val[:k] for key, val in a["extras"].items()}
    return b


def test_point_in_time_truncation_and_perturbation():
    a = busy_market()
    full = target_weights(view(a), PARAMS)
    assert np.abs(full).sum() > 0
    rng = np.random.default_rng(99)
    for k in (70, 101, 150, 199, 233):
        tr = target_weights(view(cut(a, k)), PARAMS)
        assert np.array_equal(tr, full[:k])
        b = dict(a)
        b["ret_oc"] = a["ret_oc"].copy()
        b["ret_oc"][k:] = rng.normal(0, 0.05, b["ret_oc"][k:].shape)
        fl = a["extras"]["ea_reaction_day"].copy()
        fl[k:] = (rng.random(fl[k:].shape) < 0.3).astype(np.float32)
        b["extras"] = dict(a["extras"], ea_reaction_day=fl)
        pert = target_weights(view(b), PARAMS)
        assert np.array_equal(pert[:k], full[:k])


def test_missing_flag_raises():
    a = make()
    del a["extras"]["ea_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(view(a), PARAMS)


def test_synthetic_runs_and_gross():
    d = synthetic()
    T, N = len(d.dates), len(d.instruments)
    rng = np.random.default_rng(0)
    d2 = DataView(dates=d.dates, instruments=d.instruments, asset_class=d.asset_class,
                  ret_co=d.ret_co, ret_oc=d.ret_oc, tradable=d.tradable, listed=d.listed,
                  delisting=d.delisting, close=d.close, dollar_volume=d.dollar_volume,
                  universe=np.asarray(d.listed, dtype=bool),
                  extras={"ea_reaction_day": (rng.random((T, N)) < 0.02).astype(np.float32),
                          "liq_rank": np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))},
                  series=d.series, cash_ret=d.cash_ret)
    w = target_weights(d2, PARAMS)
    assert w.shape == (T, N)
    assert np.all(np.isfinite(w))
    assert np.all(np.abs(w).sum(axis=1) <= 1.0 + 1e-12)
    assert np.abs(w).sum() > 0
    assert np.all(w[~np.asarray(d2.universe)] == 0.0)
