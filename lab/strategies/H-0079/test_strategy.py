import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, PAIRS, zscores, run_book, diag_events


# ---------------------------------------------------------------- helpers

def _pair_returns(T=400, beta=0.8, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.01, T)
    y = beta * x + rng.normal(0.0, 0.003, T)
    return y[:, None], x[:, None]


def _ones(T, K=1):
    return np.ones((T, K), dtype=bool)


def _names():
    names = []
    for c, a in PAIRS:
        for s in (c, a):
            if s not in names:
                names.append(s)
    return tuple(names)


def _view(T=420, seed=3, shocks=()):
    names = _names()
    N = len(names)
    rng = np.random.default_rng(seed)
    ret_oc = rng.normal(0.0, 0.008, (T, N))
    ret_co = rng.normal(0.0, 0.002, (T, N))
    for row, name, size in shocks:
        ret_oc[row, names.index(name)] += size
    tradable = np.ones((T, N), dtype=bool)
    listed = np.ones((T, N), dtype=bool)
    close = np.full((T, N), 20.0)
    return DataView(
        dates=np.arange(T).astype("datetime64[D]"),
        instruments=names,
        asset_class=tuple("us_etf" for _ in names),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((T, N), dtype=bool),
        close=close,
        dollar_volume=np.full((T, N), 1e6),
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(T),
    )


# ---------------------------------------------------------------- score

def test_history_rules_first_valid_row():
    y, x = _pair_returns()
    z = zscores(y, x, _ones(len(y)), _ones(len(y)))
    # beta needs 200 joint sessions (row 199); sd needs 200 5-session values ending at t-5,
    # the first 5-session value is at row 4, so t - 5 - 4 + 1 >= 200 -> t >= 208
    assert np.all(np.isnan(z[:208, 0]))
    assert np.isfinite(z[208, 0])


def test_zscore_matches_direct_computation():
    y, x = _pair_returns(T=500)
    z = zscores(y, x, _ones(500), _ones(500))
    for t in (260, 333, 499):
        xs, ys = x[t - 251:t + 1, 0], y[t - 251:t + 1, 0]
        b = np.cov(xs, ys, ddof=1)[0, 1] / np.var(xs, ddof=1)
        b = min(max(b, 0.25), 2.0)
        e = [y[s - 4:s + 1, 0].sum() - b * x[s - 4:s + 1, 0].sum() for s in range(t - 5 - 251, t - 4)]
        sd = np.std(e, ddof=1)
        et = y[t - 4:t + 1, 0].sum() - b * x[t - 4:t + 1, 0].sum()
        assert z[t, 0] == pytest.approx(et / sd, rel=1e-6)


def test_beta_is_clipped():
    y, x = _pair_returns(T=400, beta=5.0)
    z = zscores(y, x, _ones(400), _ones(400))
    t = 350
    b = 2.0
    e = [y[s - 4:s + 1, 0].sum() - b * x[s - 4:s + 1, 0].sum() for s in range(t - 5 - 251, t - 4)]
    et = y[t - 4:t + 1, 0].sum() - b * x[t - 4:t + 1, 0].sum()
    assert z[t, 0] == pytest.approx(et / np.std(e, ddof=1), rel=1e-6)


def test_shock_gives_negative_z_and_missing_bar_gives_no_score():
    y, x = _pair_returns(T=400)
    y[350, 0] -= 0.05
    z = zscores(y, x, _ones(400), _ones(400))
    assert z[350, 0] < -3.0
    bx = _ones(400)
    bx[347, 0] = False                       # anchor without a bar inside the last 5 sessions
    z2 = zscores(y, x, _ones(400), bx)
    assert np.all(np.isnan(z2[347:352, 0]))
    assert np.isfinite(z2[352, 0])


def test_late_listing_no_score():
    y, x = _pair_returns(T=500)
    by = _ones(500)
    by[:150, 0] = False                      # CEF listed at row 150
    z = zscores(y, x, by, _ones(500))
    assert np.all(np.isnan(z[:150 + 208, 0]))
    assert np.isfinite(z[150 + 208, 0])


# ---------------------------------------------------------------- book

def _book(z, hold=5, z_entry=2.0, tradable=None, gone=None):
    T, K = z.shape
    tr = _ones(T, K) if tradable is None else tradable
    gn = np.zeros((T, K), dtype=bool) if gone is None else gone
    return run_book(z, z_entry, hold, _ones(T, K), tr, gn)


def test_slot_cap_most_negative_first_and_hold():
    z = np.zeros((30, 3))
    z[5] = [-3.0, -2.5, -2.1]
    ev, held = _book(z)
    assert ev[5].tolist() == [True, True, True]
    assert held[5:10, 0].tolist() == [1, 1, 1, 1, 1] and held[10, 0] == 0
    assert held[5:10, 1].tolist() == [1, 1, 1, 1, 1] and held[10, 1] == 0
    assert held[:, 2].sum() == 0             # dropped, not queued
    assert held[:5].sum() == 0


def test_no_event_within_hold_days_and_slot_reuse():
    z = np.zeros((30, 3))
    z[5] = [-3.0, -2.5, -2.1]
    z[7, 2] = -3.0                           # last event of fund 2 at row 5: blocked
    z[10, 2] = -3.0                          # 5 sessions after: still blocked (previous 5 include 5)
    z[11, 2] = -3.0                          # free and both slots exited at open 11
    ev, held = _book(z)
    assert not ev[7, 2] and not ev[10, 2] and ev[11, 2]
    assert held[11:16, 2].tolist() == [1, 1, 1, 1, 1] and held[16, 2] == 0


def test_freed_slot_usable_on_exit_day():
    z = np.zeros((30, 3))
    z[5, :2] = -3.0
    z[10, 2] = -3.0                          # exits ordered at row 10, new entry same open
    ev, held = _book(z)
    assert ev[10, 2] and held[10, 2] == 1
    assert held.sum(axis=1).max() <= 2


def test_untradable_entry_is_skipped():
    z = np.zeros((30, 3))
    z[5, 0] = -3.0
    z[5, 1] = -2.5
    tr = _ones(30, 3)
    tr[6, 0] = False
    z[6, 2] = -3.0                           # takes the slot freed by the skip
    ev, held = _book(z, tradable=tr)
    assert held[5, 0] == 1 and held[6:, 0].sum() == 0
    assert ev[6, 2] and held[6, 2] == 1


def test_stuck_exit_keeps_slot_occupied():
    z = np.zeros((30, 3))
    z[5, :2] = -3.0
    tr = _ones(30, 3)
    tr[11, :2] = False                       # neither can exit at open 11
    z[11, 2] = -3.0                          # both slots still busy: dropped
    z[17, 2] = -3.0
    ev, held = _book(z, tradable=tr)
    assert held[10:, 0].sum() == 0
    assert ev[11, 2] and held[11, 2] == 0
    assert ev[17, 2] and held[17, 2] == 1


def test_hold_and_z_entry_params():
    z = np.zeros((30, 2))
    z[5, 0] = -2.2
    _, h5 = _book(z, hold=5)
    _, h3 = _book(z, hold=3)
    assert h5[:, 0].sum() == 5 and h3[:, 0].sum() == 3
    _, h25 = _book(z, z_entry=2.5)
    assert h25.sum() == 0


# ---------------------------------------------------------------- full strategy

def test_target_weights_same_anchor_adds_and_gross():
    # large shocks and a high threshold so that no random crossing occupies the slots
    view = _view(shocks=((380, "NUV", -0.3), (380, "NEA", -0.3)))
    w = target_weights(view, {"hold_days": 5, "z_entry": 8.0})
    nuv, nea, mub = view.col("NUV"), view.col("NEA"), view.col("MUB")
    assert w[380, nuv] == 0.25 and w[380, nea] == 0.25 and w[380, mub] == -0.5
    assert w[384, nuv] == 0.25 and w[385, nuv] == 0.0 and w[385, mub] == 0.0
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12
    assert np.all(np.abs(w[:208]) == 0)
    assert np.all(w[:, list(view.col(a) for _, a in PAIRS)] <= 0)   # anchors only short
    assert np.all(w[:, list(view.col(c) for c, _ in PAIRS)] >= 0)   # CEFs only long


def test_params_change_weights_and_diagnostic_matches():
    view = _view(seed=5)
    w = target_weights(view, PARAMS)
    assert not np.array_equal(w, target_weights(view, {"hold_days": 3, "z_entry": 2.0}))
    assert not np.array_equal(w, target_weights(view, {"hold_days": 5, "z_entry": 1.5}))
    mask, side = diag_events(view, PARAMS)
    assert side is None
    entries = (w > 0) & ~np.vstack([np.zeros((1, w.shape[1]), dtype=bool), w[:-1] > 0])
    assert np.all(mask[entries])             # every entry is an event
    anchors = sorted(set(view.col(a) for _, a in PAIRS))
    assert not mask[:, anchors].any()


def test_truncation_invariance():
    view = _view(seed=7)
    w = target_weights(view, PARAMS)
    for cut in (250, 333):
        sub = DataView(
            dates=view.dates[:cut], instruments=view.instruments, asset_class=view.asset_class,
            ret_co=view.ret_co[:cut], ret_oc=view.ret_oc[:cut], tradable=view.tradable[:cut],
            listed=view.listed[:cut], delisting=view.delisting[:cut], close=view.close[:cut],
            dollar_volume=view.dollar_volume[:cut], universe=None, extras={}, series={},
            cash_ret=view.cash_ret[:cut],
        )
        assert np.array_equal(target_weights(sub, PARAMS), w[:cut])
