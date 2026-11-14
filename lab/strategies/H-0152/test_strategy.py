import numpy as np
import pytest
from dataclasses import dataclass, field
from typing import Any

from strategy import target_weights, PARAMS, compute_events, _betas, _daily_returns


@dataclass
class View:
    dates: Any
    instruments: tuple
    ret_co: Any
    ret_oc: Any
    tradable: Any
    listed: Any
    close: Any
    universe: Any
    extras: dict = field(default_factory=dict)
    series: dict = field(default_factory=dict)

    def col(self, name):
        return self.instruments.index(name)


def make_view(T=700, N=4, seed=1):
    rng = np.random.default_rng(seed)
    dates = np.arange(T).astype("datetime64[D]")
    mkt = rng.normal(0, 0.01, T)
    ret = mkt[:, None] * np.linspace(0.6, 1.6, N)[None, :] + rng.normal(0, 0.01, (T, N))
    ret_co = ret / 2.0
    ret_oc = (1 + ret) / (1 + ret_co) - 1
    shares = np.full((T, N), 100.0)
    return View(
        dates=dates,
        instruments=tuple("E%d" % (i + 1) for i in range(N)),
        ret_co=ret_co,
        ret_oc=ret_oc,
        tradable=np.ones((T, N), bool),
        listed=np.ones((T, N), bool),
        close=np.full((T, N), 10.0),
        universe=np.ones((T, N), bool),
        extras={"sf1_arq_sharesbas": shares.astype(np.float32)},
    )


def month_starts(dates):
    m = dates.astype("datetime64[M]")
    out = np.zeros(len(dates), bool)   # row 0 left out: the framework may or may not mark it
    out[1:] = m[1:] != m[:-1]
    return np.nonzero(out)[0]


def base_case():
    v = make_view()
    sh = v.extras["sf1_arq_sharesbas"]
    # E1: change at 250 (ratio 1.05, too small), then 1.2x at 300 -> event at 300
    sh[250:, 0] = 105.0
    sh[300:, 0] = 126.0
    # E2: 1.2x at 300 but previous value unchanged since row 0 -> filing gap, no event
    sh[300:, 1] = 120.0
    # E3: change at 250, then 2.0x at 300 -> split guard, no event
    sh[250:, 2] = 105.0
    sh[300:, 2] = 210.0
    # E4: change at 250, then 1.2x at 300 but not a member at 300 -> no event
    sh[250:, 3] = 105.0
    sh[300:, 3] = 126.0
    v.universe[300, 3] = False
    return v


def test_event_rules():
    v = base_case()
    ev = compute_events(v, PARAMS)
    assert ev.sum() == 1 and ev[300, 0]
    # issuance_min above the ratio removes the event
    assert compute_events(v, {**PARAMS, "issuance_min": 0.25}).sum() == 0
    assert compute_events(v, {**PARAMS, "issuance_min": 0.19}).sum() == 1
    assert compute_events(v, {**PARAMS, "issuance_min": 0.05}).sum() == 1  # 1.05 at 250 has no prior change


def test_ratio_cap_and_split_guard():
    v = make_view()
    sh = v.extras["sf1_arq_sharesbas"]
    sh[:250, :] = 90.0            # change at 250 (no event there: previous value never changed)
    sh[300:, 0] = 100.0 * 3.5     # above the 3.0 cap
    sh[300:, 1] = 100.0 * 1.52    # within 3 % of 1.5
    sh[300:, 2] = 100.0 * 1.56    # outside 3 % of 1.5 -> event
    sh[300:, 3] = 100.0 * 2.5     # event
    ev = compute_events(v, PARAMS)
    assert list(np.nonzero(ev[300])[0]) == [2, 3]


def test_trailing_return_and_valid_days():
    v = base_case()
    v.ret_co[100:300, 0] = -0.01
    v.ret_oc[100:300, 0] = -0.01   # about -98 % over the window -> no event
    assert compute_events(v, PARAMS).sum() == 0
    v = base_case()
    v.close[0:200, 0] = np.nan      # only 101 valid rows (200..300) in the 252 window
    assert compute_events(v, PARAMS).sum() == 0


def test_one_event_per_hold_window():
    v = base_case()
    sh = v.extras["sf1_arq_sharesbas"]
    sh[400:, 0] = 126.0 * 1.2       # second jump 100 sessions later
    ev = compute_events(v, {**PARAMS, "hold_days": 252})
    assert ev[:, 0].sum() == 1
    ev = compute_events(v, {**PARAMS, "hold_days": 50})
    assert ev[300, 0] and ev[400, 0]


def test_book_weights_and_exit():
    v = base_case()
    w = target_weights(v, PARAMS)
    reb = month_starts(v.dates)
    nonreb = np.setdiff1d(np.arange(1, len(v.dates)), reb)
    assert np.all(np.isnan(w[nonreb]))
    for t in reb:
        row = w[t]
        assert np.all(np.isfinite(row))
        assert np.abs(row).sum() <= 1.0 + 1e-9
        if t < 300 or t - 300 >= 252:
            assert np.all(row == 0)
        else:
            # one short name: cap binds, s = 0.05
            assert row[0] == pytest.approx(-0.05)
            assert np.all(row[1:] > 0)
            assert np.allclose(row[1:], row[1])
            r, valid = _daily_returns(v)
            mv = v.universe & valid
            m = np.where(mv, r, 0).sum(1) / mv.sum(1)
            bs = _betas(r, valid, m, t, np.array([0])).mean()
            bl = _betas(r, valid, m, t, np.array([1, 2, 3])).mean()
            assert 0.05 * bs == pytest.approx(row[1:].sum() * bl)
    # shorter hold exits earlier
    w2 = target_weights(v, {**PARAMS, "hold_days": 126})
    first_after = reb[reb >= 300][0]
    exit_reb = reb[reb - 300 >= 126][0]
    assert w2[first_after, 0] < 0
    assert w2[exit_reb, 0] == 0 and np.all(w2[exit_reb] == 0)


def test_universe_exit_between_rebalances():
    v = base_case()
    reb = month_starts(v.dates)
    t0 = reb[reb >= 300][0]
    t1 = reb[reb > t0][0]
    k = t0 + 5
    assert k < t1
    v.universe[k:k + 3, 1] = False       # hedge name leaves for 3 rows
    w = target_weights(v, PARAMS)
    assert w[k, 1] == 0 and w[k + 1, 1] == 0
    assert np.isnan(w[k, 0]) and np.isnan(w[k, 2])
    assert np.all(np.isnan(w[t0 + 1:k]))


def test_beta_neutral_uncapped():
    v = make_view(T=700, N=30, seed=3)
    sh = v.extras["sf1_arq_sharesbas"]
    sh[250:, :] = 105.0
    sh[300:, :15] = 105.0 * 1.3
    w = target_weights(v, PARAMS)
    t = month_starts(v.dates)
    t = t[t >= 300][0]
    row = w[t]
    assert np.all(row[:15] < 0) and np.all(row[15:] > 0)
    assert np.abs(row).sum() == pytest.approx(1.0)
    r, valid = _daily_returns(v)
    m = np.where(v.universe & valid, r, 0).sum(1) / (v.universe & valid).sum(1)
    bs = _betas(r, valid, m, t, np.arange(15)).mean()
    bl = _betas(r, valid, m, t, np.arange(15, 30)).mean()
    assert -row[:15].sum() * bs == pytest.approx(row[15:].sum() * bl)


def test_beta_rule():
    T = 300
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, T)
    r = np.stack([2.0 * m, 5.0 * m, 0.1 * m, 1.5 * m], axis=1)
    valid = np.ones((T, 4), bool)
    valid[:, 3] = False
    valid[100:, 3] = True              # exactly 200 valid rows in the window -> enough
    b = _betas(r, valid, m, T - 1, np.arange(4))
    assert b[0] == pytest.approx(2.0) and b[1] == pytest.approx(2.5) and b[2] == pytest.approx(0.3)
    assert b[3] == pytest.approx(1.5)
    valid[:, 3] = False
    valid[150:, 3] = True              # 150 rows -> beta 1.0
    assert _betas(r, valid, m, T - 1, np.array([3]))[0] == pytest.approx(1.0)


def test_point_in_time_truncation():
    v = base_case()
    w = target_weights(v, PARAMS)
    for k in (305, 340, 450):
        vk = View(dates=v.dates[:k], instruments=v.instruments, ret_co=v.ret_co[:k], ret_oc=v.ret_oc[:k],
                  tradable=v.tradable[:k], listed=v.listed[:k], close=v.close[:k], universe=v.universe[:k],
                  extras={"sf1_arq_sharesbas": v.extras["sf1_arq_sharesbas"][:k]})
        wk = target_weights(vk, PARAMS)
        assert np.array_equal(np.isnan(wk), np.isnan(w[:k]))
        assert np.allclose(np.nan_to_num(wk), np.nan_to_num(w[:k]))


def test_missing_inputs_raise():
    v = base_case()
    v.extras = {}
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
    v = base_case()
    v.universe = None
    with pytest.raises(ValueError):
        target_weights(v, PARAMS)
