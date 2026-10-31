import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, compute_signal, CEFS, ANCHOR, events

NAMES = (ANCHOR,) + CEFS          # MUB first: columns must be found by name, not position
T = 600
SHOCK_ROW = 400


def make_view(ret_oc, listed=None, tradable=None, dv=None, names=NAMES):
    t_len, n = ret_oc.shape
    if listed is None:
        listed = np.ones((t_len, n), dtype=bool)
    if tradable is None:
        tradable = listed.copy()
    if dv is None:
        dv = np.where(listed, 5.0e6, np.nan)
    dates = (np.arange(t_len) + 14000).astype("datetime64[D]")
    return DataView(
        dates=dates,
        instruments=tuple(names),
        asset_class=tuple("us_etf" for _ in names),
        ret_co=np.zeros((t_len, n)),
        ret_oc=ret_oc,
        tradable=tradable,
        listed=listed,
        delisting=np.zeros((t_len, n), dtype=bool),
        close=np.where(listed, 10.0, np.nan),
        dollar_volume=dv,
        universe=None,
        extras={},
        series={},
        cash_ret=np.zeros(t_len),
    )


def noisy_returns(seed=7, names=NAMES):
    rng = np.random.default_rng(seed)
    mub = rng.normal(0.0, 0.004, T)
    r = np.zeros((T, len(names)))
    for j, name in enumerate(names):
        if name == ANCHOR:
            r[:, j] = mub
        else:
            r[:, j] = (1.0 + 0.02 * (j % 10)) * mub + rng.normal(0.0, 0.006, T)
    return r


def col(name, names=NAMES):
    return names.index(name)


def shock_setup():
    """Seven liquid funds (CEFS[0..6]) shocked on SHOCK_ROW, all others illiquid; funds untradable on rows
    390..399 so no slot is open before the shock."""
    r = noisy_returns()
    tradable = np.ones(r.shape, dtype=bool)
    dv = np.full(r.shape, 5.0e6)
    for j, name in enumerate(CEFS):
        c = col(name)
        if j < 7:
            tradable[390:400, c] = False
            r[SHOCK_ROW, c] = -0.08 - 0.01 * j
        else:
            dv[:, c] = 1.0e5            # illiquid -> never eligible
    return r, tradable, dv


def test_first_valid_row_depends_on_formation_days():
    data = make_view(noisy_returns())
    c = CEFS.index("NUV")
    for k in (3, 5, 10):
        sig = compute_signal(data, {"formation_days": k, "z_entry": 2.0})
        z = sig["z"][:, c]
        first = 249 + k                      # beta from row 249, sigma windows need rows t-250-k+1 >= 0
        assert np.all(np.isnan(z[:first]))
        assert np.isfinite(z[first])
        assert not sig["eligible"][:first].any()


def test_beta_estimate_and_clipping():
    r = noisy_returns()
    rng = np.random.default_rng(1)
    m = r[:, col(ANCHOR)]
    r[:, col("NUV")] = 1.2 * m
    r[:, col("NEA")] = 3.0 * m
    r[:, col("NVG")] = 0.1 * m + rng.normal(0.0, 0.001, T)
    sig = compute_signal(make_view(r), PARAMS)
    beta = sig["beta"]
    assert np.all(np.isnan(beta[:249]))
    assert beta[300, CEFS.index("NUV")] == pytest.approx(1.2, abs=0.02)
    assert beta[300, CEFS.index("NEA")] == pytest.approx(1.75)
    assert beta[300, CEFS.index("NVG")] == pytest.approx(0.5)


def test_capacity_ranking_hold_and_hedge():
    r, tradable, dv = shock_setup()
    data = make_view(r, tradable=tradable, dv=dv)
    sig = compute_signal(data, PARAMS)
    w = target_weights(data, PARAMS)
    shocked = list(range(7))
    zz = sig["z"][SHOCK_ROW, shocked]
    assert np.all(zz <= -2.0)
    expect = sorted(np.array(shocked)[np.argsort(zz, kind="stable")[:5]].tolist())
    held = [j for j in range(len(CEFS)) if w[SHOCK_ROW, col(CEFS[j])] > 0]
    assert held == expect
    for j in expect:
        assert w[SHOCK_ROW, col(CEFS[j])] == pytest.approx(0.07)
    hedge = -0.07 * sum(sig["beta"][SHOCK_ROW, j] for j in expect)
    assert w[SHOCK_ROW, col(ANCHOR)] == pytest.approx(hedge)
    # no slot open before the shock (funds untradable on 390..399)
    assert np.all(w[390:SHOCK_ROW] == 0.0)
    # held unchanged through decision rows t+1..t+4, closed at decision t+5
    for t in range(SHOCK_ROW + 1, SHOCK_ROW + 5):
        assert np.array_equal(w[t], w[SHOCK_ROW])
    for j in expect:
        assert w[SHOCK_ROW + 5, col(CEFS[j])] == 0.0


def test_untradable_fund_closes_slot_and_hedge_and_frees_it():
    r, tradable, dv = shock_setup()
    data0 = make_view(r, tradable=tradable, dv=dv)
    w0 = target_weights(data0, PARAMS)
    held = [j for j in range(len(CEFS)) if w0[SHOCK_ROW, col(CEFS[j])] > 0]
    f0 = held[0]
    tradable[SHOCK_ROW + 2, col(CEFS[f0])] = False
    data = make_view(r, tradable=tradable, dv=dv)
    sig = compute_signal(data, PARAMS)
    w = target_weights(data, PARAMS)
    t = SHOCK_ROW + 2
    assert w[t - 1, col(CEFS[f0])] == pytest.approx(0.07)
    assert w[t, col(CEFS[f0])] == 0.0
    # the freed slot goes to the most negative z among eligible funds without a slot
    others = [j for j in range(7) if j not in held]
    cands = [j for j in others if sig["event"][t, j]]
    slots = [(j, SHOCK_ROW) for j in held if j != f0]
    if cands:
        newcomer = min(cands, key=lambda j: sig["z"][t, j])
        slots.append((newcomer, t))
        assert w[t, col(CEFS[newcomer])] == pytest.approx(0.07)
    hedge = -0.07 * sum(sig["beta"][s, j] for j, s in slots)
    assert w[t, col(ANCHOR)] == pytest.approx(hedge)


def test_late_listing():
    r = noisy_returns()
    listed = np.ones(r.shape, dtype=bool)
    c = col("NEA")
    listed[:300, c] = False
    r[:300, c] = 0.0
    data = make_view(r, listed=listed)
    sig = compute_signal(data, PARAMS)
    z = sig["z"][:, CEFS.index("NEA")]
    assert np.all(np.isnan(z[:554]))
    assert np.isfinite(z[554])
    w = target_weights(data, PARAMS)
    assert np.all(w[:554, c] == 0.0)


def test_liquidity_filter():
    r = noisy_returns()
    dv = np.full(r.shape, 5.0e6)
    c = col("NUV")
    dv[:, c] = 9.0e5
    data = make_view(r, dv=dv)
    sig = compute_signal(data, PARAMS)
    assert not sig["eligible"][:, CEFS.index("NUV")].any()
    assert np.all(target_weights(data, PARAMS)[:, c] == 0.0)


def test_missing_data_raises():
    r = noisy_returns()
    r[350, col("NUV")] = np.nan
    with pytest.raises(ValueError):
        target_weights(make_view(r), PARAMS)
    names = tuple(n for n in NAMES if n != "DSM")
    r2 = noisy_returns(names=names)
    with pytest.raises(Exception):
        target_weights(make_view(r2, names=names), PARAMS)


def test_book_bounds_and_params_matter():
    data = make_view(noisy_returns())
    w = target_weights(data, PARAMS)
    fcols = [col(n) for n in CEFS]
    m = col(ANCHOR)
    assert np.all(w[:, fcols] >= 0.0)
    assert np.all(w[:, m] <= 0.0)
    assert np.all((w[:, fcols] > 0).sum(axis=1) <= 5)
    assert np.all(np.abs(w).sum(axis=1) <= 5 * 0.07 * 2.75 + 1e-12)
    assert (w[:, fcols] > 0).any()
    for name, alt in (("formation_days", 3), ("formation_days", 10), ("z_entry", 1.5), ("z_entry", 2.5)):
        p = dict(PARAMS)
        p[name] = alt
        assert not np.array_equal(target_weights(data, p), w)


def test_point_in_time_truncation():
    r = noisy_returns()
    data = make_view(r)
    w = target_weights(data, PARAMS)
    for t in (300, 420, 555):
        rt = r[: t + 1]
        wt = target_weights(make_view(rt), PARAMS)
        assert np.array_equal(wt, w[: t + 1])


def test_diagnostic_events():
    r, tradable, dv = shock_setup()
    data = make_view(r, tradable=tradable, dv=dv)
    mask, side = events(data, PARAMS)
    assert side is None
    assert not mask[:, col(ANCHOR)].any()
    # all seven shocked funds are events on the shock row (no slot cap in the event study)
    for j in range(7):
        assert mask[SHOCK_ROW, col(CEFS[j])]
        assert not mask[SHOCK_ROW + 1: SHOCK_ROW + 5, col(CEFS[j])].any()
    assert not mask[:, [col(n) for n in CEFS[7:]]].any()
