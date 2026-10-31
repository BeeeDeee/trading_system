import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS, events

FUNDS = ["MXF", "MXE", "KF", "TWN", "IFN", "IIF", "CHN", "TDF", "GCH", "JOF", "JEQ", "SWZ", "GF",
         "IF", "TTF", "SGF", "ISL", "TKF", "EMF", "BZF", "SNF", "IRL"]
ANCHORS = ["EWW", "EWY", "EWT", "EPI", "FXI", "EWJ", "EWL", "EWG", "ECH", "IDX", "THD", "EWS", "EWM",
           "EIS", "TUR", "EEM", "EWZ", "EWP", "EIRL"]
ALL = FUNDS + ANCHORS
T = 220
T0 = 150  # even row: the alternating noise adds +0.001 to the 5-session sum


def make(names=ALL, shocks=None, dv=1e6, edit=None):
    """Funds alternate +-0.1 % per day, anchors are flat; shocks = {(name, row): ret_oc}."""
    n = len(names)
    dates = np.arange(10957, 10957 + T).astype("datetime64[D]")
    ret_oc = np.zeros((T, n))
    for j, nm in enumerate(names):
        if nm in FUNDS:
            ret_oc[:, j] = 0.001 * np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    for (nm, row), x in (shocks or {}).items():
        ret_oc[row, names.index(nm)] += x
    f = dict(
        ret_co=np.zeros((T, n)), ret_oc=ret_oc,
        tradable=np.ones((T, n), bool), listed=np.ones((T, n), bool), delisting=np.zeros((T, n), bool),
        close=np.full((T, n), 20.0), dollar_volume=np.full((T, n), float(dv)),
    )
    if edit:
        edit(f, names)
    return DataView(dates=dates, instruments=tuple(names), asset_class=tuple("us_etf" for _ in names),
                    universe=None, extras={}, series={}, cash_ret=np.zeros(T), **f)


def col(names, nm):
    return names.index(nm)


def test_single_event_book():
    v = make(shocks={("MXF", T0): -0.05})
    w = target_weights(v, PARAMS)
    h = PARAMS["hold_days"]
    assert np.all(w[:T0] == 0)
    assert np.allclose(w[T0:T0 + h, col(ALL, "MXF")], 0.25)
    assert np.allclose(w[T0:T0 + h, col(ALL, "EWW")], -0.25)
    assert np.all(w[T0 + h:] == 0)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_hold_days_param():
    v = make(shocks={("MXF", T0): -0.05})
    for h in (3, 10):
        w = target_weights(v, {"hold_days": h, "z_entry": 2.0})
        held = np.nonzero(w[:, col(ALL, "MXF")])[0]
        assert list(held) == list(range(T0, T0 + h))


def test_z_entry_param():
    # row T0: S ~= log(1-0.006) + 0.001 ~= -0.0050, sigma ~= 0.001 -> z ~= -2.24
    # row T0+1: S ~= log(1-0.006) - 0.001 ~= -0.0070 -> z ~= -3.1
    v = make(shocks={("KF", T0): -0.006})
    w2 = target_weights(v, {"hold_days": 5, "z_entry": 2.0})
    w25 = target_weights(v, {"hold_days": 5, "z_entry": 2.5})
    w35 = target_weights(v, {"hold_days": 5, "z_entry": 3.5})
    assert list(np.nonzero(w2[:, col(ALL, "KF")])[0]) == list(range(T0, T0 + 5))
    assert w2[T0, col(ALL, "EWY")] == -0.25
    assert list(np.nonzero(w25[:, col(ALL, "KF")])[0]) == list(range(T0 + 1, T0 + 6))
    assert np.all(w35 == 0)


def test_slot_cap_ranking_shared_anchor_no_queue():
    v = make(shocks={("MXF", T0): -0.06, ("MXE", T0): -0.08, ("KF", T0): -0.04})
    w = target_weights(v, PARAMS)
    assert np.allclose(w[T0:T0 + 5, col(ALL, "MXF")], 0.25)
    assert np.allclose(w[T0:T0 + 5, col(ALL, "MXE")], 0.25)
    assert np.allclose(w[T0:T0 + 5, col(ALL, "EWW")], -0.5)   # shared anchor adds
    assert np.all(w[:, col(ALL, "KF")] == 0)                  # skipped, not queued
    assert np.all(w[:, col(ALL, "EWY")] == 0)
    assert np.abs(w).sum(axis=1).max() <= 1.0 + 1e-12


def test_ranking_most_negative_first():
    v = make(shocks={("MXF", T0): -0.03, ("KF", T0): -0.08, ("TWN", T0): -0.05})
    w = target_weights(v, PARAMS)
    assert w[T0, col(ALL, "KF")] == 0.25 and w[T0, col(ALL, "TWN")] == 0.25
    assert np.all(w[:, col(ALL, "MXF")] == 0)


def test_no_new_event_on_held_fund():
    v = make(shocks={("MXF", T0): -0.05, ("MXF", T0 + 7): -0.05})
    w10 = target_weights(v, {"hold_days": 10, "z_entry": 2.0})
    assert list(np.nonzero(w10[:, col(ALL, "MXF")])[0]) == list(range(T0, T0 + 10))
    w3 = target_weights(v, {"hold_days": 3, "z_entry": 2.0})
    assert list(np.nonzero(w3[:, col(ALL, "MXF")])[0]) == list(range(T0, T0 + 3)) + list(range(T0 + 7, T0 + 10))


def test_dollar_volume_floor():
    v = make(shocks={("MXF", T0): -0.05}, dv=5e4)
    assert np.all(target_weights(v, PARAMS) == 0)


def test_missing_bar_in_formation_blocks_event():
    def edit(f, names):
        f["tradable"][T0 - 2, names.index("EWW")] = False
        f["close"][T0 - 2, names.index("EWW")] = np.nan
    v = make(shocks={("MXF", T0): -0.05}, edit=edit)
    assert np.all(target_weights(v, PARAMS) == 0)


def test_warmup_and_late_listing():
    v = make(shocks={("MXF", 60): -0.05})            # fewer than 100 sigma sessions
    assert np.all(target_weights(v, PARAMS) == 0)

    def edit(f, names):
        j = names.index("MXF")
        f["listed"][:110, j] = False
        f["tradable"][:110, j] = False
        f["close"][:110, j] = np.nan
        f["dollar_volume"][:110, j] = np.nan
    v = make(shocks={("MXF", T0): -0.05}, edit=edit)   # only ~36 sessions before t-5
    assert np.all(target_weights(v, PARAMS) == 0)


def test_missing_ticker_raises_and_no_data_fund_never_eligible():
    names = [n for n in ALL if n != "GCH"]
    with pytest.raises(Exception):
        target_weights(make(names=names), PARAMS)
    with pytest.raises(Exception):
        events(make(names=names), PARAMS)

    def edit(f, nm):                                   # GCH present but without any price
        j = nm.index("GCH")
        f["listed"][:, j] = False
        f["tradable"][:, j] = False
        f["close"][:, j] = np.nan
        f["dollar_volume"][:, j] = np.nan
    v = make(shocks={("MXF", T0): -0.05, ("GCH", T0): -0.09}, edit=edit)
    w = target_weights(v, PARAMS)
    assert w[T0, col(ALL, "MXF")] == 0.25
    assert np.all(w[:, col(ALL, "GCH")] == 0)


def test_removed_pairs_and_peers_never_trade():
    assert "CH" not in ALL and "MF" not in ALL
    v = make(shocks={("ECH", T0): 0.08, ("EWM", T0): 0.08})
    w = target_weights(v, PARAMS)
    assert np.all(w == 0)
    mask, _ = events(v, PARAMS)
    assert not mask[:, col(ALL, "ECH")].any() and not mask[:, col(ALL, "EWM")].any()


def test_point_in_time_truncation():
    v = make(shocks={("MXF", T0): -0.05, ("KF", T0 + 3): -0.07})
    w = target_weights(v, PARAMS)
    cut = T0 + 1
    f = dict(ret_co=v.ret_co[:cut], ret_oc=v.ret_oc[:cut], tradable=v.tradable[:cut], listed=v.listed[:cut],
             delisting=v.delisting[:cut], close=v.close[:cut], dollar_volume=v.dollar_volume[:cut])
    vt = DataView(dates=v.dates[:cut], instruments=v.instruments, asset_class=v.asset_class, universe=None,
                  extras={}, series={}, cash_ret=np.zeros(cut), **f)
    assert np.array_equal(target_weights(vt, PARAMS), w[:cut])


def test_diagnostic_marks_both_legs():
    v = make(shocks={("MXF", T0): -0.05, ("MXE", T0): -0.06, ("KF", T0): -0.07})
    mask, side = events(v, PARAMS)
    assert mask.sum() == 5                            # 3 funds + EWW + EWY (no slot cap)
    assert mask[T0, col(ALL, "MXF")] and side[T0, col(ALL, "MXF")] == 1.0
    assert mask[T0, col(ALL, "EWW")] and side[T0, col(ALL, "EWW")] == -1.0
    assert mask[T0, col(ALL, "KF")] and side[T0, col(ALL, "EWY")] == -1.0
