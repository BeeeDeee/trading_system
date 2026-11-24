import numpy as np
import pytest

from lab.framework.data import DataView, synthetic
from strategy import PARAMS, target_weights

_SYN = synthetic()


def make_view(T=80, N=8, dv=1e6):
    names = tuple(_SYN.instruments[:N]) if N <= len(_SYN.instruments) else tuple(
        _SYN.instruments[0] + str(i) for i in range(N))
    return dict(
        dates=_SYN.dates[:T].copy(),
        instruments=names,
        asset_class=tuple("us_equity" for _ in range(N)),
        ret_co=np.zeros((T, N)),
        ret_oc=np.zeros((T, N)),
        tradable=np.ones((T, N), dtype=bool),
        listed=np.ones((T, N), dtype=bool),
        delisting=np.zeros((T, N), dtype=bool),
        close=np.full((T, N), 10.0),
        dollar_volume=np.full((T, N), dv),
        universe=np.ones((T, N), dtype=bool),
        extras={"ea_reaction_day": np.zeros((T, N), dtype=np.float32),
                "liq_rank": np.tile(np.arange(1, N + 1, dtype=np.float32), (T, 1))},
        series={},
        cash_ret=np.zeros(T),
    )


def view(d):
    return DataView(**d)


def put_event(d, t, j, gap=-0.05, oc=0.0, vol=5e6):
    d["ret_co"][t, j] = gap
    d["ret_oc"][t, j] = oc
    d["dollar_volume"][t, j] = vol


def test_params_match_card():
    assert PARAMS == {"gap_pct": 3, "hold_days": 5}


def test_basic_slot():
    d = make_view()
    put_event(d, 70, 0)
    w = target_weights(view(d), PARAMS)
    assert w.shape == (80, 8)
    assert np.all(w[:70] == 0)
    assert np.allclose(w[70:75, 0], 0.10)
    assert np.allclose(w[70:75, 1:], -0.10 / 7)
    assert np.all(w[75:] == 0)
    assert np.allclose(w[70:75].sum(axis=1), 0.0)


def test_intraday_continuation_kills_event():
    d = make_view()
    put_event(d, 70, 0, oc=-0.001)
    assert np.all(target_weights(view(d), PARAMS) == 0)


def test_volume_threshold():
    d = make_view()
    put_event(d, 70, 0, vol=3.99e6)
    assert np.all(target_weights(view(d), PARAMS) == 0)
    d = make_view()
    put_event(d, 70, 0, vol=4.0e6)
    assert target_weights(view(d), PARAMS)[70, 0] > 0


def test_volume_median_excludes_today_and_needs_50_values():
    # window t-63..t-1 = rows 7..69; 14 NaN -> 49 valid -> no event; 13 NaN -> 50 valid -> event
    d = make_view()
    put_event(d, 70, 0)
    d["dollar_volume"][7:21, 0] = np.nan
    assert np.all(target_weights(view(d), PARAMS) == 0)
    d = make_view()
    put_event(d, 70, 0)
    d["dollar_volume"][7:20, 0] = np.nan
    assert target_weights(view(d), PARAMS)[70, 0] > 0
    # row 6 is outside the window: NaNs there do not matter
    d = make_view()
    put_event(d, 70, 0)
    d["dollar_volume"][0:7, 0] = np.nan
    assert target_weights(view(d), PARAMS)[70, 0] > 0
    # a median raised by a high-volume history kills the event
    d = make_view()
    put_event(d, 70, 0)
    d["dollar_volume"][30:69, 0] = 2e6
    assert np.all(target_weights(view(d), PARAMS) == 0)


def test_first_rows_without_history():
    d = make_view()
    put_event(d, 40, 0)      # only 40 prior rows < 50
    assert np.all(target_weights(view(d), PARAMS) == 0)
    d = make_view(T=10)
    put_event(d, 5, 0)
    assert np.all(target_weights(view(d), PARAMS) == 0)


@pytest.mark.parametrize("row,value,expect", [
    (70, 1.0, False), (69, 1.0, False), (68, 1.0, False), (67, 1.0, True),
    (70, np.nan, False), (69, np.nan, True), (71, 1.0, True)])
def test_earnings_exclusion(row, value, expect):
    d = make_view()
    put_event(d, 70, 0)
    d["extras"]["ea_reaction_day"][row, 0] = value
    w = target_weights(view(d), PARAMS)
    assert (w[70, 0] > 0) == expect


def test_market_adjusted_gap():
    d = make_view()
    d["ret_co"][70, :] = -0.05
    d["dollar_volume"][70, :] = 5e6
    assert np.all(target_weights(view(d), PARAMS) == 0)
    # median over members only: non-members with large positive gaps do not move it
    d = make_view()
    put_event(d, 70, 0, gap=-0.035)
    d["universe"][:, 7] = False
    d["ret_co"][70, 7] = 0.5
    w = target_weights(view(d), PARAMS)
    assert w[70, 0] > 0
    assert w[70, 7] == 0


def test_gap_param():
    d = make_view()
    put_event(d, 70, 0, gap=-0.035)
    assert target_weights(view(d), {"gap_pct": 3, "hold_days": 5})[70, 0] > 0
    assert np.all(target_weights(view(d), {"gap_pct": 4, "hold_days": 5}) == 0)
    d = make_view()
    put_event(d, 70, 0, gap=-0.025)
    assert np.all(target_weights(view(d), {"gap_pct": 3, "hold_days": 5}) == 0)
    assert target_weights(view(d), {"gap_pct": 2, "hold_days": 5})[70, 0] > 0


def test_hold_param():
    d = make_view()
    put_event(d, 60, 0)
    for h in (3, 5, 7):
        w = target_weights(view(d), {"gap_pct": 3, "hold_days": h})
        assert np.allclose(w[60:60 + h, 0], 0.10)
        assert np.all(w[60 + h:, :] == 0)


def test_no_reentry_while_slot_open():
    d = make_view()
    put_event(d, 60, 0)
    put_event(d, 62, 0)      # inside the open slot: ignored
    put_event(d, 65, 0)      # 65 - 60 = 5 = hold: still within the previous 5 sessions
    put_event(d, 66, 0)      # new slot
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[60:65, 0], 0.10)
    assert w[65, 0] == 0
    assert np.allclose(w[66:71, 0], 0.10)
    assert np.all(w[71:] == 0)


def test_early_close_on_universe_exit_and_halt():
    d = make_view()
    put_event(d, 60, 0)
    d["universe"][62:, 0] = False
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[60:62, 0], 0.10)
    assert np.all(w[62:] == 0)
    d = make_view()
    put_event(d, 60, 0)
    d["tradable"][63, 0] = False
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[60:63, 0], 0.10)
    assert np.all(w[63:] == 0)


def test_hedge_member_leaves_basket():
    d = make_view()
    put_event(d, 60, 0)
    d["universe"][62:, 3] = False
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[60:62, 3], -0.10 / 7)
    assert np.all(w[62:, 3] == 0)
    assert np.allclose(w[62:65, 1], -0.10 / 7)


def test_non_member_or_untraded_no_event():
    d = make_view()
    put_event(d, 70, 0)
    d["universe"][70, 0] = False
    assert np.all(target_weights(view(d), PARAMS)[:, 0] == 0)
    d = make_view()
    put_event(d, 70, 0)
    d["tradable"][70, 0] = False
    assert np.all(target_weights(view(d), PARAMS) == 0)
    d = make_view()
    put_event(d, 70, 0)
    d["ret_oc"][70, 0] = np.nan
    assert np.all(target_weights(view(d), PARAMS) == 0)


def test_hedge_is_100_most_liquid_without_event():
    N = 130
    d = make_view(N=N)
    d["extras"]["liq_rank"][:] = np.arange(N, 0, -1, dtype=np.float32)   # last column most liquid
    put_event(d, 70, N - 1)      # the most liquid stock has the event: not in its own hedge
    w = target_weights(view(d), PARAMS)
    assert np.isclose(w[70, N - 1], 0.10)
    short = np.nonzero(w[70] < 0)[0]
    assert short.size == 100
    assert set(short.tolist()) == set(range(N - 101, N - 1))
    assert np.allclose(w[70, short], -0.001)


def test_slot_cap_scales_all_slots():
    N = 30
    d = make_view(N=N)
    for j in range(7):
        put_event(d, 70, j)
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[70, :7], 0.10 * 5 / 7)
    assert np.abs(w[70]).sum() <= 1.0 + 1e-12
    assert np.isclose(np.abs(w[70]).sum(), 1.0)
    # exactly five slots: full size, gross 1
    d = make_view(N=N)
    for j in range(5):
        put_event(d, 70, j)
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[70, :5], 0.10)
    assert np.abs(w[70]).sum() <= 1.0


def test_staggered_slots_scale_back():
    N = 30
    d = make_view(N=N)
    for j in range(5):
        put_event(d, 60, j)
    put_event(d, 62, 5)
    w = target_weights(view(d), PARAMS)
    assert np.allclose(w[60:62, 0], 0.10)
    # j0 is long in its own slot and in the hedge basket of the slot opened on 62 (29 names)
    assert np.allclose(w[62:65, 0], (0.10 - 0.10 / 29) * 5 / 6)
    # j5 is in the hedge basket (25 names) of each of the five slots opened on 60
    assert np.allclose(w[62:65, 5], (0.10 - 5 * 0.10 / 25) * 5 / 6)
    assert np.allclose(w[65:67, 5], 0.10)
    assert np.all(np.abs(w).sum(axis=1) <= 1.0)


def test_missing_inputs_raise():
    d = make_view()
    d["universe"] = None
    with pytest.raises(Exception):
        target_weights(view(d), PARAMS)
    d = make_view()
    del d["extras"]["ea_reaction_day"]
    with pytest.raises(KeyError):
        target_weights(view(d), PARAMS)


def test_truncation_point_in_time():
    d = make_view(T=80, N=10)
    rng = np.random.default_rng(0)
    d["ret_co"] = rng.normal(0, 0.03, (80, 10))
    d["ret_oc"] = rng.normal(0, 0.01, (80, 10))
    d["dollar_volume"] = np.exp(rng.normal(14, 0.8, (80, 10)))
    full = target_weights(view(d), PARAMS)
    for cut in (55, 63, 70, 77):
        dd = make_view(T=cut, N=10)
        for k in ("ret_co", "ret_oc", "dollar_volume"):
            dd[k] = d[k][:cut].copy()
        part = target_weights(view(dd), PARAMS)
        assert np.array_equal(part, full[:cut])
