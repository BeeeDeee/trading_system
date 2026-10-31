import numpy as np
import pytest

from lab.framework.data import DataView
from strategy import target_weights, PARAMS

SUNDAY0 = 18266          # day number of a Sunday (1970-01-01 + 18266 days); row r is a Sunday iff r % 7 == 0


def build(perps, spots, t_len=400, start=SUNDAY0):
    """perps: list of (name, first_row); spots: list of (name, first_row). Universe = listed perps."""
    names = [p[0] for p in perps] + [s[0] for s in spots]
    firsts = [p[1] for p in perps] + [s[1] for s in spots]
    classes = ["crypto_perp"] * len(perps) + ["crypto_spot"] * len(spots)
    n = len(names)
    rows = np.arange(t_len)[:, None]
    listed = rows >= np.array(firsts)[None, :]
    close = np.where(listed, 100.0, np.nan)
    univ = listed & (np.array(classes) == "crypto_perp")[None, :]
    arrays = dict(
        dates=(np.arange(t_len) + start).astype("datetime64[D]"),
        ret_co=np.zeros((t_len, n)), ret_oc=np.zeros((t_len, n)),
        close=close, dollar_volume=np.where(listed, 1e6, np.nan),
        tradable=listed.copy(), listed=listed.copy(), delisting=np.zeros((t_len, n), dtype=bool),
        universe=univ,
    )
    return names, classes, arrays


def view(names, classes, arrays, upto=None):
    a = {k: (v[:upto] if upto is not None else v) for k, v in arrays.items()}
    return DataView(
        dates=a["dates"], instruments=tuple(names), asset_class=tuple(classes),
        ret_co=a["ret_co"], ret_oc=a["ret_oc"], tradable=a["tradable"], listed=a["listed"],
        delisting=a["delisting"], close=a["close"], dollar_volume=a["dollar_volume"],
        universe=a["universe"], extras={}, series={}, cash_ret=np.zeros(len(a["dates"])),
    )


# panel: perps start at row 200 of the view, so spot history before the panel is 200 days
PERPS = [("AAAUSDT.P", 200), ("BBBUSDT.P", 300), ("CCCUSDT.P", 300), ("DDDUSDT.P", 300),
         ("EEEUSDT.P", 200), ("FFFUSDT.P", 215)]
SPOTS = [("AAAUSDT", 0), ("BBBUSDT", 0), ("CCCUSDT", 150), ("FFFUSDT", 0)]
# BBB: perp launch row 300, spot age 300 days -> event; CCC: spot age 150 -> event only if min age <= 150
# DDD: no spot pair -> never; FFF: launch 15 days after panel start -> pre-existing, never; AAA, EEE pre-existing


def col(names, name):
    return names.index(name)


def test_weights_primary():
    names, classes, arr = build(PERPS, SPOTS, t_len=420)
    w = target_weights(view(names, classes, arr), PARAMS)
    assert w.shape == (420, len(names))
    sundays = np.arange(420) % 7 == 0
    assert np.isnan(w[~sundays]).all()          # no universe exits in this setup
    assert not np.isnan(w[sundays]).any()
    b = col(names, "BBBUSDT.P")
    # window ages 7..63 after launch row 300 -> Sundays 308 .. 357
    short_rows = [r for r in range(0, 420, 7) if w[r, b] < 0]
    assert short_rows == list(range(308, 364, 7))
    r = 308
    assert w[r, b] == pytest.approx(-0.10)                # k = 1: min(0.5, 0.10)
    hedge = [col(names, x) for x in ("AAAUSDT.P", "CCCUSDT.P", "DDDUSDT.P", "EEEUSDT.P", "FFFUSDT.P")]
    assert np.allclose(w[r, hedge], 0.10 / 5)
    assert np.allclose(w[r, len(PERPS):], 0.0)            # spot columns never traded
    assert w[r].sum() == pytest.approx(0.0)
    # no short leg -> flat book (no hedge)
    assert np.allclose(w[301], 0.0)                       # Sunday after launch, age 1: not yet
    assert np.allclose(w[294], 0.0)
    assert np.allclose(w[364], 0.0)                       # window ended (age 64), never re-entered
    assert (np.abs(w[sundays]).sum(axis=1) <= 1.0 + 1e-12).all()


def test_params_change_output():
    names, classes, arr = build(PERPS, SPOTS, t_len=420)
    dv = view(names, classes, arr)
    b, c = col(names, "BBBUSDT.P"), col(names, "CCCUSDT.P")
    w28 = target_weights(dv, {"hold_days": 28, "min_spot_age_days": 180})
    assert [r for r in range(0, 420, 7) if w28[r, b] < 0] == list(range(308, 336, 7))   # ages 8..35
    w90 = target_weights(dv, {"hold_days": 56, "min_spot_age_days": 90})
    assert np.allclose(w90[308, [b, c]], -0.10)               # CCC (spot age 150) now an event, k = 2
    w365 = target_weights(dv, {"hold_days": 56, "min_spot_age_days": 180})
    assert w365[308, c] > 0                                   # CCC in the hedge at the primary value


def test_preexisting_and_unmatched_never_events():
    names, classes, arr = build(PERPS, SPOTS, t_len=420)
    w = target_weights(view(names, classes, arr), {"hold_days": 84, "min_spot_age_days": 10})
    for x in ("FFFUSDT.P", "DDDUSDT.P", "AAAUSDT.P", "EEEUSDT.P"):
        assert not (w[:, col(names, x)] < 0).any()


def test_cap_and_split():
    perps = [("H%dUSDT.P" % i, 0) for i in range(3)] + [("L%dUSDT.P" % i, 100) for i in range(8)]
    spots = [("L%dUSDT" % i, 0) for i in range(8)]
    names, classes, arr = build(perps, spots, t_len=200, start=SUNDAY0)
    w = target_weights(view(names, classes, arr), {"hold_days": 56, "min_spot_age_days": 30})
    r = 112                                              # age 12
    shorts = w[r, 3:11]
    assert np.allclose(shorts, -0.5 / 8)                  # k = 8: 0.0625 < 0.10
    assert np.allclose(w[r, 0:3], 0.5 / 3)
    assert np.abs(w[r]).sum() == pytest.approx(1.0)


def test_missing_bar_skipped_and_universe_exit():
    names, classes, arr = build(PERPS, SPOTS, t_len=420)
    b, e = col(names, "BBBUSDT.P"), col(names, "EEEUSDT.P")
    arr["close"][315, e] = np.nan                         # missing bar on a Sunday: skipped that week
    arr["universe"][320:330, b] = False                   # leaves the universe on a Tuesday, back later
    w = target_weights(view(names, classes, arr), PARAMS)
    assert w[315, e] == 0.0
    assert np.allclose(w[315, [col(names, x) for x in ("AAAUSDT.P", "CCCUSDT.P", "DDDUSDT.P", "FFFUSDT.P")]],
                       0.10 / 4)
    assert w[320, b] == 0.0 and np.isnan(np.delete(w[320], b)).all()     # closed on the first non-member row
    assert np.allclose(w[322], 0.0)                       # Sunday: BBB not a member -> k = 0 -> flat book
    assert w[336, b] == pytest.approx(-0.10)              # rejoined inside its window: remaining part


def test_point_in_time_truncation():
    names, classes, arr = build(PERPS, SPOTS, t_len=420)
    full = target_weights(view(names, classes, arr), PARAMS)
    for cut in (250, 300, 301, 307, 308, 309, 330, 357, 360):
        part = target_weights(view(names, classes, arr, upto=cut), PARAMS)
        np.testing.assert_array_equal(np.isnan(part), np.isnan(full[:cut]))
        np.testing.assert_allclose(np.nan_to_num(part), np.nan_to_num(full[:cut]))


def test_fail_loudly_without_spot():
    names, classes, arr = build(PERPS, [], t_len=420)
    with pytest.raises(ValueError):
        target_weights(view(names, classes, arr), PARAMS)


def test_censored_spot_history():
    # view starts with the perp panel: spot pairs trading on row 0 have an unknown (>= row 0) listing date
    perps = [("AAAUSDT.P", 0), ("BBBUSDT.P", 100), ("CCCUSDT.P", 250), ("DDDUSDT.P", 100), ("EEEUSDT.P", 0)]
    spots = [("AAAUSDT", 0), ("BBBUSDT", 0), ("CCCUSDT", 0)]
    names, classes, arr = build(perps, spots, t_len=330)
    dv = view(names, classes, arr)
    b, c = col(names, "BBBUSDT.P"), col(names, "CCCUSDT.P")
    w = target_weights(dv, PARAMS)                         # no crash
    # BBB: launch row 100, spot age only known to be >= 100 < 180 -> unproven -> never an event
    assert not (w[:, b] < 0).any()
    # CCC: launch row 250, spot age >= 250 >= 180 -> proven -> short on Sundays with age 7..63
    assert [r for r in range(0, 330, 7) if w[r, c] < 0] == list(range(259, 314, 7))
    # with min age 90 BBB's lower bound 100 suffices; with 365 CCC's lower bound 250 does not
    w90 = target_weights(dv, {"hold_days": 56, "min_spot_age_days": 90})
    assert [r for r in range(0, 330, 7) if w90[r, b] < 0] == list(range(112, 164, 7))
    w365 = target_weights(dv, {"hold_days": 56, "min_spot_age_days": 365})
    assert not (w365 < 0).any()
    assert np.allclose(w365[~np.isnan(w365)], 0.0)
