import numpy as np
import pytest
from dataclasses import dataclass

from strategy import target_weights, PARAMS, compute_events, month_end_flags, build_weights


def _bdates(n):
    # weekdays starting Thu 1970-01-01 (epoch 0), no literals
    base = np.zeros(1, dtype="datetime64[D]")[0]
    return np.busday_offset(base, np.arange(n), roll="forward")


@dataclass
class FakeView:
    dates: np.ndarray
    instruments: tuple
    universe: np.ndarray
    tradable: np.ndarray
    extras: dict


# ---------------------------------------------------------------- calendar
def test_month_end_weekday_rule():
    d = _bdates(70)
    f = month_end_flags(d)
    # Jan 1970: Thu 1st .. Fri 30th = rows 0..21; Feb: Mon 2nd .. Fri 27th = rows 22..41
    assert f[21] and f[41]
    assert f.sum() == 3  # Jan, Feb, Mar (Mar 31 1970 is Tue = row 63)
    assert f[63]


def test_month_end_holiday_fallback():
    d = _bdates(30)
    d = np.delete(d, 21)  # Fri Jan 30 is a "holiday": last session is Thu Jan 29 (row 20)
    f = month_end_flags(d)
    assert not f[20]
    assert f[21]  # first session of February decides instead
    assert f[:21].sum() == 0


def test_month_end_point_in_time():
    d = _bdates(70)
    full = month_end_flags(d)
    for k in range(1, 70):
        assert np.array_equal(month_end_flags(d[:k]), full[:k])


# ---------------------------------------------------------------- events
def _event_inputs():
    T, N = 520, 4
    rnd = np.full((T, N), 10.0)
    assets = np.full((T, N), 100.0)
    # stock 0: +20 % R&D, flat assets at row 255 -> event; a further filing at 258 is suppressed by (e)
    rnd[255:, 0] = 12.0
    assets[258:, 0] = 101.0
    rnd[507:, 0] = 15.0      # 507 - 255 = 252 -> still inside the 252-session block, suppressed
    assets[508:, 0] = 102.0  # 508 - 255 = 253 -> fires
    # stock 1: +8 % R&D -> event only if growth_min <= 0.08
    rnd[255:, 1] = 10.8
    # stock 2: R&D +20 % but assets +20 % -> intensity growth 0 -> never
    rnd[255:, 2] = 12.0
    assets[255:, 2] = 120.0
    # stock 3: no R&D history a year earlier (NaN) -> never
    rnd[:200, 3] = np.nan
    rnd[255:, 3] = 20.0
    return rnd, assets


def test_events_rules():
    rnd, assets = _event_inputs()
    elig = np.ones(rnd.shape, dtype=bool)
    ev = compute_events(rnd, assets, elig, 0.1)
    assert list(np.flatnonzero(ev[:, 0])) == [255, 508]
    assert ev[:, 1].sum() == 0
    assert ev[:, 2].sum() == 0
    assert ev[:, 3].sum() == 0


def test_events_growth_min_param():
    rnd, assets = _event_inputs()
    elig = np.ones(rnd.shape, dtype=bool)
    ev = compute_events(rnd, assets, elig, 0.05)
    assert list(np.flatnonzero(ev[:, 1])) == [255]
    ev2 = compute_events(rnd, assets, elig, 0.25)
    # growth 20 % at 255 is too small; at 507 R&D 15 vs 12 a year earlier = +25 % fires
    assert list(np.flatnonzero(ev2[:, 0])) == [507]


def test_events_low_intensity_and_nan_to_number():
    T = 300
    rnd = np.full((T, 2), 1.0)
    assets = np.full((T, 2), 100.0)
    rnd[260:, 0] = 4.0         # +300 % but intensity 4 % < 5 % -> no event
    rnd[:270, 1] = np.nan      # NaN -> number counts as a filing, but R1 is NaN -> no event
    ev = compute_events(rnd, assets, np.ones((T, 2), bool), 0.1)
    assert ev.sum() == 0
    rnd2 = np.full((T, 1), np.nan)
    assets2 = np.full((T, 1), np.nan)
    rnd2[:10] = 10.0
    assets2[:10] = 100.0       # stale -> NaN, then a new filing reappears (NaN -> number)
    rnd2[270:] = 13.0
    assets2[270:] = 100.0
    ev2 = compute_events(rnd2, assets2, np.ones((T, 1), bool), 0.1)
    assert ev2.sum() == 0      # R1 at row 18 is NaN (stale) -> never fires, nothing filled


def test_events_need_membership_at_filing():
    rnd, assets = _event_inputs()
    elig = np.ones(rnd.shape, dtype=bool)
    elig[255, 0] = False
    ev = compute_events(rnd, assets, elig, 0.1)
    # 255 not eligible -> no event there; 258 is a new filing (assets change) and qualifies
    assert list(np.flatnonzero(ev[:, 0])) == [258]


def test_events_point_in_time():
    rnd, assets = _event_inputs()
    elig = np.ones(rnd.shape, dtype=bool)
    full = compute_events(rnd, assets, elig, 0.1)
    for k in (100, 256, 300, 508, 509):
        part = compute_events(rnd[:k], assets[:k], elig[:k], 0.1)
        assert np.array_equal(part, full[:k])


# ---------------------------------------------------------------- book
def test_build_weights_known_rows():
    T, N = 10, 3
    ev = np.zeros((T, N), bool)
    ev[1, 0] = True
    ev[6, 1] = True
    member = np.ones((T, N), bool)
    member[6, 0] = False
    trad = np.ones((T, N), bool)
    dec = np.zeros(T, bool)
    dec[[4, 9]] = True
    w = build_weights(ev, member, trad, dec, 5)
    assert np.allclose(w[4], [0.05, 0.0, 0.0])
    assert np.allclose(w[9], [0.0, 0.05, 0.0])
    assert np.isnan(w[5]).all()
    assert w[6, 0] == 0.0 and np.isnan(w[6, 1]) and np.isnan(w[6, 2])
    assert np.isnan(w[:4]).all()
    # longer hold keeps stock 0 at row 9
    w2 = build_weights(ev, member, trad, dec, 9)
    assert np.allclose(w2[9], [0.05, 0.05, 0.0])


def test_build_weights_untradable_at_month_end_excluded():
    T, N = 5, 2
    ev = np.zeros((T, N), bool)
    ev[0, :] = True
    member = np.ones((T, N), bool)
    trad = np.ones((T, N), bool)
    trad[4, 1] = False
    dec = np.zeros(T, bool)
    dec[4] = True
    w = build_weights(ev, member, trad, dec, 21)
    assert np.allclose(w[4], [0.05, 0.0])


def test_gross_capped_when_many_active():
    T, N = 3, 25
    ev = np.zeros((T, N), bool)
    ev[0] = True
    dec = np.array([False, False, True])
    w = build_weights(ev, np.ones((T, N), bool), np.ones((T, N), bool), dec, 21)
    assert np.allclose(w[2], 1 / 25)
    assert abs(w[2].sum() - 1.0) < 1e-12
    assert (w[2] >= 0).all()


# ---------------------------------------------------------------- end to end
def _view(T=600):
    rnd, assets = _event_inputs()
    pad = T - rnd.shape[0]
    rnd = np.vstack([rnd, np.repeat(rnd[-1:], pad, 0)])
    assets = np.vstack([assets, np.repeat(assets[-1:], pad, 0)])
    N = rnd.shape[1]
    return FakeView(
        dates=_bdates(T),
        instruments=tuple("E%d" % i for i in range(N)),
        universe=np.ones((T, N), bool),
        tradable=np.ones((T, N), bool),
        extras={"sf1_art_rnd": rnd.astype(np.float32), "sf1_art_assets": assets.astype(np.float32)},
    )


def test_target_weights_end_to_end_and_params():
    v = _view()
    w = target_weights(v, PARAMS)
    assert w.shape == (600, 4)
    dec = ~np.isnan(w).all(axis=1)
    assert np.array_equal(dec, month_end_flags(v.dates))
    fin = np.nan_to_num(w)
    assert (fin >= 0).all() and (fin.sum(axis=1) <= 1 + 1e-12).all()
    # first month-end after row 255 holds stock 0 at 5 %
    first = np.flatnonzero(dec & (np.arange(600) >= 255))[0]
    assert np.allclose(w[first], [0.05, 0.0, 0.0, 0.0])
    w6 = target_weights(v, {"growth_min": 0.1, "hold_months": 1})
    assert not np.array_equal(np.nan_to_num(w6), fin)
    w05 = target_weights(v, {"growth_min": 0.05, "hold_months": 12})
    assert np.allclose(w05[first], [0.05, 0.05, 0.0, 0.0])


def test_missing_inputs_raise():
    v = _view()
    v.extras = {"sf1_art_rnd": v.extras["sf1_art_rnd"]}
    with pytest.raises(KeyError):
        target_weights(v, PARAMS)
