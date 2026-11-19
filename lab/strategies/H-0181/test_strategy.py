import numpy as np
import pytest

from strategy import target_weights, PARAMS


class FakeData:
    def __init__(self, T, N, start_day=10966):   # 10966 = 2000-01-10, no month start for 20 days
        self.dates = np.arange(start_day, start_day + T).astype("datetime64[D]")
        self.instruments = tuple("E%d" % i for i in range(N))
        self.universe = np.ones((T, N), dtype=bool)
        self.tradable = np.ones((T, N), dtype=bool)
        self.listed = np.ones((T, N), dtype=bool)
        self.delisting = np.zeros((T, N), dtype=bool)
        self.extras = {"sp500_member": np.zeros((T, N), dtype=np.float32)}

    def col(self, name):
        return self.instruments.index(name)


def sp(d):
    return d.extras["sp500_member"]


def test_params():
    assert PARAMS == {"hold_days": 252}


def test_deletion_long_with_hedge_and_expiry():
    d = FakeData(15, 5)
    sp(d)[:7, 0] = 1.0              # deleted after the close of row 7
    w = target_weights(d, {"hold_days": 3})
    assert np.all(w[0] == 0)        # first row: flat decision
    assert np.all(np.isnan(w[1:7]))
    assert w[7, 0] == pytest.approx(0.05)
    assert np.allclose(w[7, 1:], -0.05 / 4)
    assert np.nansum(w[7]) == pytest.approx(0.0)
    assert np.all(np.isnan(w[8:10]))
    assert np.all(w[10] == 0)       # closed at the open after 3 sessions (decision row 7 + 3)
    assert np.all(np.isnan(w[11:]))


def test_addition_short_and_paired_slots_no_hedge():
    d = FakeData(15, 4)
    sp(d)[:7, 0] = 1.0
    sp(d)[7:, 1] = 1.0
    w = target_weights(d, {"hold_days": 5})
    assert w[7, 0] == pytest.approx(0.05)
    assert w[7, 1] == pytest.approx(-0.05)
    assert np.all(w[7, 2:] == 0)
    assert np.all(w[12] == 0)


def test_events_require_universe_tradable_and_skip_first_rows():
    d = FakeData(15, 4)
    sp(d)[:7, 0] = 1.0                 # deletion at row 7, but not a member
    d.universe[7, 0] = False
    sp(d)[:8, 1] = 1.0                 # deletion at row 8, not tradable
    d.tradable[8, 1] = False
    sp(d)[:3, 2] = 1.0                 # deletion at row 3: inside the first 5 sessions
    w = target_weights(d, {"hold_days": 5})
    assert np.all(np.isnan(w[1:]))


def test_forced_exit_on_universe_exit_and_untradable():
    d = FakeData(15, 6)
    sp(d)[:7, 0] = 1.0
    sp(d)[:7, 1] = 1.0
    d.universe[9:, 0] = False          # leaves LIQ-500 at row 9
    d.tradable[8, 1] = False           # no price on the entry day: entry skipped
    w = target_weights(d, {"hold_days": 100})
    assert w[7, 0] == pytest.approx(0.05) and w[7, 1] == pytest.approx(0.05)
    assert np.allclose(w[7, 2:], -0.1 / 4)
    assert w[8, 1] == 0 and w[8, 0] == pytest.approx(0.05)
    assert np.allclose(w[8, 2:], -0.05 / 4)
    assert np.all(w[9] == 0)
    assert np.all(np.isnan(w[10:]))


def test_delisting_closes_slot():
    d = FakeData(15, 4)
    sp(d)[:7, 0] = 1.0
    d.delisting[10, 0] = True
    w = target_weights(d, {"hold_days": 100})
    assert w[7, 0] == pytest.approx(0.05)
    assert np.all(w[10] == 0)


def test_opposite_event_replaces_slot():
    d = FakeData(20, 4)
    sp(d)[:7, 0] = 1.0                 # deletion row 7
    sp(d)[9:, 0] = 1.0                 # re-addition row 9
    w = target_weights(d, {"hold_days": 4})
    assert w[7, 0] == pytest.approx(0.05)
    assert w[9, 0] == pytest.approx(-0.05)
    assert np.allclose(w[9, 1:], 0.05 / 3)
    assert np.all(np.isnan(w[10:13]))
    assert np.all(w[13] == 0)          # hold counted from the new event


def test_side_cap_scales_pro_rata():
    d = FakeData(12, 30)
    sp(d)[:7, :10] = 1.0               # 10 deletions on row 7
    w = target_weights(d, {"hold_days": 100})
    assert np.allclose(w[7, :10], 0.045)
    assert np.allclose(w[7, 10:], -0.45 / 20)
    assert np.sum(np.abs(w[7])) <= 1.0 + 1e-12
    assert np.sum(w[7]) == pytest.approx(0.0)


def test_month_start_reweights_hedge():
    d = FakeData(40, 5, start_day=10966)    # 2000-01-10 .. 2000-02-18; 2000-02-01 is row 22
    sp(d)[:7, 0] = 1.0
    d.universe[15:, 4] = False             # basket member leaves; no slot change
    w = target_weights(d, {"hold_days": 100})
    assert np.all(np.isnan(w[15]))
    assert w[22, 0] == pytest.approx(0.05)
    assert np.allclose(w[22, 1:4], -0.05 / 3)
    assert w[22, 4] == 0


def test_hold_days_changes_output():
    d = FakeData(15, 4)
    sp(d)[:7, 0] = 1.0
    a = target_weights(d, {"hold_days": 3})
    b = target_weights(d, {"hold_days": 5})
    assert not np.array_equal(np.nan_to_num(a, nan=9), np.nan_to_num(b, nan=9))


def test_point_in_time_truncation():
    rng = np.random.default_rng(0)
    T, N = 60, 12
    d = FakeData(T, N)
    flags = (rng.random((T, N)) < 0.03).cumsum(axis=0) % 2
    sp(d)[:] = flags
    d.universe[:] = rng.random((T, N)) < 0.9
    full = target_weights(d, {"hold_days": 10})
    for cut in (10, 25, 40):
        e = FakeData(cut, N)
        sp(e)[:] = sp(d)[:cut]
        e.universe[:] = d.universe[:cut]
        part = target_weights(e, {"hold_days": 10})
        assert np.array_equal(np.nan_to_num(part, nan=9), np.nan_to_num(full[:cut], nan=9))
    g = np.nan_to_num(full)
    assert np.all(np.abs(g).sum(axis=1) <= 1.0 + 1e-9)
    assert np.allclose(g.sum(axis=1), 0.0)

