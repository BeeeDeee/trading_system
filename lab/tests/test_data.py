import numpy as np
import pytest

from lab.framework import costs, data
from lab.framework.data import DataView


def view(dates, names, cls, seed=0):
    rng = np.random.default_rng(seed)
    T, N = len(dates), len(names)
    return DataView(np.asarray(dates, dtype="datetime64[D]"), tuple(names), (cls,) * N,
                    rng.normal(0, 0.01, (T, N)), rng.normal(0, 0.01, (T, N)), np.ones((T, N), bool),
                    np.ones((T, N), bool), np.zeros((T, N), bool), np.full((T, N), 10.0),
                    rng.uniform(1e6, 1e9, (T, N)))


def test_merge_equity_and_crypto_calendars_keeps_total_returns():
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-03-01"))
    etf = view(days[np.is_busday(days)], ["SPY"], "us_etf")
    btc = view(days, ["BTCUSDT"], "crypto_spot", seed=1)
    m = data.merge([etf, btc])
    assert len(m.dates) == len(days) and m.instruments == ("SPY", "BTCUSDT")
    weekend = ~np.is_busday(m.dates)
    assert not m.tradable[weekend, 0].any() and (m.ret_co[weekend, 0] == 0).all()
    tr = lambda v, j: np.prod((1 + v.ret_co[:, j]) * (1 + v.ret_oc[:, j]))  # noqa: E731
    assert tr(m, 0) == pytest.approx(tr(etf, 0)) and tr(m, 1) == pytest.approx(tr(btc, 0))


def test_slice_and_perturb_keep_the_past():
    v = data.synthetic()
    t = 1000
    p = data.perturb_after(v, t, np.random.default_rng(0))
    np.testing.assert_array_equal(p.ret_oc[: t + 1], v.ret_oc[: t + 1])
    np.testing.assert_array_equal(p.series["signal"][: t + 1], v.series["signal"][: t + 1])
    assert not np.array_equal(p.ret_oc[t + 1:], v.ret_oc[t + 1:])
    assert v.slice(t + 1).shape == (t + 1, v.shape[1])


def test_crypto_top_n_is_point_in_time():
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-06-01"))
    v = view(days, [f"C{i}" for i in range(12)], "crypto_spot")
    full = data.crypto_top_n(v, 5)
    for t in (40, 80, 140):
        np.testing.assert_array_equal(data.crypto_top_n(v.slice(t + 1), 5)[t], full[t])
    assert (full[40:].sum(axis=1) == 5).all()


def test_crypto_costs_use_volume_up_to_the_previous_day():
    days = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-04-01"))
    v = view(days, ["X"], "crypto_spot")
    base = costs.cost_rates(v)
    v.dollar_volume[50, 0] = 1e3            # a volume collapse on day 50 ...
    after = costs.cost_rates(v)
    assert after[50, 0] == base[50, 0]      # ... is not known when trading at day 50's open
    assert after[51:81, 0].max() >= base[51:81, 0].max()


def test_synthetic_edge_is_where_it_was_planted():
    v = data.synthetic()
    sig = v.series["signal"][:-1]
    corr = [np.corrcoef(sig, v.ret_oc[1:, v.col(a)])[0, 1] for a in ("S00", "S10")]
    assert corr[0] > 0.15 and abs(corr[1]) < 0.05


def test_one_day_spikes_are_dropped_and_real_moves_kept():
    import polars as pl
    from datetime import date, timedelta
    days = [date(2020, 1, 1) + timedelta(d) for d in range(6)]
    px = pl.DataFrame({"permaticker": [1] * 6 + [2] * 6, "date": days * 2,
                       "closeadj": [10.0, 10, 40, 10.2, 10, 10,      # spike on day 2 (vendor error)
                                    10.0, 10, 4, 4.1, 4, 4]})       # a real crash that stays
    out = data.drop_spikes(px)
    assert out.filter(pl.col("permaticker") == 1)["closeadj"].to_list() == [10, 10, 10.2, 10, 10]
    assert out.filter(pl.col("permaticker") == 2).height == 6
