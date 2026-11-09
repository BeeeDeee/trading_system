import dataclasses
import numpy as np
import pytest
from lab.framework.api import period_starts
from lab.framework.data import DataView, synthetic
from lab.framework.ml import cs_rank, forward_return, walk_forward
from strategy import HORIZON, PARAMS, RETRAIN, model_inputs, tcn_score, target_weights


def make_view(T=1100, N=60, seed=7):
    """Hand-made market with cross-sectional volume dispersion (the lab try market has none, K-0135)."""
    base = synthetic()
    rng = np.random.default_rng(seed)
    dates = np.asarray(base.dates)[:T]
    ret_co = rng.standard_normal((T, N)) * 0.005
    ret_oc = rng.standard_normal((T, N)) * 0.015
    listed = np.ones((T, N), dtype=bool)
    listed[:500, 55:] = False                       # late listings
    ret_co[~listed] = 0.0
    ret_oc[~listed] = 0.0
    dv = np.exp(17 + rng.standard_normal((T, N)))
    dv[~listed] = np.nan
    close = np.where(listed, 50.0, np.nan)
    uni = listed.copy()
    uni[900:, 0] = False                            # leaves the universe
    return DataView(dates=dates, instruments=tuple(f"S{i:02d}" for i in range(N)),
                    asset_class=tuple("us_equity" for _ in range(N)), ret_co=ret_co, ret_oc=ret_oc,
                    tradable=listed.copy(), listed=listed, delisting=np.zeros((T, N), dtype=bool),
                    close=close, dollar_volume=dv, universe=uni, extras={}, series={}, cash_ret=np.zeros(T))


def truncate(data, n):
    return dataclasses.replace(
        data, dates=data.dates[:n], ret_co=data.ret_co[:n], ret_oc=data.ret_oc[:n], tradable=data.tradable[:n],
        listed=data.listed[:n], delisting=data.delisting[:n], close=data.close[:n],
        dollar_volume=data.dollar_volume[:n], universe=data.universe[:n], cash_ret=data.cash_ret[:n])


@pytest.fixture(scope="module")
def run():
    data = make_view()
    return data, target_weights(data, PARAMS)


def test_book_on_handmade_market(run):
    data, w = run
    T = len(data.dates)
    dec = np.zeros(T, dtype=bool)
    dec[np.asarray(period_starts(data.dates, "W"))] = True
    g = np.nansum(np.abs(w), axis=1)
    assert (g <= 1.0 + 1e-9).all()
    assert np.isfinite(w[dec]).all()                 # every weekly decision row is decided
    act = np.nonzero(dec & (g > 0))[0]
    assert act.size > 20
    assert act[0] > 252                              # nothing before a model is fitted
    assert np.allclose(w[dec & ~(g > 0)], 0.0)       # flat before the first fit
    for t in act:
        assert np.sum(w[t][w[t] > 0]) == pytest.approx(0.5)
        assert np.sum(w[t][w[t] < 0]) == pytest.approx(-0.5)
        n_long = int((w[t] > 0).sum())
        assert n_long == int((w[t] < 0).sum()) and n_long >= 1
    # between decisions only NaN (keep) or 0 (exit of a name that left the universe)
    off = w[~dec]
    assert np.all(np.isnan(off) | (off == 0.0))
    # never a nonzero target on a non-member
    nz = np.isfinite(w) & (w != 0)
    assert not (nz & ~data.universe).any()
    # late listings never weighted before they have a full feature history
    assert not nz[:600, 55:].any()


def test_point_in_time_truncation(run):
    data, w = run
    cut = 900
    wt = target_weights(truncate(data, cut), PARAMS)
    np.testing.assert_array_equal(np.isnan(w[:cut]), np.isnan(wt))
    np.testing.assert_allclose(np.nan_to_num(w[:cut]), np.nan_to_num(wt), atol=1e-12)


def test_seq_len_changes_sequence_mask():
    data = make_view(T=300, N=30)
    _, _, s5 = model_inputs(data, 5)
    _, _, s20 = model_inputs(data, 20)
    assert s5.sum() > s20.sum()
    assert not (s20 & ~s5).any()


def test_ridge_control_scores_same_rows():
    data = make_view()
    X, vol, seq_ok = model_inputs(data, PARAMS["seq_len"])
    st = np.where(seq_ok, np.asarray(tcn_score(data, X, PARAMS["seq_len"]), dtype=float), np.nan)
    y = cs_rank(forward_return(data, HORIZON), data.universe)
    Xr = np.where(seq_ok[:, :, None], X, np.nan).astype(np.float32)
    sr = np.asarray(walk_forward(Xr, y, horizon=HORIZON, model="ridge", params={}, retrain_every=RETRAIN,
                                 universe=data.universe), dtype=float)
    sr = np.where(seq_ok, sr, np.nan)
    ft = np.isfinite(st)
    fr = np.isfinite(sr)
    assert ft.sum() > 0 and fr.sum() > 0
    assert (ft & fr).sum() >= 0.95 * max(ft.sum(), fr.sum())
