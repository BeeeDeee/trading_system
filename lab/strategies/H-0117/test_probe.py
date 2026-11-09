"""Integration test on a widened synthetic market (3 x 31 columns, noisy dollar volume, all listed names members):
the plain synthetic market has identical dollar volume across stocks (volume z-score NaN) and too few names
for 5-name legs, so the model path and the book would never run there."""
import dataclasses
import numpy as np

from strategy import PARAMS, target_weights, _inputs, _score, _decision_days, _book


def _wide():
    from lab.framework.data import synthetic
    d = synthetic()
    rng = np.random.default_rng(5)
    k = 3

    def tile(a, noise=None):
        a = np.asarray(a)
        out = np.concatenate([a] * k, axis=1)
        if noise is not None:
            out = out + rng.normal(0, noise, out.shape) * np.isfinite(out)
        return out

    T, N = np.asarray(d.ret_co).shape
    dv = tile(d.dollar_volume) * rng.uniform(0.5, 2.0, (T, N * k))
    listed = tile(d.listed).astype(bool)
    return dataclasses.replace(
        d,
        instruments=tuple("X%d" % i for i in range(N * k)),
        asset_class=tuple(list(d.asset_class) * k),
        ret_co=tile(d.ret_co, 0.004), ret_oc=tile(d.ret_oc, 0.008),
        tradable=tile(d.tradable).astype(bool), listed=listed, delisting=tile(d.delisting),
        close=tile(d.close), dollar_volume=dv, universe=listed.copy(),
    )


def test_wide_market_book_and_events():
    data = _wide()
    u, rco, roc, r, dv = _inputs(data)
    score, elig = _score(data, PARAMS, u, rco, roc, r, dv)
    assert np.isfinite(score).any()
    assert not np.isfinite(score[~elig]).any()            # only eligible rows are scored
    first_score = int(np.argmax(np.isfinite(score).any(axis=1)))
    assert first_score > 63                                # nothing before features + first fit

    w = np.asarray(target_weights(data, PARAMS))
    ps, days = _decision_days(data.dates)
    dec = np.zeros(w.shape[0], dtype=bool)
    dec[days] = True
    assert np.isnan(w[~dec]).all()                         # held between weekly decisions
    assert not np.isnan(w[dec]).any()
    g = np.abs(w[dec]).sum(axis=1)
    assert (g <= 1.0).all()
    inv = days[g > 0]
    assert inv.size > 50 and inv[0] >= first_score       # flat before the first fit
    for d in inv:
        assert (w[d] > 0).sum() >= 5 and (w[d] < 0).sum() >= 5
        assert abs(np.abs(w[d]).sum() - 1.0) < 1e-6
        lg = w[d][w[d] > 0].sum()
        assert 0.3 - 1e-6 <= lg <= 0.7 + 1e-6
        assert u[d][w[d] != 0].all() and elig[d][w[d] != 0].all()

    # entry/exit params change the book (same scores)
    e = float(PARAMS["entry_frac"])
    x = float(PARAMS["exit_frac"])
    W1, ev, side = _book(score, elig, r, u, days, e, x)
    W2, _, _ = _book(score, elig, r, u, days, 0.2, x)
    W3, _, _ = _book(score, elig, r, u, days, e, 0.3)
    assert not np.allclose(W1, W2) and not np.allclose(W1, W3)
    # buffer lowers turnover relative to no buffer (exit = entry)
    W0, _, _ = _book(score, elig, r, u, days, e, e)
    to = lambda W: np.abs(np.diff(W[days], axis=0)).sum()
    assert to(W1) < to(W0)

    # event logic (copied into diagnostic.py): new entries only, on the leg they enter
    assert ev.sum() > 0
    assert set(np.unique(side[ev])) <= {1.0, -1.0}
    assert (np.sign(w[ev]) == side[ev]).all()
    for a, b in zip(days[:-1], days[1:]):
        cont = (np.sign(w[b]) == np.sign(w[a])) & (w[b] != 0)
        assert not ev[b][cont].any()
