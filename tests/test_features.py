"""Featury na fixním CSV proti naivní referenční implementaci + look-ahead test (SPEC §5.1)."""

import math

import numpy as np
import pytest

from arena.features import BarArrays, BookArrays, compute

IDX = [400, 777, 1000, 1499]


def ref_z(hist, cur):
    hist = [h for h in hist if math.isfinite(h)]
    mu = sum(hist) / len(hist)
    sd = math.sqrt(sum((h - mu) ** 2 for h in hist) / len(hist))
    return (cur - mu) / sd


def ref_features(b: BarArrays, i: int, btc: BarArrays | None = None):
    c, h, l, qv, tbqv = (list(map(float, a)) for a in (b.c, b.h, b.l, b.qv, b.tbqv))
    r = [math.nan] + [math.log(c[j] / c[j - 1]) for j in range(1, len(c))]
    out = {"ret1": r[i], "c": c[i]}
    hist = r[i - 288:i]
    mu = sum(hist) / 288
    out["sigma288"] = math.sqrt(sum((x - mu) ** 2 for x in hist) / 288)
    # Wilder ATR přes posledních 288 TR, seed = průměr prvních 14
    trs = [max(h[j] - l[j], abs(h[j] - c[j - 1]), abs(l[j] - c[j - 1])) for j in range(i - 287, i + 1)]
    atr = sum(trs[:14]) / 14
    for tr in trs[14:]:
        atr = (atr * 13 + tr) / 14
    out["atr14_pct"] = atr / c[i] * 100
    lv = [math.log(q) for q in qv]
    out["lvz288"] = ref_z(lv[i - 288:i], lv[i])
    for n in (12, 24, 48):
        out[f"don_hi_{n}"] = max(h[i - n:i])
        out[f"don_lo_{n}"] = min(l[i - n:i])
    cvd = [math.nan] * len(c)
    for j in range(11, len(c)):
        cvd[j] = sum(2 * tbqv[k] - qv[k] for k in range(j - 11, j + 1)) / sum(qv[k] for k in range(j - 11, j + 1))
    out["cvd12"] = cvd[i]
    out["cvd12_z"] = ref_z(cvd[i - 288:i], cvd[i])
    if i + 1 >= 1152:
        e = sum(c[i - 1151:i - 575]) / 576
        for x in c[i - 575:i + 1]:
            e = 2 / 577 * x + (1 - 2 / 577) * e
        out["ema_576"] = e
    if btc is not None:
        cb = list(map(float, btc.c))
        rb = [math.nan] + [math.log(cb[j] / cb[j - 1]) for j in range(1, len(cb))]
        out["ret1_btc"] = rb[i]
        hb = rb[i - 288:i]
        mb = sum(hb) / 288
        out["sigma288_btc"] = math.sqrt(sum((x - mb) ** 2 for x in hb) / 288)
        ya = r[i - 288:i]
        my = sum(ya) / 288
        out["beta288"] = sum((x - mb) * (y - my) for x, y in zip(hb, ya)) / sum((x - mb) ** 2 for x in hb)
    return out


@pytest.mark.parametrize("i", IDX)
def test_features_match_reference(btc, eth, i):
    got = compute(eth.upto(i), btc=btc.upto(i))
    ref = ref_features(eth, i, btc)
    for k, v in ref.items():
        assert got[k] == pytest.approx(v, rel=1e-9, abs=1e-12), k


def test_warmup_gives_nan_not_garbage(btc):
    f = compute(btc.upto(100))
    assert math.isnan(f["sigma288"]) and math.isnan(f["lvz288"]) and math.isnan(f["ema_576"])
    assert math.isfinite(f["don_hi_24"]) and math.isfinite(f["atr14_pct"])


def _book_for(bars: BarArrays, seed=0):
    rng = np.random.default_rng(seed)
    n = len(bars)
    ok = (rng.random(n) > 0.05).astype(np.int8)
    return BookArrays({p: rng.normal(0, 0.2, n) for p in (0.2, 1.0, 2.0)}, ok)


def _perturb_after(b: BarArrays, t: int, seed=1) -> BarArrays:
    rng = np.random.default_rng(seed)
    arrs = []
    for a in (b.o, b.h, b.l, b.c, b.qv, b.tbqv):
        a = a.copy()
        a[t + 1:] *= rng.uniform(0.5, 1.5, len(a) - t - 1)
        arrs.append(a)
    return BarArrays(b.open_time.copy(), *arrs)


@pytest.mark.parametrize("t", [600, 1152, 1300])
def test_no_look_ahead(btc, eth, t):
    """Změna libovolného baru s indexem > t nezmění featury ani signály v čase ≤ t."""
    from arena.config import load_config
    from arena.strategies.base import BarContext
    from arena.strategies.registry import build_strategies

    book = _book_for(eth)
    eth2, btc2 = _perturb_after(eth, t), _perturb_after(btc, t, seed=2)
    book2 = BookArrays({p: np.r_[a[: t + 1], a[t + 1:] * -3] for p, a in book.obi.items()},
                       np.r_[book.ok[: t + 1], 1 - book.ok[t + 1:]])
    strategies = build_strategies(load_config())
    for i in (t - 5, t):
        f1 = compute(eth.upto(i), book.upto(i), btc.upto(i))
        f2 = compute(eth2.upto(i), book2.upto(i), btc2.upto(i))
        assert f1.keys() == f2.keys()
        for k in f1:
            assert f1[k] == f2[k] or (math.isnan(f1[k]) and math.isnan(f2[k])), k
        for s in strategies.values():
            if "ETHUSDT" not in s.symbols:
                continue
            ctx1 = BarContext("ETHUSDT", int(eth.open_time[i]), f1, f"{s.variant_id}.ETHUSDT")
            ctx2 = BarContext("ETHUSDT", int(eth.open_time[i]), f2, f"{s.variant_id}.ETHUSDT")
            assert s.on_bar(ctx1) == s.on_bar(ctx2), s.variant_id


def test_obi_z_needs_200_valid_and_current_ok(btc):
    n = len(btc)
    x = np.linspace(-0.1, 0.1, n)
    ok = np.ones(n, dtype=np.int8)
    ok[n - 120:n - 1] = 0       # v okně 288 jen 169 platných
    f = compute(btc, BookArrays({1.0: x}, ok))
    assert math.isnan(f["obi1_z"]) and f["book_ok"] == 1.0
    ok[:] = 1
    ok[-1] = 0
    f = compute(btc, BookArrays({1.0: x}, ok))
    assert f["book_ok"] == 0.0 and math.isnan(f["obi1"]) and math.isnan(f["obi1_z"])
