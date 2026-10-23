"""`obi` z bookDepth: náš pipeline (parse → bar_book → features) = algoritmus research/obi_prior/analyze.py."""

import gzip
import math

import numpy as np
import pytest
from numpy.lib.stride_tricks import sliding_window_view as swv

from arena.data.book_validate import parse_bookdepth_csv
from arena.data.recorder import aggregate_bar_book
from arena.features import BarArrays, book_arrays, compute
from tests.conftest import DATA

DAYS = ("2026-10-20", "2026-10-21")
D0 = 1_792_454_400_000          # 2026-10-20 00:00 UTC


def research_obi(bd: np.ndarray, ot: np.ndarray, band: float) -> np.ndarray:
    """Doslovně podle research/obi_prior/analyze.py: features() – část book."""
    n = len(ot)
    close_t = ot + 300_000
    bid = bd[bd[:, 1] == -band]
    ask = bd[bd[:, 1] == band]
    tb, ta = bid[:, 0], ask[:, 0]
    common, ib, ia = np.intersect1d(tb, ta, return_indices=True)
    imb = (bid[ib, 2] - ask[ia, 2]) / (bid[ib, 2] + ask[ia, 2])
    idx = np.searchsorted(ot, common, side="right") - 1
    ok = (idx >= 0) & (common < close_t[np.clip(idx, 0, n - 1)])
    s = np.bincount(idx[ok], weights=imb[ok], minlength=n)
    cnt = np.bincount(idx[ok], minlength=n)
    return np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)


def research_rz(x, w=288):
    out = np.full_like(x, np.nan)
    win = swv(x[:-1], w)
    mu = np.nanmean(win, axis=1)
    sd = np.nanstd(win, axis=1)
    out[w:] = (x[w:] - mu) / np.where(sd > 0, sd, np.nan)
    return out


@pytest.fixture(scope="module")
def data():
    texts = [gzip.open(DATA / f"BTCUSDT-bookDepth-{d}.csv.gz", "rt").read() for d in DAYS]
    # research formát: (ts, pct, notional)
    import datetime as dt
    rows = []
    for t in texts:
        for ln in t.splitlines()[1:]:
            p = ln.split(",")
            ts = dt.datetime.strptime(p[0], "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() * 1000
            rows.append((ts, float(p[1]), float(p[3])))
    bd = np.array(rows)
    parsed = {}
    for t in texts:
        parsed.update(parse_bookdepth_csv(t))
    ot = np.arange(D0, D0 + 2 * 86_400_000, 300_000, dtype=np.int64)
    return bd, parsed, ot


def ours(parsed, ot, pcts=(0.2, 1.0, 2.0)):
    flat = [{"ts": ts, "pct": p, "bid_notional": b, "ask_notional": a}
            for ts, bands in parsed.items() for p, (b, a) in bands.items()]
    flat.sort(key=lambda r: r["ts"])
    times = np.array([r["ts"] for r in flat])
    bb = []
    for t in ot:
        lo, hi = np.searchsorted(times, t), np.searchsorted(times, t + 300_000)
        for r in aggregate_bar_book("BTCUSDT", int(t), flat[lo:hi], pcts, source="bookDepth"):
            bb.append(dict(zip(("symbol", "open_time", "pct", "obi", "n_samples", "book_ok", "source"), r)))
    return book_arrays(ot, bb, pcts), bb


@pytest.mark.parametrize("band", [0.2, 1.0, 2.0])
def test_bar_obi_identical_to_research(data, band):
    bd, parsed, ot = data
    ref = research_obi(bd, ot, band)
    ba, _ = ours(parsed, ot)
    mine = ba.obi[band]
    assert np.isfinite(ref).sum() > 570
    np.testing.assert_allclose(mine, ref, rtol=1e-12, equal_nan=True)


def test_book_ok_coverage(data):
    _, parsed, ot = data
    ba, bb = ours(parsed, ot)
    assert ba.ok.mean() > 0.99      # bookDepth má 10 vzorků na bar
    assert all(r["n_samples"] in (9, 10, 11) for r in bb if r["pct"] == 1.0)


def test_obi_z_matches_research_when_all_bars_ok(data):
    bd, parsed, ot = data
    ba, _ = ours(parsed, ot)
    ref_z = research_rz(research_obi(bd, ot, 1.0))
    dummy = BarArrays(ot, *(np.ones(len(ot)) for _ in range(6)))
    checked = 0
    for i in range(300, len(ot)):
        if not ba.ok[i - 288:i + 1].all():
            continue
        z = compute(dummy.upto(i), ba.upto(i))["obi1_z"]
        assert z == pytest.approx(ref_z[i], rel=1e-9), i
        checked += 1
    assert checked > 200
