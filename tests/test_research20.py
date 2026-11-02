"""Research 20: forward data construction."""

import numpy as np
import pytest

from qlab.research20 import data as D


def fake_chart(rows, divs=(), splits=()):
    """rows: [(date 'YYYY-MM-DD', o, h, l, c)]; UTC midnight timestamps, gmtoffset 0."""
    ts = [int(np.datetime64(d, "s").astype(int)) for d, *_ in rows]
    q = {k: [r[i + 1] for r in rows] for i, k in enumerate(("open", "high", "low", "close"))}
    ev = {"dividends": {str(t): {"date": t, "amount": a} for t, a in
                        ((int(np.datetime64(d, "s").astype(int)), a) for d, a in divs)},
          "splits": {str(t): {"date": t, "numerator": n, "denominator": m} for t, (n, m) in
                     ((int(np.datetime64(d, "s").astype(int)), x) for d, x in splits)}}
    return {"chart": {"result": [{"meta": {"gmtoffset": 0}, "timestamp": ts, "indicators": {"quote": [q]}, "events": ev}]}}


def test_tr_with_dividend_matches_sharadar_convention():
    rows = [("2026-01-02", 100, 101, 99, 100), ("2026-01-05", 99.5, 101, 99, 100.5), ("2026-01-06", 100.5, 102, 100, 101)]
    p = D.parse_yahoo(fake_chart(rows, divs=[("2026-01-05", 1.0)]))
    b = D.tr_bars(p)
    assert b["ret_co"][0] == 0.0
    np.testing.assert_allclose(b["tr"][1], (100.5 + 1.0) / 100 - 1)        # (close + D) / prev close - 1
    np.testing.assert_allclose(b["ret_oc"][1], 100.5 / 99.5 - 1)            # dividend not in the intraday part
    np.testing.assert_allclose(b["tr"][2], 101 / 100.5 - 1)                 # no dividend: plain price return
    np.testing.assert_allclose((1 + b["ret_co"]) * (1 + b["ret_oc"]) - 1, b["tr"])


def test_parse_drops_incomplete_rows_and_reads_splits():
    j = fake_chart([("2026-01-02", 10, 11, 9, 10), ("2026-01-05", 5, 6, 4, 5.5)], splits=[("2026-01-05", (1, 2))])
    j["chart"]["result"][0]["indicators"]["quote"][0]["open"][0] = None
    p = D.parse_yahoo(j)
    assert len(p["dates"]) == 1 and p["split"][0] == 0.5


def test_gate_stats_counts_missing_dates_and_basis_points():
    d1 = np.array(["2026-01-02", "2026-01-05", "2026-01-06"], dtype="datetime64[D]")
    d2 = d1[:2]
    s = D.gate_stats(d1, np.array([0.01, 0.02, 0.03]), d2, np.array([0.01, 0.0201]))
    assert s["n_common"] == 2 and s["n_only_yahoo"] == 1 and s["n_only_sharadar"] == 0
    assert s["max_abs_bps"] == pytest.approx(1.0)
