"""tools/prereg_trend_eval.py on synthetic repos with known answers (docs/PREREGISTRATION_TREND.md)."""
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "engine"))
import prereg_trend_eval as E  # noqa: E402
from cpb import canon  # noqa: E402


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def build_repo(price, start="2026-12-01", end="2028-11-30", hist_from="2026-01-01", fee=0.0, miss_on=None):
    """price(date) -> BTC close. History from hist_from; daily decision runs start..end.
    The primary follows the SMA200 signal of asof (= run date - 1); equity moves with BTC while held."""
    repo = tempfile.mkdtemp()
    closes, rows, d = {}, [], hist_from
    while d <= end:
        c = price(d)
        closes[d] = c
        rows.append([d, c, c, c, c, 1.0, 1e9])
        d = canon.add_days(d, 1)
    write(os.path.join(repo, "data", "history", "BTC.json"), rows)
    eq_t, eq_b, held, d, prev_c = 10000.0, 10000.0, False, start, None
    while d <= end:
        asof = canon.add_days(d, -1)
        c = closes[asof]
        if prev_c is not None:
            eq_b *= c / prev_c
            if held:
                eq_t *= c / prev_c
        sig = E.sma_signal(closes, asof)
        if sig != held:
            eq_t *= 1 - fee
            held = sig
        shown = (not held) if d == miss_on else held
        pos = {"BTC": {"qty": 1.0}} if shown else {}
        mark = lambda e, h: {"equity": e, "exposure": 1.0 if h else 0.0, "positions": {"BTC": {"qty": 1.0}} if h else {},  # noqa: E731
                             "fees": 0.0, "slippage": 0.0}
        close = {E.PRIMARY: {s: mark(eq_t, held) for s in ("base", "stress")},
                 E.BENCH: {s: mark(eq_b, True) for s in ("base", "stress")}}
        led = {"date": d, "close": close, "after": {E.PRIMARY: {"base": {"positions": pos}}}}
        rd = os.path.join(repo, "runs", d)
        write(os.path.join(rd, "run.json"), {"date": d, "status": "ok"})
        write(os.path.join(rd, "inputs.json"), {"date": d, "asof": asof, "mode": "decision"})
        write(os.path.join(rd, "ledger.json"), led)
        prev_c = c
        d = canon.add_days(d, 1)
    return repo


def day_index(d):
    return canon.days_between("2026-01-01", d)


def boom_bust(d):
    """Up 0.2 %/day until mid 2027, a 60 % crash over 4 months, then a slow recovery."""
    i = day_index(d)
    if i < 560:
        return 50000 * math.exp(0.002 * i)
    if i < 680:
        return 50000 * math.exp(0.002 * 560) * math.exp(-0.0076 * (i - 560))
    return 50000 * math.exp(0.002 * 560) * math.exp(-0.0076 * 120) * math.exp(0.001 * (i - 680))


def gap_crash(d):
    """Steady rise, then a 45 % drop in one day that never recovers: no daily filter can step aside in time."""
    base = 50000 * math.exp(0.001 * day_index(d))
    return base * (0.55 if d >= "2027-09-01" else 1.0)


def steady(d):
    return 50000 * math.exp(0.0008 * day_index(d))


class TestTrendEval(unittest.TestCase):
    def tearDown(self):
        for r in getattr(self, "repos", []):
            shutil.rmtree(r, ignore_errors=True)

    def repo(self, *a, **k):
        r = build_repo(*a, **k)
        self.repos = getattr(self, "repos", []) + [r]
        return r

    def test_sma_signal_needs_full_history(self):
        closes = {canon.add_days("2027-01-01", -k): 100.0 + (k == 0) for k in range(200)}
        self.assertTrue(E.sma_signal(closes, "2027-01-01"))
        del closes[canon.add_days("2027-01-01", -150)]
        self.assertFalse(E.sma_signal(closes, "2027-01-01"))        # a gap -> cash, never filled in

    def test_filter_avoiding_crash_passes(self):
        res = E.evaluate(self.repo(boom_bust), today="2028-12-15")
        self.assertEqual(res["status"], "splněno", res)
        self.assertTrue(res["bear_market"])
        self.assertFalse(res["extended"])
        self.assertLess(res["base"]["dd_ratio"], 0.6)
        self.assertGreaterEqual(res["signal_changes"], 1)
        self.assertEqual(res["execution_misses"], [])

    def test_one_day_crash_is_rejected(self):
        res = E.evaluate(self.repo(gap_crash), today="2028-12-15")
        self.assertTrue(res["bear_market"])
        self.assertFalse(res["criteria"]["1_dd_base"])
        self.assertEqual(res["status"], "zamítnuto")

    def test_missed_execution_fails_criterion_4(self):
        res = E.evaluate(self.repo(boom_bust, miss_on="2027-03-01"), today="2028-12-15")
        self.assertFalse(res["criteria"]["4_execution"])
        self.assertEqual(res["execution_misses"][0][0], "2027-03-01")
        self.assertEqual(res["status"], "zamítnuto")

    def test_no_bear_market_extends_window_once(self):
        res = E.evaluate(self.repo(steady, end="2029-11-30"), today="2029-12-15")
        self.assertTrue(res["extended"])
        self.assertEqual(res["end"], E.EXT_END)
        self.assertFalse(res["bear_market"])
        self.assertTrue(any("bez medvědího trhu" in n for n in res["notes"]))

    def test_interim_is_never_a_verdict_and_does_not_extend(self):
        res = E.evaluate(self.repo(steady, end="2027-06-30"), today="2027-07-01")
        self.assertEqual(res["status"], "průběžně")
        self.assertFalse(res["extended"])
        self.assertFalse(res["valid"])

    def test_start_outside_window_is_invalid(self):
        res = E.evaluate(self.repo(steady, start="2027-01-05", end="2027-03-01"), today="2027-03-02")
        self.assertEqual(res["status"], "neplatné")
        self.assertEqual(res["start"], "2027-01-05")

    def test_no_variant_yet(self):
        repo = tempfile.mkdtemp()
        self.repos = [repo]
        self.assertEqual(E.evaluate(repo, today="2026-10-03")["status"], "nezačalo")


if __name__ == "__main__":
    unittest.main()
