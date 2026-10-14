"""Tests of the server runner. Run:  cd server && python3 -m unittest discover -s tests -q
No network, no LLM: Yahoo/Nasdaq are replaced by fakes, scores are injected."""
import datetime as dt
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from evening import cal, llm, prices, report, runner  # noqa: E402
from evening.store import Store, read_json  # noqa: E402

spec = importlib.util.spec_from_file_location("simulate", os.path.join(runner.REPO, "tools", "simulate.py"))
simulate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(simulate)
CFG = read_json(runner.CONFIG)
U = CFG["universe"]
D = dt.date


class Calendar(unittest.TestCase):
    def test_2026_holidays(self):
        h = {str(k) for k in cal.holidays(2026)}
        self.assertEqual(h, {"2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19", "2026-07-03",
                             "2026-09-07", "2026-11-26", "2026-12-25"})

    def test_early_close_and_weekend(self):
        self.assertTrue(cal.is_early_close(D(2026, 11, 27)))
        self.assertTrue(cal.is_early_close(D(2026, 12, 24)))
        self.assertFalse(cal.is_trading_day(D(2026, 10, 10)))
        self.assertEqual(cal.prev_trading_day(D(2026, 9, 8)), D(2026, 9, 4))
        self.assertEqual([str(x) for x in cal.trading_days_between(D(2026, 10, 8), D(2026, 10, 13))], ["2026-10-09", "2026-10-12", "2026-10-13"])


class Prompt(unittest.TestCase):
    def test_steps_4_to_6_verbatim(self):
        text = open(runner.PROMPT).read()
        p = llm.build_prompt(text, "HEADER")
        body = p.split("\n\n", 1)[1]
        self.assertTrue(body.startswith("## 4. Research"))
        self.assertIn("## 6. Decisions of the", body)
        self.assertNotIn("## 7.", body)
        self.assertIn(body.strip(), text)               # a literal slice of the pinned prompt


class Pins(unittest.TestCase):
    def test_pinned_hashes_match(self):
        code = runner.pinned_code()
        self.assertEqual(code["source"], "server")
        self.assertEqual(len(code["engine_sha256"]), 64)


class Prices(unittest.TestCase):
    def test_day_prices_leaves_out_missing(self):
        rows = {"SPY": {"2026-09-22": [1, 2, 0.5, 1.5]}, "QQQ": {"2026-09-21": [1, 1, 1, 1]}}
        t = prices.day_prices(rows, ["SPY", "QQQ"], "2026-09-22")
        self.assertEqual(t["close"], {"SPY": 1.5})


class FakeWorld(unittest.TestCase):
    """A synthetic market on real trading days (2026-09-08 .. 09-21 stored, runs for 09-22 and later)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        sim = simulate.run(CFG, days=10, seed=5, start=D(2026, 9, 8))
        st = Store(self.tmp)
        for d in sim["days"]:
            st.put("days", d["date"], d)
        for t in sim["trades"]:
            st.put("trades", f"{t['variant']}-{t['ticker']}-{t['entry_date']}", t)
        for s in sim["scores"]:
            st.put("scores", s["date"], s)
        for o in sim["ohlc"]:
            st.put("prices", o["date"], o)
        for r in sim["runs"]:
            st.put("runs", r["date"], r)
        st.put_history(sim["history"], sim["days"][-1]["date"])
        self.sim, self.st = sim, st
        self.last_close = sim["ohlc"][-1]["close"]
        self.scores = os.path.join(self.tmp, "scores_in.json")
        sc = {"date": "x", "regime": {"label": "Test", "summary": "t"}, "scores": {t: {"trend": 0, "mr": 0, "news": 0, "conviction": 3, "event": False, "note": "n"} for t in U}}
        json.dump(sc, open(self.scores, "w"))
        self.rows = {}
        self.patches = []
        for name, fn in (("fetch_all", lambda u, r="6mo": (self.rows, [])),
                         ("nasdaq_rows", self.fake_nasdaq), ("yahoo_splits", lambda tk, rng="1mo": {})):
            self.patches.append((name, getattr(prices, name)))
            setattr(prices, name, fn)
        self.addCleanup(self.restore)

    def restore(self):
        for name, fn in self.patches:
            setattr(prices, name, fn)

    def fake_nasdaq(self, tk, days=12):
        return {d: v for d, v in self.rows.get(tk, {}).items()}

    def market(self, dates, bad=()):
        px = dict(self.last_close)
        for d in dates:
            for t in U:
                if t in bad:
                    continue
                px[t] = round(px[t] * 1.004, 2)
                self.rows.setdefault(t, {})[d] = [px[t], round(px[t] * 1.002, 2), round(px[t] * 0.998, 2), px[t]]

    def run_day(self, date, **kw):
        return runner.run(self.tmp, date=date, force=True, wait_s=0, scores_file=self.scores, **kw)

    def test_normal_day(self):
        self.market(["2026-09-22"])
        rec = self.run_day("2026-09-22")
        self.assertIn(rec["status"], ("ok", "warning"), rec)
        day = self.st.get("days", "2026-09-22")
        self.assertEqual(day["code"]["source"], "server")
        self.assertEqual(self.st.get("runs", "2026-09-22")["spy_crosscheck"]["diff_pct"], 0.0)
        self.assertEqual(self.st.get("prices", "2026-09-22")["date"], "2026-09-22")
        self.assertEqual(self.run_day("2026-09-22")["status"], "exists")          # idempotent

    def test_catchup(self):
        self.market(["2026-09-22", "2026-09-23", "2026-09-24"])
        rec = self.run_day("2026-09-24")
        self.assertIn(rec["status"], ("ok", "warning"), rec)
        self.assertEqual(rec["catchup_days"], ["2026-09-22", "2026-09-23"])
        self.assertEqual(self.st.get("runs", "2026-09-22")["status"], "catchup")
        self.assertEqual(self.st.get("days", "2026-09-23")["regime"]["label"], "Doplněno zpětně")
        self.assertIsNotNone(self.st.get("days", "2026-09-24"))

    def test_closed_day(self):
        rec = self.run_day("2026-09-26")                                           # Saturday
        self.assertEqual(rec["status"], "closed")
        self.assertIsNone(self.st.get("days", "2026-09-26"))

    def test_missing_data_fails_and_writes_nothing_else(self):
        self.market(["2026-09-22"], bad=U[:8])
        rec = self.run_day("2026-09-22")
        self.assertEqual(rec["status"], "failed", rec)
        self.assertIsNone(self.st.get("days", "2026-09-22"))
        self.assertEqual(self.st.get("runs", "2026-09-22")["status"], "failed")

    def test_crosscheck_disagreement_fails(self):
        self.market(["2026-09-22"])
        real = self.fake_nasdaq
        prices.nasdaq_rows = lambda tk, days=12: {d: [v[0], v[1], v[2], v[3] * 1.01] for d, v in real(tk).items()}
        rec = self.run_day("2026-09-22")
        self.assertEqual(rec["status"], "failed")
        self.assertIn("Křížová kontrola", rec["message"])

    def test_nasdaq_lag_saves_day_and_confirms_later(self):
        self.market(["2026-09-22", "2026-09-23"])
        full = self.fake_nasdaq
        prices.nasdaq_rows = lambda tk, days=12: {d: v for d, v in full(tk).items() if d < "2026-09-22"}   # Nasdaq lags
        rec = self.run_day("2026-09-22")
        self.assertEqual(rec["status"], "ok", rec)
        self.assertIsNotNone(self.st.get("days", "2026-09-22"))
        self.assertTrue(self.st.get("runs", "2026-09-22")["spy_crosscheck"]["pending"])
        prices.nasdaq_rows = lambda tk, days=12: {d: v for d, v in full(tk).items() if d < "2026-09-23"}
        rec = self.run_day("2026-09-23")
        c = self.st.get("runs", "2026-09-22")["spy_crosscheck"]
        self.assertEqual((c["pending"], c["diff_pct"]), (False, 0.0), c)
        self.assertNotIn("Zpětná", rec["message"])

    def test_late_nasdaq_disagreement_is_reported(self):
        self.market(["2026-09-22", "2026-09-23"])
        full = self.fake_nasdaq
        prices.nasdaq_rows = lambda tk, days=12: {d: v for d, v in full(tk).items() if d < "2026-09-22"}
        self.run_day("2026-09-22")
        prices.nasdaq_rows = lambda tk, days=12: {d: ([v[0], v[1], v[2], v[3] * 1.01] if d == "2026-09-22" else v) for d, v in full(tk).items() if d < "2026-09-23"}
        rec = self.run_day("2026-09-23")
        self.assertEqual(rec["status"], "warning", rec)
        self.assertIn("Zpětná kontrola 2026-09-22", rec["message"])

    def test_dry_run_leaves_real_data(self):
        self.market(["2026-09-22"])
        self.run_day("2026-09-22", dry_run=True)
        self.assertIsNone(self.st.get("days", "2026-09-22"))
        self.assertIsNotNone(Store(self.tmp + ".dry").get("days", "2026-09-22"))
        shutil.rmtree(self.tmp + ".dry", ignore_errors=True)

    def test_notification_and_report(self):
        self.market(["2026-09-22"])
        self.run_day("2026-09-22")
        txt = runner.notification(self.tmp, "2026-09-22")
        self.assertTrue(txt.startswith(("OK – 2026-09-22", "UPOZORNĚNÍ – 2026-09-22")), txt)
        web = os.path.join(self.tmp, "web")
        os.makedirs(web)
        rel = report.publish(self.tmp, web)
        html = open(os.path.join(rel, "index.html"), encoding="utf-8").read()
        self.assertNotIn("fonts.googleapis", html)
        self.assertNotIn("window.claude", html)
        self.assertIn("Content-Security-Policy", html)
        self.assertTrue(os.path.islink(os.path.join(web, "current")))
        self.assertEqual(len(json.load(open(os.path.join(rel, "data.json")))["days"]), 11)


if __name__ == "__main__":
    unittest.main()
