"""Invariant tests (stdlib unittest):  python -m unittest discover -s tests -v

A 30-day simulation runs through the real engine (runner.execute_run) once per test session; most tests
inspect its outputs. Day 10 is skipped (catch-up), day 5 has a failing LLM step, NEAR is delisted on day 20.
"""
import copy
import glob
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import simulate  # noqa: E402
from cpb import canon, chain, check, corrections, ledger as L, llm, null, pipeline, portfolio, replay, report, runner  # noqa: E402

DAYS, SKIP, LLM_FAIL, DELIST = 30, (10,), (5,), 20
START = "2026-06-01"
_SIM = {}


def sim():
    if not _SIM:
        tmp = tempfile.mkdtemp(prefix="cpb-test-")
        repo, res, world = simulate.run(tmp, days=DAYS, skip=SKIP, llm_fail=LLM_FAIL, delist_day=DELIST, report=False, start=START)
        _SIM.update(repo=repo, res=res, world=world, tmp=tmp)
    return _SIM


def run_dirs(repo, statuses=("ok", "warning", "catchup")):
    out = []
    for p in sorted(glob.glob(os.path.join(repo, "runs", "*", "run.json"))):
        r = canon.read_json(p)
        if r["status"] in statuses:
            out.append((r, os.path.dirname(p)))
    return out


class TestSimulationInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s = sim()
        cls.repo, cls.res, cls.cfg = s["repo"], s["res"], canon.read_json(os.path.join(s["repo"], "config", "config.json"))

    def test_all_runs_succeed(self):
        self.assertEqual(len(self.res), DAYS - len(SKIP))
        self.assertTrue(all(st in ("ok", "warning") for _, st, _ in self.res), self.res)

    def test_cash_never_negative_and_equity_consistent(self):
        for r, d in run_dirs(self.repo):
            led = canon.read_json(os.path.join(d, "ledger.json"))
            for when in ("close", "after"):
                for pid, scns in (led[when] or {}).items():
                    for scn, m in scns.items():
                        self.assertGreaterEqual(m["cash"], -1e-6, f"{r['date']} {pid} {scn} záporná hotovost")
                        self.assertAlmostEqual(m["equity"], m["cash"] + sum(p["value"] for p in m["positions"].values()), places=4)

    def test_weights_sum_and_cap(self):
        caps = {v["id"]: v.get("max_weight", self.cfg["rules"]["max_weight"]) for v in self.cfg["variants"]}
        for r, d in run_dirs(self.repo, ("ok", "warning")):
            dec = canon.read_json(os.path.join(d, "decisions.json"))
            for pid, x in dec["portfolios"].items():
                t = x["targets"] or {}
                self.assertLessEqual(sum(t.values()), 1 + 1e-9, f"{r['date']} {pid} součet vah > 100 %")
                if pid in caps:
                    self.assertTrue(all(w <= caps[pid] + 1e-9 for w in t.values()), f"{r['date']} {pid} váha nad limit")
                    self.assertTrue(all(w >= self.cfg["rules"]["min_weight"] - 1e-9 for w in t.values()))
            led = canon.read_json(os.path.join(d, "ledger.json"))
            for pid, scns in led["after"].items():
                self.assertLessEqual(scns["base"]["exposure"], 1 + 1e-9)

    def test_costs_nonnegative_and_charged(self):
        n = 0
        for r, d in run_dirs(self.repo):
            for t in canon.read_json(os.path.join(d, "fills.json")):
                self.assertGreaterEqual(t["fee_usd"], 0)
                self.assertGreaterEqual(t["slippage_usd"], 0)
                if t["scenario"] == "base":
                    self.assertGreater(t["fee_usd"] + t["slippage_usd"], 0)
                    self.assertTrue(t["fill_price"] > t["snapshot_price"] if t["side"] == "BUY" else t["fill_price"] < t["snapshot_price"])
                if t["scenario"] == "gross":
                    self.assertEqual(t["fee_usd"], 0)
                n += 1
        self.assertGreater(n, 100)
        last = run_dirs(self.repo)[-1][1]
        led = canon.read_json(os.path.join(last, "ledger.json"))["after"]
        for pid in ("zaklad", "nahodny", "b_ew20"):
            e = {s: led[pid][s]["equity"] for s in ("gross", "base", "stress")}
            self.assertGreater(e["gross"], e["base"])
            self.assertGreater(e["base"], e["stress"])

    def test_snapshot_after_lock(self):
        for r, d in run_dirs(self.repo, ("ok", "warning")):
            dec = canon.read_json(os.path.join(d, "decisions.json"))
            snap = canon.read_json(os.path.join(d, "snapshot.json"))
            self.assertGreater(snap["fetched_ms"], dec["locked_ms"])
            self.assertGreater(snap["server_ms"], dec["locked_ms"])
            self.assertGreater(snap["fetched_at"], dec["locked_at"])

    def test_hash_chain_and_replay(self):
        self.assertEqual(chain.verify(self.repo), [])
        probs, state, hist, series = replay.replay(self.repo)
        self.assertEqual(probs, [])
        self.assertEqual(replay.compare_final(self.repo, state, hist), [])
        self.assertEqual(len(series), DAYS)

    def test_tamper_is_detected(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = os.path.join(tmp, "repo")
            shutil.copytree(self.repo, cp)
            d = run_dirs(cp)[5][1]
            p = os.path.join(d, "fills.json")
            fills = canon.read_json(p)
            fills[0]["fee_usd"] = 0.0
            canon.write_json(p, fills)
            self.assertTrue(any("soubory běhu" in x for x in chain.verify(cp)))
            # re-sealing the edited record breaks the link of the next one
            rp = os.path.join(d, "run.json")
            rec = canon.read_json(rp)
            rec["files"] = chain.files_digest(d)
            rec["this_hash"] = chain.record_hash(rec)
            canon.write_json(rp, rec)
            self.assertTrue(any("this_hash" in x or "prev_hash" in x for x in chain.verify(cp)))
        finally:
            shutil.rmtree(tmp)

    def test_catchup_day_has_no_decisions(self):
        missed = canon.add_days(START, SKIP[0])
        d = os.path.join(self.repo, "runs", missed)
        rec = canon.read_json(os.path.join(d, "run.json"))
        self.assertEqual(rec["status"], "catchup")
        self.assertFalse(os.path.exists(os.path.join(d, "decisions.json")))
        self.assertFalse(os.path.exists(os.path.join(d, "llm")))
        for t in canon.read_json(os.path.join(d, "fills.json")):
            self.assertIn(t["reason"], ("stop", "delisting"))
        nxt = canon.read_json(os.path.join(self.repo, "runs", canon.add_days(missed, 1), "run.json"))
        self.assertEqual(nxt["catchup_days"], [missed])

    def test_llm_failure_holds_llm_variants(self):
        day = canon.add_days(START, LLM_FAIL[0])
        d = os.path.join(self.repo, "runs", day)
        rec = canon.read_json(os.path.join(d, "run.json"))
        self.assertEqual(rec["status"], "warning")
        dec = canon.read_json(os.path.join(d, "decisions.json"))
        self.assertFalse(dec["llm_ok"])
        llm_ids = {v["id"] for v in self.cfg["variants"] if v["uses_llm"]}
        for pid in llm_ids:
            self.assertIsNone(dec["portfolios"][pid]["targets"], pid)
        self.assertIsNotNone(dec["portfolios"]["mech_momentum"]["targets"])
        self.assertIsNotNone(dec["portfolios"]["nahodny"]["targets"])
        fills = canon.read_json(os.path.join(d, "fills.json"))
        self.assertFalse([t for t in fills if t["portfolio"] in llm_ids and t["reason"] in ("signal", "rebalance")])
        self.assertFalse(os.path.exists(os.path.join(d, "scores.json")))

    def test_delisting_forced_sale_and_never_rebought(self):
        dl = canon.add_days(START, DELIST)
        sold = [t for r, d in run_dirs(self.repo) for t in canon.read_json(os.path.join(d, "fills.json")) if t["reason"] == "delisting"]
        self.assertTrue(sold, "žádný nucený prodej při delistingu")
        self.assertTrue(all(t["coin"] == "NEAR" for t in sold))
        for r, d in run_dirs(self.repo):
            if r["date"] <= dl:
                continue
            fills = canon.read_json(os.path.join(d, "fills.json"))
            self.assertFalse([t for t in fills if t["coin"] == "NEAR" and t["side"] == "BUY"], r["date"])
            led = canon.read_json(os.path.join(d, "ledger.json"))
            for pid, scns in (led["after"] or led["close"]).items():
                self.assertNotIn("NEAR", scns["base"]["positions"], f"{r['date']} {pid}")

    def test_stops_and_max_hold_fire(self):
        reasons = {t["reason"] for r, d in run_dirs(self.repo) for t in canon.read_json(os.path.join(d, "fills.json"))}
        self.assertIn("max_hold", reasons)
        stops = [t for r, d in run_dirs(self.repo) for t in canon.read_json(os.path.join(d, "fills.json")) if t["reason"] == "stop"]
        self.assertTrue(all(t["portfolio"] == "stop" for t in stops))

    def test_determinism_byte_identical(self):
        tmp = tempfile.mkdtemp(prefix="cpb-det-")
        try:
            repo2, _, _ = simulate.run(tmp, days=DAYS, skip=SKIP, llm_fail=LLM_FAIL, delist_day=DELIST, report=False, start=START)
            for sub in ("data/chain.jsonl", "data/state.json"):
                a = open(os.path.join(self.repo, sub), "rb").read()
                b = open(os.path.join(repo2, sub), "rb").read()
                self.assertEqual(a, b, sub)
            for p in glob.glob(os.path.join(self.repo, "runs", "*", "*.json")):
                q = os.path.join(repo2, os.path.relpath(p, self.repo))
                self.assertEqual(open(p, "rb").read(), open(q, "rb").read(), p)
        finally:
            shutil.rmtree(tmp)

    def test_fills_at_bid_ask(self):
        n = 0
        for r, d in run_dirs(self.repo, ("ok", "warning")):
            snap = canon.read_json(os.path.join(d, "snapshot.json"))
            u = canon.read_json(os.path.join(d, "inputs.json"))["universe"]["coins"]
            feats = canon.read_json(os.path.join(d, "features.json"))
            self.assertTrue({c for c in u if feats.get(c)} <= set(snap["prices"]), "snímek musí mít všechny obchodovatelné coiny")
            for t in canon.read_json(os.path.join(d, "fills.json")):
                if t["reason"] in ("stop", "delisting"):
                    continue
                bid, ask = snap["quotes"][t["coin"]]
                self.assertAlmostEqual(t["snapshot_price"], (bid + ask) / 2)
                if t["scenario"] == "gross":
                    self.assertAlmostEqual(t["fill_price"], t["snapshot_price"])
                elif t["side"] == "BUY":
                    self.assertEqual(t["touch_price"], ask)
                    self.assertGreater(t["fill_price"], ask)
                else:
                    self.assertEqual(t["touch_price"], bid)
                    self.assertLess(t["fill_price"], bid)
                n += 1
        self.assertGreater(n, 100)

    def test_null_replicates_variant_with_actual_picks(self):
        good = [(r, d) for r, d in run_dirs(self.repo)]
        days = null.load_days(self.repo, good)
        H = null.Hist(pipeline.load_history(self.repo))
        for v in self.cfg["variants"]:
            if v.get("stops"):
                continue                      # stops of null paths use daily candles (documented approximation)
            eq = null.simulate_path(v, days, H, None, self.cfg["start_capital"], replicate=True)
            self.assertAlmostEqual(eq[days[-1]["date"]], days[-1]["close"][v["id"]], places=2, msg=v["id"])
        res = null.null_percentiles(days, pipeline.load_history(self.repo), self.cfg["variants"][:3], n_paths=40)
        for vid, r in res.items():
            self.assertLessEqual(r["p05"], r["p50"])
            self.assertLessEqual(r["p50"], r["p95"])
            self.assertTrue(0 <= r["percentile"] <= 100)

    def test_public_dir_only_dashboard_files(self):
        out = tempfile.mkdtemp()
        try:
            report.build(self.repo, self.cfg, out, null_paths=50)
            self.assertEqual(sorted(os.listdir(out)), ["data.json", "index.html"])
            html = open(os.path.join(out, "index.html"), encoding="utf-8").read()
            self.assertIn("Content-Security-Policy", html)
            self.assertNotIn("http://", html.split("<script>")[0].replace("http://www.w3.org", ""))
            open(os.path.join(out, "secret.key"), "w").write("x")
            with self.assertRaises(RuntimeError):
                report.build(self.repo, self.cfg, out, null_paths=50)
            web = os.path.join(out, "web")
            os.remove(os.path.join(out, "secret.key"))
            report.publish(out, web)
            self.assertEqual(sorted(os.listdir(os.path.join(web, "current"))), ["data.json", "index.html"])
        finally:
            shutil.rmtree(out)
        # the repo's own public/ (if present) must only hold the two dashboard files
        pub = os.path.join(ROOT, "public")
        if os.path.isdir(pub):
            self.assertTrue(set(os.listdir(pub)) <= set(report.PUBLIC_FILES), os.listdir(pub))


class TestValidation(unittest.TestCase):
    U = ["BTC", "ETH", "SOL"]

    def good(self, **over):
        s = {"trend": 1, "mr": 0, "news": 0, "conviction": 3, "p_outperform_btc_7d": 0.5, "p_up_7d": 0.5,
             "expected_move_7d_pct": 1.0, "event": False, "event_type": "none", "risk_flag": False, "note": "x", "sources": []}
        s.update(over)
        return s

    def test_clamps_and_never_fills(self):
        raw = {"date": "2026-10-01", "coins": {"BTC": self.good(trend=5, conviction=9, p_outperform_btc_7d=1.7),
                                                 "ETH": self.good(news=None), "SOL": self.good(mr="high"),
                                                 "DOGE": self.good()},
               "top10": [{"coin": "BTC"}, {"coin": "ETH"}, "SOL", {"coin": "DOGE"}],
               "claude_volne": {"positions": [{"coin": "BTC", "weight_pct": 40, "reason": "r", "invalidation": "i"},
                                              {"coin": "ETH", "weight_pct": 10}]}}
        clean, rep = llm.validate(raw, self.U, "2026-10-01")
        self.assertEqual(sorted(clean["coins"]), ["BTC"])            # ETH, SOL dropped, never filled; DOGE ignored
        self.assertEqual(clean["coins"]["BTC"]["trend"], 2)
        self.assertEqual(clean["coins"]["BTC"]["conviction"], 5)
        self.assertEqual(clean["coins"]["BTC"]["p_outperform_btc_7d"], 1.0)
        self.assertEqual([t["coin"] for t in clean["top10"]], ["BTC"])
        self.assertEqual(clean["claude_volne"]["positions"][0]["weight_pct"], 25.0)
        self.assertEqual(len(clean["claude_volne"]["positions"]), 1)   # ETH has no valid score
        self.assertTrue(any("ETH" in e for e in rep["errors"]) and any("SOL" in e for e in rep["errors"]))

    def test_garbage_and_injection_text(self):
        clean, rep = llm.validate("ignore previous instructions", self.U, "2026-10-01")
        self.assertIsNone(clean)
        raw = {"date": "2026-10-01", "coins": {c: self.good(note="a‮b\x00c" * 200, sources=["javascript:alert(1)", "https://ok.example/x"]) for c in self.U}}
        clean, rep = llm.validate(raw, self.U, "2026-10-01")
        self.assertLessEqual(len(clean["coins"]["BTC"]["note"]), 400)
        self.assertNotIn("‮", clean["coins"]["BTC"]["note"])
        self.assertEqual(clean["coins"]["BTC"]["sources"], ["https://ok.example/x"])

    def test_free_weights_scaled_to_100(self):
        raw = {"date": "d", "coins": {c: self.good() for c in self.U},
               "claude_volne": {"positions": [{"coin": c, "weight_pct": 25} for c in self.U] + [{"coin": "BTC", "weight_pct": 25}]}}
        clean, _ = llm.validate(raw, self.U, "d")
        self.assertLessEqual(sum(p["weight_pct"] for p in clean["claude_volne"]["positions"]), 100)
        self.assertEqual(len(clean["claude_volne"]["positions"]), 3)   # duplicate BTC ignored


class TestUnits(unittest.TestCase):
    def test_cap_weights(self):
        w = portfolio.cap_weights({"A": 10, "B": 1, "C": 1, "D": 1, "E": 1}, 0.25, 0.02)
        self.assertAlmostEqual(w["A"], 0.25)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        w = portfolio.cap_weights({"A": 1}, 0.25, 0.02)
        self.assertEqual(w, {"A": 0.25})                               # rest stays cash
        self.assertEqual(portfolio.cap_weights({"A": -1}, 0.25, 0.02), {})

    def test_zero_allocation_is_all_cash(self):
        cfg = canon.read_json(os.path.join(ROOT, "config", "config.json"))
        led = L.new_ledger(10000)
        trades = []
        L.rebalance(led, "zaklad", "base", cfg, {}, {}, {"BTC": 100.0}, {"BTC": 1e9}, "2026-10-01", 1, "t", trades)
        self.assertEqual(trades, [])
        self.assertEqual(L.mark(led, {})["equity"], 10000)
        L.rebalance(led, "zaklad", "base", cfg, {"BTC": 0.5}, {}, {"BTC": 100.0}, {"BTC": 1e9}, "2026-10-01", 1, "t", trades)
        L.rebalance(led, "zaklad", "base", cfg, {}, {}, {"BTC": 100.0}, {"BTC": 1e9}, "2026-10-02", 2, "t", trades)
        self.assertEqual(led["positions"], {})
        self.assertLess(led["cash"], 10000)                            # round trip paid costs
        self.assertGreater(led["cash"], 9900)

    def test_band_prevents_small_rebalances(self):
        cfg = canon.read_json(os.path.join(ROOT, "config", "config.json"))
        led, trades = L.new_ledger(10000), []
        L.rebalance(led, "x", "base", cfg, {"BTC": 0.20}, {}, {"BTC": 100.0}, {"BTC": 1e9}, "d", 1, "t", trades)
        n = len(trades)
        L.rebalance(led, "x", "base", cfg, {"BTC": 0.21}, {}, {"BTC": 100.0}, {"BTC": 1e9}, "d", 2, "t", trades)
        self.assertEqual(len(trades), n)                               # 1 % change < 2 % band

    def test_redenomination_keeps_value(self):
        cfg = canon.read_json(os.path.join(ROOT, "config", "config.json"))
        led, trades = L.new_ledger(10000), []
        L.rebalance(led, "x", "base", cfg, {"PEPE": 0.25}, {}, {"PEPE": 0.00001}, {"PEPE": 1e9}, "d", 1, "t", trades)
        state = {"portfolios": {"x": {"base": led}}}
        hist = {"PEPE": [["2026-09-29", 1e-5, 1.1e-5, 0.9e-5, 1e-5, 1e9, 1e4], ["2026-09-30", 1e-2, 1.1e-2, 0.9e-2, 1e-2, 1e6, 1e4]]}
        before = L.mark(led, {"PEPE": 1e-5})["equity"]
        st2 = corrections.apply_state(copy.deepcopy(state), "PEPE", 1 / 1000, "2026-09-30", "1000PEPE")
        h2 = corrections.rebase_history(hist, "PEPE", 1 / 1000, "2026-09-30", "1000PEPE")
        after = L.mark(st2["portfolios"]["x"]["base"], {"1000PEPE": h2["1000PEPE"][0][4]})["equity"]
        self.assertAlmostEqual(before, after, places=6)
        self.assertIn("1000PEPE", st2["portfolios"]["x"]["base"]["positions"])
        self.assertEqual(check.redenomination_ratio(1e-5, 1e-2), 1e-3)
        self.assertIsNone(check.redenomination_ratio(100, 70))

    def test_invalid_data_is_rejected(self):
        cfg = canon.read_json(os.path.join(ROOT, "config", "config.json"))
        good = [["2026-09-28", 10, 11, 9, 10.5, 5, 50], ["2026-09-29", 10.5, 11, 10, 10.8, 5, 54]]
        self.assertEqual(check.check_coin("X", good, [], "2026-09-29", cfg)[0], [])
        cases = {
            "nekladná": [good[0], ["2026-09-29", 10.5, 11, 10, 0, 5, 54]],
            "high < low": [good[0], ["2026-09-29", 10.5, 9, 10, 10.8, 5, 54]],
            "chybějící": [["2026-09-27", 10, 11, 9, 10.5, 5, 50], good[1]],
            "zastaralá": [good[0]],
            "duplicitní": [good[0], good[0], good[1]],
            "nulový objem": [good[0], ["2026-09-29", 10.5, 11, 10, 10.8, 0, 0]],
            "bez potvrzení": [good[0], ["2026-09-29", 10.5, 20, 10, 19, 5, 54]],
        }
        for name, rows in cases.items():
            errs = check.check_coin("X", rows, [], "2026-09-29", cfg, confirm=lambda *a: False)[0]
            self.assertTrue(errs, name)
        errs, warns, _ = check.check_coin("X", cases["bez potvrzení"], [], "2026-09-29", cfg, confirm=lambda *a: True)
        self.assertEqual(errs, [])
        self.assertTrue(warns)
        errs = check.check_coin("X", good, [["2026-09-28", 10, 11, 9, 10.9, 5, 50]], "2026-09-29", cfg)[0]
        self.assertTrue(any("uložené historie" in e for e in errs))


class TestFailures(unittest.TestCase):
    def _repo(self):
        tmp = tempfile.mkdtemp(prefix="cpb-fail-")
        return tmp, simulate.make_repo(tmp)

    def test_bad_data_fails_run_without_state_change(self):
        tmp, repo = self._repo()
        try:
            cfg = canon.read_json(os.path.join(repo, "config", "config.json"))
            world = simulate.World(START, 5)
            clock = simulate.SimClock(canon.date_ms(START) + 20 * 60_000)

            class Bad(simulate.SimMarket):
                def daily_candles(self, coin, pair, start, end):
                    rows, src = super().daily_candles(coin, pair, start, end)
                    if coin == "ETH":
                        rows[-1] = rows[-1][:3] + [rows[-1][2] * 2] + rows[-1][4:]      # low > high
                    return rows, src
            st, rec = runner.execute_run(repo, cfg, START, lambda r, d: Bad(world, r, clock), simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "failed")
            self.assertIn("ETH", rec["error"])
            self.assertFalse(os.path.exists(os.path.join(repo, "data", "state.json")))
            self.assertFalse(os.path.exists(os.path.join(repo, "runs", START, "run.json")))
            self.assertEqual(chain.verify(repo), [])
            # a later successful attempt the same day works (failed attempt stays recorded)
            clock.set(canon.date_ms(START) + 6 * 3600_000)
            st, rec = runner.execute_run(repo, cfg, START, lambda r, d: simulate.SimMarket(world, r, clock), simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "ok")
            self.assertEqual(len(chain.entries(repo)), 2)
            st, _ = runner.execute_run(repo, cfg, START, lambda r, d: simulate.SimMarket(world, r, clock), simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "exists")                                              # idempotent
        finally:
            shutil.rmtree(tmp)

    def test_quarantine_of_unheld_coin(self):
        tmp, repo = self._repo()
        try:
            cfg = canon.read_json(os.path.join(repo, "config", "config.json"))
            world = simulate.World(START, 5)
            clock = simulate.SimClock(canon.date_ms(START) + 20 * 60_000)
            bad = {"coin": "SOL"}

            class Bad(simulate.SimMarket):
                def daily_candles(self, coin, pair, start, end):
                    rows, src = super().daily_candles(coin, pair, start, end)
                    if coin == bad["coin"]:
                        rows[-1] = rows[-1][:3] + [rows[-1][2] * 2] + rows[-1][4:]      # low > high
                    return rows, src
            f = lambda r, d: Bad(world, r, clock)
            st, rec = runner.execute_run(repo, cfg, START, f, simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "warning")
            self.assertIn("SOL", rec["checks"]["quarantine"])
            d = os.path.join(repo, "runs", START)
            self.assertFalse([t for t in canon.read_json(os.path.join(d, "fills.json")) if t["coin"] == "SOL"])
            self.assertIsNone(canon.read_json(os.path.join(d, "features.json"))["SOL"])
            self.assertNotIn("SOL", pipeline.load_history(repo))                                      # bad data not stored
            # next day SOL is fine again (re-fetched, gap filled); then a HELD coin with bad data fails the run
            bad["coin"] = None
            D2 = canon.add_days(START, 1)
            clock.set(canon.date_ms(D2) + 20 * 60_000)
            st, rec = runner.execute_run(repo, cfg, D2, f, simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "ok")
            self.assertEqual(pipeline.load_history(repo)["SOL"][-1][0], START)
            held = pipeline.held_coins(pipeline.load_state(repo))
            bad["coin"] = [c for c in held if c not in ("BTC", "ETH")][0]
            D3 = canon.add_days(START, 2)
            clock.set(canon.date_ms(D3) + 20 * 60_000)
            st, rec = runner.execute_run(repo, cfg, D3, f, simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "failed")
            self.assertIn(bad["coin"], rec["error"])
        finally:
            shutil.rmtree(tmp)

    def test_lookahead_guard(self):
        tmp, repo = self._repo()
        try:
            cfg = canon.read_json(os.path.join(repo, "config", "config.json"))
            world = simulate.World(START, 5)
            clock = simulate.SimClock(canon.date_ms(START) + 20 * 60_000)

            class Stale(simulate.SimMarket):
                def ticker_prices(self, pairs):
                    prices, server, src = super().ticker_prices(pairs)
                    return prices, canon.date_ms(START), src                             # server time before the lock
            st, rec = runner.execute_run(repo, cfg, START, lambda r, d: Stale(world, r, clock), simulate.fake_llm(world), clock, {"tag": "t"})
            self.assertEqual(st, "failed")
            self.assertIn("lookahead", rec["error"])
        finally:
            shutil.rmtree(tmp)

    def test_pin_mismatch_detected(self):
        tmp = tempfile.mkdtemp()
        try:
            cp = os.path.join(tmp, "r")
            for d in ("engine", "config", "task", "dashboard"):
                shutil.copytree(os.path.join(ROOT, d), os.path.join(cp, d), ignore=shutil.ignore_patterns("__pycache__"))
            for f in ("run_daily.sh", "VERSION"):
                shutil.copy(os.path.join(ROOT, f), os.path.join(cp, f))
            canon.write_text(os.path.join(cp, "MANIFEST.sha256"), runner.manifest_text(cp))
            os.environ["CPB_PIN_FILE"] = os.path.join(tmp, "pin.json")
            v, probs = runner.version_info(cp, require_pin=False)
            self.assertEqual(probs, [])
            canon.write_json(os.environ["CPB_PIN_FILE"], {"tag": v["tag"], "manifest_sha256": v["manifest_sha256"]})
            self.assertEqual(runner.version_info(cp)[1], [])
            with open(os.path.join(cp, "config", "config.json"), "a") as f:
                f.write(" ")
            self.assertTrue(runner.version_info(cp)[1])
        finally:
            os.environ.pop("CPB_PIN_FILE", None)
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
