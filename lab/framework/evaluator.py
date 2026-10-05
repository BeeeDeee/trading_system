"""The real gate evaluator (G0-G4) over qlab. Deterministic; thresholds only from gates.yaml.

Data flow of one hypothesis:
- `Context` loads every instrument the strategy and its benchmark need into one DataView (union calendar),
  with the dev end = the day before the earliest `holdout_from` of the strategy's datasets.
- G0-G3 only ever pass the strategy rows up to the dev end. G4 runs the same code over the full history and
  scores only the holdout rows (warm-up on dev rows is legitimate, the code is point-in-time by G0).
- The benchmark comes from gates.yaml (asset classes + market exposure), never from the agent.

Every metric and every check (value, threshold, pass) goes into the gate result, so a rejection can be
explained without rerunning anything.
"""

import hashlib
import itertools
import json
import time
from dataclasses import dataclass, field, replace
from datetime import date

import numpy as np

from qlab.validation.stats import expected_max_sharpe, pbo_cscv, probabilistic_sharpe

from lab.framework import catalog, costs, data, engine, metrics, strategy
from lab.framework.blackboard import Lab
from lab.framework.gates import Outcome, Trial, strategy_sha
from lab.framework.api import period_starts
from lab.framework.data import DataView

BENCHMARKS = {"SPY_TR": {"SPY": 1.0}, "SIXTY_FORTY": {"SPY": 0.6, "IEF": 0.4}, "BTC_HOLD": {"BTCUSDT": 1.0},
              "SIXTY_FORTY_BTC5": {"SPY": 0.57, "IEF": 0.38, "BTCUSDT": 0.05}, "TBILL": {},
              "SYN_MKT": {"MKT": 1.0}}
SOURCE = {"SPY": "sharadar_sfp", "IEF": "sharadar_sfp", "BTCUSDT": "binance_spot_1d", "MKT": "synthetic_market"}
BENCH_GROUP = {"crypto_perp": "crypto_spot"}
TRADABLE = {"sharadar_sfp", "sharadar_sep", "binance_spot_1d", "binance_perp_1d", "synthetic_market"}


@dataclass
class Run:
    targets: np.ndarray      # (T, N_strategy) as returned, rows up to `end`
    returns: np.ndarray      # (T,) net returns over the view rows up to `end`
    sim: engine.SimResult


@dataclass
class Context:
    hid: str
    card: dict
    view: DataView | None = None     # full width (strategy + benchmark instruments), full history
    strat_cols: list[str] = field(default_factory=list)
    bench_name: str = ""
    bench_weights: dict = field(default_factory=dict)
    runtime_limit_s: float = 1200.0
    dev_end: int = 0                 # number of dev rows
    data_end: int = 0
    path: object = None
    code_sha: str = ""
    runs: dict = field(default_factory=dict)



class QlabEvaluator:
    name = "qlab-v1"

    def __init__(self, overrides: dict | None = None, view_loader=None, require_canaries: bool = True):
        """`overrides` (tests only) shrink expensive settings such as bootstrap or random-entry runs; they are
        written into every result. `view_loader(dataset, instruments)` replaces `data.load` in tests.
        With `require_canaries` (the default) no gate runs unless the canaries passed for the current
        framework hash (`lab canaries`): a changed judge must prove itself before it judges."""
        self.overrides = overrides or {}
        self.require_canaries = require_canaries
        self.load = view_loader or data.load
        self._ctx: dict[tuple, Context] = {}

    def ingest(self, lab: Lab, payload: dict) -> dict:
        """Real ingest of an Archivist dataset (lab.framework.ingest); raises IngestError (a ValueError)."""
        from lab.framework import ingest
        ds = payload["dataset"]
        src = lab.paths.lab / "data" / "sources" / f"{ds}.py"
        return ingest.ingest(lab.paths.home, lab.paths.catalog, ds, payload["catalog_entry"], src)

    def paper_ready(self, lab: Lab, hid: str) -> bool:
        return False  # the paper runner arrives in step 4

    # ------------------------------------------------------------------ entry point

    def evaluate(self, lab: Lab, hid: str, card: dict, gate: str, th: dict) -> Outcome:
        if self.require_canaries:
            from lab.framework.canaries import passed_for_current
            if not passed_for_current(lab):
                raise RuntimeError("canaries have not passed for the current framework; run `lab canaries`")
        th = _merge(th, self.overrides.get(gate, {}))
        try:
            ctx = self._context(lab, hid, card)
        except (NotImplementedError, KeyError) as e:   # catalog has it, no loader yet / unknown instrument
            lab.send("ALERT", "gatekeeper", "human", hid, {"severity": "warning", "text": f"{gate}: {e}"})
            return Outcome(False, {"error": str(e)}, f"{gate.lower()}_no_data_adapter")
        out = {"G0": self._g0, "G1": self._g1, "G2": self._g2, "G3": self._g3, "G4": self._g4}[gate](lab, ctx, th)
        if self.overrides.get(gate):
            out.metrics["test_overrides"] = self.overrides[gate]
        return out

    # ------------------------------------------------------------------ context and runs

    def _context(self, lab: Lab, hid: str, card: dict) -> Context:
        code = strategy_sha(lab, hid)
        key = (hid, lab.hypothesis(hid)["version"], code)
        if key in self._ctx:
            return self._ctx[key]
        import yaml
        gates_cfg = yaml.safe_load(lab.paths.gates.read_text())
        cat = catalog.load(lab.paths.catalog)
        reqs = card["data_requirements"]
        universe = card["universe"]
        unknown = [r["dataset"] for r in reqs if r["dataset"] not in TRADABLE
                   and cat.get(r["dataset"], {}).get("loader") != "generic"]
        if unknown:
            raise NotImplementedError(f"no data adapter yet for {unknown}")
        signal_ds = [r["dataset"] for r in reqs if r["dataset"] not in TRADABLE]
        per_ds = strategy_instruments(card)
        strat_ds = list(per_ds)

        exposure = card["market_exposure"]
        # one benchmark per market: spot and perps are both crypto (BTC), stocks with ETFs are US (60/40)
        classes = sorted({BENCH_GROUP.get(cat[d]["asset_class"], cat[d]["asset_class"]) for d in strat_ds})
        if classes == ["us_equity", "us_etf"]:
            classes = ["us_etf"]
        bcfg = gates_cfg["benchmarks"]
        bench = bcfg[exposure] if exposure != "long_only" else bcfg["long_only"][
            classes[0] if len(classes) == 1 else "cross_asset"]
        bweights = BENCHMARKS[bench]
        load_ds = {d: (None if v is None else list(v)) for d, v in per_ds.items()}
        for inst in bweights:
            ds = SOURCE[inst]
            if ds in load_ds and load_ds[ds] is not None and inst not in load_ds[ds]:
                load_ds[ds].append(inst)
            elif ds not in load_ds:
                load_ds[ds] = [inst]
        views, strat_set = [], set()
        for ds, inst in load_ds.items():
            v = data.sharadar_sep(universe) if ds == "sharadar_sep" and inst is None else self.load(ds, inst)
            if universe["kind"] == "crypto_top_n" and ds == "binance_spot_1d":
                v = replace(v, universe=data.crypto_top_n(v, universe["n"]))
            if ds in per_ds:
                strat_set |= set(v.instruments if per_ds[ds] is None else per_ds[ds])
            views.append(v)
        view = data.merge(views)
        for ds in dict.fromkeys(signal_ds):
            view = data.attach_generic(view, lab.paths.home, ds, cat[ds])
        all_ds = strat_ds + signal_ds
        start = max(data.as_date(cat[d]["range"][0]) for d in strat_ds)
        if any(r.get("period") for r in reqs):
            start = max([start] + [data.as_date(r["period"][0]) for r in reqs if r.get("period")])
        end = min(data.as_date(cat[d]["range"][1]) for d in all_ds)
        view = view.between(start, end)
        synthetic_only = set(load_ds) == {"synthetic_market"}
        view = data.with_cash(view, None if synthetic_only else data.tbill_rates())
        holdout = min(data.as_date(cat[d]["holdout_from"]) for d in all_ds)
        strat_cols = [c for c in view.instruments if c in strat_set]
        limit = _merge(gates_cfg["G0"], self.overrides.get("G0", {}))["max_runtime_min"] * 60.0
        ctx = Context(hid, card, view, strat_cols, bench, bweights, limit,
                      dev_end=int(np.searchsorted(view.dates, holdout, side="left")), data_end=len(view.dates),
                      path=lab.paths.strategies / hid / "strategy.py", code_sha=code or "")
        self._ctx = {key: ctx}   # one hypothesis at a time: keeps memory flat on the small machine
        return ctx

    def _targets(self, ctx: Context, params: dict, end: int, universe=None) -> np.ndarray:
        view = ctx.view.slice(end).columns(ctx.strat_cols)
        if universe is not None:
            view = replace(view, universe=universe[:end])
        module = strategy.load(ctx.path)
        with strategy.time_limit(ctx.runtime_limit_s):
            w = np.asarray(module.target_weights(view, dict(params)), dtype=float)
        if w.shape != view.shape:
            raise ValueError(f"target_weights returned shape {w.shape}, expected {view.shape}")
        return w

    def _full_width(self, ctx: Context, w: np.ndarray) -> np.ndarray:
        full = np.full((w.shape[0], len(ctx.view.instruments)), np.nan)
        decided = ~np.isnan(w).all(axis=1)
        full[decided] = 0.0
        for k, c in enumerate(ctx.strat_cols):
            full[:, ctx.view.instruments.index(c)] = np.where(decided, w[:, k], np.nan)
        return full

    def _simulate(self, ctx: Context, full_targets: np.ndarray, end: int, cost_mult: float = 1.0):
        v = ctx.view.slice(end)
        return engine.simulate(v.ret_co, v.ret_oc, full_targets, costs.cost_rates(v, cost_mult), v.tradable,
                               v.delisting, v.cash_ret, funding=v.extras.get("funding_paid"))

    def run(self, ctx: Context, params: dict, end: int, cost_mult: float = 1.0, universe=None) -> Run:
        key = (json.dumps(params, sort_keys=True), end, cost_mult, None if universe is None else id(universe))
        if key not in ctx.runs:
            w = self._targets(ctx, params, end, universe)
            sim = self._simulate(ctx, self._full_width(ctx, w), end, cost_mult)
            if params != self.primary(ctx.card) or cost_mult != 1.0 or universe is not None:
                # variants (neighbors, cost stress, alt universe) are only scored on returns: keep them light,
                # a (T, N) targets + holdings pair is ~0.3 GB for single stocks
                w, sim = None, replace(sim, held=None)
            ctx.runs[key] = Run(w, sim.returns, sim)
        return ctx.runs[key]

    def benchmark(self, ctx: Context, end: int) -> np.ndarray:
        key = ("benchmark", end)
        if key not in ctx.runs:
            v = ctx.view.slice(end)
            if not ctx.bench_weights:
                ctx.runs[key] = np.asarray(v.cash_ret)
            else:
                t = np.full(v.shape, np.nan)
                rows = period_starts(v.dates, "M")
                t[rows] = 0.0
                for inst, w in ctx.bench_weights.items():
                    t[rows, v.instruments.index(inst)] = w
                sim = engine.simulate(v.ret_co, v.ret_oc, t, costs.cost_rates(v), v.tradable, v.delisting,
                                      v.cash_ret)
                ctx.runs[key] = sim.returns
        return ctx.runs[key]

    @staticmethod
    def primary(card: dict) -> dict:
        return {k: p["value"] for k, p in card.get("signal", {}).get("params", {}).items()}

    def config_sha(self, ctx: Context, params: dict) -> str:
        return hashlib.sha256(json.dumps({"hid": ctx.hid, "code": ctx.code_sha, "params": params},
                                         sort_keys=True).encode()).hexdigest()[:16]

    def window(self, ctx: Context, run: Run, start: int = 0) -> slice:
        """Scored rows: from the first execution after the first decision (or `start`) to the end of the run."""
        decided = np.flatnonzero(~np.isnan(run.targets).all(axis=1))
        first = int(decided[0]) + 1 if len(decided) else len(run.returns)
        return slice(max(first, start), len(run.returns))

    # ------------------------------------------------------------------ G0 integrity

    def _g0(self, lab, ctx: Context, th) -> Outcome:
        return integrity(ctx.card, ctx.path, ctx.view, ctx.strat_cols, self.primary(ctx.card), ctx.dev_end, th,
                         ctx.runtime_limit_s)

    # ------------------------------------------------------------------ scoring
    # Sharpe ratios are of returns in excess of cash, so a cash benchmark has Sharpe 0 and a strategy that is
    # partly in cash is compared on risk-adjusted terms, not on raw return.

    def excess(self, ctx: Context, r: np.ndarray, win: slice) -> np.ndarray:
        return np.asarray(r)[win] - np.asarray(ctx.view.cash_ret)[win]

    def sr(self, ctx: Context, r: np.ndarray, win: slice, ppy: float) -> float:
        return metrics.sharpe(self.excess(ctx, r, win), ppy)

    def _scores(self, ctx: Context, run: Run, bench: np.ndarray, win: slice) -> dict:
        ppy = metrics.periods_per_year(ctx.view.dates[win])
        r, b = run.returns[win], bench[win]
        return {"sharpe": self.sr(ctx, run.returns, win, ppy), "bench_sharpe": self.sr(ctx, bench, win, ppy),
                "cagr": metrics.cagr(r, ppy), "bench_cagr": metrics.cagr(b, ppy),
                "max_dd": metrics.max_dd(r), "bench_max_dd": metrics.max_dd(b),
                "mean_exposure": float(run.sim.gross[win].mean()), "ppy": ppy, "n_obs": len(r),
                "start": str(ctx.view.dates[win][0]), "end": str(ctx.view.dates[win][-1])}

    # ------------------------------------------------------------------ G1 basic

    def _g1(self, lab, ctx: Context, th) -> Outcome:
        params, end = self.primary(ctx.card), ctx.dev_end
        run, bench = self.run(ctx, params, end), self.benchmark(ctx, end)
        win = self.window(ctx, run)
        s = self._scores(ctx, run, bench, win)
        lower = metrics.bootstrap_sharpe_diff_lower(self.excess(ctx, run.returns, win), self.excess(ctx, bench, win),
                                                    th["sharpe_diff_ci_level"], th["bootstrap_samples"],
                                                    th["bootstrap_mean_block_days"], ppy=s["ppy"])
        decisions = int((run.sim.turnover[win] > 1e-9).sum())
        checks = {
            "sharpe_excess": _check(s["sharpe"] - s["bench_sharpe"], ">=", th["min_sharpe_excess"]),
            "sharpe_diff_ci_lower": _check(lower, ">", 0.0),
            "position_entries": _check(metrics.position_entries(run.sim.held[win]), ">=", th["min_position_entries"]),
            "decision_dates": _check(decisions, ">=", th["min_decision_dates"]),
            "mean_exposure": _check(s["mean_exposure"], ">=", th["min_mean_exposure"]),
        }
        if ctx.bench_weights:
            checks["max_dd_vs_benchmark"] = _check(s["max_dd"], "<=",
                                                   th["max_drawdown_vs_benchmark"] * s["bench_max_dd"])
        else:
            checks["sharpe_vs_tbill"] = _check(s["sharpe"], ">=", 0.5)
        return _outcome("g1", s | {"benchmark": ctx.bench_name}, checks,
                        [Trial(self.config_sha(ctx, params), _per_period(self.excess(ctx, run.returns, win)),
                               win.stop - win.start)])

    # ------------------------------------------------------------------ G2 robustness

    def neighbors(self, card: dict, steps: int = 1, max_combined: int = 3) -> list[dict]:
        """One-step grid neighbors of the primary configuration: all combinations for <= max_combined
        parameters, otherwise one parameter at a time."""
        params = card.get("signal", {}).get("params", {})
        if not params:
            return []
        names = list(params)
        idx = {k: params[k]["grid"].index(params[k]["value"]) for k in names}
        out = []
        for delta in itertools.product(*[range(-steps, steps + 1)] * len(names)):
            if not any(delta) or (len(names) > max_combined and sum(map(bool, delta)) > 1):
                continue
            cfg = {}
            for k, d in zip(names, delta):
                i = idx[k] + d
                if not 0 <= i < len(params[k]["grid"]):
                    break
                cfg[k] = params[k]["grid"][i]
            else:
                if cfg not in out:
                    out.append(cfg)
        return out

    def _g2(self, lab, ctx: Context, th) -> Outcome:
        params, end = self.primary(ctx.card), ctx.dev_end
        run, bench = self.run(ctx, params, end), self.benchmark(ctx, end)
        win = self.window(ctx, run)
        base = self._scores(ctx, run, bench, win)
        ppy, sb = base["ppy"], base["bench_sharpe"]
        checks, m, trials = {}, {"benchmark": ctx.bench_name, "sharpe": base["sharpe"], "bench_sharpe": sb}, []

        # parameter neighborhood
        nb = self.neighbors(ctx.card, th["params"]["neighbor_steps"], th["params"]["max_combined_params"])
        nb_sharpes = []
        for cfg in nb:
            r = self.run(ctx, cfg, end)
            nb_sharpes.append(self.sr(ctx, r.returns, win, ppy))
            trials.append(Trial(self.config_sha(ctx, cfg), _per_period(self.excess(ctx, r.returns, win)),
                                win.stop - win.start))
        m["neighbors"] = [{"params": c, "sharpe": round(s, 4)} for c, s in zip(nb, nb_sharpes)]
        if nb:
            share = float(np.mean(np.array(nb_sharpes) >= sb))
            ratio = float(np.median(nb_sharpes) / base["sharpe"]) if base["sharpe"] > 0 else 0.0
            checks["neighbors_beating_benchmark"] = _check(share, ">=", th["params"]["min_share_beating_benchmark"])
            checks["median_neighbor_sharpe_ratio"] = _check(ratio, ">=", th["params"]["min_median_sharpe_ratio"])
        else:
            checks["neighbors"] = _na("card declares no parameters")

        # sub-periods
        sub = th["subperiods"]
        parts = np.array_split(np.arange(win.start, win.stop), sub["n_blocks"])
        beats = [self.sr(ctx, run.returns, slice(p[0], p[-1] + 1), ppy) >= self.sr(ctx, bench, slice(p[0], p[-1] + 1), ppy)
                 for p in parts]
        blocks, share = metrics.block_log_return_share(run.returns[win], sub["n_blocks"])
        m["subperiods"] = {"beats_benchmark": beats, "log_return": [round(float(b), 4) for b in blocks]}
        checks["subperiods_beating_benchmark"] = _check(sum(beats), ">=", sub["min_blocks_beating_benchmark"])
        checks["max_subperiod_share"] = _check(share, "<=", sub["max_block_share_of_log_return"])

        # alternative universe
        uni = ctx.card["universe"]
        if uni["kind"] == "crypto_top_n":
            n_alt = int(round(uni["n"] * 2.5))
            v = ctx.view
            mask = np.zeros(v.shape, dtype=bool)
            crypto = [i for i, c in enumerate(v.asset_class) if c.startswith("crypto")]
            sub_view = replace(v.columns([v.instruments[i] for i in crypto]), universe=None)
            mask[:, crypto] = data.crypto_top_n(sub_view, n_alt)
            alt = self.run(ctx, params, end, universe=mask[:, [v.instruments.index(c) for c in ctx.strat_cols]])
            alt_sharpe = self.sr(ctx, alt.returns, win, ppy)
            m["alt_universe"] = {"kind": "crypto_top_n", "n": n_alt, "sharpe": alt_sharpe}
            checks["alt_universe_vs_benchmark"] = _check(alt_sharpe, ">=", sb)
            checks["alt_universe_ratio"] = _check(alt_sharpe / base["sharpe"] if base["sharpe"] > 0 else 0.0, ">=",
                                                  th["alt_universe"]["min_sharpe_ratio"])
        elif uni["kind"] in ("liq_n", "sp500") and "alt_universe" in ctx.view.extras:
            v = ctx.view
            mask = np.nan_to_num(v.extras["alt_universe"]) > 0.5
            alt = self.run(ctx, params, end, universe=mask[:, [v.instruments.index(c) for c in ctx.strat_cols]])
            alt_sharpe = self.sr(ctx, alt.returns, win, ppy)
            m["alt_universe"] = {"kind": "sp500" if uni["kind"] == "liq_n" else "liq_n", "sharpe": alt_sharpe}
            checks["alt_universe_vs_benchmark"] = _check(alt_sharpe, ">=", sb)
            checks["alt_universe_ratio"] = _check(alt_sharpe / base["sharpe"] if base["sharpe"] > 0 else 0.0, ">=",
                                                  th["alt_universe"]["min_sharpe_ratio"])
        else:
            checks["alt_universe"] = _na(f"no alternative universe defined for kind {uni['kind']!r} yet")

        # cost stress
        stressed = self.run(ctx, params, end, cost_mult=th["costs"]["stress_multiplier"])
        m["sharpe_costs_x3"] = self.sr(ctx, self.run(ctx, params, end, cost_mult=th["costs"]["report_multiplier"]).returns,
                                       win, ppy)
        checks["sharpe_costs_x2_vs_benchmark"] = _check(self.sr(ctx, stressed.returns, win, ppy), ">=", sb)

        # random-entry null
        pct, null = self._random_entry(ctx, run, end, win, ppy, th["random_entry"]["runs"])
        m["random_entry"] = {"runs": len(null), "median_sharpe": float(np.median(null)) if null else None,
                             "p95_sharpe": float(np.quantile(null, 0.95)) if null else None}
        checks["random_entry_percentile"] = _check(pct, ">=", th["random_entry"]["min_percentile"])

        # publication decay: Sharpe excess over the benchmark before and after the first publication year
        years = [r["year"] for r in ctx.card.get("references", []) if r.get("year")]
        if years:
            split = np.datetime64(f"{min(years) + 1}-01-01")
            k = win.start + int(np.searchsorted(ctx.view.dates[win], split))
            if (k - win.start) / ppy >= 3 and (win.stop - k) / ppy >= 3:
                pre_w, post_w = slice(win.start, k), slice(k, win.stop)
                pre = self.sr(ctx, run.returns, pre_w, ppy) - self.sr(ctx, bench, pre_w, ppy)
                post = self.sr(ctx, run.returns, post_w, ppy) - self.sr(ctx, bench, post_w, ppy)
                m["publication_decay"] = {"split": str(split), "pre_sharpe_excess": pre, "post_sharpe_excess": post}
                checks["post_publication_ratio"] = (
                    _check(post / pre, ">=", th["publication_decay"]["min_post_pre_ratio"]) if pre > 0
                    else _na("no pre-publication edge to decay"))
            else:
                checks["post_publication_ratio"] = _na("publication year not inside dev with 3 years on each side")
        return _outcome("g2", m, checks, trials)

    def _random_entry(self, ctx, run: Run, end, win, ppy, n_runs) -> tuple[float, list[float]]:
        """Same exposure, turnover and holding periods, random assets (>= 5 active) or random timing."""
        w = run.targets
        active_cols = np.flatnonzero(np.nan_to_num(np.abs(w)).sum(axis=0) > 0)
        rng = np.random.default_rng(12345)
        sv = ctx.view.slice(end).columns(ctx.strat_cols)
        # random assets come from the same pool the strategy chooses from: the card's universe when there is
        # one (otherwise a LIQ-500 book is compared with random illiquid stocks that cost several times more)
        pool = (sv.listed if sv.universe is None else sv.universe)[win]
        eligible = np.union1d(np.flatnonzero(pool.mean(axis=0) > 0.5), active_cols)
        null = []
        for _ in range(n_runs):
            if len(active_cols) >= 5 and len(eligible) > len(active_cols):
                perm = np.arange(w.shape[1])
                perm[eligible] = rng.permutation(eligible)
                fake = np.full_like(w, np.nan)
                fake[:, perm] = w
            else:
                fake = np.roll(w, int(rng.integers(63, max(64, len(w) - 63))), axis=0)
            null.append(self.sr(ctx, self._simulate(ctx, self._full_width(ctx, fake), end).returns, win, ppy))
        s = self.sr(ctx, run.returns, win, ppy)
        return float(np.mean(np.array(null) < s)) if null else 0.0, null

    # ------------------------------------------------------------------ G3 multiple testing

    def _g3(self, lab, ctx: Context, th) -> Outcome:
        """Deflated test: P(true Sharpe > benchmark Sharpe + expected best Sharpe of N unskilled trials)."""
        params, end = self.primary(ctx.card), ctx.dev_end
        run, bench = self.run(ctx, params, end), self.benchmark(ctx, end)
        win = self.window(ctx, run)
        ex = self.excess(ctx, run.returns, win)
        sb = _per_period(self.excess(ctx, bench, win))
        family = ctx.card["family"]
        family_sharpes = np.array(lab.family_trial_sharpes(family), dtype=float)   # merged families included
        n_family = lab.family_trials(family)
        prior = th["literature_prior_trials"] if ctx.card.get("references") else 0
        n = max(n_family + prior, th["min_trials"])
        var = max(float(family_sharpes.var(ddof=1)) if len(family_sharpes) > 1 else 0.0, 1.0 / len(ex))
        hurdle = sb + expected_max_sharpe(n, var)
        dsr = probabilistic_sharpe(ex, hurdle)
        checks = {"dsr": _check(dsr, ">=", th["min_dsr"])}
        m = {"n_family_trials": n_family, "literature_prior": prior, "n_trials": n, "sr_variance": var,
             "sharpe_per_period": _per_period(ex), "bench_sharpe_per_period": sb, "hurdle_per_period": hurdle,
             "dsr": dsr}
        configs = [params] + self.neighbors(ctx.card)
        if len(configs) >= 2:
            mat = np.column_stack([self.excess(ctx, self.run(ctx, c, end).returns, win) for c in configs])
            pbo, _ = pbo_cscv(mat, th["pbo_blocks"])
            m["pbo"], m["pbo_configs"] = pbo, len(configs)
            if th.get("max_pbo") is not None:
                checks["pbo"] = _check(pbo, "<=", th["max_pbo"])
        return _outcome("g3", m, checks, [])

    # ------------------------------------------------------------------ G4 holdout

    def _g4(self, lab, ctx: Context, th) -> Outcome:
        params = self.primary(ctx.card)
        dev_run, dev_bench = self.run(ctx, params, ctx.dev_end), self.benchmark(ctx, ctx.dev_end)
        dev = self._scores(ctx, dev_run, dev_bench, self.window(ctx, dev_run))
        full, bench = self.run(ctx, params, ctx.data_end), self.benchmark(ctx, ctx.data_end)
        win = self.window(ctx, full, start=ctx.dev_end)
        if win.stop - win.start < 60:
            return Outcome(False, {"error": "holdout shorter than 60 rows"}, "g4_too_short")
        s = self._scores(ctx, full, bench, win)
        attempts = lab.family_holdout_attempts(ctx.card["family"])
        levels = th["sharpe_diff_ci_levels"]
        level = levels[min(attempts, len(levels) - 1)]
        lower = metrics.bootstrap_sharpe_diff_lower(self.excess(ctx, full.returns, win), self.excess(ctx, bench, win),
                                                    level, 2000, 20, ppy=s["ppy"])
        stressed = self.run(ctx, params, ctx.data_end, cost_mult=2.0)
        checks = {
            "sharpe_vs_benchmark": _check(s["sharpe"] - s["bench_sharpe"], ">=", th["min_sharpe_vs_benchmark"]),
            "sharpe_vs_dev": _check(s["sharpe"], ">=", th["min_sharpe_vs_dev"] * dev["sharpe"]),
            "costs_x2_sharpe_vs_benchmark": _check(self.sr(ctx, stressed.returns, win, s["ppy"]) - s["bench_sharpe"],
                                                   ">=", th["cost_stress_min_sharpe_vs_benchmark"]),
            "max_dd_vs_dev": _check(s["max_dd"], "<=", th["max_drawdown_vs_dev"] * dev["max_dd"]),
            "sharpe_diff_ci_lower": _check(lower, ">", 0.0),
        }
        return _outcome("g4", s | {"dev_sharpe": dev["sharpe"], "ci_level": level, "family_attempts_before": attempts},
                        checks, [])


# ---------------------------------------------------------------------- helpers

def _check(value, op: str, threshold) -> dict:
    v = float(value)
    ok = {">=": v >= threshold, ">": v > threshold, "<=": v <= threshold}[op] and np.isfinite(v)
    return {"value": round(v, 6) if np.isfinite(v) else str(v), "op": op, "threshold": round(float(threshold), 6),
            "pass": bool(ok)}


def _na(reason: str) -> dict:
    return {"pass": True, "not_applicable": reason}


def _outcome(prefix: str, metrics_: dict, checks: dict, trials: list) -> Outcome:
    failed = [k for k, c in checks.items() if not c["pass"]]
    clean = json.loads(json.dumps(metrics_ | {"checks": checks}, default=_jsonable))
    return Outcome(not failed, clean, f"{prefix}_{failed[0]}" if failed else None, trials)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.datetime64, date)):
        return str(o)
    return str(o)


def _per_period(x: np.ndarray) -> float:
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd) if sd > 0 else 0.0


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) else v
    return out



# ---------------------------------------------------------------------------- instruments of a card

def strategy_instruments(card: dict) -> dict[str, list[str] | None]:
    """dataset -> the strategy's instruments from it (None = every instrument, for crypto_top_n).

    Only tradable datasets give instruments. A requirement's own `instruments`, else the universe's
    instruments for the first tradable requirement. Several
    requirements on the same dataset add up (a card may list SPY and IEF as two requirements with
    different periods); a later requirement without instruments adds nothing."""
    universe, per_ds = card["universe"], {}
    for r in card["data_requirements"]:
        ds = r["dataset"]
        if ds not in TRADABLE:
            continue   # signal-only dataset (Archivist ingest): series, not instruments
        inst = r.get("instruments") or (universe.get("instruments") if not per_ds else None)
        if universe["kind"] == "crypto_top_n" and ds == "binance_spot_1d":
            inst = None
        if universe["kind"] in ("liq_n", "sp500") and ds == "sharadar_sep":
            inst = None
        if ds not in per_ds:
            per_ds[ds] = list(inst) if inst else None
        elif inst and per_ds[ds] is not None:
            per_ds[ds] += [i for i in inst if i not in per_ds[ds]]
    return per_ds


# ---------------------------------------------------------------------------- G0 on any view

def integrity(card: dict, path, view: DataView, strat_cols: list[str], params: dict, end: int, th: dict,
              runtime_limit_s: float) -> Outcome:
    """G0: static scan, crash/timeout, weight schema, universe, determinism, look-ahead (truncation +
    perturbation of qlab.validation.leakage). The gate runner calls it on the real dev rows; `lab try` calls it
    on a synthetic market inside the Builder's workspace, so the Builder never needs real data to pass G0."""
    source = path.read_text() if path.exists() else ""
    universe = set(card["universe"].get("instruments") or strat_cols)
    problems = strategy.scan(source, universe, set(view.instruments))
    m = {"problems": problems}
    if not source:
        return Outcome(False, m | {"problems": ["strategy.py missing"]}, "g0_missing")
    if problems:
        return Outcome(False, m, "g0_static")
    sv = view.slice(end).columns(strat_cols)

    def targets(v: DataView) -> np.ndarray:
        module = strategy.load(path)
        with strategy.time_limit(runtime_limit_s):
            w = np.asarray(module.target_weights(v, dict(params)), dtype=float)
        if w.shape != v.shape:
            raise ValueError(f"target_weights returned shape {w.shape}, expected {v.shape}")
        return w

    t0 = time.monotonic()
    try:
        w = targets(sv)
    except strategy.StrategyTimeout as e:
        return Outcome(False, m | {"error": str(e)}, "g0_timeout")
    except Exception as e:  # noqa: BLE001 - any crash of agent code is a G0 failure
        return Outcome(False, m | {"error": f"{type(e).__name__}: {e}"}, "g0_crash")
    m["runtime_s"] = round(time.monotonic() - t0, 3)
    errors = engine.validate_targets(w, w.shape)
    decided = ~np.isnan(w).all(axis=1)
    if errors:
        return Outcome(False, m | {"error": "; ".join(errors)}, "g0_weights")
    if not decided.any():
        return Outcome(False, m | {"error": "no decision row"}, "g0_no_decisions")
    if sv.universe is not None:
        outside = decided[:, None] & ~sv.universe & (np.abs(np.nan_to_num(w)) > 1e-12)
        if outside.any():
            t, j = np.argwhere(outside)[0]
            return Outcome(False, m | {"error": f"weight on {strat_cols[j]} outside the universe on "
                                                 f"{sv.dates[t]}"}, "g0_outside_universe")
    if not np.array_equal(w, targets(sv), equal_nan=True):
        return Outcome(False, m, "g0_nondeterministic")
    bad = _lookahead(targets, sv, w, th.get("leakage_cut_days", 12))
    m["lookahead_failures"] = bad
    if bad["truncation"] or bad["perturbation"]:
        return Outcome(False, m, "g0_lookahead")
    return Outcome(True, m)


def _lookahead(targets, view: DataView, full: np.ndarray, n_cuts: int) -> dict:
    """Truncation + perturbation test of qlab.validation.leakage, on the DataView."""
    rng = np.random.default_rng(0)
    end = len(view.dates)
    decided = np.flatnonzero(~np.isnan(full).all(axis=1))
    lo = int(decided[0]) if len(decided) else 1
    cuts = set(rng.integers(lo, end - 1, n_cuts).tolist()) | {end - 2}
    # sparse strategies (weekly/monthly decisions) only reveal a peek on the rows where they decide:
    # add as many cuts on rows where the weights change
    held = np.where(np.isnan(full), 0.0, full)
    changes = np.flatnonzero(np.abs(np.diff(held, axis=0)).sum(axis=1) > 1e-12) + 1
    changes = changes[(changes >= lo) & (changes < end - 1)]
    if len(changes):
        cuts |= set(rng.choice(changes, min(n_cuts, len(changes)), replace=False).tolist())
    cuts = sorted(cuts)
    same = lambda a, b: np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True)  # noqa: E731
    trunc, pert = [], []
    for t in cuts:
        if not same(targets(view.slice(t + 1))[t], full[t]):
            trunc.append(str(view.dates[t]))
        if not same(targets(data.perturb_after(view, t, rng))[: t + 1], full[: t + 1]):
            pert.append(str(view.dates[t]))
    return {"cuts": len(cuts), "truncation": trunc, "perturbation": pert}
