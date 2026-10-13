"""The scalping branch: strategies on 15m (later 5m) crypto perpetual candles, judged by deterministic gates.

A card is intraday when `signal.timeframes` contains "5m" or "15m" (`is_intraday`). Everything else in the lab
stays daily. The branch reuses the funnel (G0 -> G4, same states, same trial counting, same holdout rule); only the
data, the engine and the checks differ, because a scalper is judged by its trades and its costs, not by daily
Sharpe against a benchmark:

    strategy file     lab/strategies/<id>/strategy.py   def target_positions(data, params) -> (T, N)
    data              IntradayView (intraday_data.py), dev = rows before 2023-01-01, holdout = G4 only
    engine            `simulate`: vectorized, constant-fraction weights, fill at the next candle's open
    costs             taker fee + slippage by trailing liquidity, per side, charged on every change of position
    benchmark         cash (a scalper must earn its keep in absolute terms)
    controls (G2)     random timing (positions circularly shifted per instrument), +1 candle delay, 2x costs,
                      parameter neighbors, three sub-periods
    forward data      none (the paper bots are daily): such a card can pass at most G4

Timing: row t is a candle that has closed; the decision of row t is filled at the OPEN of candle t + 1 (+ `delay`).
"""

import hashlib
import json
import time
from dataclasses import dataclass

import numpy as np

from qlab.validation.stats import expected_max_sharpe, pbo_cscv, probabilistic_sharpe

from lab.framework import metrics, strategy
from lab.framework.evaluator import QlabEvaluator, _check, _merge, _outcome, _per_period
from lab.framework.gates import Outcome, Trial, strategy_sha
from lab.framework.intraday_data import FIELDS, HOLDOUT_FROM, IntradayView

INTRADAY_TF = {"5m", "15m"}
ALLOWED_IMPORTS = strategy.ALLOWED_IMPORTS | {"lab.framework.intraday_api"}
WEIGHT_TOL = 1e-9


def is_intraday(card: dict) -> bool:
    return bool(INTRADAY_TF & set((card.get("signal") or {}).get("timeframes") or []))


def card_tf(card: dict) -> str:
    tfs = INTRADAY_TF & set(card["signal"]["timeframes"])
    return "5m" if "5m" in tfs else "15m"                     # the finest timeframe the signal uses


# ---------------------------------------------------------------------------- costs

def cost_rates(view: IntradayView, cfg: dict, multiplier: float = 1.0) -> np.ndarray:
    """(T, N) cost per unit of traded value: fee + slippage tier by the 30-day median daily quote volume up to the
    previous UTC day (point-in-time)."""
    bpd = view.bars_per_day
    days = view.shape[0] // bpd
    daily = view.qv[: days * bpd].reshape(days, bpd, -1).sum(axis=1).astype(float)
    window = 30
    med = np.full(daily.shape, np.nan)
    for d in range(1, days):
        med[d] = np.median(daily[max(0, d - window):d], axis=0)
    slip = np.full(daily.shape, cfg["tiers"][-1][1])
    for floor, bps in reversed(cfg["tiers"]):
        slip = np.where(med >= floor, bps, slip)
    slip = np.where(np.isfinite(med), slip, cfg["tiers"][-1][1])
    bps = np.repeat(cfg["fee_bps"] + slip, bpd, axis=0)
    full = np.full(view.shape, cfg["fee_bps"] + cfg["tiers"][-1][1])
    full[: len(bps)] = bps
    return multiplier * full / 1e4


# ---------------------------------------------------------------------------- engine

def _ffill_rows(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaN along axis 0; leading NaN -> 0 (flat). Only the columns that contain NaN are touched."""
    nan = np.isnan(a)
    cols = np.flatnonzero(nan.any(axis=0))
    if not len(cols):
        return a
    out = np.array(a, copy=True)
    sub, ok = a[:, cols], ~nan[:, cols]
    idx = np.where(ok, np.arange(len(a))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    filled = np.take_along_axis(np.where(ok, sub, 0.0), idx, axis=0)
    first = np.where(ok.any(axis=0), ok.argmax(axis=0), len(a))
    filled[np.arange(len(a))[:, None] < first[None, :]] = 0.0
    out[:, cols] = filled
    return out


@dataclass
class Sim:
    returns: np.ndarray        # (T,) net return per candle
    pos: np.ndarray            # (T, N) position held during candle t (after the fill at its open)
    contrib: np.ndarray        # (T, N) net contribution per instrument
    turnover: np.ndarray       # (T,) sum |change of position|
    cost: np.ndarray           # (T,) costs paid


def validate_targets(w: np.ndarray, shape: tuple[int, int]) -> list[str]:
    if w.shape != shape:
        return [f"targets shape {w.shape} != {shape}"]
    rows = w[~np.isnan(w).all(axis=1)]
    errors = []
    if np.isinf(rows).any():
        errors.append("infinite weight")
    if (np.nansum(np.abs(rows), axis=1) > 1 + WEIGHT_TOL).any():
        errors.append("gross exposure above 1 (no leverage)")
    return errors


def simulate(view: IntradayView, targets: np.ndarray, rates: np.ndarray, delay: int = 0) -> Sim:
    """Targets -> returns. Decision at the close of candle t is filled at the open of candle t + 1 + delay; where
    the instrument cannot trade the position is carried. A position is a constant fraction of equity (weights
    are not rebalanced for drift inside a trade: scalping holds are short). Funding is paid by longs, received by
    shorts, on the candle that contains a settlement."""
    errors = validate_targets(targets, view.shape)
    if errors:
        raise ValueError("; ".join(errors))
    held = _ffill_rows(targets)                       # the target in force after the close of row t
    want = np.zeros_like(held)
    k = 1 + delay
    want[k:] = held[:-k] if k < len(held) else 0.0    # what the fill at the open of row t aims at
    tradable = view.tradable
    pos = _ffill_rows(np.where(tradable, want, np.nan))
    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    gap = prev * view.ret_co + pos * view.ret_oc
    cost = np.abs(pos - prev) * rates
    contrib = gap - cost - pos * np.nan_to_num(view.funding)
    return Sim(contrib.sum(axis=1), pos, contrib, np.abs(pos - prev).sum(axis=1), cost.sum(axis=1))


def trades(sim: Sim) -> dict:
    """Per-trade statistics from the positions: a trade is a run of one sign in one instrument. Returns arrays
    `ret` (net return on the position, fraction), `bars` (candles held), `side`, `entry_row`."""
    pos, contrib = sim.pos, sim.contrib
    prev = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    out = {"ret": [], "bars": [], "side": [], "entry_row": []}
    for j in range(pos.shape[1]):
        p, q = pos[:, j], prev[:, j]
        active = (p != 0) | (q != 0)
        if not active.any():
            continue
        start = (p != 0) & (np.sign(p) != np.sign(q))
        tid = np.cumsum(start)
        sel = active & (tid > 0)
        n = tid.max()
        pnl = np.bincount(tid[sel], weights=contrib[sel, j], minlength=n + 1)[1:]
        bars = np.bincount(tid[sel], minlength=n + 1)[1:]
        rows = np.flatnonzero(start)
        size = np.abs(p[rows])
        out["ret"].append(pnl / np.where(size > 0, size, 1.0))
        out["bars"].append(bars)
        out["side"].append(np.sign(p[rows]))
        out["entry_row"].append(rows)
    return {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items()}


def random_timing(pos_targets: np.ndarray, rng, min_shift: int) -> np.ndarray:
    """Null for the entry timing: every instrument's target series circularly shifted by its own random offset.
    Keeps the number of trades, their lengths and sides; destroys the link to the market. Rows whose shifted
    positions add up to more than gross 1 are scaled down (the null may not use leverage either)."""
    from lab.framework.intraday_api import normalize_gross
    T = len(pos_targets)
    out = np.empty_like(pos_targets)
    for j in range(pos_targets.shape[1]):
        out[:, j] = np.roll(pos_targets[:, j], int(rng.integers(min_shift, max(min_shift + 1, T - min_shift))))
    return normalize_gross(out)


# ---------------------------------------------------------------------------- synthetic market (lab try, tests)

SYN_NAMES = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT", "FFFUSDT")


def synthetic_view(tf: str = "15m", days: int = 90, names=SYN_NAMES, seed: int = 7) -> IntradayView:
    rng = np.random.default_rng(seed)
    step = {"15m": 15, "5m": 5}[tf]
    bpd = 1440 // step
    T, N = days * bpd, len(names)
    vol = rng.uniform(0.0008, 0.0025, N) * np.sqrt(15.0 / step)
    r = rng.standard_t(4, (T, N)) * vol / np.sqrt(2.0)
    r += 0.15 * np.roll(r.mean(axis=1, keepdims=True), 1, axis=0)             # a little common factor
    c = 100 * np.exp(np.cumsum(r, axis=0))
    o = np.vstack([c[:1], c[:-1]]) * np.exp(rng.normal(0, vol / 10, (T, N)))
    h = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, (T, N))))
    l = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, (T, N))))
    qv = np.exp(rng.normal(14.5, 0.8, (T, N))) * (1 + 0.5 * np.sin(np.arange(T) * 2 * np.pi / bpd))[:, None]
    nt = (qv / 2000).round()
    tbq = qv * np.clip(0.5 + 0.2 * np.tanh(r / vol), 0.05, 0.95)
    funding = np.zeros((T, N))
    funding[:: 8 * 60 // step] = 1e-4
    times = np.datetime64("2024-01-01T00:00", "m") + np.arange(T) * np.timedelta64(step, "m")
    real = np.ones((T, N), dtype=bool)
    f = lambda x: x.astype(np.float32)  # noqa: E731
    return IntradayView(times=times, tf=tf, instruments=tuple(names), o=f(o), h=f(h), l=f(l), c=f(c), qv=f(qv),
                        nt=f(nt), tbq=f(tbq), funding=f(funding), real=real, listed=real.copy(),
                        universe=real.copy())


# ---------------------------------------------------------------------------- G0 integrity

def _perturbed(view: IntradayView, cut: int, rng) -> IntradayView:
    """Rows after `cut` replaced by noise (prices rescaled, volumes redrawn)."""
    from dataclasses import replace
    n = len(view.times) - cut - 1
    fields = {}
    for k in FIELDS:
        x = np.array(getattr(view, k), copy=True)
        if k in ("o", "h", "l", "c"):
            x[cut + 1:] *= np.exp(rng.normal(0, 0.05, (n, x.shape[1]))).astype(x.dtype)
        elif k != "funding":
            x[cut + 1:] *= rng.uniform(0.2, 5.0, (n, x.shape[1])).astype(x.dtype)
        fields[k] = x
    return replace(view, **fields)


def integrity(card: dict, path, view: IntradayView, params: dict, th: dict, runtime_limit_s: float = 1200.0,
              instruments_all: set | None = None) -> Outcome:
    source = path.read_text() if path.exists() else ""
    if not source:
        return Outcome(False, {"problems": ["strategy.py missing"]}, "g0_missing")
    allowed = set(card["universe"].get("instruments") or view.instruments)
    problems = strategy.scan(source, allowed, set(instruments_all or view.instruments), imports=ALLOWED_IMPORTS)
    m = {"problems": problems}
    if problems:
        return Outcome(False, m, "g0_static")

    def targets(v: IntradayView) -> np.ndarray:
        module = strategy.load(path, entry="target_positions")
        with strategy.time_limit(runtime_limit_s):
            w = np.asarray(module.target_positions(v, dict(params)), dtype=float)
        if w.shape != v.shape:
            raise ValueError(f"target_positions returned shape {w.shape}, expected {v.shape}")
        return w

    t0 = time.monotonic()
    try:
        w = targets(view)
    except strategy.StrategyTimeout as e:
        return Outcome(False, m | {"error": str(e)}, "g0_timeout")
    except Exception as e:  # noqa: BLE001 - any crash of agent code is a G0 failure
        return Outcome(False, m | {"error": f"{type(e).__name__}: {e}"}, "g0_crash")
    m["runtime_s"] = round(time.monotonic() - t0, 3)
    errors = validate_targets(w, w.shape)
    if errors:
        return Outcome(False, m | {"error": "; ".join(errors)}, "g0_weights")
    decided = ~np.isnan(w).all(axis=1)
    if not decided.any():
        return Outcome(False, m | {"error": "no decision row"}, "g0_no_decisions")
    outside = decided[:, None] & ~view.universe & (np.abs(np.nan_to_num(w)) > 1e-12)
    if outside.any():
        t, j = np.argwhere(outside)[0]
        return Outcome(False, m | {"error": f"position in {view.instruments[j]} outside the universe at row {t}"},
                       "g0_outside_universe")
    if not np.array_equal(w, targets(view), equal_nan=True):
        return Outcome(False, m, "g0_nondeterministic")
    rng = np.random.default_rng(2026)
    T = len(view.times)
    cuts = np.sort(rng.integers(T // 4, T - 2, th.get("leakage_cuts", 8)))
    bad = {"truncation": [], "perturbation": []}
    for cut in cuts:
        cut = int(cut)
        if not np.array_equal(targets(view.truncate(cut + 1))[: cut + 1], w[: cut + 1], equal_nan=True):
            bad["truncation"].append(cut)
        if not np.array_equal(targets(_perturbed(view, cut, rng))[: cut + 1], w[: cut + 1], equal_nan=True):
            bad["perturbation"].append(cut)
    m["lookahead_failures"] = bad
    if bad["truncation"] or bad["perturbation"]:
        return Outcome(False, m, "g0_lookahead")
    return Outcome(True, m)


# ---------------------------------------------------------------------------- the evaluator

class IntradayEvaluator:
    """Gate runner of the scalping branch. Same `evaluate(lab, hid, card, gate, th)` protocol as QlabEvaluator."""

    name = "qlab-intraday"

    def __init__(self, cfg: dict | None = None, view_loader=None):
        self.cfg_override = cfg or {}
        self.view_loader = view_loader or self._load_view
        self._cache: dict = {}
        self._rates: dict = {}

    # data ---------------------------------------------------------------------------------------
    @staticmethod
    def _load_view(card: dict, stage: str) -> IntradayView:
        from lab.framework import data, intraday_data
        tf = card_tf(card)
        uni = card["universe"]
        top_n = int(uni.get("n") or 20)
        out_dir = data.DATA_ROOT / "derived" / "binance_2026-10-03" / f"lab_intraday_{tf}"
        view = intraday_data.load(out_dir, top_n=top_n, stage=stage, symbols=uni.get("instruments") or None)
        keep = np.flatnonzero(view.universe.any(axis=0))              # columns that are ever members
        return _columns(view, keep)

    def context(self, card: dict, stage: str = "dev") -> IntradayView:
        key = (json.dumps(card["universe"], sort_keys=True), card_tf(card), stage)
        if key not in self._cache:
            self._cache = {key: self.view_loader(card, stage)}
        return self._cache[key]

    # runs ---------------------------------------------------------------------------------------
    @staticmethod
    def primary(card: dict) -> dict:
        return {k: p["value"] for k, p in card.get("signal", {}).get("params", {}).items()}

    def targets(self, lab, hid: str, view: IntradayView, params: dict) -> np.ndarray:
        module = strategy.load(lab.paths.strategies / hid / "strategy.py", entry="target_positions")
        with strategy.time_limit(1200.0):
            return np.asarray(module.target_positions(view, dict(params)), dtype=float)

    def rates(self, view, cfg, mult: float) -> np.ndarray:
        key = (id(view), mult)
        if key not in self._rates:
            self._rates = {k: v for k, v in self._rates.items() if k[0] == id(view)} | {key: cost_rates(view, cfg["cost"], mult)}
        return self._rates[key]

    def run(self, lab, hid, view, params, cfg, cost_mult=1.0, delay=0, w=None) -> Sim:
        w = self.targets(lab, hid, view, params) if w is None else w
        return simulate(view, w, self.rates(view, cfg, cost_mult), delay)

    def config_sha(self, lab, hid: str, params: dict) -> str:
        return hashlib.sha256(json.dumps({"hid": hid, "code": strategy_sha(lab, hid), "params": params},
                                         sort_keys=True).encode()).hexdigest()[:16]

    @staticmethod
    def window(w: np.ndarray, n: int, start: int = 0) -> slice:
        decided = np.flatnonzero(~np.isnan(w).all(axis=1))
        first = int(decided[0]) + 1 if len(decided) else n
        return slice(max(first, start), n)

    def neighbors(self, card: dict, steps: int = 1, max_combined: int = 3) -> list[dict]:
        return QlabEvaluator.neighbors(self, card, steps, max_combined)

    def ingest(self, lab, payload):                                 # Archivist ingest is for daily datasets
        raise NotImplementedError("intraday data is built by the framework (lab intraday-build)")

    # gates --------------------------------------------------------------------------------------
    def evaluate(self, lab, hid: str, card: dict, gate: str, th: dict) -> Outcome:
        cfg = _merge(lab_cfg(lab), self.cfg_override)
        if gate in ("G5",):
            return Outcome(False, {"error": "no forward data for intraday cards"}, "g5_no_forward_data")
        fn = {"G0": self._g0, "G1": self._g1, "G2": self._g2, "G3": self._g3, "G4": self._g4}[gate]
        try:
            return fn(lab, hid, card, _merge(cfg.get(gate, {}), {}), cfg)
        except FileNotFoundError as e:                       # e.g. the 5m panel has not been built
            lab.send("ALERT", "gatekeeper", "human", hid, {"severity": "warning", "text": f"{gate}: {e}"})
            return Outcome(False, {"error": str(e)}, f"{gate.lower()}_no_data_adapter")

    def _g0(self, lab, hid, card, th, cfg) -> Outcome:
        view = self.context(card)
        return integrity(card, lab.paths.strategies / hid / "strategy.py", view, self.primary(card), th)

    def _stats(self, sim: Sim, win: slice, ppy: float) -> dict:
        r = sim.returns[win]
        tr = trades(Sim(sim.returns[win], sim.pos[win], sim.contrib[win], sim.turnover[win], sim.cost[win]))
        n = len(tr["ret"])
        return {"sharpe": metrics.sharpe(r, ppy), "max_dd": metrics.max_dd(r), "n_trades": n,
                "mean_trade_net": float(tr["ret"].mean()) if n else 0.0,
                "median_bars_held": float(np.median(tr["bars"])) if n else 0.0,
                "mean_exposure": float((np.abs(sim.pos[win]).sum(axis=1) > 0).mean()),
                "turnover_per_day": float(sim.turnover[win].sum() / max(len(r) / ppy * 365.0, 1e-9)),
                "cost_share_of_gross": float(sim.cost[win].sum() / max(abs(sim.contrib[win].sum() + sim.cost[win].sum()), 1e-12)),
                "ppy": ppy, "rows": int(win.stop - win.start), "_trades": tr}

    def _g1(self, lab, hid, card, th, cfg) -> Outcome:
        view, params = self.context(card), self.primary(card)
        w = self.targets(lab, hid, view, params)
        sim = self.run(lab, hid, view, params, cfg, w=w)
        win = self.window(w, len(view.times))
        s = self._stats(sim, win, view.periods_per_year())
        tr = s.pop("_trades")
        lower = (metrics.bootstrap_mean_lower(tr["ret"], th["trade_ci_level"], th["bootstrap_samples"], 1.5)
                 if len(tr["ret"]) > 20 else float("-inf"))
        checks = {
            "trades": _check(s["n_trades"], ">=", th["min_trades"]),
            "sharpe": _check(s["sharpe"], ">=", th["min_sharpe"]),
            "mean_trade_net_ci_lower": _check(lower, ">", 0.0),
            "max_drawdown": _check(s["max_dd"], "<=", th["max_drawdown"]),
        }
        sharpe_pp = _per_period(sim.returns[win])
        return _outcome("g1", s | {"benchmark": "cash", "mean_trade_net_ci_lower": lower}, checks,
                        [Trial(self.config_sha(lab, hid, params), sharpe_pp, win.stop - win.start)])

    def _g2(self, lab, hid, card, th, cfg) -> Outcome:
        view, params = self.context(card), self.primary(card)
        w = self.targets(lab, hid, view, params)
        base = self.run(lab, hid, view, params, cfg, w=w)
        win = self.window(w, len(view.times))
        ppy = view.periods_per_year()
        sr = lambda sim: metrics.sharpe(sim.returns[win], ppy)  # noqa: E731
        s0 = sr(base)
        checks, m, trials = {}, {"sharpe": s0}, []
        nb = self.neighbors(card, th["params"]["neighbor_steps"], th["params"]["max_combined_params"])
        nb_sharpes = []
        for cfg_n in nb:
            sim = self.run(lab, hid, view, cfg_n, cfg)
            nb_sharpes.append(sr(sim))
            trials.append(Trial(self.config_sha(lab, hid, cfg_n), _per_period(sim.returns[win]), win.stop - win.start))
        m["neighbors"] = [{"params": c, "sharpe": round(s, 4)} for c, s in zip(nb, nb_sharpes)]
        if nb:
            checks["neighbors_positive"] = _check(float(np.mean(np.array(nb_sharpes) > 0)), ">=",
                                                  th["params"]["min_share_positive"])
            ratio = float(np.median(nb_sharpes) / s0) if s0 > 0 else 0.0
            checks["median_neighbor_sharpe_ratio"] = _check(ratio, ">=", th["params"]["min_median_sharpe_ratio"])
        delayed = self.run(lab, hid, view, params, cfg, delay=th["delay"]["bars"], w=w)
        m["sharpe_delayed"] = sr(delayed)
        checks["delay_sharpe_ratio"] = _check(m["sharpe_delayed"] / s0 if s0 > 0 else 0.0, ">=",
                                              th["delay"]["min_sharpe_ratio"])
        stressed = self.run(lab, hid, view, params, cfg, cost_mult=cfg["cost"]["stress_multiplier"], w=w)
        m["sharpe_costs_x2"] = sr(stressed)
        checks["sharpe_costs_x2"] = _check(m["sharpe_costs_x2"], ">", 0.0)
        # three sub-periods of the dev window
        edges = np.linspace(win.start, win.stop, th["subperiods"]["n_blocks"] + 1).astype(int)
        blocks = [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
        bs = [metrics.sharpe(base.returns[b], ppy) for b in blocks]
        logs = np.array([np.log1p(base.returns[b]).sum() for b in blocks])
        share = float(logs.max() / logs.sum()) if logs.sum() > 0 else 1.0
        m["block_sharpes"] = [round(x, 3) for x in bs]
        checks["blocks_positive"] = _check(sum(x > 0 for x in bs), ">=", th["subperiods"]["min_blocks_positive"])
        checks["max_block_share_of_log_return"] = _check(share, "<=", th["subperiods"]["max_block_share"])
        # random timing null
        rng = np.random.default_rng(12345)
        shift = view.bars_per_day * 5
        null = []
        for _ in range(th["random_entry"]["runs"]):
            fake = random_timing(np.nan_to_num(_ffill_rows(w)), rng, shift)
            null.append(sr(self.run(lab, hid, view, params, cfg, w=fake)))
        pct = float(np.mean(np.array(null) < s0)) if null else 0.0
        m["random_timing"] = {"runs": len(null), "median_sharpe": float(np.median(null)) if null else None,
                              "p95_sharpe": float(np.quantile(null, 0.95)) if null else None}
        checks["random_timing_percentile"] = _check(pct, ">=", th["random_entry"]["min_percentile"])
        return _outcome("g2", m, checks, trials)

    def _g3(self, lab, hid, card, th, cfg) -> Outcome:
        view, params = self.context(card), self.primary(card)
        w = self.targets(lab, hid, view, params)
        sim = self.run(lab, hid, view, params, cfg, w=w)
        win = self.window(w, len(view.times))
        ex = sim.returns[win]
        family = card["family"]
        family_sharpes = np.array(lab.family_trial_sharpes(family), dtype=float)
        n_family = lab.family_trials(family)
        prior = th["literature_prior_trials"] if card.get("references") else 0
        n = max(n_family + prior, th["min_trials"])
        var = max(float(family_sharpes.var(ddof=1)) if len(family_sharpes) > 1 else 0.0, 1.0 / len(ex))
        hurdle = expected_max_sharpe(n, var)
        dsr = probabilistic_sharpe(ex, hurdle)
        m = {"n_family_trials": n_family, "literature_prior": prior, "n_trials": n, "sr_variance": var,
             "sharpe_per_period": _per_period(ex), "hurdle_per_period": hurdle, "dsr": dsr}
        configs = [params] + self.neighbors(card)
        if len(configs) >= 2:
            mat = np.column_stack([self.run(lab, hid, view, c, cfg).returns[win] for c in configs])
            m["pbo"], m["pbo_configs"] = pbo_cscv(mat, th["pbo_blocks"])[0], len(configs)
        return _outcome("g3", m, {"dsr": _check(dsr, ">=", th["min_dsr"])}, [])

    def _g4(self, lab, hid, card, th, cfg) -> Outcome:
        params = self.primary(card)
        dev_view = self.context(card, "dev")
        dev_w = self.targets(lab, hid, dev_view, params)
        dev = self._stats(self.run(lab, hid, dev_view, params, cfg, w=dev_w),
                          self.window(dev_w, len(dev_view.times)), dev_view.periods_per_year())
        view = self.context(card, "full")
        w = self.targets(lab, hid, view, params)
        start = int((view.times < np.datetime64(HOLDOUT_FROM, "m")).sum())
        win = slice(start, len(view.times))
        if win.stop - win.start < view.bars_per_day * 60:
            return Outcome(False, {"error": "holdout shorter than 60 days"}, "g4_too_short")
        sim = self.run(lab, hid, view, params, cfg, w=w)
        stressed = self.run(lab, hid, view, params, cfg, cost_mult=cfg["cost"]["stress_multiplier"], w=w)
        s = self._stats(sim, win, view.periods_per_year())
        tr = s.pop("_trades")
        attempts = lab.family_holdout_attempts(card["family"])
        level = th["trade_ci_levels"][min(attempts, len(th["trade_ci_levels"]) - 1)]
        lower = (metrics.bootstrap_mean_lower(tr["ret"], level, 2000, 1.5) if len(tr["ret"]) > 20 else float("-inf"))
        checks = {
            "sharpe": _check(s["sharpe"], ">=", th["min_sharpe"]),
            "sharpe_vs_dev": _check(s["sharpe"], ">=", th["min_sharpe_vs_dev"] * dev["sharpe"]),
            "costs_x2_sharpe": _check(metrics.sharpe(stressed.returns[win], s["ppy"]), ">", 0.0),
            "max_dd_vs_dev": _check(s["max_dd"], "<=", th["max_drawdown_vs_dev"] * max(dev["max_dd"], 1e-9)),
            "mean_trade_net_ci_lower": _check(lower, ">", 0.0),
        }
        return _outcome("g4", s | {"dev_sharpe": dev["sharpe"], "ci_level": level,
                                   "family_attempts_before": attempts}, checks, [])


def lab_cfg(lab) -> dict:
    import yaml
    return yaml.safe_load(lab.paths.gates.read_text())["intraday"]


def _columns(view: IntradayView, keep) -> IntradayView:
    from dataclasses import replace
    keep = list(keep)
    kw = {k: getattr(view, k)[:, keep] for k in (*FIELDS, "real", "listed", "universe")}
    return replace(view, instruments=tuple(view.instruments[k] for k in keep), **kw)


# ---------------------------------------------------------------------------- `lab try` for the Builder

def dry_run(ws) -> tuple[bool, list[str]]:
    """G0 integrity and the grid neighbors on a synthetic market in the Builder's workspace (no real data)."""
    import yaml
    from lab.framework import dryrun
    card = yaml.safe_load((ws / "card.yaml").read_text())
    cfg = yaml.safe_load((ws / "gates.yaml").read_text())["intraday"]
    path = ws / "strategy" / "strategy.py"
    names = tuple(card["universe"].get("instruments") or SYN_NAMES)
    tf = card_tf(card)
    view = synthetic_view(tf, days=60, names=names, seed=int(hashlib.sha256(
        json.dumps(card.get("signal", {}), sort_keys=True, default=str).encode()).hexdigest()[:8], 16))
    params = IntradayEvaluator.primary(card)
    lines = [f"synthetic market: {len(names)} perpetuals, {tf} candles, {len(view.times)} rows (60 days; random walks "
             f"with fat tails, random volume; no real data is shown)"]
    out = integrity(card, path, view, params, cfg["G0"])
    ok = out.passed
    if ok:
        lines.append(f"G0 integrity on synthetic data: passed ({out.metrics.get('runtime_s')} s, "
                     f"{cfg['G0']['leakage_cuts']} look-ahead cuts)")
        module = strategy.load(path, entry="target_positions")
        w = np.asarray(module.target_positions(view, dict(params)), dtype=float)
        sim = simulate(view, w, cost_rates(view, cfg["cost"]))
        tr = trades(sim)
        decided = int((~np.isnan(w).all(axis=1)).sum())
        lines.append("trading statistics (primary config, synthetic data; check them against the card):")
        lines += [f"  decision rows: {decided}", f"  trades: {len(tr['ret'])} in 60 days "
                  f"(the judge needs >= {cfg['G1']['min_trades']} on the real dev period, about 3 years)",
                  f"  median candles held: {float(np.median(tr['bars'])) if len(tr['ret']) else 0.0}",
                  f"  mean time in the market: {float((np.abs(sim.pos).sum(axis=1) > 0).mean()):.3f}",
                  f"  turnover per day (sum of |change|): {float(sim.turnover.sum() / 60):.2f}"]
        for c in IntradayEvaluator().neighbors(card):
            try:
                with strategy.time_limit(600.0):
                    wn = np.asarray(module.target_positions(view, params | c), dtype=float)
                errs = validate_targets(wn, view.shape)
                if errs:
                    raise ValueError("; ".join(errs))
            except Exception as e:  # noqa: BLE001
                ok = False
                lines.append(f"grid neighbor {c}: FAILED {type(e).__name__}: {e}")
        lines.append(f"grid neighbors: {len(IntradayEvaluator().neighbors(card))} configurations run")
    else:
        lines.append(f"G0 integrity on synthetic data: FAILED {out.reason_code}")
        lines += [f"  {k}: {v}" for k, v in out.metrics.items() if v]
    ok_tests, test_lines = dryrun.run_tests(ws, card, {})
    return ok and ok_tests, lines + test_lines


class RoutingEvaluator:
    """Daily cards go to the QlabEvaluator, cards with a 5m / 15m timeframe to the IntradayEvaluator."""

    name = "qlab"

    def __init__(self, daily=None, intraday=None):
        self.daily = daily or QlabEvaluator()
        self.intraday = intraday or IntradayEvaluator()

    def evaluate(self, lab, hid, card, gate, th):
        return (self.intraday if is_intraday(card) else self.daily).evaluate(lab, hid, card, gate, th)

    def paper_ready(self, lab, hid) -> bool:
        return False if is_intraday(lab.card(hid)) else self.daily.paper_ready(lab, hid)

    def __getattr__(self, item):                        # ingest, paper_forward, paper_run, ... stay daily
        return getattr(self.daily, item)
