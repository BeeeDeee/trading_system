"""Statistiky (SPEC §13). Jen numpy, float. Vstupem jsou řádky tabulky `trades`.

Definice (zapsané i v CHANGELOG P4):
- náklady obchodu v bps = poplatky + entry slip + exit slip − funding (funding je v `net` se znaménkem);
  hrubá hrana `edge_bps = net_bps + cost_bps` (pohyb ceny od rozhodnutí k úrovni výstupu); net při k× nákladech
  = edge − k × cost; break-even = průměrné náklady.
- denní P&L = součet net_pnl obchodů uzavřených ten den UTC, dny bez obchodu = 0 (od prvního do posledního dne).
- Sharpe = mean/sd denního P&L × √365; max DD z kumulativního denního P&L; Calmar = 365 × mean / max DD.
- CI průměru net_bps: blokový bootstrap po dnech UTC (2000 vzorků, seed 20261021).
- Δ vs kontrola: rozdíl průměrů net_bps, bootstrap po dnech (dny se losují společně); p jednostranné P(Δ* ≤ 0).
- Holm přes varianty (bez kontrol); DSR (Bailey & López de Prado 2014) s N = počet variant a rozptylem denních
  Sharpe napříč variantami."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DAY_MS = 86_400_000
SEED = 20261021
N_BOOT = 2000
EULER = 0.5772156649015329
_N = NormalDist()


@dataclass(frozen=True)
class T:
    """Zjednodušený obchod pro statistiku."""

    slot_id: str
    symbol: str
    variant: str
    family: str
    side: str
    entry_time: int
    exit_time: int
    exit_reason: str
    notional: float
    net_pnl: float
    gross_bps: float
    net_bps: float
    fee_bps: float
    entry_slip_bps: float
    exit_slip_bps: float
    funding_bps: float
    mfe_bps: float
    mae_bps: float
    tainted: int

    @property
    def cost_bps(self) -> float:
        return self.fee_bps + self.entry_slip_bps + self.exit_slip_bps - self.funding_bps

    @property
    def edge_bps(self) -> float:
        return self.net_bps + self.cost_bps

    @property
    def day(self) -> int:
        return self.exit_time // DAY_MS


def from_rows(rows: Iterable[Mapping[str, Any]]) -> list[T]:
    out = []
    for r in rows:
        slot = str(r["slot_id"])
        variant, symbol = slot.rsplit(".", 1)
        n = float(r["notional"]) or 1.0
        fee = (float(r["entry_fee"] or 0) + float(r["exit_fee"] or 0)) / n * 1e4
        out.append(T(slot, symbol, variant, variant.split(".")[0], str(r["side"]), int(r["entry_time"]),
                     int(r["exit_time"]), str(r["exit_reason"]), n, float(r["net_pnl"]), float(r["gross_bps"]),
                     float(r["net_bps"]), fee, float(r["entry_slip_bps"] or 0), float(r["exit_slip_bps"] or 0),
                     float(r["funding"] or 0) / n * 1e4, float(r["mfe_bps"] or 0), float(r["mae_bps"] or 0),
                     int(r["tainted"] or 0)))
    return out


# --- primitiva -----------------------------------------------------------------------------------

def _days_index(trades: Sequence[T]) -> tuple[np.ndarray, list[int]]:
    days = sorted({t.day for t in trades})
    pos = {d: i for i, d in enumerate(days)}
    return np.array([pos[t.day] for t in trades], dtype=np.int64), days


def block_bootstrap_mean(values: np.ndarray, day_idx: np.ndarray, n_days: int, n_boot: int = N_BOOT,
                         seed: int = SEED) -> np.ndarray:
    """Bootstrap průměru: losují se celé dny (s opakováním), průměr přes obchody vylosovaných dnů."""
    rng = np.random.default_rng(seed)
    sums = np.bincount(day_idx, weights=values, minlength=n_days)
    cnts = np.bincount(day_idx, minlength=n_days).astype(float)
    pick = rng.integers(0, n_days, size=(n_boot, n_days))
    s, c = sums[pick].sum(axis=1), cnts[pick].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / c, np.nan)


def daily_pnl(trades: Sequence[T]) -> np.ndarray:
    if not trades:
        return np.zeros(0)
    d0 = min(t.day for t in trades)
    d1 = max(t.day for t in trades)
    out = np.zeros(d1 - d0 + 1)
    for t in trades:
        out[t.day - d0] += t.net_pnl
    return out


def sharpe(daily: np.ndarray, annualize: bool = True) -> float:
    if len(daily) < 2:
        return math.nan
    sd = float(np.std(daily, ddof=1))
    if sd == 0:
        return math.nan
    sr = float(np.mean(daily)) / sd
    return sr * math.sqrt(365) if annualize else sr


def max_drawdown(daily: np.ndarray) -> float:
    if len(daily) == 0:
        return 0.0
    eq = np.cumsum(daily)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    return float(np.max(peak - eq))


def holm(pvals: Mapping[str, float]) -> dict[str, float]:
    """Holm–Bonferroni adjustované p (monotónní). NaN se neúčastní."""
    items = sorted(((p, k) for k, p in pvals.items() if math.isfinite(p)))
    m = len(items)
    out: dict[str, float] = {k: math.nan for k in pvals}
    running = 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float, sr_var: float, n_trials: int) -> float:
    """DSR: P(skutečné SR > SR*), SR* = očekávané maximum z N pokusů při rozptylu `sr_var` (vše denní, ne ročně)."""
    if not all(math.isfinite(x) for x in (sr, skew, kurt, sr_var)) or n_obs < 3 or n_trials < 2 or sr_var < 0:
        return math.nan
    sr0 = math.sqrt(sr_var) * ((1 - EULER) * _N.inv_cdf(1 - 1 / n_trials)
                               + EULER * _N.inv_cdf(1 - 1 / (n_trials * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if denom <= 0:
        return math.nan
    return _N.cdf((sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(denom))


def _moments(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 3 or np.std(x) == 0:
        return math.nan, math.nan
    z = (x - x.mean()) / x.std()
    return float(np.mean(z ** 3)), float(np.mean(z ** 4))


# --- souhrn skupiny ------------------------------------------------------------------------------

@dataclass
class GroupStats:
    key: str
    n: int
    days: int
    trades_per_day: float
    exposure_pct: float
    avg_hold_min: float
    exit_mix: dict[str, int]
    gross_mean: float
    gross_median: float
    net_mean: float
    net_median: float
    net_sd: float
    win_rate: float
    mfe_mean: float
    mae_mean: float
    net_ci: tuple[float, float]
    cost_bps: dict[str, float]
    edge_mean: float
    net_at_1_5x: float
    net_at_2x: float
    break_even_edge: float
    sharpe: float
    sharpe_daily: float
    max_dd: float
    calmar: float
    n_req: dict[str, float]
    days_left: dict[str, float]
    daily_skew: float
    daily_kurt: float
    by_symbol: dict[str, dict[str, float]] = field(default_factory=dict)
    vs_control: dict[str, float] = field(default_factory=dict)
    p_holm: float = math.nan
    dsr: float = math.nan
    badge: str = ""
    monthly_ok: bool = True

    def to_json(self) -> dict[str, Any]:
        def clean(v: Any) -> Any:
            if isinstance(v, float):
                return None if not math.isfinite(v) else round(v, 4)
            if isinstance(v, dict):
                return {k: clean(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [clean(x) for x in v]
            return v
        return {k: clean(v) for k, v in self.__dict__.items()}


def group_stats(key: str, trades: Sequence[T], n_symbols: int = 1, period_ms: int | None = None) -> GroupStats:
    n = len(trades)
    nan = math.nan
    if n == 0:
        return GroupStats(key, 0, 0, 0.0, 0.0, nan, {}, nan, nan, nan, nan, nan, nan, nan, nan, (nan, nan), {}, nan,
                          nan, nan, nan, nan, nan, 0.0, nan, {"z2": nan, "z3": nan}, {"z2": nan, "z3": nan}, nan, nan)
    net = np.array([t.net_bps for t in trades])
    gross = np.array([t.gross_bps for t in trades])
    t0 = min(t.entry_time for t in trades)
    t1 = max(t.exit_time for t in trades)
    span_ms = period_ms or max(t1 - t0, DAY_MS)
    n_days_span = span_ms / DAY_MS
    hold = np.array([t.exit_time - t.entry_time for t in trades], dtype=float)
    day_idx, days = _days_index(trades)
    boot = block_bootstrap_mean(net, day_idx, len(days))
    ci = (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5))) if len(days) >= 2 else (nan, nan)
    cost = {
        "fee": float(np.mean([t.fee_bps for t in trades])),
        "entry_slip": float(np.mean([t.entry_slip_bps for t in trades])),
        "exit_slip": float(np.mean([t.exit_slip_bps for t in trades])),
        "funding": float(np.mean([-t.funding_bps for t in trades])),
    }
    cost["total"] = sum(cost.values())
    edge = float(np.mean([t.edge_bps for t in trades]))
    dp = daily_pnl(trades)
    sd = float(np.std(net, ddof=1)) if n > 1 else nan
    mean = float(np.mean(net))
    tpd = n / n_days_span
    n_req = {f"z{z}": ((z * sd) / mean) ** 2 if mean > 0 and math.isfinite(sd) else math.inf for z in (2, 3)}
    days_left = {k: max(0.0, (v - n) / tpd) if math.isfinite(v) and tpd > 0 else math.inf for k, v in n_req.items()}
    mdd = max_drawdown(dp)
    skew, kurt = _moments(dp)
    mix: dict[str, int] = {}
    for t in trades:
        mix[t.exit_reason] = mix.get(t.exit_reason, 0) + 1
    by_sym: dict[str, dict[str, float]] = {}
    for sym in sorted({t.symbol for t in trades}):
        xs = [t.net_bps for t in trades if t.symbol == sym]
        by_sym[sym] = {"n": len(xs), "net_mean": float(np.mean(xs)),
                       "gross_mean": float(np.mean([t.gross_bps for t in trades if t.symbol == sym]))}
    return GroupStats(
        key=key, n=n, days=len(days), trades_per_day=tpd,
        exposure_pct=float(hold.sum() / (span_ms * n_symbols) * 100),
        avg_hold_min=float(hold.mean() / 60_000), exit_mix=mix,
        gross_mean=float(gross.mean()), gross_median=float(np.median(gross)),
        net_mean=mean, net_median=float(np.median(net)), net_sd=sd,
        win_rate=float((net > 0).mean() * 100), mfe_mean=float(np.mean([t.mfe_bps for t in trades])),
        mae_mean=float(np.mean([t.mae_bps for t in trades])), net_ci=ci, cost_bps=cost, edge_mean=edge,
        net_at_1_5x=edge - 1.5 * cost["total"], net_at_2x=edge - 2 * cost["total"], break_even_edge=cost["total"],
        sharpe=sharpe(dp), sharpe_daily=sharpe(dp, annualize=False), max_dd=mdd,
        calmar=(365 * float(dp.mean()) / mdd) if mdd > 0 else nan, n_req=n_req, days_left=days_left,
        daily_skew=skew, daily_kurt=kurt, by_symbol=by_sym, monthly_ok=_monthly_ok(trades),
    )


def _monthly_ok(trades: Sequence[T]) -> bool:
    """Pravidlo 14.2/5: žádný kalendářní měsíc s net < −3 sd měsíčního P&L."""
    import datetime as dt
    months: dict[str, float] = {}
    for t in trades:
        m = dt.datetime.fromtimestamp(t.exit_time / 1000, dt.timezone.utc).strftime("%Y-%m")
        months[m] = months.get(m, 0.0) + t.net_pnl
    vals = np.array(list(months.values()))
    if len(vals) < 2:
        return True
    sd = float(np.std(vals, ddof=1))
    return bool(np.all(vals >= -3 * sd))


def vs_control(var_trades: Sequence[T], ctl_trades: Sequence[T]) -> dict[str, float]:
    """Δ průměrů net_bps (varianta − kontrola), bootstrap přes společné dny; p = P(Δ* ≤ 0)."""
    if len(var_trades) < 2 or len(ctl_trades) < 2:
        return {"delta": math.nan, "ci_lo": math.nan, "ci_hi": math.nan, "p": math.nan}
    days = sorted({t.day for t in var_trades} | {t.day for t in ctl_trades})
    pos = {d: i for i, d in enumerate(days)}
    nd = len(days)

    def agg(tr: Sequence[T]) -> tuple[np.ndarray, np.ndarray]:
        idx = np.array([pos[t.day] for t in tr])
        return (np.bincount(idx, weights=[t.net_bps for t in tr], minlength=nd),
                np.bincount(idx, minlength=nd).astype(float))

    sv, cv = agg(var_trades)
    sc, cc = agg(ctl_trades)
    rng = np.random.default_rng(SEED + 1)
    pick = rng.integers(0, nd, size=(N_BOOT, nd))
    with np.errstate(invalid="ignore", divide="ignore"):
        d = sv[pick].sum(1) / cv[pick].sum(1) - sc[pick].sum(1) / cc[pick].sum(1)
    d = d[np.isfinite(d)]
    delta = float(np.mean([t.net_bps for t in var_trades]) - np.mean([t.net_bps for t in ctl_trades]))
    return {"delta": delta, "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)),
            "p": float((np.sum(d <= 0) + 1) / (len(d) + 1))}


# --- žebříček ------------------------------------------------------------------------------------

def leaderboard(trades: Sequence[T], variants: Mapping[str, Sequence[str]], live_variants: Iterable[str] = (),
                calibrated: bool = False, period_ms: int | None = None) -> dict[str, GroupStats]:
    """Statistiky všech variant (sloučeně přes symboly) + kontroly, Δ vs kontrola, Holm, DSR a badge.

    `variants`: variant_id → symboly. `calibrated`: nákladový model kalibrovaný ≥ 30 live obchody na symbol."""
    by_var: dict[str, list[T]] = {}
    for t in trades:
        by_var.setdefault(t.variant, []).append(t)
    out = {v: group_stats(v, by_var.get(v, []), len(syms), period_ms) for v, syms in variants.items()}
    real = [v for v in variants if not v.startswith("ctl.")]
    pvals: dict[str, float] = {}
    for v in real:
        ctl = f"ctl.{v.split('.')[0]}"
        out[v].vs_control = vs_control(by_var.get(v, []), by_var.get(ctl, []))
        pvals[v] = out[v].vs_control["p"]
    adj = holm(pvals)
    srs = [out[v].sharpe_daily for v in real if math.isfinite(out[v].sharpe_daily)]
    sr_var = float(np.var(srs, ddof=1)) if len(srs) >= 2 else math.nan
    live = set(live_variants)
    for v in real:
        g = out[v]
        g.p_holm = adj.get(v, math.nan)
        g.dsr = deflated_sharpe(g.sharpe_daily, len(daily_pnl(by_var.get(v, []))), g.daily_skew, g.daily_kurt,
                                sr_var, len(real))
        g.badge = badge(g, by_var.get(v, []), v.startswith("ll."), calibrated, v in live)
    for v in variants:
        if v.startswith("ctl."):
            out[v].badge = "KONTROLA"
    return out


def badge(g: GroupStats, trades: Sequence[T], family_c: bool, calibrated: bool, live: bool) -> str:
    if live:
        return "LIVE"
    if g.n < 200:
        return "PŘÍLIŠ BRZY"
    if math.isfinite(g.net_ci[1]) and g.net_ci[1] < 0:
        return "ZÁPORNÁ"
    clean = [t for t in trades if not t.tainted]
    pos_syms = sum(1 for s in g.by_symbol.values() if s["net_mean"] > 0)
    need_syms = 3 if family_c else 4
    if (len(clean) >= 500 and len(clean) == len(trades) and calibrated and g.net_at_1_5x > 0
            and math.isfinite(g.p_holm) and g.p_holm < 0.05 and pos_syms >= need_syms and g.monthly_ok):
        return "KANDIDÁT"
    return "NEJASNÁ"


def heatmap(trades: Sequence[T]) -> dict[str, dict[str, dict[str, float]]]:
    """varianta × symbol: průměr net_bps a n."""
    out: dict[str, dict[str, list[float]]] = {}
    for t in trades:
        out.setdefault(t.variant, {}).setdefault(t.symbol, []).append(t.net_bps)
    return {v: {s: {"n": len(xs), "net_mean": float(np.mean(xs))} for s, xs in d.items()} for v, d in out.items()}
