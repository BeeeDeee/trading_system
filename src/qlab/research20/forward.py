"""Forward panel and paper runs of F1 (research 19 H1b) and F2 (H2) (prereg §3-§6).

The panel is `panel_r19_etf` (Sharadar, to 2026-09-25) extended by days from the daily raw snapshots. Each day's bar is
frozen at first sight: taken from the earliest snapshot that contains it together with the previous close (so the
return is computed inside one fetch, where splits cancel). Evaluation stops at the first day missing for any ticker.
"""

import gzip
import json
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.costs import cash_returns
from qlab.research19 import etf_trend as ET
from qlab.research19 import rsi2 as R2
from qlab.research19 import setup as S19
from qlab.research20 import data as D
from qlab.research20 import evaluate as E
from qlab.research20 import setup as S


def snapshot_dirs(raw: Path = S.RAW) -> list[Path]:
    return sorted(p for p in raw.glob("*") if (p / "manifest.json").exists())


def load_ledger(raw: Path = S.RAW, after: date = S.SHARADAR_END) -> dict[str, dict]:
    """{ticker: {date: (open, close, ret_co, ret_oc)}} for days after `after`, frozen at first sight."""
    led: dict[str, dict] = {tk: {} for tk in S19.ALL_ETF}
    cut = np.datetime64(after, "D")
    for d in snapshot_dirs(raw):
        snap = np.datetime64(d.name, "D")
        for tk in S19.ALL_ETF:
            f = d / f"{tk}.json.gz"
            if not f.exists():
                continue
            b = D.tr_bars(p := D.parse_yahoo(json.loads(gzip.decompress(f.read_bytes()))))
            for i in range(1, len(b["dates"])):
                day = b["dates"][i]
                if day > cut and day <= snap and day not in led[tk]:
                    led[tk][day] = (float(p["open"][i]), float(p["close"][i]), float(b["ret_co"][i]), float(b["ret_oc"][i]))
    return led


def complete_days(led: dict[str, dict]) -> list[np.datetime64]:
    """Sorted days present for every ticker, up to (excluding) the first day missing for some ticker."""
    allv = sorted(set().union(*(set(v) for v in led.values())))
    out = []
    for d in allv:
        if all(d in led[tk] for tk in led):
            out.append(d)
        else:
            break
    return out


def fred_cash(dates: np.ndarray, raw: Path = S.RAW) -> np.ndarray:
    """Daily T-bill return for `dates` from the Sharadar-era FRED file plus all snapshot files (later files win)."""
    import polars as pl
    files = sorted(Path("data/raw").glob("fred_*/DTB3.csv")) + sorted(raw.glob("*/fred_DTB3.csv"))
    rate = {}
    for f in files:
        df = pl.read_csv(f, null_values=".", schema_overrides={"DTB3": pl.Float64})
        for d, v in zip(df["observation_date"].str.to_date().to_numpy(), df["DTB3"].to_numpy()):
            if v is not None and np.isfinite(v):
                rate[d] = v
    ks = np.array(sorted(rate), dtype="datetime64[D]")
    return cash_returns(dates, ks, np.array([rate[k] for k in ks]))


def append_days(panel: Panel, cash: np.ndarray, tickers: list[str], led: dict[str, dict],
                days: list[np.datetime64]) -> tuple[Panel, np.ndarray]:
    """Panel extended by `days` (all tickers present) and the cash vector extended with the same T-bill rule."""
    if not days:
        return panel, cash
    n, T = len(days), panel.shape[0]
    co, oc = np.zeros((n, len(tickers))), np.zeros((n, len(tickers)))
    cl = np.full((n, len(tickers)), np.nan)
    for j, tk in enumerate(tickers):
        for i, d in enumerate(days):
            o, c, rco, roc = led[tk][d]
            co[i, j], oc[i, j], cl[i, j] = rco, roc, c
    new = Panel(np.concatenate([panel.dates, np.array(days, dtype="datetime64[D]")]), panel.assets,
                np.vstack([panel.ret_co, co]), np.vstack([panel.ret_oc, oc]),
                np.vstack([panel.tradable, np.ones((n, len(tickers)), bool)]),
                np.vstack([panel.listed, np.ones((n, len(tickers)), bool)]),
                np.vstack([panel.delisting, np.zeros((n, len(tickers)), bool)]),
                np.vstack([panel.close_u, cl]), np.vstack([panel.dollar_volume, np.full((n, len(tickers)), np.nan)]))
    full_cash = fred_cash(new.dates)
    return new, np.concatenate([cash, full_cash[T:]])


def build_panel(raw: Path = S.RAW) -> tuple[Panel, np.ndarray, list[str], int]:
    """(panel incl. forward days, cash, tickers, index of T0). Raises if T0 is not (yet) in the data."""
    p, ex, tickers = S19.load_etf()
    led = load_ledger(raw)
    p2, cash = append_days(p, np.asarray(ex["cash_ret"]), tickers, led, complete_days(led))
    t0 = int(np.searchsorted(p2.dates, np.datetime64(S.T0, "D")))
    if t0 >= len(p2.dates) or p2.dates[t0] != np.datetime64(S.T0, "D"):
        raise SystemExit(f"T0 {S.T0} is not in the data yet (last day {p2.dates[-1]})")
    return p2, cash, tickers, t0


def run_f1(p: Panel, cash: np.ndarray, tickers: list[str], t0: int, cost_mult: float = 1.0) -> np.ndarray:
    """Daily returns of F1 from T0 (first decision on the close of the day before T0, filled at the open of T0)."""
    cfg = replace(ET.Config(), cost_mult=cost_mult)
    return ET.run_panel(p, tickers, cfg, t0 - 1, cash).returns[t0:]


def run_f2(p: Panel, cash: np.ndarray, tickers: list[str], t0: int, cost_mult: float = 1.0, **kw):
    """Event simulation of F2 from T0 (first decision on the close of T0). -> SimOutput."""
    return R2.run_panel(p, tickers, replace(R2.Config(), cost_mult=cost_mult, **kw), t0, cash)


def forward_metrics(p, cash, tickers, t0, end: int) -> dict:
    """All statistics for days [t0, end) of the panel (end exclusive)."""
    p, cash = p.slice(end), cash[:end]
    c = cash[t0:]
    out = {"asof": str(p.dates[-1]), "n_days": int(end - t0)}
    f1 = {m: run_f1(p, cash, tickers, t0, m) for m in (1.0, S.BAR_COST_MULT, 3.0)}
    sh = {m: E.excess_sharpe(r, c) for m, r in f1.items()}
    boots = E.boot_sharpe(f1[1.0], c)
    out["F1"] = {"sharpe_x1": sh[1.0], "sharpe_x1.5": sh[S.BAR_COST_MULT], "sharpe_x3": sh[3.0], "p": E.p_value(boots),
                 "up90": E.upper90(boots), "n_days": len(c), "returns_total_x1": float(np.prod(1 + f1[1.0]) - 1)}
    f2 = {m: run_f2(p, cash, tickers, t0, m) for m in (1.0, S.BAR_COST_MULT, 3.0)}
    r2 = {}
    for m, o in f2.items():
        tr = o.trades.filter(o.trades["reason"] != "end")
        r2[m] = {"sharpe": E.excess_sharpe(o.returns, cash[t0:]), "n_trades": tr.height,
                 "mean_trade": float(tr["net_ret"].mean()) if tr.height else 0.0, "tr": tr}
    tr1 = r2[1.0]["tr"]
    b2 = E.boot_trade_means(tr1["net_ret"].to_numpy(), tr1["entry_day"].to_numpy()) if tr1.height >= 2 else np.zeros(2)
    out["F2"] = {"sharpe_x1": r2[1.0]["sharpe"], "sharpe_x1.5": r2[S.BAR_COST_MULT]["sharpe"], "sharpe_x3": r2[3.0]["sharpe"],
                 "mean_trade_x1": r2[1.0]["mean_trade"], "mean_trade_x1.5": r2[S.BAR_COST_MULT]["mean_trade"],
                 "mean_trade_x3": r2[3.0]["mean_trade"], "n_trades": r2[1.0]["n_trades"], "p": E.p_value(b2),
                 "up90": E.upper90(b2), "win_rate": float((tr1["net_ret"] > 0).mean()) if tr1.height else None,
                 "avg_hold": float(tr1["held"].mean()) if tr1.height else None,
                 "time_in_market": float(f2[1.0].exposure.mean())}
    spy = ET.columns(tickers, ["SPY"])[0]
    r_spy = (1 + p.ret_co[t0:, spy]) * (1 + p.ret_oc[t0:, spy]) - 1
    out["SPY"] = {"sharpe": E.excess_sharpe(r_spy, c), "total": float(np.prod(1 + r_spy) - 1)}
    return out
