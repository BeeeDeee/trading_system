"""Research 5 / R5.3: selection, robustness and kill criteria on the development period.

Usage: python scripts/r5_evaluate.py sharadar_YYYY-MM-DD [n_random_sims]
Needs runs/research5/grid.parquet (scripts/r5_grid.py).
Output: docs/research5/results/dev_summary.json, data/derived/<snapshot>/research5/*.csv
Follows docs/research5/PREREGISTRATION.md §8–§10; every run is recorded in the research 5 registry.
"""

import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import Panel, load_panel
from qlab.pipeline import apply_delisting_scenario
from qlab.research5.report import portfolio_stats, spy_returns, trade_stats, yearly
from qlab.research5.setup import STAGES, load_stage, log_experiment, registry
from qlab.research5.strategy import (DEFAULT, GRID, Config, Context, neighbors, random_benchmark,
                                     run)
from qlab.research5.universe import sic_division
from qlab.validation.stats import deflated_sharpe

snapshot = sys.argv[1]
n_sims = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
t0 = time.time()
der = Path("data/derived") / snapshot
res_dir = Path("docs/research5/results")
csv_dir = der / "research5"
res_dir.mkdir(parents=True, exist_ok=True)
csv_dir.mkdir(parents=True, exist_ok=True)
ST_REVERSAL_R1 = 432  # st_reversal candidates of research 1 (pre-registration §10.5)

ctx = load_stage(snapshot, "dev")
spy = spy_returns(ctx)
reg = registry()
grid = pl.read_parquet("runs/research5/grid.parquet")
grid_ret = np.load("runs/research5/grid_returns.npy", mmap_mode="r")
PARAMS = list(GRID)


def key(cfg: Config) -> tuple:
    return tuple(getattr(cfg, p) for p in PARAMS)


sharpe_of = {tuple(r[p] for p in PARAMS): r["sharpe"] for r in grid.iter_rows(named=True)}
configs = [Config(**{p: r[p] for p in PARAMS}) for r in grid.iter_rows(named=True)]


def n_nondefault(cfg: Config) -> int:
    return sum(getattr(cfg, p) != getattr(DEFAULT, p) for p in PARAMS)


# --- selection (§8) -----------------------------------------------------------------------------
hood = {c: float(np.median([sharpe_of[key(n)] for n in [c, *neighbors(c)]])) for c in configs}
selected = max(configs, key=lambda c: (round(hood[c], 2), -n_nondefault(c)))
sel_sharpe = sharpe_of[key(selected)]
nb_sharpes = [sharpe_of[key(n)] for n in neighbors(selected)]
print(f"selected {key(selected)}: Sharpe {sel_sharpe:.2f}, neighbourhood median "
      f"{hood[selected]:.2f}")


def evaluate(cfg: Config, c: Context = ctx, note: str = "", with_trades: bool = False):
    reg.record("other", {"stage": "dev", **asdict(cfg), "note": note}, 1, f"research5 dev {note}")
    out = run(c, cfg)
    stats = {**portfolio_stats(out, spy if c is ctx else None), **trade_stats(out.trades)}
    log_experiment("dev", asdict(cfg), {k: stats.get(k) for k in
                                        ("sharpe", "cagr", "max_drawdown", "n_trades",
                                         "gross_bps", "cost_bps")}, note)
    return (out, stats) if with_trades else stats


sel_out, sel_stats = evaluate(selected, note="selected", with_trades=True)
def_out, def_stats = evaluate(DEFAULT, note="default", with_trades=True)
results = {"selected": asdict(selected), "selected_stats": sel_stats, "default_stats": def_stats,
           "neighbourhood_median_sharpe": hood[selected], "neighbour_sharpes": nb_sharpes}

# --- heatmaps (§9.1) ----------------------------------------------------------------------------
heat = {}
for a, b in (("entry_z", "max_hold"), ("lookback_ret", "exit_sma"), ("min_adv", "max_positions")):
    fixed = [p for p in PARAMS if p not in (a, b)]
    sub = grid.filter(pl.all_horizontal([pl.col(p) == getattr(selected, p) for p in fixed]))
    heat[f"{a}__{b}"] = sub.select(a, b, "sharpe").sort(a, b).to_dicts()
results["heatmaps"] = heat

# --- variants (§9.2, §9.5) ----------------------------------------------------------------------
variants = {
    "cost_x2": replace(selected, cost_mult=2.0),
    "cost_x3": replace(selected, cost_mult=3.0),
    "cost_tiers_research1": replace(selected, cost="tiers"),
    "entry_t_plus_2": replace(selected, entry_delay=2),
    "moc_upper_bound": replace(selected, entry_delay=0),
    "spy_filter": replace(selected, market_filter=True),
    "atr_stop_2": replace(selected, atr_stop=2.0),
    "vol_target_12": replace(selected, vol_target=0.12),
    "rsi2": replace(selected, signal="rsi2"),
    "ibs": replace(selected, signal="ibs"),
    "cash_tbill": replace(selected, cash="tbill"),
    "capital_10m": replace(selected, capital=10e6),
    "default_cost_x2": replace(DEFAULT, cost_mult=2.0),
}
results["variants"] = {name: evaluate(cfg, note=name) for name, cfg in variants.items()}
print(f"variants done, {time.time() - t0:.0f}s", flush=True)

# Delisting scenarios: same features, terminal returns replaced.
delistings = pl.read_parquet(der / "delistings.parquet")
for scen in ("optimistic", "pessimistic"):
    p2 = apply_delisting_scenario(ctx.panel, delistings, scen)
    c2 = Context(p2, ctx.extra, ctx.start, ctx.cache_dir, ctx.spy)
    results["variants"][f"delisting_{scen}"] = evaluate(selected, c2, f"delisting_{scen}")

# Earnings exclusion (§9.6): compare on trades with signal from 2005 on.
first_2005 = int(np.searchsorted(ctx.panel.dates, np.datetime64("2005-01-01")))
c05 = Context(ctx.panel, ctx.extra, first_2005, ctx.cache_dir, ctx.spy)
results["earnings"] = {
    "with_earnings_2005_2014": evaluate(selected, c05, "earnings_included_2005"),
    "without_earnings_2005_2014": evaluate(replace(selected, exclude_earnings=True), c05,
                                           "earnings_excluded_2005"),
}

# Short side, trade level only (§9.5): gross short = -gross long; costs as for the long side.
short_out = run(ctx, replace(selected, side="short"))
reg.record("other", {"stage": "dev", **asdict(selected), "side": "short"}, 1, "research5 short")
st = short_out.trades.filter(pl.col("reason") != "end")
cost_rt = ((st["entry_cost"] + st["exit_cost"]) / st["entry_value"]).to_numpy()
g = -st["gross_ret"].to_numpy()
results["short_trade_level"] = {"n_trades": st.height, "gross_bps": float(g.mean() * 1e4),
                                "net_bps": float((g - cost_rt).mean() * 1e4),
                                "cost_bps": float(cost_rt.mean() * 1e4),
                                "win_rate": float(((g - cost_rt) > 0).mean())}

# ETF universe (§9.5): the 24 ETFs of research 4 on their own panel.
p4, x4 = load_panel(der / "panel_r4")
cols = json.loads((der / "panel_r4" / "columns.json").read_text())["etf"]
etf_cols = np.array(sorted(cols.values()))
end = int(np.searchsorted(p4.dates, np.datetime64(STAGES["dev"][1]), side="right"))
pe = Panel(p4.dates[:end], p4.assets[etf_cols],
           *(np.asarray(getattr(p4, f)[:end][:, etf_cols]) for f in
             ("ret_co", "ret_oc", "tradable", "listed", "delisting", "close_u", "dollar_volume")))
dv = np.nan_to_num(pe.dollar_volume.astype(float))
c = np.cumsum(np.vstack([np.zeros((1, dv.shape[1])), dv]), axis=0)
adv20 = np.full(dv.shape, np.nan)
adv20[19:] = (c[20:] - c[:-20]) / 20
days = np.cumsum(pe.listed, axis=0)
etf_extra = {"base_ok": pe.listed & (days >= 252) & (np.nan_to_num(pe.close_u) >= 5),
             "adv20": adv20, "high_u": np.full(pe.shape, np.nan), "low_u": np.full(pe.shape, np.nan),
             "earn8k": np.zeros(pe.shape, bool), "cash_ret": np.zeros(end)}
spy_etf = int(np.flatnonzero(etf_cols == cols["SPY"])[0])
c_etf = Context(pe, etf_extra, int(np.searchsorted(pe.dates, np.datetime64(STAGES["dev"][0]))),
                der / "r5_cache" / "dev_etf", spy_etf)
results["variants"]["etf_universe"] = evaluate(selected, c_etf, "etf_universe")
print(f"robustness variants done, {time.time() - t0:.0f}s", flush=True)

# --- breakdowns (§9.4) --------------------------------------------------------------------------
yr = yearly(sel_out)
results["yearly"] = yr.to_dicts()
dates = sel_out.dates.astype("datetime64[D]")
r = sel_out.returns


def seg(lo: str, hi: str) -> dict:
    m = (dates >= np.datetime64(lo)) & (dates <= np.datetime64(hi))
    x = r[m]
    return {"days": int(m.sum()), "ann_ret": float((np.prod(1 + x) ** (252 / max(len(x), 1))) - 1),
            "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 1 else None}


results["regimes"] = {"2000-2002": seg("2000-01-01", "2002-12-31"),
                      "2008-2009": seg("2008-01-01", "2009-12-31"),
                      "pre_2002": seg("1999-01-01", "2001-12-31"),
                      "post_2002": seg("2002-01-01", "2014-12-31")}
m_rest = ~(((dates >= np.datetime64("2000-01-01")) & (dates <= np.datetime64("2002-12-31")))
           | ((dates >= np.datetime64("2008-01-01")) & (dates <= np.datetime64("2009-12-31"))))
x = r[m_rest]
results["regimes"]["rest"] = {"days": int(m_rest.sum()),
                              "sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(252))}

tr = sel_out.trades.filter(pl.col("reason") != "end")
sic = np.asarray(ctx.extra["sic"][tr["signal_day"].to_numpy(), tr["asset"].to_numpy()])
tr = tr.with_columns(sector=pl.Series(sic_division(sic).tolist()),
                     adv_q=pl.col("adv20").qcut(5, labels=[f"Q{i}" for i in range(1, 6)]),
                     year=pl.Series(ctx.panel.dates[tr["signal_day"].to_numpy()]).dt.year(),
                     cost=(pl.col("entry_cost") + pl.col("exit_cost")) / pl.col("entry_value"))
agg = lambda by: (tr.group_by(by).agg(n=pl.len(), gross_bps=pl.col("gross_ret").mean() * 1e4,  # noqa: E731
                                      net_bps=pl.col("net_ret").mean() * 1e4,
                                      cost_bps=pl.col("cost").mean() * 1e4,
                                      win=(pl.col("net_ret") > 0).mean()).sort(by))
results["by_adv_quintile"] = agg("adv_q").with_columns(pl.col("adv_q").cast(pl.Utf8)).to_dicts()
results["by_sector"] = agg("sector").to_dicts()
results["trades_by_year"] = agg("year").to_dicts()

# --- kill criteria on dev (§10) -----------------------------------------------------------------
n_trials = reg.total_configs()
daily_sr = np.array([np.nanmean(grid_ret[:, j]) / np.nanstd(grid_ret[:, j], ddof=1)
                     for j in range(grid_ret.shape[1])])
sr_var = float(np.nanvar(daily_sr, ddof=1))
dsr = deflated_sharpe(r, n_trials, sr_var)
dsr_r1 = deflated_sharpe(r, n_trials + ST_REVERSAL_R1, sr_var)
reg.record("other", {"stage": "dev", "random_benchmark": asdict(selected), "n": n_sims}, 1,
           "research5 random benchmark")
rand = random_benchmark(ctx, selected, sel_out, n_sims=n_sims, seed=0)
p95 = float(np.quantile(rand, 0.95))
near = sum(s >= 0.7 * sel_sharpe for s in nb_sharpes) / len(nb_sharpes)
yl = yr.filter(pl.col("year") <= 2002)["log_ret"].sum()
kill = {
    "K1_gross_lt_2x_cost": sel_stats["gross_to_cost"] < 2.0,
    "K3_cagr_cost_x2_le_0": results["variants"]["cost_x2"]["cagr"] <= 0,
    "K5_dsr_lt_0_95": dsr < 0.95,
    "K6_not_above_random_p95": sel_stats["sharpe"] <= p95,
    "K7_isolated_spike": near < 2 / 3,
}
kill = {k: bool(v) for k, v in kill.items()}
results["kill_dev"] = kill
results["kill_inputs"] = {
    "gross_to_cost": sel_stats["gross_to_cost"], "cagr_cost_x2": results["variants"]["cost_x2"]["cagr"],
    "dsr": dsr, "dsr_with_r1_trials": dsr_r1, "n_trials": n_trials, "sr_variance_daily": sr_var,
    "random_sharpe_p50": float(np.median(rand)), "random_sharpe_p95": p95,
    "strategy_sharpe": sel_stats["sharpe"], "neighbours_ge_70pct": near,
    "log_ret_1999_2002_share_of_dev": float(yl / yr["log_ret"].sum()) if yr["log_ret"].sum() > 0 else None,
}
results["grid_summary"] = {"median_sharpe": float(grid["sharpe"].median()),
                           "share_sharpe_gt_0": float((grid["sharpe"] > 0).mean()),
                           "max_sharpe": float(grid["sharpe"].max()),
                           "median_gross_to_cost": float(grid["gross_to_cost"].median())}
np.save(csv_dir / "random_benchmark_sharpe.npy", rand)

# --- trade list (brief §10) ---------------------------------------------------------------------
tick = (pl.read_parquet(Path("data/parquet") / snapshot / "tickers.parquet")
        .filter(pl.col("table") == "SEP").select(pl.col("permaticker").cast(pl.Int64), "ticker"))
perm = ctx.panel.assets
t_all = sel_out.trades.with_columns(
    permaticker=pl.Series(perm[sel_out.trades["asset"].to_numpy()]),
    entry_date=pl.Series(ctx.panel.dates[sel_out.trades["entry_day"].to_numpy()]).cast(pl.Date),
    exit_date=pl.Series(ctx.panel.dates[sel_out.trades["exit_day"].to_numpy()]).cast(pl.Date))
opens = (pl.scan_parquet(der / "bars" / "*.parquet")
         .filter(pl.col("permaticker").is_in(pl.Series(np.unique(t_all["permaticker"])).implode()))
         .select("permaticker", "date", "open_u", "close_u").collect())
t_all = (t_all.join(tick, on="permaticker", how="left")
         .join(opens.rename({"date": "entry_date", "open_u": "entry_px"}).drop("close_u"),
               on=["permaticker", "entry_date"], how="left")
         .join(opens.rename({"date": "exit_date", "open_u": "exit_px"}).drop("close_u"),
               on=["permaticker", "exit_date"], how="left")
         .select("ticker", "entry_date", "entry_px", "exit_date", "exit_px", "reason",
                 "gross_ret", "net_ret", "held", "score", "adv20"))
t_all.write_csv(csv_dir / "trades_dev_selected.csv")
yr.write_csv(csv_dir / "yearly_dev_selected.csv")

(res_dir / "dev_summary.json").write_text(json.dumps(results, indent=2, default=float))
print(json.dumps({"kill_dev": kill, **results["kill_inputs"]}, indent=2, default=float))
print(f"done in {time.time() - t0:.0f}s")
