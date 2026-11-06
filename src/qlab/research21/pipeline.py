"""Research 21 pipeline (prereg §5-§7): S0 partition by proxy filters, books, controls, robustness, verdict."""

from dataclasses import replace

import numpy as np
import polars as pl

from qlab.research5.costs import period_cost
from qlab.research5.sim import SimOutput
from qlab.research19.evaluate import trade_ci_by_entry_day
from qlab.research21 import evaluate as EV
from qlab.research21 import setup as S
from qlab.research21 import strategy as ST
from qlab.research21.strategy import Config, Context
from qlab.validation.stats import deflated_sharpe

S0 = Config()
PERIOD_FILTERS = {"E": "e", "C": "e", "V": "full", "G": "full", "I": "full"}   # E and C only from E_START


def plan() -> list[dict]:
    """Every configuration to register (prereg §5). `counted` = one of the N = 6 primary pre-registered trials."""
    out = [{"id": "S0", "role": "control S0", "counted": True, "config": S0.as_dict()}]
    for f in S.FILTERS:
        out.append({"id": f"book_{f}", "role": f"allowed-only book {f}", "counted": True,
                    "config": replace(S0, name=f"book_{f}", allow=f).as_dict()})
    out += [
        {"id": "C0", "role": "control C0 (regime stocks, random order)", "counted": False,
         "config": replace(S0, name="c0", control="c0").as_dict()},
        {"id": "r1", "role": "robustness: fill at the close (MOC)", "counted": False,
         "config": replace(S0, name="r1", entry_delay=0).as_dict()},
        {"id": "r2", "role": "robustness: E window [t-2, t]", "counted": False,
         "config": replace(S0, name="r2", e_window="e2").as_dict()},
    ]
    for tag, kw in (("r3a", {"vol_ratio": 2.0}), ("r3b", {"vol_ratio": 4.0}), ("r3c", {"gap_z": -1.5}),
                    ("r3d", {"gap_z": -3.0}), ("r3e", {"idio_z": -1.5}), ("r3f", {"idio_z": -3.0})):
        out.append({"id": tag, "role": f"robustness: thresholds {kw}", "counted": False,
                    "config": replace(S0, name=tag, **kw).as_dict()})
    out += [
        {"id": "r4a", "role": "robustness: ADV >= 20M", "counted": False, "config": replace(S0, name="r4a", adv_min=20e6).as_dict()},
        {"id": "r4b", "role": "robustness: ADV >= 100M", "counted": False, "config": replace(S0, name="r4b", adv_min=100e6).as_dict()},
        {"id": "r5", "role": "robustness: no SMA200", "counted": False, "config": replace(S0, name="r5", regime_filter=False).as_dict()},
        {"id": "r6", "role": "robustness: without 2000-2002 and 2008-2009", "counted": False, "config": {**S0.as_dict(), "name": "r6", "exclude_years": [2000, 2001, 2002, 2008, 2009]}},
        {"id": "r7", "role": "robustness: costs x3", "counted": False, "config": replace(S0, name="r7", cost_mult=3.0).as_dict()},
    ]
    return out


# --------------------------------------------------------------------------- metrics

def sim_metrics(out: SimOutput, cash: np.ndarray | None = None) -> dict:
    r = np.asarray(out.returns, float)
    ex = r - (0.0 if cash is None else cash)
    sd = ex.std(ddof=1)
    years = len(r) / S.ANN
    nav = np.asarray(out.nav, float)
    t = out.trades
    return {"sharpe": float(ex.mean() / sd * np.sqrt(S.ANN)) if sd > 0 else 0.0,
            "sharpe_per_period": float(ex.mean() / sd) if sd > 0 else 0.0,
            "cagr": float(nav[-1] ** (1 / years) - 1), "max_dd": float((nav / np.maximum.accumulate(nav) - 1).min()),
            "exposure": float(np.mean(out.exposure)), "turnover_per_year": float(np.sum(out.turnover) / years),
            "n_trades": int(t.height),
            "mean_net_trade": float(t["net_ret"].mean()) if t.height else float("nan"),
            "win_rate": float((t["net_ret"] > 0).mean()) if t.height else float("nan"),
            "mean_hold": float(t["held"].mean()) if t.height else float("nan"),
            "first": str(out.dates[0]), "last": str(out.dates[-1])}


def spy_book(ctx: Context) -> SimOutput:
    p = ctx.panel
    r = ((1 + np.asarray(p.ret_co[ctx.start:ctx.end, ctx.spy])) * (1 + np.asarray(p.ret_oc[ctx.start:ctx.end, ctx.spy])) - 1)
    n = len(r)
    return SimOutput(p.dates[ctx.start:ctx.end], r, np.cumprod(1 + r), np.ones(n), np.ones(n, int), np.zeros(n),
                     np.zeros(n), pl.DataFrame({"net_ret": [], "held": []}, schema={"net_ret": pl.Float64, "held": pl.Int64}))


def since(ctx: Context, out: SimOutput, d) -> SimOutput:
    """The same book restricted to days >= d (nav renormalised) and trades signalled on or after d."""
    k = int(np.searchsorted(out.dates, np.datetime64(d, "D")))
    nav = out.nav[k:] / (out.nav[k - 1] if k > 0 else 1.0)
    thr = S.index_of(ctx.panel.dates, d)
    return SimOutput(out.dates[k:], out.returns[k:], nav, out.exposure[k:], out.n_positions[k:], out.costs[k:],
                     out.turnover[k:], out.trades.filter(pl.col("signal_day") >= thr))


# --------------------------------------------------------------------------- filter evaluation

def trade_costs(ctx: Context, trades: pl.DataFrame, cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    """(entry, exit) cost rate per trade at multiplier 1 as the simulator applied it (delisting / end legs cost 0)."""
    dates = ctx.panel.dates
    c_open, c_close = period_cost(dates, True, 1.0), period_cost(dates, False, 1.0)
    ce = (c_close if cfg.entry_delay == 0 else c_open)[trades["entry_day"].to_numpy()]
    cx = c_open[trades["exit_day"].to_numpy()]
    cx = np.where(trades["reason"].is_in(["delisted", "end"]).to_numpy(), 0.0, cx)
    return ce * cfg.cost_mult, cx * cfg.cost_mult


def in_period(ctx: Context, trades: pl.DataFrame, flag: str) -> pl.DataFrame:
    if PERIOD_FILTERS[flag] == "e":
        return trades.filter(pl.col("signal_day") >= S.index_of(ctx.panel.dates, S.E_START))
    return trades


def evaluate_filters(ctx: Context, cfg: Config, tagged: pl.DataFrame, with_c2: bool, reps: int = S.BOOT_REPS) -> dict:
    ce_all, cx_all = trade_costs(ctx, tagged, cfg)
    tagged = tagged.with_columns(pl.Series("ce", ce_all), pl.Series("cx", cx_all))
    res = {}
    for f in S.FILTERS:
        t = in_period(ctx, tagged, f)
        thirds = S.THIRDS_E if PERIOD_FILTERS[f] == "e" else S.THIRDS
        allowed = t.filter(~pl.col(f"b_{f}"))
        if with_c2 and allowed.height:
            sims = EV.c2_means(ctx.feats, ctx.extra, ctx.panel, ctx.panel.dates, allowed["signal_day"].to_numpy(),
                               (allowed["exit_day"] - allowed["entry_day"]).to_numpy(), cfg.adv_min, cfg.cost_mult)
            p95, med = float(np.quantile(sims, 0.95)), float(np.median(sims))
        else:
            p95, med = float("nan"), float("nan")
        r = EV.filter_criteria(t, f, t["ce"].to_numpy(), t["cx"].to_numpy(), p95, thirds, reps)
        r["c2_median"] = med
        res[f] = r
    return res


def partition_brief(ctx: Context, cfg: Config, tagged: pl.DataFrame) -> dict:
    """Trade-level partition summary for robustness variants: counts, mean allowed / blocked, spread (no CI)."""
    out = {}
    for f in S.FILTERS:
        t = in_period(ctx, tagged, f)
        blk = t[f"b_{f}"].to_numpy().astype(bool)
        net = t["net_ret"].to_numpy()
        a, b = net[~blk], net[blk]
        out[f] = {"n_allowed": int(len(a)), "n_blocked": int(len(b)),
                  "mean_allowed": float(a.mean()) if len(a) else None, "mean_blocked": float(b.mean()) if len(b) else None,
                  "spread": float(a.mean() - b.mean()) if len(a) and len(b) else None}
    return out


def book_summary(ctx: Context, out: SimOutput) -> dict:
    k = int(np.searchsorted(ctx.panel.dates, out.dates[0]))
    return sim_metrics(out, np.asarray(ctx.extra["cash_ret"][k:k + len(out.dates)]))


# --------------------------------------------------------------------------- smoke (S0 only, no outcome by group)

def smoke(ctx: Context) -> dict:
    """S0 descriptive numbers and partition SHARES only. Results of the groups (allowed vs blocked) are not computed here,
    so a smoke run cannot be used to look at the outcome of the hypotheses before the single full run."""
    out = ST.run(ctx, S0)
    tagged = ST.tag_trades(ctx, out.trades, S0)
    shares = {}
    for f in S.FILTERS:
        t = in_period(ctx, tagged, f)
        shares[f] = {"n": t.height, "share_blocked": float(t[f"b_{f}"].mean()) if t.height else None}
    ci = trade_ci_by_entry_day(out.trades["net_ret"].to_numpy(), out.trades["entry_day"].to_numpy(), 20.0, 200)
    return {"s0": book_summary(ctx, out), "s0_mean_net_trade_ci": list(ci), "partition_shares": shares,
            "n_candidates": int(len(ST.candidates(ctx, S0).t))}


# --------------------------------------------------------------------------- full evaluation

def evaluate(ctx: Context, reps: int = S.BOOT_REPS, log=print) -> dict:
    res: dict = {}
    out0 = ST.run(ctx, S0)
    tagged0 = ST.tag_trades(ctx, out0.trades, S0)
    log(f"S0: {out0.trades.height} trades")
    res["s0"] = {**book_summary(ctx, out0),
                 "mean_net_trade_ci": list(trade_ci_by_entry_day(out0.trades["net_ret"].to_numpy(),
                                                                 out0.trades["entry_day"].to_numpy(), 20.0, reps)),
                 "since_2005": book_summary(ctx, since(ctx, out0, S.E_START))}
    res["spy"] = sim_metrics(spy_book(ctx))
    res["filters"] = evaluate_filters(ctx, S0, tagged0, True, reps)
    log("filters evaluated")

    books, per_period = {"S0": out0}, {"S0": res["s0"]["sharpe_per_period"]}
    for f in S.FILTERS:
        books[f] = ST.run(ctx, replace(S0, name=f"book_{f}", allow=f))
        m = book_summary(ctx, books[f])
        res["filters"][f]["book"] = m
        per_period[f] = m["sharpe_per_period"]
        log(f"book {f}: {m['n_trades']} trades, sharpe {m['sharpe']:.2f}")
    var = float(np.var(list(per_period.values()), ddof=1))
    cash = np.asarray(ctx.extra["cash_ret"][ctx.start:ctx.end])
    res["dsr"] = {k: {"n6": float(deflated_sharpe(np.asarray(b.returns) - cash, 6, var)),
                      "n_cumulative": float(deflated_sharpe(np.asarray(b.returns) - cash, 6 + S.PRIOR_STOCK_TRIALS, var))}
                  for k, b in books.items()}
    res["dsr"]["sr_variance_per_period"] = var

    c0 = ST.run(ctx, replace(S0, name="c0", control="c0"))
    res["c0"] = book_summary(ctx, c0)
    log("c0 done")

    rob: dict = {}
    base_tag = ST.tag_trades
    for tag, cfg in (("r2", replace(S0, e_window="e2")), ("r3a", replace(S0, vol_ratio=2.0)), ("r3b", replace(S0, vol_ratio=4.0)),
                     ("r3c", replace(S0, gap_z=-1.5)), ("r3d", replace(S0, gap_z=-3.0)),
                     ("r3e", replace(S0, idio_z=-1.5)), ("r3f", replace(S0, idio_z=-3.0))):
        rob[tag] = {"partition": partition_brief(ctx, cfg, base_tag(ctx, out0.trades, cfg))}
    rob["r6"] = {"partition": partition_brief(
        ctx, S0, tagged0.filter(~pl.col("year").is_in([2000, 2001, 2002, 2008, 2009])))}
    for tag, cfg in (("r1", replace(S0, entry_delay=0)), ("r4a", replace(S0, adv_min=20e6)), ("r4b", replace(S0, adv_min=100e6)),
                     ("r5", replace(S0, regime_filter=False)), ("r7", replace(S0, cost_mult=3.0))):
        o = ST.run(ctx, cfg)
        rob[tag] = {"s0": book_summary(ctx, o), "partition": partition_brief(ctx, cfg, ST.tag_trades(ctx, o.trades, cfg))}
        log(f"robustness {tag} done")
    res["robustness"] = rob

    passed = [f for f in S.FILTERS if res["filters"][f]["verdict"] == "nazek"]
    res["verdict"] = {
        "filters": {f: res["filters"][f]["verdict"] for f in S.FILTERS},
        "criteria_passed": {f: res["filters"][f]["criteria_passed"] for f in S.FILTERS},
        "decision": ("(b) kandidat na forward: " + ", ".join(passed)) if passed else
                    "(a) zadny filtr neprosel: proxy uroven nepridava informaci, LLM klasifikator se nad timto signalem nestavi",
        "n_primary": 6}
    return res
