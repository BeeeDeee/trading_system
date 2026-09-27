"""STR-TF metrics (brief §10): per trade and per portfolio."""

import numpy as np
import polars as pl

from qlab.research5.sim import SimOutput
from qlab.validation.metrics import summary

TRADING_DAYS = 252


def trade_stats(trades: pl.DataFrame) -> dict[str, float]:
    tr = trades.filter(pl.col("reason") != "end")
    if tr.is_empty():
        return {"n_trades": 0}
    cost = ((tr["entry_cost"] + tr["exit_cost"]) / tr["entry_value"]).to_numpy()
    gross = tr["gross_ret"].to_numpy()
    net = tr["net_ret"].to_numpy()
    return {
        "n_trades": tr.height,
        "win_rate": float((net > 0).mean()),
        "gross_bps": float(gross.mean() * 1e4),
        "net_bps": float(net.mean() * 1e4),
        "median_net_bps": float(np.median(net) * 1e4),
        "cost_bps": float(cost.mean() * 1e4),
        "gross_to_cost": float(gross.mean() / cost.mean()) if cost.mean() > 0 else np.inf,
        "avg_hold": float(tr["held"].mean()),
        **{f"exit_{r}": float((tr["reason"] == r).mean())
           for r in ("exit_signal", "time_stop", "stop", "delisted")},
    }


def portfolio_stats(out: SimOutput, spy: np.ndarray | None = None) -> dict[str, float]:
    r = out.returns
    s = summary(r, None, out.turnover, out.costs, out.exposure)
    s["sharpe"] = float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
    s["t_stat"] = float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if r.std() > 0 else 0.0
    s["avg_positions"] = float(out.n_positions.mean())
    if spy is not None:
        x = np.asarray(spy, dtype=float)
        beta = float(np.cov(r, x)[0, 1] / x.var(ddof=1))
        s["beta"] = beta
        s["corr_spy"] = float(np.corrcoef(r, x)[0, 1])
        s["alpha_annual"] = float((r.mean() - beta * x.mean()) * TRADING_DAYS)
    return s


def yearly(out: SimOutput) -> pl.DataFrame:
    return (pl.DataFrame({"date": out.dates, "r": out.returns})
            .with_columns(pl.col("date").cast(pl.Date))
            .group_by(pl.col("date").dt.year().alias("year"))
            .agg(ret=(pl.col("r") + 1).product() - 1, log_ret=(pl.col("r") + 1).log().sum(),
                 sharpe=pl.col("r").mean() / pl.col("r").std() * np.sqrt(TRADING_DAYS))
            .sort("year"))


def spy_returns(ctx) -> np.ndarray:
    p = ctx.panel
    r = (1.0 + np.asarray(p.ret_co[:, ctx.spy])) * (1.0 + np.asarray(p.ret_oc[:, ctx.spy])) - 1.0
    return r[ctx.start:ctx.end]
