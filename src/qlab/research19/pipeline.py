"""Registration plan and the single evaluation run of research 19 (prereg §5-§7).

`plan()` lists every configuration that is registered before anything is computed. `evaluate(data, ...)` computes
all hypotheses, controls and criteria for a `Data` bundle: the real one (`load_real`) for `r19_run.py full`, a
synthetic one (`synthetic_data`) for `smoke`. Nothing here selects among configurations: the primary
configuration of each hypothesis is fixed by the pre-registration.
"""

from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from qlab.data.panel import Panel
from qlab.engine.vector import simulate as vector_simulate
from qlab.features.basic import tr_index
from qlab.research9.panel import cost_rate
from qlab.research9.strategy import hold as hold_decisions
from qlab.research9.strategy import monthly_5050
from qlab.research10.sim import monthly_mix
from qlab.research19 import combine, crypto_trend as CT, etf_trend as ET, evaluate as EV, overlays as OV
from qlab.research19 import rsi2 as R2, setup as S
from qlab.research19.signals import tom_window
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import deflated_sharpe

CRYPTO_ANN, ETF_ANN = S.CRYPTO_YEAR, S.ETF_YEAR


# ------------------------------------------------------------------------------------- registration plan
def plan() -> list[dict]:
    """Every configuration of the study: {id, role, counted, crypto, config}. `counted` = enters the DSR variance."""
    h1a, h1b, h2 = CT.Config(), ET.Config(), R2.Config()
    out: list[dict] = []

    def add(id_, role, cfg, crypto=False, counted=None):
        out.append({"id": id_, "role": role, "counted": role != "control" if counted is None else counted,
                    "crypto": crypto, "config": cfg.as_dict() if hasattr(cfg, "as_dict") else {**cfg, "study": "research19"}})

    add("H1a", "primary", h1a, True)
    add("H1b", "primary", h1b)
    add("H2", "primary", h2)
    add("H3a", "primary", {"id": "H3a", "tom": [S.TOM_BEFORE, S.TOM_AFTER], "base": "H2", "bps": S.ETF_COST_BPS})
    add("H3b", "primary", {"id": "H3b", "target": S.VOL_TARGET_PORTFOLIO, "window": S.VOL_TARGET_WINDOW, "base": "H4"}, True)
    add("H3c", "primary", {"id": "H3c", "vol_mult": S.CRASH_VOL_MULT, "scale": S.CRASH_SCALE, "base": "H4"}, True)
    add("H4", "primary", {"id": "H4", "sleeves": ["H1a", "H1b", "H2"], "w": [1 / 3] * 3, "bps": S.ETF_COST_BPS}, True)
    add("H1a-BTC", "secondary", replace(h1a, name="h1a_btc", btc_only=True), True)
    add("H2-SPY", "secondary", replace(h2, name="h2_spy", tickers=("SPY",), max_positions=1))
    for k, cfg in H1A_ROBUST.items():
        add(f"H1a/{k}", "robustness", cfg, True)
    for k, cfg in BTC_ROBUST.items():
        add(f"H1a-BTC/{k}", "robustness", cfg, True)
    for k, cfg in H1B_ROBUST.items():
        add(f"H1b/{k}", "robustness", cfg)
    for k, cfg in H2_ROBUST.items():
        add(f"H2/{k}", "robustness", cfg)
    add("H4/cost2x", "robustness", {"id": "H4/cost2x", "sleeve_cost_mult": 2.0}, True)
    add("H4/invvol", "robustness", {"id": "H4/invvol", "weights": "inverse volatility 252d monthly"}, True)
    for k in ("H1a/btc_hold", "H1a/btc_eth_5050", "H1a/ew_top20", "H1a/null_shift", "H1b/ew16", "H1b/60_40",
              "H1b/spy", "H1b/null_shift", "H2/c1_regime_long", "H2/c2_random_entries", "H4/60_40", "H4/spy",
              "H4/b_c_5050"):
        add(k, "control", {"id": k})
    return out


H1A_ROBUST = {"band_none": replace(CT.Config(), band=None), "cost25bps": replace(CT.Config(), cost_const=0.0025),
              "cost2x": replace(CT.Config(), cost_mult=2.0), "tbill": replace(CT.Config(), tbill_cash=True),
              "top10": replace(CT.Config(), universe_n=10), "top40": replace(CT.Config(), universe_n=40),
              "no_age": replace(CT.Config(), min_age=0), "delay1": replace(CT.Config(), delay=1)}
BTC_ROBUST = {"band_none": replace(CT.Config(btc_only=True), band=None),
              "cost2x": replace(CT.Config(btc_only=True), cost_mult=2.0)}
H1B_ROBUST = {"band_none": replace(ET.Config(), band=None), "cost2x": replace(ET.Config(), cost_mult=2.0),
              "cost3x": replace(ET.Config(), cost_mult=3.0), "vol15": replace(ET.Config(), vol_target=0.15),
              "etf9": replace(ET.Config(), tickers=tuple(S.ETF_TREND_9))}
H2_ROBUST = {"r1_open_fills": replace(R2.Config(), entry_delay=1, exit_at_close=False),
             "r2_exit_rsi70": replace(R2.Config(), exit_rule="rsi70"),
             "r3_sum2": replace(R2.Config(), entry_rule="sum2"),
             "r4_cost2x": replace(R2.Config(), cost_mult=2.0), "r4_cost3x": replace(R2.Config(), cost_mult=3.0),
             "r5_no_sma200": replace(R2.Config(), regime_filter=False),
             "r6_index4": replace(R2.Config(), tickers=tuple(S.RSI_INDEX_4)),
             "r7_pos2": replace(R2.Config(), max_positions=2)}


# ------------------------------------------------------------------------------------------------- data
@dataclass
class Data:
    pc: Panel
    qv: np.ndarray
    syms: list[str]
    tbill_c: np.ndarray
    pe: Panel
    tick: list[str]
    cash_e: np.ndarray
    periods: dict
    reps: dict
    snapshots: dict


def real_periods() -> dict:
    return {"H1A": S.PERIOD_H1A, "H1A_RECENT": S.PERIOD_H1A_RECENT, "H1A_POST_PUB": S.PERIOD_H1A_POST_PUB,
            "H1B": S.PERIOD_H1B, "H1B_POST": S.PERIOD_H1B_POST, "H2": S.PERIOD_H2, "H2_POST": S.PERIOD_H2_POST,
            "TOM": S.PERIOD_TOM, "H4": S.PERIOD_H4}


def real_reps() -> dict:
    return {"boot": S.BOOT_REPS, "null_trend": S.NULL_REPS_TREND, "null_h2": S.NULL_REPS_H2}


def load_real() -> Data:
    import io

    import polars as pl

    from qlab.research8.panel import tbill_daily
    pc, ex, syms = S.load_crypto()
    fred = pl.read_csv(io.BytesIO((Path("data/raw") / S.BINANCE / "fred_DTB3.csv").read_bytes()), infer_schema_length=0)
    pe, exe, tick = S.load_etf()
    return Data(pc, np.asarray(ex["qv"]), syms, tbill_daily(fred, pc.dates), pe, tick, np.asarray(exe["cash_ret"]),
                real_periods(), real_reps(), {"sharadar": S.SHARADAR, "binance": S.BINANCE})


# ---------------------------------------------------------------------------------------------- helpers
def tr_ret(p: Panel, j: int) -> np.ndarray:
    return (1.0 + np.asarray(p.ret_co[:, j])) * (1.0 + np.asarray(p.ret_oc[:, j])) - 1.0


def span(p: Panel, per: tuple[date, date]) -> slice:
    d = p.dates
    return slice(int(np.searchsorted(d, np.datetime64(per[0], "D"))),
                 int(np.searchsorted(d, np.datetime64(per[1], "D"), side="right")))


def maxdd(r: np.ndarray) -> float:
    return max_drawdown(np.cumprod(1.0 + np.asarray(r, float)))[0]


def yearly(r: np.ndarray, dates: np.ndarray) -> dict:
    y = dates.astype("datetime64[Y]").astype(int) + 1970
    return {int(k): float(np.prod(1.0 + r[y == k]) - 1.0) for k in np.unique(y)}


def per_period_sharpe(r, cash=0.0) -> float:
    x = np.asarray(r, float) - cash
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else 0.0


class Ctx:
    """Shared state of one evaluation: data, caches, collected Sharpes for the DSR variance."""

    def __init__(self, data: Data):
        self.d = data
        self.reps = data.reps
        self.per = data.periods
        self.series: dict[str, np.ndarray] = {}      # id -> (returns, cash) used for DSR variance
        self.members: dict[tuple, np.ndarray] = {}
        self.btc = data.syms.index("BTCUSDT") if "BTCUSDT" in data.syms else 0
        self.eth = data.syms.index("ETHUSDT") if "ETHUSDT" in data.syms else 1
        self.costs: dict[float, np.ndarray] = {}
        pc = data.pc
        self.hc = span(pc, self.per["H1A"])
        self.first_c = self.hc.start - 1
        pe = data.pe
        self.he = span(pe, self.per["H1B"])
        self.first_e = self.he.start - 1
        self.spy = data.tick.index("SPY")
        self.r_spy = tr_ret(pe, self.spy)
        self.r_ief = tr_ret(pe, data.tick.index("IEF"))

    def reg(self, id_: str, r: np.ndarray, cash=0.0) -> None:
        self.series[id_] = (np.asarray(r, float), cash if np.isscalar(cash) else np.asarray(cash, float))

    # ---- H1a family
    def members_for(self, cfg: CT.Config) -> np.ndarray:
        key = (cfg.universe_n, cfg.min_age)
        if key not in self.members:
            self.members[key] = CT.monthly_members(self.d.pc, self.d.qv, cfg.universe_n, cfg.min_age,
                                                   CT.excluded_mask(self.d.syms))
        return self.members[key]

    def cost_matrix(self, cfg: CT.Config) -> np.ndarray:
        if cfg.cost_const is not None:
            return np.full(self.d.pc.shape, cfg.cost_const)
        if cfg.cost_mult not in self.costs:
            self.costs[cfg.cost_mult] = cost_rate(self.d.qv, cfg.cost_mult)
        return self.costs[cfg.cost_mult]

    def crypto(self, cfg: CT.Config, scale=None, E_override=None, cache=None):
        kw = dict(tbill=self.d.tbill_c, scale=scale, E_override=E_override, signal_cache=cache,
                  cost=self.cost_matrix(cfg))
        if cfg.btc_only:
            return CT.run_panel(self.d.pc, self.d.qv, self.d.syms, cfg, self.first_c, **kw)
        return CT.run_panel(self.d.pc, self.d.qv, self.d.syms, cfg, self.first_c, members=self.members_for(cfg), **kw)

    def etf(self, cfg: ET.Config, scale=None, E_override=None, cache=None):
        return ET.run_panel(self.d.pe, self.d.tick, cfg, self.first_e, self.d.cash_e, E_override, scale, cache)

    def rsi(self, cfg: R2.Config, **kw):
        start = span(self.d.pe, self.per["H2"]).start
        return R2.run_panel(self.d.pe, self.d.tick, cfg, start, self.d.cash_e, **kw), start


def bench_6040(ctx: Ctx, sl: slice) -> np.ndarray:
    R = np.column_stack([ctx.r_spy[sl], ctx.r_ief[sl]])
    return combine.mix(R, ctx.d.pe.dates[sl], np.array([0.6, 0.4]))


# ----------------------------------------------------------------------------------------------- H1a
def eval_h1a(ctx: Ctx) -> dict:
    d, hc, B, rng = ctx.d, ctx.hc, ctx.reps["boot"], np.random.default_rng(S.SEED)
    dates = d.pc.dates[hc]
    res = ctx.crypto(CT.Config())
    r = res.returns[hc]
    ctx.reg("H1a", r)
    pt, lo, hi = EV.ci_sharpe(r, CRYPTO_ANN, S.BOOT_BLOCK_CRYPTO, reps=B)
    # controls
    c1 = ctx.cost_matrix(CT.Config())
    btc_hold = vector_simulate(d.pc, hold_decisions(d.pc, ctx.btc, ctx.first_c), c1).returns[hc]
    mix5050 = vector_simulate(d.pc, monthly_5050(d.pc, ctx.btc, ctx.eth, ctx.first_c), c1).returns[hc]
    ew = vector_simulate(d.pc, CT.ew_members_decisions(d.pc, ctx.members_for(CT.Config()), ctx.first_c), c1).returns[hc]
    alpha, beta, t_alpha = EV.alpha_hac(r, btc_hold, 30)
    rec = span(d.pc, ctx.per["H1A_RECENT"])
    post = span(d.pc, ctx.per["H1A_POST_PUB"])
    sub = lambda x, s: x[s.start - hc.start:s.stop - hc.start]  # noqa: E731
    # robustness
    rob = {}
    for k, cfg in H1A_ROBUST.items():
        rr = ctx.crypto(cfg).returns[hc]
        cash = d.tbill_c[hc] if cfg.tbill_cash else 0.0
        rob[k] = {"sharpe": EV.sharpe_ann(rr, CRYPTO_ANN, cash), "maxdd": maxdd(rr)}
        ctx.reg(f"H1a/{k}", rr, cash)
    # circular-shift null
    members = ctx.members_for(CT.Config())
    cols = np.flatnonzero(members.any(axis=0))
    E, sig = CT.signals(d.pc, CT.Config(), cols)
    f_, l_ = EV.listed_spans(d.pc.listed)
    null = np.array([EV.sharpe_ann(ctx.crypto(CT.Config(), E_override=EV.circular_shift(E, f_, l_, rng),
                                              cache=(E, sig)).returns[hc], CRYPTO_ANN)
                     for _ in range(ctx.reps["null_trend"])])
    # BTC-only
    btc_cfg = CT.Config(btc_only=True)
    rb = ctx.crypto(btc_cfg).returns[hc]
    ctx.reg("H1a-BTC", rb)
    rob_btc = {}
    for k, cfg in BTC_ROBUST.items():
        rr = ctx.crypto(cfg).returns[hc]
        rob_btc[k] = {"sharpe": EV.sharpe_ann(rr, CRYPTO_ANN), "maxdd": maxdd(rr)}
        ctx.reg(f"H1a-BTC/{k}", rr)
    ctx.reg("H1a/btc_hold", btc_hold, 0.0); ctx.reg("H1a/btc_eth_5050", mix5050, 0.0); ctx.reg("H1a/ew_top20", ew, 0.0)
    ctx.crypto_res = res
    q = np.sort(r)[::-1]
    metrics = dict(sharpe=pt, sharpe_lo=lo, sharpe_hi=hi, sharpe_btc=EV.sharpe_ann(btc_hold, CRYPTO_ANN), alpha=alpha,
                   alpha_t=t_alpha, beta=beta, sharpe_recent=EV.sharpe_ann(sub(r, rec), CRYPTO_ANN),
                   sharpe_cost25=rob["cost25bps"]["sharpe"], sharpe_cost2x=rob["cost2x"]["sharpe"],
                   null_p95=EV.percentile_of(null), dsr=None)
    return {"metrics": metrics, "returns": r, "null_sharpes": null, "robustness": rob, "robustness_btc": rob_btc,
            "descriptive": {"maxdd": maxdd(r), "maxdd_btc": maxdd(btc_hold), "exposure_mean": float(res.exposure[hc].mean()),
                            "turnover_per_year": float(res.turnover[hc].sum() / (len(r) / 365)),
                            "costs_per_year": float(res.costs[hc].sum() / (len(r) / 365)),
                            "yearly": yearly(r, dates), "yearly_btc": yearly(btc_hold, dates),
                            "sharpe_post_pub_2025_04": EV.sharpe_ann(sub(r, post), CRYPTO_ANN),
                            "profit_share_best_5pct_days": float(q[:max(1, len(q) // 20)].clip(min=0).sum() / max(r.clip(min=0).sum(), 1e-300)),
                            "sharpe_btc_only": EV.sharpe_ann(rb, CRYPTO_ANN), "sharpe_btc_eth_5050": EV.sharpe_ann(mix5050, CRYPTO_ANN),
                            "sharpe_ew_top20": EV.sharpe_ann(ew, CRYPTO_ANN)}}


# ----------------------------------------------------------------------------------------------- H1b
def eval_h1b(ctx: Ctx) -> dict:
    d, he, B, rng = ctx.d, ctx.he, ctx.reps["boot"], np.random.default_rng(S.SEED)
    dates, cash = d.pe.dates[he], d.cash_e[he]
    res = ctx.etf(ET.Config())
    r = res.returns[he]
    ctx.reg("H1b", r, cash)
    pt, lo, hi = EV.ci_sharpe(r, ETF_ANN, S.BOOT_BLOCK_ETF, cash, reps=B)
    b6040 = bench_6040(ctx, he)
    yb, yr = yearly(b6040, dates), yearly(r, dates)
    post = span(d.pe, ctx.per["H1B_POST"])
    s2010 = lambda x, c: EV.sharpe_ann(x[post.start - he.start:post.stop - he.start], ETF_ANN, c[post.start - he.start:post.stop - he.start])  # noqa: E731
    rob = {}
    for k, cfg in H1B_ROBUST.items():
        rr = ctx.etf(cfg).returns[he]
        rob[k] = {"sharpe": EV.sharpe_ann(rr, ETF_ANN, cash), "maxdd": maxdd(rr)}
        ctx.reg(f"H1b/{k}", rr, cash)
    cols = ET.columns(d.tick, ET.Config().tickers)
    E, sig = ET.signals(d.pe, cols, ET.Config())
    f_, l_ = EV.listed_spans(d.pe.listed)
    null = np.array([EV.sharpe_ann(ctx.etf(ET.Config(), E_override=EV.circular_shift(E, f_, l_, rng), cache=(E, sig))
                                   .returns[he], ETF_ANN, cash) for _ in range(ctx.reps["null_trend"])])
    ew = vector_simulate(d.pe, _ew_etf(ctx), ET.etf_cost(d.pe.dates, d.pe.shape), d.cash_e).returns[he]
    spy = ctx.r_spy[he]
    ctx.reg("H1b/ew16", ew, cash); ctx.reg("H1b/60_40", b6040, cash); ctx.reg("H1b/spy", spy, cash)
    ctx.etf_res = res
    metrics = dict(sharpe=pt, sharpe_lo=lo, sharpe_hi=hi, sharpe_2010=s2010(r, cash),
                   ret_2008=yr.get(2008, np.nan), bench_2008=yb.get(2008, np.nan), ret_2022=yr.get(2022, np.nan),
                   bench_2022=yb.get(2022, np.nan), sharpe_cost3x=rob["cost3x"]["sharpe"],
                   corr_spy=float(np.corrcoef(r, spy)[0, 1]))
    return {"metrics": metrics, "returns": r, "null_sharpes": null, "robustness": rob,
            "descriptive": {"maxdd": maxdd(r), "maxdd_6040": maxdd(b6040), "exposure_mean": float(res.exposure[he].mean()),
                            "yearly": yr, "yearly_6040": yb, "sharpe_6040": EV.sharpe_ann(b6040, ETF_ANN, cash),
                            "sharpe_spy": EV.sharpe_ann(spy, ETF_ANN, cash), "sharpe_ew16": EV.sharpe_ann(ew, ETF_ANN, cash),
                            "null_p95": EV.percentile_of(null)}}


def _ew_etf(ctx: Ctx):
    from qlab.engine.vector import Decisions
    p = ctx.d.pe
    cols = ET.columns(ctx.d.tick, S.ETF_TREND)
    month = p.dates.astype("datetime64[M]")
    days = np.array([t for t in range(ctx.first_e, len(p.dates) - 1) if month[t + 1] != month[t] or t == ctx.first_e])
    w = np.zeros((len(days), p.shape[1]))
    for k, t in enumerate(days):
        live = [c for c in cols if p.listed[t, c] and p.tradable[t, c]]
        w[k, live] = 1.0 / len(live) if live else 0.0
    return Decisions(days, w)


# ------------------------------------------------------------------------------------------------ H2
def eval_h2(ctx: Ctx) -> dict:
    d, B, rng = ctx.d, ctx.reps["boot"], np.random.default_rng(S.SEED)
    cfg = R2.Config()
    out, start = ctx.rsi(cfg)
    cash = d.cash_e[start:]
    dates = d.pe.dates[start:]
    tr = out.trades
    net, ed = tr["net_ret"].to_numpy(), tr["entry_day"].to_numpy()
    mean_trade, tlo, thi = EV.trade_ci_by_entry_day(net, ed, reps=B)
    post_i = span(d.pe, ctx.per["H2_POST"]).start
    post = ed >= post_i
    post_r = slice(post_i - start, None)
    ctx.reg("H2", out.returns, cash)
    # controls
    c1 = R2.c1_regime_long(d.pe, d.tick, cfg, start, d.cash_e)
    r_c1 = c1.returns[start:]
    cols = np.array([d.tick.index(t) for t in cfg.tickers])
    ind = R2.indicators(d.pe, cols)
    with np.errstate(invalid="ignore"):
        regime = (ind.P > ind.sma200) & ind.valid
    zero = np.zeros(d.pe.shape)
    c2_mean, c2_sh = [], []
    for _ in range(ctx.reps["null_h2"]):
        c = R2.c2_candidates(tr["asset"].to_numpy(), tr["held"].to_numpy(), regime, start, d.pe.shape[0], rng)
        o, _ = ctx.rsi(cfg, ind=ind, cands=c, exit_sig=zero)
        c2_mean.append(float(o.trades["net_ret"].mean()) if o.trades.height else np.nan)
        c2_sh.append(EV.sharpe_ann(o.returns, ETF_ANN, cash))
    ctx.reg("H2/c1_regime_long", r_c1, cash)
    # robustness
    rob = {}
    for k, rc in H2_ROBUST.items():
        o, _ = ctx.rsi(rc)
        rob[k] = {"mean_trade": float(o.trades["net_ret"].mean()), "n_trades": o.trades.height,
                  "sharpe": EV.sharpe_ann(o.returns, ETF_ANN, cash), "maxdd": maxdd(o.returns)}
        ctx.reg(f"H2/{k}", o.returns, cash)
    # secondary: SPY only
    spy_cfg = replace(cfg, name="h2_spy", tickers=("SPY",), max_positions=1)
    o_spy, _ = ctx.rsi(spy_cfg)
    ctx.reg("H2-SPY", o_spy.returns, cash)
    t_spy = o_spy.trades
    ctx.h2_out, ctx.h2_start = out, start
    profit = (tr["exit_value"] - tr["entry_value"]).to_numpy()
    yrs = dates[tr["exit_day"].to_numpy() - start].astype("datetime64[Y]").astype(int) + 1970
    early = (yrs >= 1999) & (yrs <= 2002)
    metrics = dict(mean_trade=mean_trade, trade_lo=tlo, trade_hi=thi, mean_trade_post=float(net[post].mean()),
                   sharpe=EV.sharpe_ann(out.returns, ETF_ANN, cash),
                   sharpe_post=EV.sharpe_ann(out.returns[post_r], ETF_ANN, cash[post_r]),
                   c2_p95=EV.percentile_of(np.array(c2_mean)[np.isfinite(c2_mean)]), sharpe_c1=EV.sharpe_ann(r_c1, ETF_ANN, cash),
                   maxdd=maxdd(out.returns), maxdd_c1=maxdd(r_c1), mean_trade_cost2x=rob["r4_cost2x"]["mean_trade"],
                   mean_trade_r1=rob["r1_open_fills"]["mean_trade"])
    return {"metrics": metrics, "robustness": rob,
            "controls": {"c2_mean_trade": np.array(c2_mean), "c2_sharpe": np.array(c2_sh)},
            "descriptive": {"n_trades": tr.height, "win_rate": float((net > 0).mean()), "avg_hold": float(tr["held"].mean()),
                            "time_in_market": float(out.exposure.mean()), "yearly": yearly(out.returns, dates),
                            "profit_share_1999_2002": float(profit[early].sum() / profit.sum()) if profit.sum() else None,
                            "spy_only": {"n_trades": t_spy.height, "win_rate": float((t_spy["net_ret"] > 0).mean()),
                                         "avg_hold": float(t_spy["held"].mean()), "time_in_market": float(o_spy.exposure.mean()),
                                         "sharpe": EV.sharpe_ann(o_spy.returns, ETF_ANN, cash)},
                            "reference_research": {"win_rate": 0.82, "avg_hold": 3.7, "time_in_market": 0.10, "sharpe": 0.62}}}


# ---------------------------------------------------------------------------------------------- H3a / H4 / H3b / H3c
def eval_h3a(ctx: Ctx) -> dict:
    d, B = ctx.d, ctx.reps["boot"]
    tom = tom_window(d.pe.dates, S.TOM_BEFORE, S.TOM_AFTER)
    sl = span(d.pe, ctx.per["TOM"])
    diff, lo, hi = EV.mean_diff_ci(ctx.r_spy[sl], tom[sl], S.BOOT_BLOCK_ETF, reps=B)
    out, start = ctx.h2_out, ctx.h2_start
    n = d.pe.shape[0]
    r_h2, expo = np.zeros(n), np.zeros(n)
    r_h2[start:], expo[start:] = out.returns, out.exposure
    r_ov = OV.h3a_overlay(r_h2, expo, ctx.r_spy, d.cash_e, tom)
    a, b, c = r_h2[sl] - d.cash_e[sl], r_ov[sl] - d.cash_e[sl], d.cash_e[sl]
    dd, dlo, dhi = EV.paired_ci(b, a, ETF_ANN, S.BOOT_BLOCK_ETF, reps=B)
    ctx.reg("H3a", r_ov[sl], c)
    m = dict(tom_diff=diff, tom_lo=lo, tom_hi=hi, diff=dd, diff_lo=dlo, diff_hi=dhi, maxdd_overlay=maxdd(r_ov[sl]),
             maxdd_base=maxdd(r_h2[sl]), sharpe_overlay=EV.sharpe_ann(r_ov[sl], ETF_ANN, c),
             sharpe_base=EV.sharpe_ann(r_h2[sl], ETF_ANN, c))
    return {"metrics": m, "criteria": EV.criteria_h3a(m), "tom_days_share": float(tom[sl].mean()),
            "tom_mean_daily": float(ctx.r_spy[sl][tom[sl]].mean()), "other_mean_daily": float(ctx.r_spy[sl][~tom[sl]].mean())}


def sleeves(ctx: Ctx, cost_mult=1.0, scale_a=None, scale_b=None):
    """Daily returns of the three sleeves on the US trading days of the H4 period (+ those days' cash)."""
    d = ctx.d
    sl = span(d.pe, ctx.per["H4"])
    us = d.pe.dates[sl]
    ra = ctx.crypto(replace(CT.Config(), cost_mult=cost_mult), scale=scale_a).returns
    rb = ctx.etf(replace(ET.Config(), cost_mult=cost_mult), scale=scale_b).returns[sl]
    rc_cfg = replace(R2.Config(), cost_mult=cost_mult)
    o, start = ctx.rsi(rc_cfg)
    rc = np.zeros(d.pe.shape[0]); rc[start:] = o.returns
    a = combine.align_to_calendar(d.pc.dates, ra, us)
    return np.column_stack([a, rb, rc[sl]]), d.cash_e[sl], sl


def eval_h4(ctx: Ctx) -> dict:
    d, B = ctx.d, ctx.reps["boot"]
    R, cash, sl = sleeves(ctx)
    us = d.pe.dates[sl]
    w = np.full(3, 1 / 3)
    h4 = combine.mix(R, us, w)
    b6040 = bench_6040(ctx, sl)
    spy = ctx.r_spy[sl]
    bc = combine.mix(R[:, 1:], us, np.array([0.5, 0.5]))
    ctx.reg("H4", h4, cash); ctx.reg("H4/60_40", b6040, cash); ctx.reg("H4/spy", spy, cash); ctx.reg("H4/b_c_5050", bc, cash)
    R2x, _, _ = sleeves(ctx, cost_mult=2.0)
    h4x = combine.mix(R2x, us, w)
    h4iv = combine.mix(R, us, combine.inverse_vol_weights(252))
    ctx.reg("H4/cost2x", h4x, cash); ctx.reg("H4/invvol", h4iv, cash)
    sh = lambda x: EV.sharpe_ann(x, ETF_ANN, cash)  # noqa: E731
    dd, dlo, dhi = EV.paired_ci(h4 - cash, b6040 - cash, ETF_ANN, S.BOOT_BLOCK_ETF, reps=B)
    sleeve_sh = [sh(R[:, i]) for i in range(3)]
    m = dict(sharpe=sh(h4), best_sleeve=max(sleeve_sh), diff=dd, diff_lo=dlo, diff_hi=dhi, maxdd=maxdd(h4),
             maxdd_6040=maxdd(b6040), sharpe_cost2x=sh(h4x), sharpe_6040=sh(b6040))
    C = np.corrcoef(np.column_stack([R, spy]).T)
    crash = spy < -0.02
    C_crash = np.corrcoef(np.column_stack([R, spy])[crash].T) if crash.sum() > 5 else None
    y22 = us.astype("datetime64[Y]").astype(int) + 1970 == 2022
    C_22 = np.corrcoef(np.column_stack([R, spy])[y22].T) if y22.sum() > 5 else None
    cov = np.cov(R.T)
    rc = (w * (cov @ w)) / (w @ cov @ w)
    ctx.h4 = (R, cash, sl, h4)
    return {"metrics": m, "criteria": EV.criteria_h4(m), "returns": h4,
            "descriptive": {"sleeve_sharpes": dict(zip("ABC", sleeve_sh)), "sharpe_spy": sh(spy), "sharpe_b_c_5050": sh(bc),
                            "sharpe_invvol": sh(h4iv), "maxdd_sleeves": [maxdd(R[:, i]) for i in range(3)],
                            "corr_total": C, "corr_spy_down_2pct": C_crash, "corr_2022": C_22, "risk_contribution": rc,
                            "labels": ["A_crypto", "B_etf_trend", "C_rsi2", "SPY"], "yearly": yearly(h4, us),
                            "yearly_6040": yearly(b6040, us)}}


def eval_h3bc(ctx: Ctx) -> dict:
    d, B = ctx.d, ctx.reps["boot"]
    R, cash, sl, h4 = ctx.h4
    us = d.pe.dates[sl]
    r3b, k = OV.h3b_vol_target(h4, cash)
    ctx.reg("H3b", r3b, cash)
    dd, lo, hi = EV.paired_ci(r3b - cash, h4 - cash, ETF_ANN, S.BOOT_BLOCK_ETF, reps=B)
    mb = dict(diff=dd, diff_lo=lo, diff_hi=hi, maxdd_overlay=maxdd(r3b), maxdd_base=maxdd(h4),
              sharpe_overlay=EV.sharpe_ann(r3b, ETF_ANN, cash), sharpe_base=EV.sharpe_ann(h4, ETF_ANN, cash))
    # H3c: crash filter on the weights of the two trend sleeves, then H4 again
    Pc = tr_index(d.pc)[:, ctx.btc]
    vc = np.asarray(d.pc.listed & d.pc.tradable)[:, ctx.btc]
    Pe = tr_index(d.pe)[:, ctx.spy]
    ve = np.asarray(d.pe.listed & d.pe.tradable)[:, ctx.spy]
    crash_c, crash_e = OV.crash_state(Pc, vc, CRYPTO_ANN), OV.crash_state(Pe, ve, ETF_ANN)
    Rc, _, _ = sleeves(ctx, 1.0, OV.crash_scale(crash_c), OV.crash_scale(crash_e))
    h3c = combine.mix(Rc, us, np.full(3, 1 / 3))
    ctx.reg("H3c", h3c, cash)
    dd, lo, hi = EV.paired_ci(h3c - cash, h4 - cash, ETF_ANN, S.BOOT_BLOCK_ETF, reps=B)
    mc = dict(diff=dd, diff_lo=lo, diff_hi=hi, maxdd_overlay=maxdd(h3c), maxdd_base=maxdd(h4),
              sharpe_overlay=EV.sharpe_ann(h3c, ETF_ANN, cash), sharpe_base=EV.sharpe_ann(h4, ETF_ANN, cash))
    return {"H3b": {"metrics": mb, "criteria": EV.criteria_overlay(mb), "mean_k": float(k.mean())},
            "H3c": {"metrics": mc, "criteria": EV.criteria_overlay(mc),
                    "crash_share_crypto": float(crash_c.mean()), "crash_share_etf": float(crash_e.mean())}}


# ------------------------------------------------------------------------------------------------- DSR / driver
def add_dsr(ctx: Ctx, out: dict) -> dict:
    """Per-period Sharpe variance over all counted configurations; DSR with N = 7 (prereg N_meth, used by the
    criteria), with N = all counted configurations and with the cumulative crypto N (89 + crypto configurations)."""
    ids = [p["id"] for p in plan() if p["counted"]]
    srs = np.array([per_period_sharpe(*ctx.series[i]) for i in ids if i in ctx.series])
    var = float(srs.var(ddof=1))
    n_all = len(ids)
    n_crypto = S.CUM_CRYPTO_N_PRIOR + sum(1 for p in plan() if p["counted"] and p["crypto"])
    res = {"sr_variance_per_period": var, "n_primary": 7, "n_all": n_all, "n_cumulative_crypto": n_crypto, "dsr": {}}
    for i in ("H1a", "H1b", "H2", "H3a", "H3b", "H3c", "H4", "H1a-BTC", "H2-SPY"):
        r, cash = ctx.series[i]
        x = r - cash
        res["dsr"][i] = {"n7": deflated_sharpe(x, 7, var), "n_all": deflated_sharpe(x, n_all, var),
                         "n_cum_crypto": deflated_sharpe(x, n_crypto, var)}
    out["h1a"]["metrics"]["dsr"] = res["dsr"]["H1a"]["n7"]
    return res


def evaluate(data: Data) -> dict:
    ctx = Ctx(data)
    out = {"h1a": eval_h1a(ctx), "h1b": eval_h1b(ctx), "h2": eval_h2(ctx)}
    out["h3a"] = eval_h3a(ctx)
    out["h4"] = eval_h4(ctx)
    out.update({"h3b": (x := eval_h3bc(ctx))["H3b"], "h3c": x["H3c"]})
    out["dsr"] = add_dsr(ctx, out)
    out["h1a"]["criteria"] = EV.criteria_h1a(out["h1a"]["metrics"])
    out["h1b"]["criteria"] = EV.criteria_h1b(out["h1b"]["metrics"])
    out["h2"]["criteria"] = EV.criteria_h2(out["h2"]["metrics"])
    out["verdict"] = {k: EV.passed(out[k]["criteria"]) for k in ("h1a", "h1b", "h2", "h3a", "h3b", "h3c", "h4")}
    return out


# --------------------------------------------------------------------------------------- synthetic (smoke)
def synthetic_data(seed: int = 19) -> Data:
    """Small synthetic bundle with the same shapes and symbol names as the real one. Used by `smoke` only."""
    from qlab.data.normalize import normalize_prices
    from qlab.data.panel import panel_from_bars
    from qlab.data.synthetic import generate_market
    rng = np.random.default_rng(seed)
    days = np.arange(np.datetime64("2008-08-01"), np.datetime64("2012-12-31") + 1)
    T = len(days)
    syms = ["BTCUSDT", "ETHUSDT", "PAXGUSDT"] + [f"S{i}USDT" for i in range(30)]
    N = len(syms)
    start = np.r_[0, 0, 100, rng.integers(0, 300, N - 3)]
    r_oc = rng.normal(0.0008, 0.035, (T, N))
    r_co = rng.normal(0.0, 0.01, (T, N))
    live = np.arange(T)[:, None] >= start[None, :]
    r_oc, r_co = np.where(live, r_oc, 0.0), np.where(live, r_co, 0.0)
    qv = np.where(live, np.exp(rng.normal(19.5, 1.0, (T, N))), np.nan)
    close = np.where(live, 100 * np.cumprod((1 + r_co) * (1 + r_oc), axis=0), np.nan)
    pc = Panel(days, np.arange(N), r_co, r_oc, live.copy(), live.copy(), np.zeros((T, N), bool), close, qv.copy())
    m = generate_market(n_assets=len(S.ALL_ETF), seed=seed, start="1999-01-04", end="2012-12-31", p_halt=0,
                        p_no_open=0, p_delist=0)
    pe = panel_from_bars(normalize_prices(m.prices, m.delistings, m.calendar), m.calendar)
    periods = {"H1A": (date(2010, 6, 1), date(2012, 12, 31)), "H1A_RECENT": (date(2012, 1, 1), date(2012, 12, 31)),
               "H1A_POST_PUB": (date(2012, 7, 1), date(2012, 12, 31)), "H1B": (date(2007, 7, 2), date(2012, 12, 31)),
               "H1B_POST": (date(2010, 1, 1), date(2012, 12, 31)), "H2": (date(1999, 11, 1), date(2012, 12, 31)),
               "H2_POST": (date(2009, 1, 1), date(2012, 12, 31)), "TOM": (date(2002, 1, 2), date(2012, 12, 31)),
               "H4": (date(2010, 6, 1), date(2012, 12, 31))}
    return Data(pc, np.nan_to_num(qv), syms, np.full(T, 0.00008), pe, list(S.ALL_ETF), np.full(pe.shape[0], 0.00008),
                periods, {"boot": 100, "null_trend": 4, "null_h2": 4}, {"synthetic": seed})
