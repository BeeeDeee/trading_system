"""Research 4 / R4.3: meta layer walk-forward, position-level simulation, statistics, diagnostics.

Usage: python scripts/r4_evaluate.py sharadar_YYYY-MM-DD
Output: data/derived/<snapshot>/research4/results/ (summary.json, yearly.csv, activation.csv,
noisy_oracle.csv, robustness.csv, weights.parquet)
Follows docs/research4/PREREGISTRATION.md; all methods are recorded in the research-4 trial
registry before any result is computed.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from qlab.data.panel import load_panel
from qlab.engine.costs import cash_returns
from qlab.engine.vector import Decisions, simulate
from qlab.pipeline import load_config
from qlab.research4.meta import (MARKET_FEATURES, SLEEVE_FEATURES, gbm_scores, market_features,
                                 period_returns, rank_ic, regime_scores, ridge_design,
                                 ridge_scores, select_top, sleeve_features, xs_rank)
from qlab.research4.sleeves import cost_rates
from qlab.selection.metrics import sharpe as col_sharpe
from qlab.validation.metrics import summary
from qlab.validation.registry import TrialRegistry
from qlab.validation.stats import bootstrap_ci, deflated_sharpe, spa_pvalue

TRAIN_FROM = "2001-01-01"
FIRST_TEST_YEAR, SPLIT_YEAR = 2006, 2020
METHODS = ("M1_FACTOR_MOM", "M2_REGIME", "M3_ML_GBM", "M4_ML_RIDGE", "M5_BLEND")
PRIMARY = "M3_ML_GBM"

snapshot = sys.argv[1]
t0 = time.time()
der = Path("data/derived") / snapshot
sdir = der / "research4" / "sleeves"
out = der / "research4" / "results"
out.mkdir(parents=True, exist_ok=True)

registry = TrialRegistry(Path("runs/registry/research4.sqlite"))
for m in METHODS:
    trial = {"research": 4, "method": m, "prereg": "docs/research4/PREREGISTRATION.md v1.0"}
    if not registry.has(trial):
        registry.record("methodology_eval", trial, note=f"research 4 {m}")
n_meth = registry.n_meth

# ------------------------------------------------------------------ inputs
panel, extra = load_panel(der / "panel_r4")
cols = json.loads((der / "panel_r4" / "columns.json").read_text())["etf"]
dates, T = panel.dates, panel.shape[0]
cash = np.asarray(extra["cash_ret"])
R = np.load(sdir / "returns.npy")
days = np.load(sdir / "days.npy")
sleeves = pl.read_parquet(sdir / "sleeves.parquet")
targets = pl.read_parquet(sdir / "targets.parquet").sort("k", "sleeve")
S, D = len(sleeves), len(days)
names = sleeves["name"].to_list()
fam_names = sleeves["family"].to_list()
fam_list = sorted(set(fam_names))
fam = np.array([fam_list.index(f) for f in fam_names])
kind = np.array(sleeves["kind"].to_list())
is_stock = kind == "stock"
cash_sleeve = names.index("cash")

# A sleeve counts from its first decision with any position (prereg §11: stock sleeves whose
# score needs years of data - lt_reversal, insider - hold nothing at first).
first_k = np.full(S, D)
fk = targets.group_by("sleeve").agg(pl.col("k").min())
first_k[fk["sleeve"].to_numpy()] = fk["k"].to_numpy()
first_k[cash_sleeve] = 0
first_row = np.where(first_k < D, days[np.minimum(first_k, D - 1)] + 1, T)
Rm = R.copy()
for s in range(S):
    Rm[:first_row[s], s] = np.nan

spy_ret = (1 + np.asarray(panel.ret_co[:, cols["SPY"]])) * (1 + np.asarray(panel.ret_oc[:, cols["SPY"]])) - 1
ief_ret = (1 + np.asarray(panel.ret_co[:, cols["IEF"]])) * (1 + np.asarray(panel.ret_oc[:, cols["IEF"]])) - 1
funds = pl.read_parquet(Path("data/parquet") / snapshot / "funds.parquet",
                        columns=["ticker", "date", "close"]).filter(pl.col("ticker") == "^VIX")
vix = (pl.DataFrame({"date": dates}).with_columns(pl.col("date").cast(pl.Date))
       .join(funds.select("date", "close"), on="date", how="left")["close"]
       .fill_null(strategy="forward").to_numpy())
rf_file = pl.read_csv(sorted(Path("data/raw").glob("fred_*/DTB3.csv"))[-1], null_values=".",
                      schema_overrides={"DTB3": pl.Float64}).drop_nulls()
rd = rf_file["observation_date"].str.to_date().to_numpy().astype("datetime64[D]")
ri = np.searchsorted(rd, dates, side="left") - 1  # published strictly before the day
tbill = np.where(ri >= 0, rf_file["DTB3"].to_numpy()[np.maximum(ri, 0)], np.nan)

scores = np.load(sdir / "scores.npz")
uni = np.asarray(extra["in_liq1000"][days])
tr200, r21 = scores["trend_200"], -scores["strev_21"]
with np.errstate(invalid="ignore"):
    breadth = np.array([np.mean(tr200[k][uni[k] & np.isfinite(tr200[k])] > 0)
                        if (uni[k] & np.isfinite(tr200[k])).any() else np.nan for k in range(D)])
    dispersion = np.array([np.nanstd(np.where(uni[k], r21[k], np.nan)) for k in range(D)])

# ------------------------------------------------------------------ monthly data set
M = period_returns(Rm, days)
cash_m = period_returns(cash[:, None], days)[:, 0]
avail = (days[:, None] - first_row[None, :] + 1 >= 252) & np.isfinite(M)
avail_known = days[:, None] - first_row[None, :] + 1 >= 252   # known at decision time
y_rel = np.where(avail, M - np.nanmean(np.where(avail, M, np.nan), axis=1, keepdims=True), np.nan)
sf = sleeve_features(Rm, M, days, spy_ret, cash)
mf = market_features(days, spy_ret, ief_ret, vix, tbill, dispersion, breadth)
print(f"features ({time.time() - t0:.0f}s)", flush=True)

years = np.asarray(dates[days], dtype="datetime64[Y]").astype(int) + 1970
k_train0 = int(np.searchsorted(dates[days], np.datetime64(TRAIN_FROM)))
test_years = list(range(FIRST_TEST_YEAR, int(years[-1]) + 1))
k_oos = int(np.argmax(years >= FIRST_TEST_YEAR))

# long format rows (k, s) for the ML models
kk, ss = np.nonzero(avail_known)
X = np.column_stack([sf[n][kk, ss] for n in SLEEVE_FEATURES]
                    + [mf[n][kk] for n in MARKET_FEATURES]
                    + [fam[ss], is_stock[ss].astype(float)])
n_num = len(SLEEVE_FEATURES) + len(MARKET_FEATURES)
cat_cols = [n_num]
y_long = y_rel[kk, ss]
Z = ridge_design(X[:, :n_num], fam[ss], len(fam_list),
                 list(range(len(SLEEVE_FEATURES), n_num)))
Z = np.hstack([Z, is_stock[ss, None].astype(float)])
state = (np.nan_to_num(mf["spy_above_sma200"]) * 2 + (np.nan_to_num(mf["vix_rel"]) > 1)).astype(int)

score = {m: np.full((D, S), np.nan) for m in METHODS}
score["M1_FACTOR_MOM"] = np.where(avail_known, sf["ret_12m"], np.nan)
score["M1_FACTOR_MOM"][:k_oos] = np.nan
for y in test_years:
    test = years == y
    k_first = int(np.argmax(test))
    train = (np.arange(D) >= k_train0) & (np.arange(D) < k_first)
    score["M2_REGIME"][test] = np.where(avail_known[test], regime_scores(y_rel, state, train,
                                                                           test)[test], np.nan)
    tr_idx = np.flatnonzero(train[kk] & np.isfinite(y_long))
    te_idx = np.flatnonzero(test[kk])
    score["M3_ML_GBM"][kk[te_idx], ss[te_idx]] = gbm_scores(X, y_long, tr_idx, te_idx, cat_cols)
    score["M4_ML_RIDGE"][kk[te_idx], ss[te_idx]] = ridge_scores(Z, y_long, tr_idx, te_idx)
    print(f"  fold {y}: {len(tr_idx)} train rows ({time.time() - t0:.0f}s)", flush=True)
blend = np.nanmean(np.stack([xs_rank(score[m], avail_known) for m in
                             ("M1_FACTOR_MOM", "M2_REGIME", "M3_ML_GBM")]), axis=0)
score["M5_BLEND"] = np.where(avail_known, blend, np.nan)
for m in METHODS:
    score[m][:k_oos] = np.nan

# ------------------------------------------------------------------ portfolios
cfg = load_config()
etf_cols = list(cols.values())
rate1 = cost_rates(extra["liq_rank"], dates, cfg["costs"], etf_cols)
tk, tsl = targets["k"].to_numpy(), targets["sleeve"].to_numpy()
tcol, tw = targets["col"].to_numpy(), targets["w"].to_numpy()
bounds = np.searchsorted(tk, np.arange(D + 1))


def holdings(W: np.ndarray) -> Decisions:
    """Position-level targets: sum over sleeves of W[k, s] x sleeve target at decision k."""
    rows = np.zeros((D - k_oos, panel.shape[1]))
    for k in range(k_oos, D):
        a, b = bounds[k], bounds[k + 1]
        np.add.at(rows[k - k_oos], tcol[a:b], W[k, tsl[a:b]] * tw[a:b])
    return Decisions(days[k_oos:], rows)


oos = slice(days[k_oos] + 1, T)
rf = cash[oos]


def run(W: np.ndarray, rate=None) -> dict:
    sim = simulate(panel, holdings(W), rate1 if rate is None else rate, cash)
    return {"returns": sim.returns[oos], "turnover": sim.turnover[oos], "costs": sim.costs[oos],
            "exposure": sim.exposure[oos]}


def fixed(weights: dict[str, float]) -> np.ndarray:
    W = np.zeros((D, S))
    for n, w in weights.items():
        W[:, names.index(n)] = w
    return W


def ew(mask: np.ndarray) -> np.ndarray:
    m = avail_known & mask[None, :]
    return m / np.maximum(m.sum(axis=1, keepdims=True), 1)


bench_W = {"SPY": fixed({"etf_SPY": 1.0}), "60_40": fixed({"etf_SPY": 0.6, "etf_IEF": 0.4}),
           "EW_ALL": ew(np.ones(S, bool)), "EW_STOCK": ew(is_stock)}
weights = {m: select_top(score[m], avail_known, fam, 5, 2) for m in METHODS}
series = {n: run(W) for n, W in {**weights, **bench_W}.items()}
print(f"simulations ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ statistics


def sharpe_diff(x):
    s = col_sharpe(x[:, :2], x[:, 2])
    return float(s[0] - s[1])


def cagr_diff(x):
    g = np.exp(np.log1p(x[:, :2]).sum(axis=0) * 252 / len(x)) - 1
    return float(g[0] - g[1])


oos_years = np.asarray(dates[oos], dtype="datetime64[Y]").astype(int) + 1970
segments = {"2006-2026": slice(0, len(rf)),
            "2006-2019": slice(0, int(np.argmax(oos_years >= SPLIT_YEAR))),
            "2020-2026": slice(int(np.argmax(oos_years >= SPLIT_YEAR)), len(rf))}
stats = {}
for seg, sl in segments.items():
    st = {}
    for n, x in series.items():
        e = {"metrics": summary(x["returns"][sl], rf[sl], x["turnover"][sl], x["costs"][sl],
                                x["exposure"][sl])}
        if n in METHODS:
            r = x["returns"][sl]
            e["dsr"] = deflated_sharpe(r - rf[sl], n_meth, 1.0 / len(r))
            for b in ("SPY", "EW_ALL"):
                z = np.column_stack([r, series[b]["returns"][sl], rf[sl]])
                e[f"sharpe_diff_vs_{b}"] = bootstrap_ci(z, sharpe_diff, n_boot=1000)
                e[f"cagr_diff_vs_{b}"] = bootstrap_ci(z, cagr_diff, n_boot=1000)
                e[f"spa_p_vs_{b}"] = spa_pvalue((r - series[b]["returns"][sl])[:, None],
                                                n_boot=1000)
        st[n] = e
    stats[seg] = st
    print(f"stats {seg} ({time.time() - t0:.0f}s)", flush=True)

p, full = stats["2006-2026"][PRIMARY], "2006-2026"
criteria = {
    "1_ci_sharpe_vs_EW_ALL": p["sharpe_diff_vs_EW_ALL"][1] > 0,
    "2_ci_sharpe_vs_SPY": p["sharpe_diff_vs_SPY"][1] > 0,
    "3_dsr": p["dsr"] >= 0.90,
    "4_both_subperiods_vs_SPY": all(
        stats[s][PRIMARY]["metrics"]["sharpe"] > stats[s]["SPY"]["metrics"]["sharpe"]
        for s in ("2006-2019", "2020-2026")),
}

# ------------------------------------------------------------------ diagnostics (linear, monthly)
mo = slice(k_oos, D)


def monthly_perf(ret_m: np.ndarray) -> dict:
    r, c = ret_m[mo], cash_m[mo]
    yrs = len(r) / 12
    ex = r - c
    return {"cagr": float(np.prod(1 + r) ** (1 / yrs) - 1),
            "sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(12))}


def linear(W):
    return np.nansum(W * np.nan_to_num(M), axis=1)


Mz = np.where(avail, M, np.nan)
diag = {"oracle_top1": monthly_perf(linear(select_top(Mz, avail, fam, 1, None))),
        "oracle_top5": monthly_perf(linear(select_top(Mz, avail, fam, 5, 2))),
        "worst_top5": monthly_perf(linear(select_top(-Mz, avail, fam, 5, 2)))}
for n, W in {**weights, **bench_W}.items():
    diag[f"linear_{n}"] = monthly_perf(linear(W))
ic = {}
for m in METHODS:
    x = rank_ic(score[m], M, avail & avail_known)[mo]
    x = x[np.isfinite(x)]
    ic[m] = {"mean": float(x.mean()), "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
             "share_positive": float((x > 0).mean())}

rng = np.random.default_rng(0)
noisy = []
rank_true = xs_rank(Mz, avail)
from statistics import NormalDist  # noqa: E402
zs = np.vectorize(lambda u: NormalDist().inv_cdf(min(max(u - 0.5 / 87, 1e-6), 1 - 1e-6)))(
    np.nan_to_num(rank_true, nan=0.5))
for sigma in (0.5, 1, 2, 3, 5, 8, 12, 20):
    ics, perf = [], []
    for rep in range(20):
        sc = np.where(avail, zs + rng.normal(0, sigma, zs.shape), np.nan)
        x = rank_ic(sc, M, avail)[mo]
        ics.append(np.nanmean(x))
        perf.append(monthly_perf(linear(select_top(sc, avail, fam, 5, 2))))
    noisy.append({"sigma": sigma, "ic": float(np.mean(ics)),
                  "cagr": float(np.mean([q["cagr"] for q in perf])),
                  "sharpe": float(np.mean([q["sharpe"] for q in perf]))})
pl.DataFrame(noisy).write_csv(out / "noisy_oracle.csv")

sel = weights[PRIMARY][mo] > 0
yrs_m = years[mo]
act = []
for f_i, f_n in enumerate(fam_list):
    row = {"family": f_n, "all": float(sel[:, fam == f_i].any(axis=1).mean())}
    for a, b in ((2006, 2009), (2010, 2014), (2015, 2019), (2020, 2022), (2023, 2026)):
        k = (yrs_m >= a) & (yrs_m <= b)
        row[f"{a}-{b}"] = float(sel[k][:, fam == f_i].any(axis=1).mean())
    act.append(row)
pl.DataFrame(act).sort("all", descending=True).write_csv(out / "activation.csv")
wrows = [{"date": str(dates[days[k]]), "method": m, "sleeve": names[s], "w": float(W[k, s])}
         for m, W in weights.items() for k in range(k_oos, D) for s in np.flatnonzero(W[k])]
pl.DataFrame(wrows).write_parquet(out / "weights.parquet")

yearly = pl.DataFrame([{"year": int(y), **{n: float(np.prod(1 + x["returns"][oos_years == y]) - 1)
                                         for n, x in series.items()}}
                       for y in np.unique(oos_years)])
yearly.write_csv(out / "yearly.csv")

# sleeve-level reference: each sleeve's own OOS performance (linear monthly)
sleeve_perf = pl.DataFrame([{"sleeve": names[s], "family": fam_names[s],
                             **monthly_perf(np.where(np.isfinite(M[:, s]), M[:, s], cash_m))}
                            for s in range(S)]).sort("sharpe", descending=True)
sleeve_perf.write_csv(out / "sleeves_oos.csv")

# ------------------------------------------------------------------ robustness (prereg §8)
rate2 = cost_rates(extra["liq_rank"], dates, cfg["costs"], etf_cols, 2.0)
variants = {"costs_x2": dict(rate=rate2), "top3": dict(k=3), "top10": dict(k=10),
            "no_family_cap": dict(cap=None), "stock_only": dict(mask=is_stock),
            "etf_only": dict(mask=~is_stock)}
rob = []
for v, o in variants.items():
    mask = o.get("mask", np.ones(S, bool))
    rate = o.get("rate")
    refs = {"SPY": series["SPY"]["returns"], "EW_ALL": series["EW_ALL"]["returns"]}
    if rate is not None:
        refs = {b: run(bench_W[b], rate)["returns"] for b in refs}
    for m in METHODS:
        W = select_top(score[m], avail_known & mask[None, :], fam, o.get("k", 5),
                       o.get("cap", 2))
        r = run(W, rate)["returns"]
        met = summary(r, rf)
        rob.append({"variant": v, "method": m, "cagr": met["cagr"], "sharpe": met["sharpe"],
                    "max_drawdown": met["max_drawdown"],
                    **{f"sharpe_diff_vs_{b}": float(sharpe_diff(np.column_stack([r, x, rf])))
                       for b, x in refs.items()}})
pl.DataFrame(rob).write_csv(out / "robustness.csv")

result = {"period": [str(dates[oos][0]), str(dates[-1])], "n_meth": n_meth, "n_sleeves": S,
          "criteria": {k: bool(v) for k, v in criteria.items()},
          "passed": bool(all(criteria.values())), "stats": stats, "rank_ic": ic,
          "diagnostics_linear_monthly": diag, "seconds": round(time.time() - t0)}
(out / "summary.json").write_text(json.dumps(result, indent=2, default=float))

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_float_precision(3)
for seg in segments:
    print(f"\n== {seg}")
    print(pl.DataFrame([{"name": n, **{k: e["metrics"][k] for k in ("cagr", "sharpe",
                                                                     "max_drawdown",
                                                                     "turnover_annual")},
                         **({"dSR_SPY_lo": e["sharpe_diff_vs_SPY"][1],
                             "dSR_EW_lo": e["sharpe_diff_vs_EW_ALL"][1], "dsr": e["dsr"]}
                            if "dsr" in e else {})}
                        for n, e in stats[seg].items()]))
print(json.dumps({"criteria": result["criteria"], "rank_ic": ic, "diag": diag}, indent=1,
                 default=float))
print(pl.DataFrame(noisy))
print(pl.read_csv(out / "activation.csv"))
print(yearly)
print(pl.read_csv(out / "robustness.csv"))
print(sleeve_perf.head(15), sleeve_perf.tail(10))
