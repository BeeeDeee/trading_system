"""Research 10, part A (prereg §4): pairs of research 4 sleeves, static 50/50 vs monthly switchers.
Only data <= 2019-12-31; selection 2001-2012, check 2013-2019.

    uv run python scripts/r10_pairs.py sharadar_2026-09-25     -> docs/research10/results/pairs.json
    uv run python scripts/r10_pairs.py sharadar_2026-09-25 posthoc2004   -> pairs_posthoc2004.json
      (post hoc, descriptive: selection window 2004-2012 so that bond ETFs are eligible; prereg §9)
"""

import itertools
import json
import sys
from pathlib import Path

import numpy as np

from qlab.research10.setup import A_CHECK, A_SELECT, CHECK_SUBPERIODS, registry, load_sleeves, window
from qlab.research10.sim import monthly_mix
from qlab.validation.metrics import max_drawdown
from qlab.validation.stats import deflated_sharpe, sharpe

ANN, ANN_M = np.sqrt(252), np.sqrt(12)
MIN_SR = 0.3
CRISIS_DD = 0.15
snapshot = sys.argv[1]
POSTHOC = len(sys.argv) > 2 and sys.argv[2] == "posthoc2004"
if POSTHOC:
    from qlab.research10.setup import B_SELECT as A_SELECT  # noqa: F811
R, dates, tab = load_sleeves(snapshot)
names, fams = tab["name"].to_list(), tab["family"].to_list()
spy, ief = names.index("etf_SPY"), names.index("etf_IEF")


def stats(r: np.ndarray) -> dict:
    nav = np.cumprod(1 + r)
    return {"cagr": float(nav[-1] ** (252 / len(r)) - 1), "sharpe": float(sharpe(r) * ANN),
            "vol": float(r.std(ddof=1) * ANN), "max_dd": float(max_drawdown(nav)[0])}


def monthly(r: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compounded calendar-month returns of the columns of r (NaN -> 0), and the month labels."""
    m = d.astype("datetime64[M]")
    labels, start = np.unique(m, return_index=True)
    g = np.log1p(np.nan_to_num(r))
    return np.expm1(np.add.reduceat(g, start, axis=0)), labels


# SPY drawdown from its own running maximum (point in time) -> crisis days
spy_nav = np.cumprod(1 + np.nan_to_num(R[:, spy]))
crisis = spy_nav / np.maximum.accumulate(spy_nav) - 1 < -CRISIS_DD

sel, chk = window(dates, *A_SELECT), window(dates, *A_CHECK)
elig = [s for s in range(len(names)) if names[s] != "cash" and not np.isnan(R[sel, s]).any()]
pairs = [(a, b) for a, b in itertools.combinations(elig, 2) if fams[a] != fams[b]]
print(f"eligible sleeves {len(elig)}, cross-family pairs {len(pairs)}, crisis days in selection "
      f"{int(crisis[sel].sum())}")
reg = registry()
cfg = {"study": "research10", "part": "A", "rule": "max 50/50 Sharpe, both SR >= 0.3", "n_pairs": len(pairs)}
if POSTHOC:
    cfg = {**cfg, "window": "2004-2012 posthoc"}
if not reg.has(cfg):
    reg.record("other", cfg, n_configs=len(pairs), note="r10 A pair scan 2001-2012")

# --- selection window: static 50/50 for every pair (chunked), single sleeves, correlations
Rs = np.nan_to_num(R[sel])
d_sel = dates[sel]
sr_single = {s: sharpe(Rs[:, s]) * ANN for s in elig}
pair_sr, pair_rows = np.empty(len(pairs)), []
for k in range(0, len(pairs), 400):
    chunk = pairs[k:k + 400]
    W = np.zeros((len(chunk), R.shape[1]))
    for i, (a, b) in enumerate(chunk):
        W[i, [a, b]] = 0.5
    mix = monthly_mix(Rs, W, d_sel)
    pair_sr[k:k + len(chunk)] = np.array([sharpe(mix[:, i]) for i in range(len(chunk))]) * ANN
pair_sr_daily = pair_sr / ANN

# --- monthly switchers (signals may use history before the window; evaluation inside it)
M, mlab = monthly(R, dates)
mlab_sel = (mlab >= np.datetime64(A_SELECT[0], "M")) & (mlab <= np.datetime64(A_SELECT[1], "M"))
mi = np.flatnonzero(mlab_sel)
cum12 = np.full_like(M, np.nan)
for i in range(12, len(M)):
    cum12[i] = np.prod(1 + M[i - 12:i], axis=0) - 1          # 12 months ending at month i-1


def switch_sr(a: int, b: int) -> dict:
    ma, mb = M[mi, a], M[mi, b]
    static = 0.5 * ma + 0.5 * mb                             # monthly rebalanced 50/50 (monthly view)
    oracle = np.maximum(ma, mb)
    prev_a, prev_b = M[mi - 1, a], M[mi - 1, b]
    mom1 = np.where(prev_a >= prev_b, ma, mb)
    mom12 = np.where(cum12[mi, a] >= cum12[mi, b], ma, mb)
    f = lambda x: float(sharpe(x) * ANN_M)                   # noqa: E731
    out = {"static": f(static), "oracle": f(oracle), "mom1": f(mom1), "mom12": f(mom12)}
    gap = out["oracle"] - out["static"]
    out["capture_mom1"] = (out["mom1"] - out["static"]) / gap if gap > 0 else float("nan")
    out["capture_mom12"] = (out["mom12"] - out["static"]) / gap if gap > 0 else float("nan")
    return out


ok = np.array([sr_single[a] >= MIN_SR and sr_single[b] >= MIN_SR for a, b in pairs])
order = [i for i in np.argsort(-pair_sr) if ok[i]]
print(f"pairs with both SR >= {MIN_SR}: {int(ok.sum())}")


def pair_info(i: int) -> dict:
    a, b = pairs[i]
    ra, rb = Rs[:, a], Rs[:, b]
    cs = crisis[sel]
    return {"a": names[a], "b": names[b], "family_a": fams[a], "family_b": fams[b],
            "sr_a": float(sr_single[a]), "sr_b": float(sr_single[b]), "sr_mix": float(pair_sr[i]),
            "corr": float(np.corrcoef(ra, rb)[0, 1]),
            "corr_crisis": float(np.corrcoef(ra[cs], rb[cs])[0, 1]) if cs.sum() > 20 else None,
            "switch_monthly": switch_sr(a, b)}


top = [pair_info(i) for i in order[:10]]
# switching potential across all qualifying pairs (descriptive)
sw = [switch_sr(*pairs[i]) for i in order]
dist = {k: {q: float(np.nanquantile([s[k] for s in sw], q)) for q in (0.1, 0.5, 0.9)}
        for k in ("static", "oracle", "mom1", "mom12", "capture_mom1", "capture_mom12")}
# lowest crisis correlation among qualifying pairs (descriptive: the "they complement each other" pairs)
cc = []
for i in order:
    a, b = pairs[i]
    cc.append((np.corrcoef(Rs[crisis[sel], a], Rs[crisis[sel], b])[0, 1], i))
most_compl = [pair_info(i) for _, i in sorted(cc)[:10]]

# library level: EW all eligible, oracle / mom1 top-1 across all eligible sleeves
E = np.array(elig)
ew_sel = monthly_mix(Rs, np.isin(np.arange(R.shape[1]), E) / len(E), d_sel)
Me = M[np.ix_(mi, E)]
lib = {"ew_all_daily": stats(ew_sel),
       "ew_all_monthly_sr": float(sharpe(Me.mean(axis=1)) * ANN_M),
       "oracle_top1_monthly_sr": float(sharpe(Me.max(axis=1)) * ANN_M),
       "mom1_top1_monthly_sr": float(sharpe(Me[np.arange(len(mi)), np.argmax(M[np.ix_(mi - 1, E)], axis=1)]) * ANN_M),
       "mom12_top1_monthly_sr": float(sharpe(Me[np.arange(len(mi)), np.argmax(np.nan_to_num(cum12[np.ix_(mi, E)], nan=-9), axis=1)]) * ANN_M)}

dsr = deflated_sharpe(monthly_mix(Rs, np.isin(np.arange(R.shape[1]), pairs[order[0]]) * 0.5, d_sel),
                      len(pairs), float(np.var(pair_sr_daily, ddof=1)))

# --- check window 2013-2019: top 10 pairs vs SPY, 60/40, EW all
Rc, d_chk = np.nan_to_num(R[chk]), dates[chk]
w6040 = np.zeros(R.shape[1]); w6040[spy], w6040[ief] = 0.6, 0.4
bench = {"SPY": Rc[:, spy], "60/40": monthly_mix(Rc, w6040, d_chk),
         "EW_all": monthly_mix(Rc, np.isin(np.arange(R.shape[1]), E) / len(E), d_chk)}


def sub_sr(r: np.ndarray) -> dict:
    out = {}
    for k, (a, b) in CHECK_SUBPERIODS.items():
        s = window(d_chk, a, b)
        out[k] = float(sharpe(r[s]) * ANN)
    return out


check = {"benchmarks": {k: {**stats(v), "subperiods_sharpe": sub_sr(v)} for k, v in bench.items()}, "pairs": []}
for rank, i in enumerate(order[:10]):
    a, b = pairs[i]
    w = np.zeros(R.shape[1]); w[[a, b]] = 0.5
    r = monthly_mix(Rc, w, d_chk)
    check["pairs"].append({"rank": rank + 1, "a": names[a], "b": names[b], **stats(r), "subperiods_sharpe": sub_sr(r),
                           "single_a": stats(Rc[:, a]), "single_b": stats(Rc[:, b])})
pa = check["pairs"][0]
b6040, bspy = check["benchmarks"]["60/40"], check["benchmarks"]["SPY"]
crit = {"1_sr_vs_spy": pa["sharpe"] > bspy["sharpe"], "2_sr_vs_6040": pa["sharpe"] > b6040["sharpe"],
        "3_mdd_vs_spy": pa["max_dd"] < bspy["max_dd"],          # max_drawdown is a positive fraction
        "4_both_subperiods_vs_6040": all(pa["subperiods_sharpe"][k] > b6040["subperiods_sharpe"][k]
                                         for k in CHECK_SUBPERIODS)}
out = {"selection": {"period": [str(x) for x in A_SELECT], "n_eligible_sleeves": len(elig), "n_pairs": len(pairs),
                     "n_qualifying": int(ok.sum()), "crisis_days": int(crisis[sel].sum()),
                     "spy": stats(Rs[:, spy]), "pair_sr_quantiles": {q: float(np.quantile(pair_sr[ok], q)) for q in (0.1, 0.5, 0.9, 1.0)},
                     "top10": top, "lowest_crisis_corr": most_compl, "switching_distribution": dist,
                     "library": lib, "dsr_P_A": float(dsr)},
       "check": {"period": [str(x) for x in A_CHECK], **check, "criteria": crit, "pass": all(crit.values())}}
Path("docs/research10/results").mkdir(parents=True, exist_ok=True)
Path(f"docs/research10/results/pairs{'_posthoc2004' if POSTHOC else ''}.json").write_text(json.dumps(out, indent=1, default=float))

print(f"\nSPY selection SR {out['selection']['spy']['sharpe']:.2f}; pair 50/50 SR quantiles {out['selection']['pair_sr_quantiles']}")
print("top 10 pairs (selection 2001-2012):")
for p_ in top:
    s = p_["switch_monthly"]
    print(f"  {p_['a']:<22} + {p_['b']:<22} SR {p_['sr_a']:.2f}/{p_['sr_b']:.2f} -> {p_['sr_mix']:.2f}  corr {p_['corr']:+.2f} "
          f"crisis {p_['corr_crisis'] if p_['corr_crisis'] is None else round(p_['corr_crisis'], 2)}  | monthly static {s['static']:.2f} "
          f"oracle {s['oracle']:.2f} mom1 {s['mom1']:.2f} mom12 {s['mom12']:.2f}")
print("lowest crisis correlation (qualifying):")
for p_ in most_compl[:5]:
    print(f"  {p_['a']:<22} + {p_['b']:<22} crisis corr {p_['corr_crisis']:+.2f}  SR mix {p_['sr_mix']:.2f}  "
          f"oracle {p_['switch_monthly']['oracle']:.2f} mom1 {p_['switch_monthly']['mom1']:.2f}")
print("switching distribution over qualifying pairs:", json.dumps(dist, indent=0))
print("library:", json.dumps(lib, indent=0))
print(f"DSR P_A (N={len(pairs)}): {dsr:.3f}")
print("\ncheck 2013-2019:")
for k, v in check["benchmarks"].items():
    print(f"  {k:<8} CAGR {v['cagr'] * 100:5.1f} %  SR {v['sharpe']:.2f}  MDD {v['max_dd'] * 100:6.1f} %  sub {v['subperiods_sharpe']}")
for p_ in check["pairs"]:
    print(f"  #{p_['rank']:<2} {p_['a']:<22} + {p_['b']:<22} CAGR {p_['cagr'] * 100:5.1f} %  SR {p_['sharpe']:.2f}  "
          f"MDD {p_['max_dd'] * 100:6.1f} %  sub {p_['subperiods_sharpe']}")
print("criteria P_A:", crit, "PASS" if out["check"]["pass"] else "FAIL")
