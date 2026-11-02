"""Research 19 data audit (PLAN §3.3) -> docs/research19/results/audit.json.

Usage: python scripts/r19_audit.py
"""

import json
import re

import numpy as np

from qlab.research9.strategy import universe
from qlab.research19 import setup as S

STABLE = re.compile(r"^(USDC|TUSD|BUSD|FDUSD|USDP|DAI|PAXG|EUR|EURI|AEUR|USDS|USD1|UST|USTC|XUSD|PYUSD|BFUSD|RLUSD)USDT$")
TOKENIZED_OR_WRAPPED = re.compile(r"^(WBTC|WBETH|BETH|WBNB|WETH|STETH|WSTETH|CBBTC|BNSOL|BTCB|RENBTC)USDT$|^[A-Z]{2,5}BUSDT$")


def etf_audit() -> dict:
    p, _, cols = S.load_etf()
    r = (1 + p.ret_co) * (1 + p.ret_oc) - 1
    out = {}
    for j, tk in enumerate(cols):
        lst = np.flatnonzero(p.listed[:, j])
        first, last = int(lst[0]), int(lst[-1])
        trad = p.tradable[first:last + 1, j]
        big = np.flatnonzero((np.abs(r[:, j]) > 0.25) & p.listed[:, j])
        exp = np.datetime64(S.EXPECTED_START[tk], "D")
        out[tk] = {"first": str(p.dates[first]), "last": str(p.dates[last]), "expected_first": str(exp),
                   "start_diff_days": int((p.dates[first] - exp).astype(int)),
                   "not_tradable_days_after_first": int((~trad).sum()),
                   "abs_ret_gt_25pct": [{"date": str(p.dates[i]), "ret": float(r[i, j])} for i in big]}
    return out


def crypto_audit() -> dict:
    p, ex, syms = S.load_crypto()
    qv = np.asarray(ex["qv"])
    T = len(p.dates)
    listed_cum = np.cumsum(np.asarray(p.listed), axis=0)
    month = p.dates.astype("datetime64[M]")
    ends = [t for t in range(T - 1) if month[t + 1] != month[t]]
    excl = np.array([x in S.EXCLUDED_NONCRYPTO for x in syms])
    members, ever, ever_noage, ever_raw, counts = {}, set(), set(), set(), {}
    member_mask = np.zeros(p.shape, dtype=bool)   # day t of a member in force (rebalance at month end t0 applies to t0..next month end)

    for t in ends:
        ok0 = np.asarray(p.tradable[t - 59:t + 1]).all(axis=0) & ~excl if t >= 59 else np.zeros(len(syms), bool)
        v0 = np.where(ok0, np.nanmean(qv[max(0, t - 29):t + 1], axis=0), -np.inf)
        u = [int(j) for j in np.argsort(-v0, kind="stable")[:S.CRYPTO_TOP_N] if ok0[j]]   # no age filter
        raw = universe(p, qv, t, S.CRYPTO_TOP_N)                                          # before exclusion
        ever_raw.update(int(j) for j in raw)
        # PLAN 4.1: age filter and exclusion BEFORE the top-20 cut
        ok = ok0 & (listed_cum[t] >= S.CRYPTO_MIN_AGE)
        v = np.where(ok, np.nanmean(qv[max(0, t - 29):t + 1], axis=0), -np.inf)
        top = [int(j) for j in np.argsort(-v, kind="stable")[:S.CRYPTO_TOP_N] if ok[j]]
        counts[str(month[t])] = len(top)
        member_mask[t:min(T, t + 32), :][:, top] |= (month[t:min(T, t + 32)] == month[t] + 1)[:, None]
        members[str(month[t])] = [syms[j] for j in top]
        ever.update(top); ever_noage.update(int(j) for j in u)
    sel = sorted(ever | ever_noage)
    r = (1 + np.asarray(p.ret_co)[:, sel]) * (1 + np.asarray(p.ret_oc)[:, sel]) - 1
    big = [{"symbol": syms[j], "date": str(p.dates[i]), "ret": float(r[i, k])}
           for k, j in enumerate(sel) for i in np.flatnonzero((np.abs(r[:, k]) > 0.60) & member_mask[:, j])]
    names = sorted(syms[j] for j in sel)
    flagged = [s for s in names if STABLE.match(s) or TOKENIZED_OR_WRAPPED.match(s)]
    return {"n_symbols_panel": len(syms), "first": str(p.dates[0]), "last": str(p.dates[-1]),
            "members_per_month": counts, "members_by_month": members,
            "ever_top20_aged": sorted(syms[j] for j in ever), "ever_top20_without_age": sorted(syms[j] for j in ever_noage),
            "ever_top20_union": names, "ever_top20_before_exclusion": sorted(syms[j] for j in ever_raw),
            "excluded": sorted(S.EXCLUDED_NONCRYPTO), "excluded_found_in_final": sorted(set(names) & S.EXCLUDED_NONCRYPTO), "flagged_by_regex": flagged, "abs_ret_gt_60pct_top20_members": big,
            "all_symbols": syms}


if __name__ == "__main__":
    res = {"etf": etf_audit(), "crypto": crypto_audit()}
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    (S.RESULTS / "audit.json").write_text(json.dumps(res, indent=1))
    print(f"{'tk':5} {'first':11} {'expected':11} diff  nontrad  big")
    for tk, d in res["etf"].items():
        print(f"{tk:5} {d['first']} {d['expected_first']} {d['start_diff_days']:4d} {d['not_tradable_days_after_first']:7d} "
              f"{len(d['abs_ret_gt_25pct'])}")
    c = res["crypto"]
    print("last ETF dates:", {d['last'] for d in res['etf'].values()})
    print(f"crypto {c['first']}..{c['last']}, members/month min {min(c['members_per_month'].values())} "
          f"max {max(c['members_per_month'].values())}")
    print("ever top20 (union):", c["ever_top20_union"])
    print("flagged:", c["flagged_by_regex"], "| excluded still present:", c["excluded_found_in_final"])
    print("crypto |r|>60%:", c["abs_ret_gt_60pct_top20_members"])
