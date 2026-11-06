"""Statistics and mechanical criteria of research 21 (prereg §6, v1.1). Pure functions of arrays, plus the c2 control."""

import numpy as np

from qlab.research19.evaluate import boot_indices
from qlab.research21 import setup as S


def net_at(gross: np.ndarray, c_entry: np.ndarray, c_exit: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Net trade return for cost multiplier k: the simulator's identity (1+g)(1-c_x)/(1+c_e) - 1 (costs are a
    fraction of traded value on both legs)."""
    return (1.0 + gross) * (1.0 - k * c_exit) / (1.0 + k * c_entry) - 1.0


def month_boot_spread(net: np.ndarray, blocked: np.ndarray, month: np.ndarray, reps: int = S.BOOT_REPS,
                      block_months: float = S.BOOT_BLOCK_MONTHS, seed: int = S.SEED,
                      level: float = S.CI_LEVEL) -> tuple[float, float, float]:
    """mean(net | allowed) - mean(net | blocked) with a stationary bootstrap over the ordered calendar months of the
    signal (all trades of a month are resampled together). Returns (point, lo, hi)."""
    net, blk = np.asarray(net, float), np.asarray(blocked, bool)
    months, inv = np.unique(month, return_inverse=True)
    sb, cb = np.bincount(inv, weights=net * blk, minlength=len(months)), np.bincount(inv, weights=blk, minlength=len(months))
    sa, ca = np.bincount(inv, weights=net * ~blk, minlength=len(months)), np.bincount(inv, weights=~blk, minlength=len(months))
    point = sa.sum() / ca.sum() - sb.sum() / cb.sum() if ca.sum() and cb.sum() else float("nan")
    rng = np.random.default_rng(seed)
    boots = []
    for idx in boot_indices(len(months), block_months, reps, rng):
        with np.errstate(invalid="ignore", divide="ignore"):
            boots.append(sa[idx].sum(axis=1) / ca[idx].sum(axis=1) - sb[idx].sum(axis=1) / cb[idx].sum(axis=1))
    b = np.concatenate(boots)
    a = (1 - level) / 2
    lo, hi = np.nanquantile(b, [a, 1 - a])
    return float(point), float(lo), float(hi)


def third_spreads(net: np.ndarray, blocked: np.ndarray, year: np.ndarray, thirds) -> list[dict]:
    out = []
    for lo, hi in thirds:
        m = (year >= lo) & (year <= hi)
        a, b = net[m & ~blocked], net[m & blocked]
        out.append({"years": [lo, hi], "n_allowed": int(len(a)), "n_blocked": int(len(b)),
                    "mean_allowed": float(a.mean()) if len(a) else None,
                    "mean_blocked": float(b.mean()) if len(b) else None,
                    "spread": float(a.mean() - b.mean()) if len(a) and len(b) else None})
    return out


def filter_criteria(trades, flag: str, cost_entry: np.ndarray, cost_exit: np.ndarray, c2_p95: float,
                    thirds, reps: int = S.BOOT_REPS) -> dict:
    """The five criteria of prereg §6 for one filter on the (already period-restricted) S0 trades of the partition.
    `trades` needs net_ret, gross_ret, year, month and the blocked column `b_<flag>`."""
    net = trades["net_ret"].to_numpy()
    blk = trades[f"b_{flag}"].to_numpy().astype(bool)
    n = len(net)
    share_blocked = float(blk.mean()) if n else float("nan")
    spread, lo, hi = month_boot_spread(net, blk, trades["month"].to_numpy(), reps) if blk.any() and (~blk).any() \
        else (float("nan"),) * 3
    a_mean = float(net[~blk].mean()) if (~blk).any() else float("nan")
    thirds_out = third_spreads(net, blk, trades["year"].to_numpy(), thirds)
    gross = trades["gross_ret"].to_numpy()
    net2 = net_at(gross, cost_entry, cost_exit, 2.0)
    a2 = float(net2[~blk].mean()) if (~blk).any() else float("nan")
    c = {
        "c1_spread_ci_lo_gt0": bool(lo > 0),
        "c2_allowed_gt0_and_gt_c2_p95": bool(a_mean > 0 and a_mean > c2_p95),
        "c3_spread_gt0_in_every_third": all(t["spread"] is not None and t["spread"] > 0 for t in thirds_out),
        "c4_cost_x2_allowed_gt0": bool(a2 > 0),
        "c5_share_ge_15pct_each": bool(min(share_blocked, 1 - share_blocked) >= S.MIN_SHARE),
    }
    judged = c["c5_share_ge_15pct_each"]
    verdict = "nelze posoudit" if not judged else ("nazek" if all(c.values()) else "ne")
    return {"n_trades": n, "n_allowed": int((~blk).sum()), "n_blocked": int(blk.sum()),
            "share_blocked": share_blocked, "mean_allowed": a_mean,
            "mean_blocked": float(net[blk].mean()) if blk.any() else float("nan"),
            "spread": spread, "spread_ci": [lo, hi], "c2_p95": c2_p95, "mean_allowed_cost_x2": a2,
            "thirds": thirds_out, "criteria": c, "criteria_passed": int(sum(c.values())), "verdict": verdict}


def c2_means(feats, extra, panel, dates, sig_day: np.ndarray, hold: np.ndarray, adv_min: float, cost_mult: float,
             sims: int = S.C2_SIMS, seed: int = S.SEED) -> np.ndarray:
    """Control c2 (prereg v1.1 (e)): for every trade of the group one random trade on the SAME signal day, a random
    security of the PIT universe with P > SMA200 that trades at the next open, the same holding period, the same
    costs. Returns the mean net return of each of `sims` simulations. Slots are ignored (trades are independent)."""
    from qlab.research5.costs import period_cost
    cols = feats.cols
    T = panel.shape[0]
    entry = sig_day + 1
    exit_ = entry + hold
    keep = exit_ < T
    sig_day, entry, exit_ = sig_day[keep], entry[keep], exit_[keep]
    days, inv = np.unique(sig_day, return_inverse=True)
    uni = (np.asarray(extra["base_ok"][days])[:, cols] & (np.asarray(extra["adv20"][days])[:, cols] >= adv_min)
           & np.asarray(feats.m["valid"])[days] & np.asarray(feats.m["regime"])[days]
           & np.asarray(panel.tradable[days + 1])[:, cols])
    lens = uni.sum(axis=1)
    off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    flat = np.nonzero(uni)[1]
    has = lens[inv] > 0
    inv, entry, exit_ = inv[has], entry[has], exit_[has]
    c = period_cost(dates, True, cost_mult)
    ce, cx = c[entry], c[exit_]
    O = feats.m["O"]
    rng = np.random.default_rng(seed)
    out = np.empty(sims)
    for k in range(sims):
        j = flat[off[inv] + np.floor(rng.random(len(inv)) * lens[inv]).astype(int)]
        g = np.asarray(O[exit_, j]) / np.asarray(O[entry, j]) - 1.0
        net = net_at(g, ce, cx)
        out[k] = np.nanmean(net)
    return out
