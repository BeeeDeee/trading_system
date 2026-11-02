"""Bootstrap p-values, Holm and hypothesis states of research 20 (prereg §7). Pure functions."""

from datetime import date, timedelta

import numpy as np

from qlab.research19 import evaluate as EV
from qlab.research20 import setup as S


def excess_sharpe(r: np.ndarray, cash: np.ndarray) -> float:
    return EV.sharpe_ann(r, 252, cash)


def boot_sharpe(r: np.ndarray, cash: np.ndarray, reps: int = S.BOOT_REPS) -> np.ndarray:
    x = np.asarray(r, float) - cash
    rng = np.random.default_rng(S.SEED)
    return np.concatenate([EV._sharpes(x[idx], 252) for idx in EV.boot_indices(len(x), S.BOOT_BLOCK, reps, rng)])


def boot_trade_means(net: np.ndarray, entry_day: np.ndarray, reps: int = S.BOOT_REPS) -> np.ndarray:
    """Bootstrap mean net trade; trades entered the same day are resampled together (blocks over ordered entry days)."""
    days, inv = np.unique(np.asarray(entry_day), return_inverse=True)
    s, c = np.bincount(inv, weights=np.asarray(net, float)), np.bincount(inv).astype(float)
    rng = np.random.default_rng(S.SEED)
    return np.concatenate([s[idx].sum(axis=1) / c[idx].sum(axis=1)
                           for idx in EV.boot_indices(len(days), S.BOOT_BLOCK, reps, rng)])


def p_value(boots: np.ndarray) -> float:
    """One-sided: (1 + #{resampled statistic <= 0}) / (B + 1)."""
    return float((1 + np.sum(boots <= 0)) / (len(boots) + 1))


def upper90(boots: np.ndarray) -> float:
    """Upper bound of the two-sided 90 % interval."""
    return float(np.quantile(boots, 0.95))


def holm(pvals: dict[str, float], alpha: float) -> dict[str, bool]:
    """Holm step-down: sorted p_(i) < alpha / (m - i); stop at the first failure."""
    out = {k: False for k in pvals}
    m = len(pvals)
    for i, (k, v) in enumerate(sorted(pvals.items(), key=lambda kv: kv[1])):
        if v < alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def look_dates(t0: date, calendar: np.ndarray) -> list[date]:
    """Last trading day on or before the day before each anniversary (12, 24, 36 months after T0)."""
    cal = np.asarray(calendar, dtype="datetime64[D]")
    out = []
    for m in S.LOOKS_MONTHS:
        ann = date(t0.year + m // 12, t0.month, t0.day) - timedelta(days=1)
        ok = cal[cal <= np.datetime64(ann, "D")]
        out.append(ok[-1].astype(object) if len(ok) else None)
    return out


def decide(look: int, hyps: dict[str, dict], alpha: float = S.ALPHA_LOOK) -> dict[str, str]:
    """States at decision `look` (1..3) for the open hypotheses (prereg §7). Each `hyps[name]` has:
    n_enough (minimum data reached), main_x1 (Sharpe for F1 / mean trade for F2 at cost x1), up90 (upper bound of its
    90 % interval), p (one-sided bootstrap p-value), plan_bar (Sharpe(x1.5) > 0, for F2 also mean trade(x1.5) > 0),
    point_ok (all point estimates at x1 > 0). Rejected hypotheses leave the Holm family at once."""
    last = look == len(S.LOOKS_MONTHS)
    out, family = {}, {}
    for k, h in hyps.items():
        if not h["n_enough"]:
            out[k] = "malo_dat"
        elif h["up90"] <= 0 or (last and not h["point_ok"]):
            out[k] = "zamitnuta"
        else:
            family[k] = h["p"]
    passed = holm(family, alpha) if family else {}
    for k in family:
        h = hyps[k]
        ok = bool(h["plan_bar"] and h["main_x1"] > 0)
        out[k] = "potvrzena" if ok and passed[k] else ("konzistentni" if ok else "neprukazne")
    if last:
        out = {k: ("nepotvrzeno" if v in ("konzistentni", "neprukazne", "malo_dat") else v) for k, v in out.items()}
    return out
