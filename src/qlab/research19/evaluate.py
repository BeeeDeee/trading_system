"""Statistics and mechanical criteria of research 19 (prereg §7). Pure functions of arrays / metric dicts."""

import numpy as np

from qlab.research19 import setup as S


def sharpe_ann(r: np.ndarray, ann: float, cash: np.ndarray | float = 0.0) -> float:
    """Mean daily excess over cash / sd * sqrt(ann)."""
    x = np.asarray(r, float) - cash
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def boot_indices(n: int, mean_block: float, reps: int, rng: np.random.Generator, chunk: int = 500):
    """Yield (<=chunk, n) index matrices of stationary-bootstrap resamples (geometric blocks, wrap-around);
    vectorised equivalent of `validation.stats.stationary_bootstrap_indices`."""
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        starts = rng.random((m, n)) < 1.0 / mean_block
        starts[:, 0] = True
        pos = rng.integers(n, size=(m, n))
        k = np.arange(n)
        last = np.maximum.accumulate(np.where(starts, k, 0), axis=1)
        base = np.take_along_axis(pos, last, axis=1)
        yield (base + (k - last)) % n
        done += m


def _sharpes(x: np.ndarray, ann: float) -> np.ndarray:
    """Sharpe per row of x (m, n)."""
    sd = x.std(axis=1, ddof=1)
    return np.where(sd > 0, x.mean(axis=1) / np.where(sd > 0, sd, 1.0), 0.0) * np.sqrt(ann)


def _interval(boots: np.ndarray, level: float) -> tuple[float, float]:
    a = (1 - level) / 2
    lo, hi = np.quantile(boots, [a, 1 - a])
    return float(lo), float(hi)


def ci_sharpe(r: np.ndarray, ann: float, block: float, cash: np.ndarray | float = 0.0, reps: int = S.BOOT_REPS,
              seed: int = S.SEED, level: float = S.CI_LEVEL) -> tuple[float, float, float]:
    x = np.asarray(r, float) - cash
    rng = np.random.default_rng(seed)
    boots = np.concatenate([_sharpes(x[idx], ann) for idx in boot_indices(len(x), block, reps, rng)])
    return sharpe_ann(x, ann), *_interval(boots, level)


def paired_ci(a: np.ndarray, b: np.ndarray, ann: float, block: float, reps: int = S.BOOT_REPS,
              seed: int = S.SEED, level: float = S.CI_LEVEL) -> tuple[float, float, float]:
    """Sharpe(a) - Sharpe(b) with both series resampled on the same days (excess returns already)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([_sharpes(a[idx], ann) - _sharpes(b[idx], ann)
                            for idx in boot_indices(len(a), block, reps, rng)])
    return sharpe_ann(a, ann) - sharpe_ann(b, ann), *_interval(boots, level)


def mean_diff_ci(x: np.ndarray, flag: np.ndarray, block: float, reps: int = S.BOOT_REPS, seed: int = S.SEED,
                 level: float = S.CI_LEVEL) -> tuple[float, float, float]:
    """mean(x | flag) - mean(x | ~flag), (x, flag) resampled together (H3a i)."""
    x, f = np.asarray(x, float), np.asarray(flag, bool)

    def stat(xx, ff):
        n1 = ff.sum(axis=1)
        m1 = (xx * ff).sum(axis=1) / np.maximum(n1, 1)
        m0 = (xx * ~ff).sum(axis=1) / np.maximum(len(f) - n1, 1)
        return m1 - m0

    rng = np.random.default_rng(seed)
    boots = np.concatenate([stat(x[idx], f[idx]) for idx in boot_indices(len(x), block, reps, rng)])
    return float(x[f].mean() - x[~f].mean()), *_interval(boots, level)


def alpha_hac(y: np.ndarray, x: np.ndarray, lags: int = 30) -> tuple[float, float, float]:
    """OLS y = a + b x + e with Newey-West (Bartlett) standard errors. Returns (alpha, beta, t_alpha)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    X = np.column_stack([np.ones(len(y)), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = X * (y - X @ beta)[:, None]
    Smat = u.T @ u
    for l in range(1, lags + 1):
        G = u[l:].T @ u[:-l]
        Smat += (1.0 - l / (lags + 1.0)) * (G + G.T)
    cov = XtX_inv @ Smat @ XtX_inv
    return float(beta[0]), float(beta[1]), float(beta[0] / np.sqrt(cov[0, 0]))


def trade_ci_by_entry_day(net: np.ndarray, entry_day: np.ndarray, block: float = S.BOOT_BLOCK_ETF,
                          reps: int = S.BOOT_REPS, seed: int = S.SEED,
                          level: float = S.CI_LEVEL) -> tuple[float, float, float]:
    """Mean net trade return; trades entered on the same day are resampled together (stationary bootstrap over
    the ordered entry days)."""
    net, ed = np.asarray(net, float), np.asarray(entry_day)
    days, inv = np.unique(ed, return_inverse=True)
    s, c = np.bincount(inv, weights=net), np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    boots = np.concatenate([s[idx].sum(axis=1) / c[idx].sum(axis=1)
                            for idx in boot_indices(len(days), block, reps, rng)])
    return float(net.mean()), *_interval(boots, level)


def circular_shift(E: np.ndarray, first: np.ndarray, last: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Cyclically shift each column of E by a random offset inside its own span [first_j, last_j]."""
    out = E.copy()
    for j in range(E.shape[1]):
        f, l = int(first[j]), int(last[j])
        if l > f:
            out[f:l + 1, j] = np.roll(E[f:l + 1, j], int(rng.integers(l - f + 1)))
    return out


def listed_spans(listed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last listed day per column (0, -1 for never listed -> empty span)."""
    L = np.asarray(listed, bool)
    any_ = L.any(axis=0)
    first = np.where(any_, L.argmax(axis=0), 0)
    last = np.where(any_, L.shape[0] - 1 - L[::-1].argmax(axis=0), 0)
    return first, last


def percentile_of(null: np.ndarray, q: float = 95.0) -> float:
    return float(np.percentile(np.asarray(null, float), q))


# ---- mechanical criteria: {criterion: (value, threshold, passed)} -----------------------------------------

def _row(value, threshold, ok):
    return (None if value is None else float(value), threshold, bool(ok))


def criteria_h1a(m: dict) -> dict:
    return {
        "1 CI lower > 0": _row(m["sharpe_lo"], 0.0, m["sharpe_lo"] > 0),
        "2a Sharpe > BTC hold": _row(m["sharpe"] - m["sharpe_btc"], 0.0, m["sharpe"] > m["sharpe_btc"]),
        "2b alpha t > 2 and alpha > 0": _row(m["alpha_t"], 2.0, m["alpha"] > 0 and m["alpha_t"] > 2.0),
        "3 Sharpe 2022+ > 0": _row(m["sharpe_recent"], 0.0, m["sharpe_recent"] > 0),
        "4a Sharpe cost 25bps > 0": _row(m["sharpe_cost25"], 0.0, m["sharpe_cost25"] > 0),
        "4b Sharpe cost x2 > 0": _row(m["sharpe_cost2x"], 0.0, m["sharpe_cost2x"] > 0),
        "5 Sharpe > null p95": _row(m["sharpe"] - m["null_p95"], 0.0, m["sharpe"] > m["null_p95"]),
        "6 DSR > 0.95": _row(m["dsr"], 0.95, m["dsr"] > 0.95),
    }


def criteria_h1b(m: dict) -> dict:
    return {
        "1 CI lower > 0": _row(m["sharpe_lo"], 0.0, m["sharpe_lo"] > 0),
        "2 Sharpe 2010+ > 0": _row(m["sharpe_2010"], 0.0, m["sharpe_2010"] > 0),
        "3a 2008 return > 60/40": _row(m["ret_2008"] - m["bench_2008"], 0.0, m["ret_2008"] > m["bench_2008"]),
        "3b 2022 return > 60/40": _row(m["ret_2022"] - m["bench_2022"], 0.0, m["ret_2022"] > m["bench_2022"]),
        "4 Sharpe cost x3 > 0": _row(m["sharpe_cost3x"], 0.0, m["sharpe_cost3x"] > 0),
        "5 corr with SPY < 0.5": _row(m["corr_spy"], 0.5, m["corr_spy"] < 0.5),
    }


def criteria_h2(m: dict) -> dict:
    return {
        "1 mean trade CI lower > 0": _row(m["trade_lo"], 0.0, m["mean_trade"] > 0 and m["trade_lo"] > 0),
        "2a post-2009 mean trade > 0": _row(m["mean_trade_post"], 0.0, m["mean_trade_post"] > 0),
        "2b post-2009 Sharpe > 0": _row(m["sharpe_post"], 0.0, m["sharpe_post"] > 0),
        "3 mean trade > c2 p95": _row(m["mean_trade"] - m["c2_p95"], 0.0, m["mean_trade"] > m["c2_p95"]),
        "4 Sharpe > c1 or maxDD < 0.5 c1": _row(m["sharpe"] - m["sharpe_c1"], 0.0,
                                               m["sharpe"] > m["sharpe_c1"] or m["maxdd"] < 0.5 * m["maxdd_c1"]),
        "5 mean trade cost x2 > 0": _row(m["mean_trade_cost2x"], 0.0, m["mean_trade_cost2x"] > 0),
        "6 mean trade r1 (open fills) > 0": _row(m["mean_trade_r1"], 0.0, m["mean_trade_r1"] > 0),
    }


def criteria_h3a(m: dict) -> dict:
    """(i) must hold, otherwise the overlay is rejected regardless of (ii)."""
    return {
        "i TOM - other CI lower > 0": _row(m["tom_lo"], 0.0, m["tom_lo"] > 0),
        "ii paired Sharpe diff CI lower > 0": _row(m["diff_lo"], 0.0, m["diff_lo"] > 0),
        "ii maxDD overlay <= base": _row(m["maxdd_overlay"] - m["maxdd_base"], 0.0, m["maxdd_overlay"] <= m["maxdd_base"]),
    }


def criteria_overlay(m: dict) -> dict:
    return {
        "paired Sharpe diff CI lower > 0": _row(m["diff_lo"], 0.0, m["diff_lo"] > 0),
        "maxDD overlay <= base": _row(m["maxdd_overlay"] - m["maxdd_base"], 0.0, m["maxdd_overlay"] <= m["maxdd_base"]),
    }


def criteria_h4(m: dict) -> dict:
    return {
        "1 Sharpe > every sleeve": _row(m["sharpe"] - m["best_sleeve"], 0.0, m["sharpe"] > m["best_sleeve"]),
        "2 paired diff vs 60/40 CI lower > 0": _row(m["diff_lo"], 0.0, m["diff_lo"] > 0),
        "3 maxDD < 60/40": _row(m["maxdd"] - m["maxdd_6040"], 0.0, m["maxdd"] < m["maxdd_6040"]),
        "4 Sharpe cost x2 > 60/40": _row(m["sharpe_cost2x"] - m["sharpe_6040"], 0.0,
                                         m["sharpe_cost2x"] > m["sharpe_6040"]),
    }


def passed(criteria: dict) -> bool:
    return all(ok for _, _, ok in criteria.values())
