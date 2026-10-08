"""H-0014 High-volume visibility premium (LIQ-500, market-neutral, weekly, 5-day hold).

Decision after the close of the last trading day of each calendar week (approximated point-in-time as
the Friday rows: weekday arithmetic on the date of row t only, exchange holidays unknown), executed at
the next open, held until the next weekly decision (NaN rows in between = keep positions).

Eligible on decision day t: universe member and tradable on t, dollar_volume present on >= 80 % of the
5 rows t-4..t and of the reference_days rows t-4-reference_days..t-5, all ret_co/ret_oc on t-4..t known.
AV = ln(mean dv t-4..t) - ln(mean dv reference window) (means over the present days).
R5 = prod_{t-4..t} (1+ret_co)(1+ret_oc) - 1.
Five R5 quintiles; inside each, top leg_quantile by AV -> long, bottom leg_quantile -> short
(floor, at least one per side). Long leg +0.5 equal weight, short leg -0.5 equal weight.
"""
import numpy as np

PARAMS = {"leg_quantile": 0.2, "reference_days": 50}

N_QUINTILES = 5
RECENT_DAYS = 5
MIN_COVERAGE = 0.8
FRIDAY = 4  # Monday = 0


def _weekday(dates):
    days = np.asarray(dates).astype("datetime64[D]").astype(np.int64)
    return (days + 3) % 7  # 1970-01-01 was a Thursday (3)


def _legs(data, params):
    """Return (decision rows (T,), long mask (T, N), short mask (T, N))."""
    ref = int(params["reference_days"])
    q = float(params["leg_quantile"])
    if ref < 1 or not (0.0 < q < 1.0):
        raise ValueError("bad parameters")

    dates = data.dates
    T = len(dates)
    N = len(data.instruments)
    if data.universe is None:
        raise ValueError("H-0014 needs the LIQ-500 point-in-time universe (data.universe is None)")
    universe = np.asarray(data.universe, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    dv_all = np.asarray(data.dollar_volume)
    ret_co = np.asarray(data.ret_co)
    ret_oc = np.asarray(data.ret_oc)

    decision = _weekday(dates) == FRIDAY
    long_m = np.zeros((T, N), dtype=bool)
    short_m = np.zeros((T, N), dtype=bool)

    first = RECENT_DAYS - 1 + ref  # t - 4 - ref >= 0
    rows = np.flatnonzero(decision & (np.arange(T) >= first))
    min_a = MIN_COVERAGE * RECENT_DAYS - 1e-9
    min_b = MIN_COVERAGE * ref - 1e-9
    for t in rows:
        a0 = t - (RECENT_DAYS - 1)
        b0 = a0 - ref
        dv_a = dv_all[a0:t + 1].astype(np.float64)
        dv_b = dv_all[b0:a0].astype(np.float64)
        pres_a = np.isfinite(dv_a)
        pres_b = np.isfinite(dv_b)
        n_a = pres_a.sum(axis=0)
        n_b = pres_b.sum(axis=0)
        s_a = np.where(pres_a, dv_a, 0.0).sum(axis=0)
        s_b = np.where(pres_b, dv_b, 0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_a = s_a / n_a
            m_b = s_b / n_b
        gross = (1.0 + ret_co[a0:t + 1].astype(np.float64)) * (1.0 + ret_oc[a0:t + 1].astype(np.float64))
        r_ok = np.isfinite(gross).all(axis=0)
        elig = (universe[t] & tradable[t] & (n_a >= min_a) & (n_b >= min_b)
                & (m_a > 0) & (m_b > 0) & np.isfinite(m_a) & np.isfinite(m_b) & r_ok)
        idx = np.flatnonzero(elig)
        n = idx.size
        if n < 2:
            continue
        r5 = np.prod(gross[:, idx], axis=0) - 1.0
        av = np.log(m_a[idx]) - np.log(m_b[idx])
        order = np.lexsort((idx, r5))  # ascending R5, ties by column
        quint = np.empty(n, dtype=np.int64)
        quint[order] = (np.arange(n) * N_QUINTILES) // n
        for qq in range(N_QUINTILES):
            sel = quint == qq
            members = idx[sel]
            m = members.size
            if m < 2:
                continue
            k = max(1, int(np.floor(q * m + 1e-9)))
            k = min(k, m // 2)
            o = np.lexsort((members, av[sel]))  # ascending AV, ties by column
            short_m[t, members[o[:k]]] = True
            long_m[t, members[o[m - k:]]] = True
    return decision, long_m, short_m


def target_weights(data, params):
    decision, long_m, short_m = _legs(data, params)
    T, N = long_m.shape
    w = np.full((T, N), np.nan)
    w[decision] = 0.0
    n_long = long_m.sum(axis=1)
    n_short = short_m.sum(axis=1)
    rows = np.flatnonzero(decision & (n_long > 0) & (n_short > 0))
    for t in rows:
        w[t, long_m[t]] = 0.5 / n_long[t]
        w[t, short_m[t]] = -0.5 / n_short[t]
    return w
