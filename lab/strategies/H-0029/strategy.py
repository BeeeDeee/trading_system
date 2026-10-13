"""H-0029 Turnover-conditioned short-term momentum (LIQ-500, dollar-neutral, weekly).

Decision dates: Fridays (last session of the calendar week, approximated by weekday arithmetic on the
date of row t alone; a week whose Friday is an exchange holiday gets no decision and the book is carried).
On each decision date t (after the close, using rows 0..t only):
  eligible  = in universe[t] & tradable[t] & price on each of rows t-F..t (returns for each of the last F
              sessions) & finite dollar_volume on rows t-F+1..t & usable sf1_arq_marketcap at t
  R         = I(t) / I(t-F) - 1  (I = total-return index from ret_co, ret_oc)
  M_t       = marketcap(t) * I(t) / I(s), s = first row on which the current marketcap value was usable
  TO        = sum(dollar_volume[t-F+1..t]) / M_t
  high-TO group = top `quantile` of eligible by TO; long = top `quantile` of the group by R,
  short = bottom `quantile` of the group by R; ties by stock id. +0.5/n_long, -0.5/n_short;
  if either leg < 5 stocks the book is flat. Non-decision rows are NaN (carry).
"""
import numpy as np
from lab.framework.api import total_return_index

PARAMS = {"formation_days": 21, "quantile": 0.2}

MIN_LEG = 5


def _weekday(dates):
    # Monday = 0 ... Sunday = 6 (1970-01-01 was a Thursday)
    d = dates.astype("datetime64[D]").astype(np.int64)
    return (d + 3) % 7


def decision_rows(dates):
    """Rows that are the last session of the calendar week: Fridays (holiday Fridays are missed)."""
    return np.flatnonzero(_weekday(dates) == 4)


def _marketcap_start_rows(mc):
    """For each (t, j): the first row on which the marketcap value held at t became usable (-1 if none)."""
    T = mc.shape[0]
    fin = np.isfinite(mc)
    change = np.zeros(mc.shape, dtype=bool)
    change[0] = fin[0]
    if T > 1:
        same = mc[1:] == mc[:-1]          # False when either side is NaN
        change[1:] = fin[1:] & ~same
    rows = np.arange(T, dtype=np.int32)[:, None]
    start = np.where(change, rows, np.int32(-1)).astype(np.int32)
    np.maximum.accumulate(start, axis=0, out=start)
    start[~fin] = -1
    return start


def _id_rank(instruments):
    order = sorted(range(len(instruments)), key=lambda j: instruments[j])
    rank = np.empty(len(instruments), dtype=np.int64)
    rank[np.asarray(order, dtype=np.int64)] = np.arange(len(instruments))
    return rank


def _top(values, idx, k, id_rank, descending):
    """The k members of idx with the highest (descending) or lowest values; ties by stock id ascending."""
    v = values[idx]
    key = -v if descending else v
    order = np.lexsort((id_rank[idx], key))
    return idx[order[:k]]


def select_legs(data, params):
    """Returns (decision row indices, list of (long_cols, short_cols) per decision row; None = flat)."""
    F = int(params["formation_days"])
    q = float(params["quantile"])
    if F < 1:
        raise ValueError("formation_days must be >= 1")

    universe = data.universe
    if universe is None:
        raise ValueError("card universe is liq_n: data.universe is required")
    # full matrices keep their dtype (RAM); per-date slices are cast to float64 below
    mc = np.asarray(data.extras["sf1_arq_marketcap"])
    close = np.asarray(data.close)
    dv = np.asarray(data.dollar_volume)
    tradable = np.asarray(data.tradable, dtype=bool)
    universe = np.asarray(universe, dtype=bool)

    tri = np.asarray(total_return_index(data))
    start = _marketcap_start_rows(mc)
    id_rank = _id_rank(data.instruments)
    N = len(data.instruments)
    cols = np.arange(N)

    rows = decision_rows(data.dates)
    out = []
    for t in rows:
        if t - F < 0:
            out.append(None)
            continue
        has_px = np.isfinite(close[t - F:t + 1]).all(axis=0)
        has_dv = np.isfinite(dv[t - F + 1:t + 1]).all(axis=0)
        s = start[t]
        mct = mc[t].astype(np.float64)
        ok = universe[t] & tradable[t] & has_px & has_dv & (s >= 0) & np.isfinite(mct) & (mct > 0)
        i_t = tri[t].astype(np.float64)
        i_f = tri[t - F].astype(np.float64)
        i_s = tri[np.maximum(s, 0), cols].astype(np.float64)
        ok &= np.isfinite(i_t) & np.isfinite(i_f) & np.isfinite(i_s) & (i_f > 0) & (i_s > 0)
        idx = np.flatnonzero(ok)
        if idx.size == 0:
            out.append(None)
            continue
        R = np.full(N, np.nan)
        TO = np.full(N, np.nan)
        R[idx] = i_t[idx] / i_f[idx] - 1.0
        m_t = mct[idx] * i_t[idx] / i_s[idx]
        dvsum = dv[t - F + 1:t + 1][:, idx].astype(np.float64).sum(axis=0)
        TO[idx] = dvsum / m_t
        good = np.isfinite(R[idx]) & np.isfinite(TO[idx]) & (m_t > 0)
        idx = idx[good]
        k_hi = int(np.floor(q * idx.size + 1e-9))
        hi = _top(TO, idx, k_hi, id_rank, descending=True)
        k_leg = int(np.floor(q * hi.size + 1e-9))
        if k_leg < MIN_LEG:
            out.append(None)
            continue
        # one total order of the group: R descending, ties by stock id ascending (legs never overlap)
        ranked = _top(R, hi, hi.size, id_rank, descending=True)
        longs = ranked[:k_leg]
        shorts = ranked[hi.size - k_leg:]
        out.append((longs, shorts))
    return rows, out


def target_weights(data, params):
    T = len(data.dates)
    N = len(data.instruments)
    w = np.full((T, N), np.nan)
    rows, legs = select_legs(data, params)
    for t, sel in zip(rows, legs):
        w[t] = 0.0
        if sel is None:
            continue
        longs, shorts = sel
        w[t, longs] = 0.5 / len(longs)
        w[t, shorts] = -0.5 / len(shorts)
    return w
