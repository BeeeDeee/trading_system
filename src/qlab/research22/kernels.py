"""Numba kernels of the research 22 evaluator.

Tie rules (part of the specification, otherwise results are not reproducible):
- `rank_rows`: equal values get the *average* rank; non-finite values stay NaN and are ignored;
- `portfolio`: a tie at the top-k boundary is resolved by the lowest compact column, which is the lowest asset
  index because the compact columns are in ascending asset order.

`portfolio` replays the position accounting of `qlab.engine.vector.simulate` (see its docstring), including positions
that cannot be sold because there is no usable open on the day.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def _tie_ranks(tmp, order, nv):
    n_rows, n_cols = tmp.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    for i in range(n_rows):
        pad = n_cols - nv[i]
        r = pad
        while r < n_cols:
            e = r
            v = tmp[i, order[i, r]]
            while e + 1 < n_cols and tmp[i, order[i, e + 1]] == v:
                e += 1
            avg = ((r - pad) + (e - pad)) / 2.0
            for q in range(r, e + 1):
                out[i, order[i, q]] = avg / nv[i]
            r = e + 1
    return out


def rank_rows(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank in [0, 1) per row (average rank for ties, NaN for non-finite input).

    numpy's SIMD argsort is ~2x faster than sorting inside numba; the tie scan is linear.
    """
    ok = np.isfinite(x)
    tmp = np.where(ok, x + np.float32(0.0), np.float32(-np.inf))  # + 0.0 turns -0.0 into +0.0
    order = np.argsort(tmp, axis=1)
    return _tie_ranks(tmp, order, ok.sum(axis=1))


@njit(cache=True)
def portfolio(score, wk, dead, idx, rate, trad, scale, topk):
    """Weekly equal-weight top-k portfolio that mirrors `engine.vector.simulate` (NAV in units of the first open).

    score (F, K) float32  compact scores, NaN = not selectable
    wk    (F, N) float64  holding return by asset, open of the day after the decision -> open after the next one
    dead  (F, N) bool     the position is paid out (delisted) during the holding week
    idx   (F, K) int64    asset index of each compact column
    rate  (F, N) float64  cost per unit of traded value at the execution day (1x), by asset
    trad  (F, N) bool     a usable open exists on the execution day
    scale float           cost multiplier (rates are multiplied inside, so cash scaling stays exact)

    As in the engine: sells first (a position that is not tradable stays, whatever the target says), buys are scaled
    down if cash is short after sells and costs, a delisting pays the position out into cash (no interest), the
    held set can therefore exceed k. Returns net weekly return (NAV at the next open / NAV at this open - 1, costs
    included), gross return of the invested positions, turnover (buys + sells over NAV at the open), number of
    positions after the trade and the NAV after the trade (= engine NAV at that close when intraday returns are 0).
    """
    n_rows, n_cols = score.shape
    n_assets = wk.shape[1]
    net = np.zeros(n_rows)
    gross = np.zeros(n_rows)
    turn = np.zeros(n_rows)
    npos = np.zeros(n_rows, dtype=np.int64)
    nav_trade = np.zeros(n_rows)
    v = np.zeros(n_assets)
    held = np.empty(n_assets, dtype=np.int64)
    is_held = np.zeros(n_assets, dtype=np.bool_)
    in_target = np.zeros(n_assets, dtype=np.bool_)
    nheld = 0
    cash = 1.0
    for i in range(n_rows):
        if i > 0:                                   # last week's holding return, paid-out names go to cash
            for m in range(nheld):
                a = held[m]
                w = wk[i - 1, a]
                if not np.isfinite(w):
                    w = 0.0
                x = v[a] * (1.0 + w)
                if dead[i - 1, a]:
                    cash += x
                    v[a] = 0.0
                else:
                    v[a] = x
        nav_open = cash
        for m in range(nheld):
            nav_open += v[held[m]]

        tmp = np.empty(n_cols, dtype=np.float32)
        nv = 0
        for j in range(n_cols):
            sc = score[i, j]
            if np.isfinite(sc):
                tmp[j] = sc
                nv += 1
            else:
                tmp[j] = -np.inf
        k = min(topk, nv)
        sel = np.empty(max(k, 1), dtype=np.int64)
        if k > 0:
            thr = np.partition(tmp, n_cols - k)[n_cols - k]
            ns = 0
            for j in range(n_cols):
                if tmp[j] > thr and ns < k:
                    sel[ns] = j
                    ns += 1
            for j in range(n_cols):
                if ns >= k:
                    break
                if tmp[j] == thr and tmp[j] > -np.inf:
                    sel[ns] = j
                    ns += 1
        target = nav_open / k if k > 0 else 0.0
        for m in range(k):
            in_target[idx[i, sel[m]]] = True

        sell_sum = 0.0
        sell_cost = 0.0
        buys = np.zeros(max(k, 1))
        for m in range(k):                          # trims and buys of the selected names
            a = idx[i, sel[m]]
            d = target - v[a]
            if d < 0.0:
                sell_sum -= d
                sell_cost -= d * rate[i, a] * scale
                v[a] += d
            else:
                buys[m] = d
        for m in range(nheld):                      # exits of everything else that can be traded
            p = held[m]
            if v[p] > 0.0 and not in_target[p] and trad[i, p]:
                sell_sum += v[p]
                sell_cost += v[p] * rate[i, p] * scale
                v[p] = 0.0
        cash += sell_sum - sell_cost
        need = 0.0
        for m in range(k):
            need += buys[m] * (1.0 + rate[i, idx[i, sel[m]]] * scale)
        if need > 0.0 and need > cash:
            f = max(cash, 0.0) / need
            for m in range(k):
                buys[m] *= f
        buy_sum = 0.0
        buy_cost = 0.0
        for m in range(k):
            a = idx[i, sel[m]]
            v[a] += buys[m]
            buy_sum += buys[m]
            buy_cost += buys[m] * rate[i, a] * scale
        cash -= buy_sum + buy_cost
        turn[i] = (buy_sum + sell_sum) / nav_open
        nav_trade[i] = nav_open - sell_cost - buy_cost

        for m in range(k):                          # new members of the held set, then drop the empty ones
            a = idx[i, sel[m]]
            in_target[a] = False
            if not is_held[a]:
                is_held[a] = True
                held[nheld] = a
                nheld += 1
        n_keep = 0
        for m in range(nheld):
            a = held[m]
            if v[a] > 0.0:
                held[n_keep] = a
                n_keep += 1
            else:
                is_held[a] = False
        nheld = n_keep
        npos[i] = nheld

        end = cash
        inv = 0.0
        for m in range(nheld):
            a = held[m]
            w = wk[i, a]
            if not np.isfinite(w):
                w = 0.0
            end += v[a] * (1.0 + w)
            inv += v[a] * w
        net[i] = end / nav_open - 1.0
        gross[i] = inv / nav_open
    return net, gross, turn, npos, nav_trade
