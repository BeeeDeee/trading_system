"""H-0193: conviction-sized insider purchases, LIQ-500 market-neutral, 5-session slots, liquid-basket hedge.

Rules (card signal.description):
  1. Event after the close of t: member of LIQ-500 and tradable on t, ins_buy_value_91d[t-1] < X <= ins_buy_value_91d[t],
     ins_buy_n_91d[t] > ins_buy_n_91d[t-1]; any NaN on t or t-1 = no event. The strict crossing (value on t-1 below X)
     is the re-fire rule: a stock fires again only after its trailing value fell back below X.
  2. Each event opens a slot on decision rows t..t+4 (entered at the open of t+1, closed at the open of t+6).
     A slot closes early from the first row s in t+1..t+4 where the stock is not a member or not tradable.
  3. Long weight per slot 0.08; with k > 5 open slots 0.40 / k.
  4. Hedge: short of the total long weight, equal weight over the 50 lowest liq_rank members at the decision date
     (all eligible members if fewer), excluding stocks with an open slot.
  5. Book recomputed only on rows where the set of open slots changes; other rows are NaN (keep positions), except
     that a held name that has left the universe is set to 0 on that element (hold only members).
"""
import numpy as np

PARAMS = {"min_buy_usd": 500000}

HOLD_SESSIONS = 5
SLOT_WEIGHT = 0.08
MAX_LONG = 0.40
HEDGE_NAMES = 50


def crossing_events(data, params):
    """(T, N) bool: the card's crossing event after the close of row t (rows <= t only)."""
    X = float(params["min_buy_usd"])
    uni = data.universe
    if uni is None:
        raise ValueError("H-0193 needs the LIQ-500 point-in-time universe")
    v = np.asarray(data.extras["ins_buy_value_91d"])
    n = np.asarray(data.extras["ins_buy_n_91d"])
    ev = np.zeros(v.shape, dtype=bool)        # row 0 has no t-1: no event
    if v.shape[0] > 1:
        vt, vp, nt, np_ = v[1:], v[:-1], n[1:], n[:-1]
        # comparisons with NaN are False, so a NaN on t or t-1 gives no event
        with np.errstate(invalid="ignore"):
            ev[1:] = (vp < X) & (vt >= X) & (nt > np_)
        ev[1:] &= np.isfinite(vt) & np.isfinite(vp) & np.isfinite(nt) & np.isfinite(np_)
    ev &= np.asarray(uni, dtype=bool)
    ev &= np.asarray(data.tradable, dtype=bool)
    return ev


def target_weights(data, params):
    ev = crossing_events(data, params)
    uni = np.asarray(data.universe, dtype=bool)
    trad = np.asarray(data.tradable, dtype=bool)
    lr = np.asarray(data.extras["liq_rank"])
    T, N = ev.shape
    alive_ok = uni & trad

    # Slots: open count per stock and row; starts / closes per row.
    count = np.zeros((T, N), dtype=np.int16)
    change = np.zeros(T, dtype=bool)
    rows, cols = np.nonzero(ev)
    for t, j in zip(rows.tolist(), cols.tolist()):
        last = min(t + HOLD_SESSIONS - 1, T - 1)
        end = last + 1                       # first row on which the slot is no longer open
        for s in range(t + 1, last + 1):
            if not alive_ok[s, j]:
                end = s
                break
        count[t:end, j] += 1
        change[t] = True
        if end < T:
            change[end] = True

    w = np.full((T, N), np.nan, dtype=np.float64)
    held = np.zeros(N, dtype=np.float64)
    for s in range(T):
        if change[s] or s == 0:
            row = np.zeros(N, dtype=np.float64)
            c = count[s]
            k = int(c.sum())
            if k > 0:
                per = SLOT_WEIGHT if k * SLOT_WEIGHT <= MAX_LONG + 1e-12 else MAX_LONG / k
                row += c * per
                total_long = per * k
                elig = uni[s] & np.isfinite(lr[s]) & (c == 0)
                m = int(elig.sum())
                if m > 0:
                    key = np.where(elig, lr[s].astype(np.float64), np.inf)
                    order = np.argsort(key, kind="stable")[: min(HEDGE_NAMES, m)]
                    row[order] -= total_long / len(order)
            w[s] = row
            held = row.copy()
        else:
            gone = (held != 0.0) & ~uni[s]
            if gone.any():
                w[s, gone] = 0.0
                held[gone] = 0.0
    return w
