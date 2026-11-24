"""H-0195 Absorbed overnight supply shock (LIQ-500).

Event after the close of session t for stock j (all conditions):
  (1) market-adjusted gap g = ret_co - cross-sectional median ret_co (LIQ-500 members with a valid value)
      <= -gap_pct / 100
  (2) ret_oc >= 0 (closed at or above the open)
  (3) dollar_volume[t] >= 4 x median dollar_volume over t-63..t-1 (>= 50 valid values)
  (4) ea_reaction_day != 1 on t-2, t-1, t and not NaN on t
  (5) LIQ-500 member on t, no accepted event of j in the previous hold_days sessions
Each event opens a slot at the open of t+1: +0.10 in j, -0.10 spread equally over the 100 most liquid
members (liq_rank on t) without an event on t. The slot is held for hold_days sessions (target rows
t .. t+hold_days-1, closed at the open of t+1+hold_days) or closed earlier at the first open after j
leaves LIQ-500 or stops trading. With more than 5 open slots, all slot weights are scaled by 5 / k.
"""
import numpy as np

PARAMS = {"gap_pct": 3, "hold_days": 5}

LONG_W = 0.10          # long weight per slot
HEDGE_W = 0.10         # total hedge weight per slot
HEDGE_N = 100          # most liquid members in the hedge basket
MAX_SLOTS = 5          # slots per side at full size (gross 1.0)
VOL_MULT = 4.0         # abnormal dollar-volume multiple
VOL_WIN = 63           # trailing sessions for the volume median
VOL_MIN_VALID = 50     # minimum valid values in the window


def _cross_median(x, valid):
    """Row-wise median of x over valid elements (NaN where a row has none)."""
    T = x.shape[0]
    xm = np.where(valid, x, np.nan)
    med = np.full(T, np.nan)
    ok = valid.any(axis=1)
    if ok.any():
        med[ok] = np.nanmedian(xm[ok], axis=1)
    return med


def _shift_rows(x, k):
    """x shifted down by k rows (row t holds row t-k); NaN on the first k rows."""
    out = np.full(x.shape, np.nan)
    if k < x.shape[0]:
        out[k:] = x[: x.shape[0] - k]
    return out


def raw_events(data, params):
    """(T, N) bool: conditions (1)-(4) plus membership and trading on t (no re-entry rule)."""
    gap = float(params["gap_pct"]) / 100.0
    uni = data.universe
    if uni is None:
        raise ValueError("H-0195 needs the point-in-time LIQ-500 universe (data.universe is None)")
    member = np.asarray(uni, dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    co = np.asarray(data.ret_co, dtype=np.float64)
    oc = np.asarray(data.ret_oc, dtype=np.float64)
    dv = np.asarray(data.dollar_volume, dtype=np.float64)
    ea = np.asarray(data.extras["ea_reaction_day"], dtype=np.float64)
    T, N = co.shape

    # (1) market-adjusted gap; a value is valid when the stock is a member, traded on t and ret_co is finite
    valid_co = member & tradable & np.isfinite(co)
    med = _cross_median(co, valid_co)
    g = co - med[:, None]

    base = valid_co & np.isfinite(oc) & np.isfinite(dv) & np.isfinite(g)
    cond = base & (g <= -gap) & (oc >= 0.0)

    # (4) earnings exclusion: not 1 on t-2, t-1, t; not NaN on t (no 8-K coverage = not eligible)
    ea1 = _shift_rows(ea, 1)
    ea2 = _shift_rows(ea, 2)
    with np.errstate(invalid="ignore"):
        ea_ok = np.isfinite(ea) & (ea != 1.0) & ~(ea1 == 1.0) & ~(ea2 == 1.0)
    cond &= ea_ok

    # (3) abnormal dollar volume, evaluated only at candidate cells
    ti, ji = np.nonzero(cond)
    out = np.zeros((T, N), dtype=bool)
    if ti.size == 0:
        return out
    offs = np.arange(-VOL_WIN, 0)                      # t-63 .. t-1
    rows = ti[:, None] + offs[None, :]
    inb = rows >= 0
    win = np.full(rows.shape, np.nan)
    win[inb] = dv[rows[inb], np.broadcast_to(ji[:, None], rows.shape)[inb]]
    nvalid = np.isfinite(win).sum(axis=1)
    enough = nvalid >= VOL_MIN_VALID
    vmed = np.full(ti.size, np.nan)
    if enough.any():
        vmed[enough] = np.nanmedian(win[enough], axis=1)
    with np.errstate(invalid="ignore"):
        vol_ok = enough & (dv[ti, ji] >= VOL_MULT * vmed)
    out[ti[vol_ok], ji[vol_ok]] = True
    return out


def accepted_events(raw, hold):
    """Apply rule (5): no accepted event of the same stock in the previous `hold` sessions."""
    T, N = raw.shape
    acc = np.zeros((T, N), dtype=bool)
    last = np.full(N, -(10 ** 9), dtype=np.int64)
    ti, ji = np.nonzero(raw)                           # row-major: increasing t
    for t, j in zip(ti.tolist(), ji.tolist()):
        if t - last[j] > hold:
            acc[t, j] = True
            last[j] = t
    return acc


def target_weights(data, params):
    hold = int(params["hold_days"])
    member = np.asarray(data.universe, dtype=bool) if data.universe is not None else None
    raw = raw_events(data, params)                     # raises if the universe is missing
    tradable = np.asarray(data.tradable, dtype=bool)
    liq = np.asarray(data.extras["liq_rank"], dtype=np.float64)
    T, N = raw.shape
    acc = accepted_events(raw, hold)

    W = np.zeros((T, N), dtype=np.float64)
    K = np.zeros(T, dtype=np.float64)                  # open slots per target row
    alive = member & tradable

    for t in np.nonzero(acc.any(axis=1))[0].tolist():
        ev = acc[t]
        # hedge basket: 100 most liquid members on t (by liq_rank) without an event on t
        elig = member[t] & np.isfinite(liq[t]) & ~ev
        cand = np.nonzero(elig)[0]
        order = np.argsort(liq[t, cand], kind="stable")
        H = cand[order[:HEDGE_N]]
        last_row = min(t + hold, T)                    # exclusive: rows t .. t+hold-1
        for j in np.nonzero(ev)[0].tolist():
            end = last_row
            for r in range(t + 1, last_row):
                if not alive[r, j]:                    # left LIQ-500 or stopped trading on r
                    end = r
                    break
            rows = np.arange(t, end)
            W[rows, j] += LONG_W
            if H.size > 0:
                hw = -HEDGE_W / H.size
                W[np.ix_(rows, H)] += hw * member[np.ix_(rows, H)]
            K[rows] += 1.0

    scale = np.ones(T)
    over = K > MAX_SLOTS
    scale[over] = MAX_SLOTS / K[over]
    W *= scale[:, None]

    # numerical safety only: gross can exceed 1 by float rounding at exactly 5 slots
    gross = np.abs(W).sum(axis=1)
    big = gross > 1.0
    if big.any():
        W[big] *= (1.0 / gross[big])[:, None] * (1.0 - 1e-12)
    return W
