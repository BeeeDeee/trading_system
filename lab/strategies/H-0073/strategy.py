"""H-0073: VIX term-structure carry. Short VIXY sized by VIX/VIX3M contango, flat in backwardation, daily.

Rules (card signal.description) and where they live:
  1. r = VIXCLS / VXVCLS from the latest usable values (series already aligned by the next_morning clock,
     so row t normally holds the close of session t-1; no extra lag)          -> ratio_and_age()
  2. weight 0 if the observation is more than 5 XNYS sessions older than t, or a value is missing or
     non-positive                                                              -> raw_target()
  3. w = -W_MAX * clip((k - r) / d, 0, 1), W_MAX = 0.8 fixed, never positive   -> raw_target()
  4. trade VIXY only if |target - current| >= 0.05, or target 0 and position not 0 (close fully)
                                                                               -> apply_band()
  5. VIXY without a bar / not tradable on session t: keep the position (NaN)   -> target_weights()
  6. SHY, IEF, TLT, AGG, GLD always 0; rest is cash                            -> target_weights()
"""
import numpy as np

PARAMS = {"k": 1.0, "d": 0.1}

W_MAX = 0.8            # fixed by the card, not a grid parameter
BAND = 0.05            # trade threshold on |target - current weight|
MAX_AGE = 5            # XNYS sessions between the observation date and t
TRADED = "VIXY"
PEERS = ("SHY", "IEF", "TLT", "AGG", "GLD")
VIX_KEY = "fred_macro.VIXCLS"
VXV_KEY = "fred_vxv_1d.VXVCLS"
_EPS = 1e-12


def _ffill_and_last_obs(x):
    """Forward-filled values and, per row, the row on which the current value first appeared.

    A new observation is a finite value that differs from the previous row (or follows a NaN). The
    series arrive forward-filled, so an unchanged value is read as 'no new print'."""
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    idx = np.arange(T)
    fin = np.isfinite(x)
    prev = np.empty(T)
    prev[0] = np.nan
    prev[1:] = x[:-1]
    new = fin & ~(np.isfinite(prev) & (x == prev))
    last_new = np.maximum.accumulate(np.where(new, idx, -1))
    last_fin = np.maximum.accumulate(np.where(fin, idx, -1))
    vals = np.where(last_fin >= 0, x[np.maximum(last_fin, 0)], np.nan)
    return vals, last_new


def ratio_and_age(vix, vxv):
    """r = VIX / VIX3M from the latest values, and the age in sessions of the older of the two prints.

    A value first seen on row s is the close of session s-1 (next_morning clock), so its age at row t is
    t - s + 1 sessions. Age is taken from the older of the two series (so a stopped series goes stale)."""
    v1, s1 = _ffill_and_last_obs(vix)
    v2, s2 = _ffill_and_last_obs(vxv)
    T = v1.shape[0]
    idx = np.arange(T)
    s = np.minimum(s1, s2)
    age = np.where(s >= 0, idx - s + 1, np.iinfo(np.int64).max).astype(np.int64)
    ok = np.isfinite(v1) & np.isfinite(v2) & (v1 > 0) & (v2 > 0)
    r = np.full(T, np.nan)
    r[ok] = v1[ok] / v2[ok]
    return r, age


def raw_target(r, age, k, d):
    """Daily target weight of VIXY before the trade band (0 when the ratio is unusable)."""
    usable = np.isfinite(r) & (age <= MAX_AGE)
    rr = np.where(usable, r, k)                      # k -> clip(0) -> weight 0
    w = -W_MAX * np.clip((k - rr) / d, 0.0, 1.0)
    w = np.where(usable, w, 0.0)
    return np.minimum(w, 0.0) + 0.0                  # never long; +0.0 turns -0.0 into 0.0


def apply_band(target, decidable):
    """Hysteresis: emit a new weight only when it differs from the held one by >= BAND, or to close fully.

    The held ('current') weight is the last emitted target (the engine holds positions until the target
    changes). Rows without a trade are NaN (keep)."""
    T = target.shape[0]
    out = np.full(T, np.nan)
    held = 0.0
    for t in range(T):
        if not decidable[t]:
            continue
        tg = float(target[t])
        if (tg == 0.0 and held != 0.0) or abs(tg - held) >= BAND - _EPS:
            out[t] = tg
            held = tg
    return out


def target_weights(data, params):
    k = float(params["k"])
    d = float(params["d"])
    T = len(data.dates)
    N = len(data.instruments)

    j = data.col(TRADED)
    peer_cols = [data.col(p) for p in PEERS]
    vix = np.asarray(data.series[VIX_KEY], dtype=float).reshape(T)
    vxv = np.asarray(data.series[VXV_KEY], dtype=float).reshape(T)

    r, age = ratio_and_age(vix, vxv)
    tgt = raw_target(r, age, k, d)

    listed = np.asarray(data.listed, dtype=bool)[:, j]
    tradable = np.asarray(data.tradable, dtype=bool)[:, j]
    has_bar = np.isfinite(np.asarray(data.close, dtype=float)[:, j])
    decidable = listed & tradable & has_bar
    if data.universe is not None:
        decidable = decidable & np.asarray(data.universe, dtype=bool)[:, j]

    w = np.zeros((T, N))
    w[:, j] = apply_band(tgt, decidable)
    for c in peer_cols:
        w[:, c] = 0.0
    return w
