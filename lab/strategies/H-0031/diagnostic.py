"""H-0031 mechanism event: earnings-release reaction session with |z| >= z_threshold.

Same event code as strategy._reaction_events (duplicated so the diagnostic needs no import
from the strategy module). side = sign(z): +1 after a positive reaction, -1 after a negative one,
the same side as the strategy's event slot.
"""
import numpy as np

VOL_WINDOW = 63
VOL_MIN_OBS = 50


def _reaction_events(ret_co, ret_oc, listed, close, universe, tradable, flag, z_threshold):
    T, N = ret_co.shape
    event = np.zeros((T, N), dtype=bool)
    side = np.zeros((T, N))

    buf_a = np.zeros((VOL_WINDOW, N))
    buf_v = np.zeros((VOL_WINDOW, N), dtype=np.int64)
    s1 = np.zeros(N)
    s2 = np.zeros(N)
    cnt = np.zeros(N, dtype=np.int64)

    for t in range(T):
        co = np.asarray(ret_co[t], dtype=np.float64)
        oc = np.asarray(ret_oc[t], dtype=np.float64)
        rt = (1.0 + co) * (1.0 + oc) - 1.0
        ok = np.asarray(listed[t], dtype=bool) & np.isfinite(np.asarray(close[t], dtype=np.float64)) & np.isfinite(rt)
        uni_t = np.asarray(universe[t], dtype=bool)
        mem = uni_t & ok
        if mem.any():
            a_ok = ok
            at = np.where(a_ok, rt - rt[mem].mean(), 0.0)
        else:
            a_ok = np.zeros(N, dtype=bool)
            at = np.zeros(N)

        enough = cnt >= VOL_MIN_OBS
        n = np.where(enough, cnt, 2).astype(np.float64)
        var = (s2 - s1 * s1 / n) / (n - 1.0)
        var = np.where(enough, np.maximum(var, 0.0), np.nan)
        sigma = np.sqrt(var)
        sig_ok = enough & np.isfinite(sigma) & (sigma > 0.0)

        zc = sig_ok & a_ok
        zt = np.zeros(N)
        zt[zc] = at[zc] / sigma[zc]
        f = np.asarray(flag[t], dtype=np.float64)
        flag_on = f == 1.0
        ev = flag_on & uni_t & np.asarray(tradable[t], dtype=bool) & zc & (np.abs(zt) >= z_threshold)
        event[t] = ev
        side[t] = np.where(ev, np.sign(zt), 0.0)

        k = t % VOL_WINDOW
        if t >= VOL_WINDOW:
            old = buf_a[k]
            s1 -= old
            s2 -= old * old
            cnt -= buf_v[k]
        buf_a[k] = at
        buf_v[k] = a_ok.astype(np.int64)
        s1 += at
        s2 += at * at
        cnt += buf_v[k]

    return event, side


def events(data, params):
    universe = data.universe
    if universe is None:
        raise ValueError("H-0031 needs the point-in-time LIQ-500 universe")
    flag = data.extras["ea_reaction_day"]
    mask, side = _reaction_events(data.ret_co, data.ret_oc, data.listed, data.close, universe,
                                  data.tradable, flag, float(params["z_threshold"]))
    return mask, side
