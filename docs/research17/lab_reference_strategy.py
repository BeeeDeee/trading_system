"""Reference implementation of docs/research17/lab_card_btc_lead_altcoins.yaml (lab intraday contract).

Not submitted to the lab: the Builder writes its own. `scripts/r17_lab_parity.py` compares any lab strategy with the
research 17 engine, and this file passes it (same entries). Written in the lab's English-code convention."""
import numpy as np
from lab.framework.intraday_api import log_return, normalize_gross, rolling_std

PARAMS = {"k_sigma": 3.0, "hold_candles": 16}


def target_positions(data, params):
    T, N = data.shape
    btc, eth = data.col("BTCUSDT"), data.col("ETHUSDT")
    c = data.c.astype(float)
    r1 = log_return(c[:, [btc]], 1)[:, 0]
    real = data.real.astype(bool)
    sigma = rolling_std(r1[:, None], 672)[:, 0]
    # all 672 one-candle returns in the window must be real candles with a real predecessor
    ok1 = real[:, btc] & np.concatenate([[False], real[:-1, btc]])
    full = np.convolve(ok1.astype(float), np.ones(672), mode="full")[:T] >= 672
    rl = log_return(c[:, [btc]], 4)[:, 0]
    s1h = np.where(full, 2.0 * sigma, np.nan)
    jump = rl >= params["k_sigma"] * s1h
    rf = log_return(c, 4)
    allreal = np.ones((T, N), bool)
    for k in range(5):
        allreal &= np.concatenate([np.zeros((k, N), bool), real[: T - k]]) if k else real
    with np.errstate(invalid="ignore", divide="ignore"):
        lag = rf / rl[:, None] < 0.5
    sig = jump[:, None] & lag & allreal & data.universe & data.listed
    sig[:, [btc, eth]] = False
    h = int(params["hold_candles"])
    w = np.zeros((T, N))
    for j in np.flatnonzero(sig.any(axis=0)):
        t, free = 0, 0
        rows = np.flatnonzero(sig[:, j])
        for t in rows:
            if t < free:
                continue
            alive = np.minimum.accumulate(data.universe[t:t + h, j].astype(float))   # closed for good once not a member
            w[t:t + h, j] = 0.1 * alive
            free = t + h + 1
    return normalize_gross(w)
