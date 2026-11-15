"""H-0167: Asia-Pacific pre-holiday run-up via US country ETFs on a 60/40 SPY/IEF core.

Decision after the close of XNYS session d (row t), executed at the open of d+1.
k_j(d) = <MIC>.sessions_to_closure of ETF j's home exchange, stamped on d (known at the close of d).
"""
import numpy as np

PARAMS = {"pre_sessions": 2, "tilt_weight": 0.25}

DATASET = "exchange_holidays_ahead_1d"
COUNTRY = ("EWJ", "EWH", "EWY", "EWT", "EWS", "EWM", "EWA")
HOME_MIC = {
    "EWJ": "XTKS",
    "EWH": "XHKG",
    "EWY": "XKRX",
    "EWT": "XTAI",
    "EWS": "XSES",
    "EWM": "XKLS",
    "EWA": "XASX",
}
MIN_PRIOR_BARS = 20
CORE_EQUITY = 0.6
CORE_BOND = 0.4


def marked_etfs(data, pre_sessions):
    """(T, 7) bool: country ETF j (COUNTRY order) is marked for the step d+1 -> d+2 after the close of d."""
    cols = [data.col(name) for name in COUNTRY]
    T = len(data.dates)
    k = np.empty((T, len(COUNTRY)), dtype=float)
    for j, name in enumerate(COUNTRY):
        s = np.asarray(data.series[DATASET + "." + HOME_MIC[name] + ".sessions_to_closure"], dtype=float)
        if s.shape[0] != T:
            raise ValueError("series length does not match the view")
        k[:, j] = s

    close = np.asarray(data.close, dtype=float)[:, cols]
    has_bar = np.isfinite(close)
    prior_bars = np.cumsum(has_bar, axis=0) - has_bar          # bars strictly before row t
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]

    # (1) eligibility
    eligible = has_bar & (prior_bars >= MIN_PRIOR_BARS) & tradable & np.isfinite(k)
    n_elig = eligible.sum(axis=1)

    # (3) common-closure exclusion: count of eligible ETFs (j included) with the same k as j
    kk = np.where(eligible, k, np.nan)
    same = (kk[:, :, None] == kk[:, None, :]) & eligible[:, None, :]
    same_cnt = same.sum(axis=2)
    common = 2 * same_cnt > n_elig[:, None]

    # (2) run-up window 2 <= k <= pre_sessions + 1
    with np.errstate(invalid="ignore"):
        in_window = (kk >= 2) & (kk <= int(pre_sessions) + 1)

    return eligible & in_window & ~common


def target_weights(data, params):
    pre = int(params["pre_sessions"])
    tilt = float(params["tilt_weight"])
    marked = marked_etfs(data, pre)

    T = len(data.dates)
    N = len(data.instruments)
    w = np.zeros((T, N), dtype=float)

    n = marked.sum(axis=1).astype(float)
    each = np.where(n > 0, np.minimum(tilt, CORE_EQUITY / np.maximum(n, 1.0)), 0.0)
    for j, name in enumerate(COUNTRY):
        w[:, data.col(name)] = np.where(marked[:, j], each, 0.0)
    w[:, data.col("SPY")] = CORE_EQUITY - n * each
    w[:, data.col("IEF")] = CORE_BOND
    return w
