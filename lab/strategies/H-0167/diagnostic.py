"""H-0167 mechanism test: entry of a country ETF into its pre-holiday run-up (rules (1)-(3), pre_sessions)."""
import numpy as np

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


def _marked(data, pre_sessions):
    # copy of strategy.marked_etfs
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
    prior_bars = np.cumsum(has_bar, axis=0) - has_bar
    tradable = np.asarray(data.tradable, dtype=bool)[:, cols]

    eligible = has_bar & (prior_bars >= MIN_PRIOR_BARS) & tradable & np.isfinite(k)
    n_elig = eligible.sum(axis=1)

    kk = np.where(eligible, k, np.nan)
    same = (kk[:, :, None] == kk[:, None, :]) & eligible[:, None, :]
    same_cnt = same.sum(axis=2)
    common = 2 * same_cnt > n_elig[:, None]

    with np.errstate(invalid="ignore"):
        in_window = (kk >= 2) & (kk <= int(pre_sessions) + 1)

    return eligible & in_window & ~common


def events(data, params):
    marked = _marked(data, int(params["pre_sessions"]))
    prev = np.zeros_like(marked)
    prev[1:] = marked[:-1]
    entry = marked & ~prev                     # marked on d, not marked on the previous session

    T = len(data.dates)
    N = len(data.instruments)
    mask = np.zeros((T, N), dtype=bool)
    tradable = np.asarray(data.tradable, dtype=bool)
    for j, name in enumerate(COUNTRY):
        c = data.col(name)
        mask[:, c] = entry[:, j] & tradable[:, c]
    if data.universe is not None:
        mask &= np.asarray(data.universe, dtype=bool)
    return mask, None
