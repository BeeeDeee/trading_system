"""Labels, feature assembly and walk-forward LightGBM of research 11 (prereg §3, §5).

Decision k is taken after the close of row days[k] and filled at the open of days[k] + 1; its label is the
open-to-open total return until the fill of decision k + 1 (delisting payouts included), as a cross-sectional
percentile among universe members. A label is usable for training only once realized, i.e. when its end
row days[k + 1] + 1 is before the first test decision's row.
"""

import numpy as np

from qlab.data.panel import Panel
from qlab.research4.meta import GBM_PARAMS
from qlab.research4.sleeves import pct_rank

PRICE = ["mom_12_1", "mom_6_1", "mom_3", "lowvol_63", "lowvol_252", "lowbeta_252", "strev_21", "strev_5",
         "ltrev_36_12", "high52", "trend_200", "trend_50"]
FUND_R4 = ["value_ep", "value_bp", "value_ebitda_ev", "value_sp", "size_large", "quality_gpa", "quality_roe",
           "quality_lowlev", "invest_lowag", "growth_rev", "growth_eps", "dividend"]
FUND_NEW = ["accruals", "net_issuance", "eps_surprise"]
INS = ["insider", "ins_sell", "ins_n_buyers", "ins_n_sellers", "ins_net_n"]
FLOW = ["io", "d_io", "d_holders", "d_breadth", "putcall"]
MODELS = {"M_P": PRICE, "M_PFI": PRICE + FUND_R4 + FUND_NEW + INS, "M_ALL": PRICE + FUND_R4 + FUND_NEW + INS + FLOW}
GROUP = {**{f: "P" for f in PRICE}, **{f: "F" for f in FUND_R4 + FUND_NEW}, **{f: "I" for f in INS},
         **{f: "H" for f in FLOW}}
TOP_N = 50
TRAIN_START = np.datetime64("2003-01-01")


def open_log_index(p: Panel, rows: np.ndarray, cols: slice) -> np.ndarray:
    """log TR index at the open of each row in `rows` (len R), columns `cols`. A -100 % move counts as a log
    step of -1000 (exp -> 0), so windows containing it return -100 % and later windows stay finite."""
    with np.errstate(divide="ignore", invalid="ignore"):
        co = np.maximum(np.log1p(np.maximum(np.asarray(p.ret_co[:, cols], dtype=float), -1.0)), -1000.0)
        oc = np.maximum(np.log1p(np.maximum(np.asarray(p.ret_oc[:, cols], dtype=float), -1.0)), -1000.0)
        cs = np.cumsum(co + oc, axis=0)
        before = np.vstack([np.zeros((1, cs.shape[1])), cs[:-1]])
        return before[rows] + co[rows]


def forward_returns(p: Panel, days: np.ndarray, chunk: int = 500) -> np.ndarray:
    """(D, N) open-to-open return of decision k (fill days[k]+1 -> fill days[k+1]+1); last row NaN."""
    D, N = len(days), p.shape[1]
    out = np.full((D, N), np.nan)
    fills = np.asarray(days) + 1
    ok = fills < p.shape[0]
    for a in range(0, N, chunk):
        L = open_log_index(p, fills[ok], slice(a, a + chunk))
        with np.errstate(invalid="ignore", over="ignore"):
            out[np.flatnonzero(ok)[:-1], a:a + chunk] = np.expm1(L[1:] - L[:-1])
    return out


def design(scores: dict[str, np.ndarray], universe: np.ndarray, feats: list[str]) -> np.ndarray:
    """(D, N, F) float32 cross-sectional percentiles of the features (NaN outside the universe)."""
    return np.stack([pct_rank(scores[f], universe).astype(np.float32) for f in feats], axis=-1)


def split(dates: np.ndarray, days: np.ndarray, Y: int) -> tuple[np.ndarray, np.ndarray]:
    """Decision indices (train, test) for test year Y: train labels are realized before the first test fill."""
    dd = np.asarray(dates, dtype="datetime64[D]")[days]
    year = dd.astype("datetime64[Y]").astype(int) + 1970
    test = np.flatnonzero(year == Y)
    if not len(test):
        return np.array([], dtype=int), test
    end_row = np.r_[np.asarray(days[1:]) + 1, np.iinfo(np.int64).max]      # row where label k is realized
    train = np.flatnonzero((dd >= TRAIN_START) & (end_row <= days[test[0]] + 1))
    return train, test


def walk_forward(X: np.ndarray, y: np.ndarray, universe: np.ndarray, dates: np.ndarray, days: np.ndarray,
                 test_years: list[int], seed: int = 0) -> tuple[np.ndarray, dict]:
    """Annual refit, expanding window from TRAIN_START. Returns (D, N) predictions for decisions in
    `test_years` (NaN elsewhere) and gain importance per year."""
    import lightgbm as lgb
    has_y = np.isfinite(y) & universe
    pred, imp = np.full(y.shape, np.nan), {}
    for Y in test_years:
        train, test = split(dates, days, Y)
        if not len(test):
            continue
        k, j = np.nonzero(has_y[train])
        k = train[k]
        model = lgb.LGBMRegressor(**{**GBM_PARAMS, "random_state": seed, "verbose": -1})
        model.fit(X[k, j], y[k, j])
        for t in test:
            m = np.flatnonzero(universe[t])
            pred[t, m] = model.predict(X[t, m])
        imp[Y] = model.booster_.feature_importance("gain").tolist()
    return pred, imp


def rank_ic(pred: np.ndarray, fwd: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Spearman correlation of prediction and realized forward return per decision (NaN if not available)."""
    out = np.full(len(pred), np.nan)
    for k in range(len(pred)):
        m = universe[k] & np.isfinite(pred[k]) & np.isfinite(fwd[k])
        if m.sum() > 30:
            a = np.argsort(np.argsort(pred[k, m])).astype(float)
            b = np.argsort(np.argsort(fwd[k, m])).astype(float)
            out[k] = np.corrcoef(a, b)[0, 1]
    return out
