"""Meta layer (research 4 pre-registration §4): which sleeves to activate each month.

Decision k is taken after the close of panel row `days[k]`; its holding period is rows
days[k]+1 .. days[k+1]. Everything known at decision k uses rows <= days[k] only:
- `M[k]` = sleeve returns over the holding period of decision k (the target, known later),
- features at k use M[:k] and daily returns up to days[k].
"""

import numpy as np

MIN_HISTORY_DAYS = 252
SLEEVE_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_1m", "vol_12m", "dd_12m",
                   "beta_12m", "sharpe_12m"]
MARKET_FEATURES = ["spy_ret_1m", "spy_ret_3m", "spy_ret_12m", "spy_above_sma200", "spy_vol_1m",
                   "spy_vol_3m", "vix", "vix_rel", "tbill", "tbill_chg_12m", "dispersion",
                   "breadth", "ief_minus_spy_3m"]


def period_returns(R: np.ndarray, days: np.ndarray) -> np.ndarray:
    """(D, S): compounded returns over each decision's holding period (NaN if any day NaN).

    The last period runs to the end of the data (partial month).
    """
    T = R.shape[0]
    ends = np.append(days[1:], T - 1)
    out = np.full((len(days), R.shape[1]), np.nan)
    for k, (a, b) in enumerate(zip(days, ends)):
        if b > a:
            out[k] = np.prod(1.0 + R[a + 1:b + 1], axis=0) - 1.0
    return out


def available(first_row: np.ndarray, days: np.ndarray) -> np.ndarray:
    """(D, S) bool: sleeve has at least MIN_HISTORY_DAYS of returns before decision k."""
    return (days[:, None] - np.asarray(first_row)[None, :] + 1) >= MIN_HISTORY_DAYS


def sleeve_features(R: np.ndarray, M: np.ndarray, days: np.ndarray, spy_ret: np.ndarray,
                    rf: np.ndarray) -> dict[str, np.ndarray]:
    """(D, S) per-sleeve features at each decision (NaN where history is missing)."""
    D, S = M.shape
    f = {n: np.full((D, S), np.nan) for n in SLEEVE_FEATURES}
    logm = np.log1p(M)
    for k, t in enumerate(days):
        for n, m in (("ret_1m", 1), ("ret_3m", 3), ("ret_6m", 6), ("ret_12m", 12)):
            if k >= m:
                f[n][k] = np.expm1(logm[k - m:k].sum(axis=0))
        if t + 1 >= 252:
            x = R[t + 1 - 252:t + 1]
            f["vol_1m"][k] = x[-21:].std(axis=0, ddof=1) * np.sqrt(252)
            f["vol_12m"][k] = x.std(axis=0, ddof=1) * np.sqrt(252)
            nav = np.cumprod(1.0 + x, axis=0)
            f["dd_12m"][k] = nav[-1] / nav.max(axis=0) - 1.0
            m = spy_ret[t + 1 - 252:t + 1]
            cov = ((x - x.mean(axis=0)) * (m - m.mean())[:, None]).sum(axis=0) / 251
            f["beta_12m"][k] = cov / m.var(ddof=1)
            ex = x - rf[t + 1 - 252:t + 1, None]
            sd = ex.std(axis=0, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                f["sharpe_12m"][k] = np.where(sd > 0, ex.mean(axis=0) / sd * np.sqrt(252), 0.0)
    return f


def market_features(days: np.ndarray, spy_ret: np.ndarray, ief_ret: np.ndarray, vix: np.ndarray,
                    tbill: np.ndarray, dispersion: np.ndarray, breadth: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """(D,) market features at each decision. Daily inputs are aligned to panel rows (value at the
    close of that row); `dispersion` and `breadth` are already per decision."""
    D = len(days)
    spy_idx = np.cumprod(1.0 + spy_ret)
    ief_idx = np.cumprod(1.0 + ief_ret)
    f = {n: np.full(D, np.nan) for n in MARKET_FEATURES}
    for k, t in enumerate(days):
        def ret(idx, n):
            return idx[t] / idx[t - n] - 1.0 if t >= n and idx[t - n] > 0 else np.nan
        f["spy_ret_1m"][k], f["spy_ret_3m"][k] = ret(spy_idx, 21), ret(spy_idx, 63)
        f["spy_ret_12m"][k] = ret(spy_idx, 252)
        if t >= 199:
            f["spy_above_sma200"][k] = float(spy_idx[t] > spy_idx[t - 199:t + 1].mean())
        if t >= 63:
            f["spy_vol_1m"][k] = spy_ret[t - 20:t + 1].std(ddof=1) * np.sqrt(252)
            f["spy_vol_3m"][k] = spy_ret[t - 62:t + 1].std(ddof=1) * np.sqrt(252)
        f["vix"][k] = vix[t]
        if t >= 251:
            f["vix_rel"][k] = vix[t] / np.nanmedian(vix[t - 251:t + 1])
            f["tbill_chg_12m"][k] = tbill[t] - tbill[t - 251]
        f["tbill"][k] = tbill[t]
        f["ief_minus_spy_3m"][k] = ret(ief_idx, 63) - f["spy_ret_3m"][k]
    f["dispersion"] = np.asarray(dispersion, dtype=float)
    f["breadth"] = np.asarray(breadth, dtype=float)
    return f


def select_top(score: np.ndarray, avail: np.ndarray, families: np.ndarray, k: int = 5,
               per_family: int | None = 2) -> np.ndarray:
    """(D, S) weights: top-k available sleeves by score, at most `per_family` per family, EW."""
    W = np.zeros(score.shape)
    for d in range(score.shape[0]):
        ok = np.flatnonzero(avail[d] & np.isfinite(score[d]))
        chosen, used = [], {}
        for s in ok[np.argsort(-score[d, ok], kind="stable")]:
            if per_family is not None and used.get(families[s], 0) >= per_family:
                continue
            chosen.append(s)
            used[families[s]] = used.get(families[s], 0) + 1
            if len(chosen) == k:
                break
        if chosen:
            W[d, chosen] = 1.0 / len(chosen)
    return W


def xs_rank(score: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Per-row percentile rank among available sleeves with a finite score."""
    out = np.full(score.shape, np.nan)
    for d in range(score.shape[0]):
        ok = np.flatnonzero(avail[d] & np.isfinite(score[d]))
        if len(ok):
            out[d, ok] = (np.argsort(np.argsort(score[d, ok], kind="stable")) + 1) / len(ok)
    return out


def rank_ic(score: np.ndarray, realized: np.ndarray, avail: np.ndarray) -> np.ndarray:
    """Per-decision Spearman correlation between score and realized return (NaN if < 5 sleeves)."""
    out = np.full(score.shape[0], np.nan)
    for d in range(score.shape[0]):
        ok = avail[d] & np.isfinite(score[d]) & np.isfinite(realized[d])
        if ok.sum() < 5:
            continue
        a = np.argsort(np.argsort(score[d, ok])).astype(float)
        b = np.argsort(np.argsort(realized[d, ok])).astype(float)
        out[d] = np.corrcoef(a, b)[0, 1]
    return out


# ---------------------------------------------------------------- scoring methods

def regime_scores(y: np.ndarray, state: np.ndarray, train: np.ndarray, test: np.ndarray,
                  prior: float = 12.0) -> np.ndarray:
    """M2: shrunk mean relative return per sleeve in the current regime state.

    `y` (D, S) relative targets (NaN = no sample), `state` (D,) int, `train`/`test` bool masks.
    """
    out = np.full(y.shape, np.nan)
    yt = np.where(train[:, None], y, np.nan)
    n_all = np.isfinite(yt).sum(axis=0)
    with np.errstate(invalid="ignore"):
        m_all = np.where(n_all > 0, np.nansum(yt, axis=0) / np.maximum(n_all, 1), 0.0)
    for s_val in np.unique(state[test]):
        rows = train & (state == s_val)
        ys = y[rows]
        n = np.isfinite(ys).sum(axis=0)
        m = np.where(n > 0, np.nansum(ys, axis=0) / np.maximum(n, 1), 0.0)
        score = (n * m + prior * m_all) / (n + prior)
        out[test & (state == s_val)] = score
    return out


GBM_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                  subsample=0.7, subsample_freq=1, colsample_bytree=0.7, random_state=0,
                  verbose=-1, n_jobs=2)


def gbm_scores(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
               cat_cols: list[int]) -> np.ndarray:
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**GBM_PARAMS)
    model.fit(X[train_idx], y[train_idx], categorical_feature=cat_cols)
    return model.predict(X[test_idx])


def ridge_design(Xnum: np.ndarray, fam: np.ndarray, n_fam: int, market_cols: list[int]
                 ) -> np.ndarray:
    """Numeric features + family one-hot + family x market interactions (unscaled)."""
    onehot = np.zeros((len(fam), n_fam))
    onehot[np.arange(len(fam)), fam] = 1.0
    inter = (onehot[:, :, None] * Xnum[:, market_cols][:, None, :]).reshape(len(fam), -1)
    return np.hstack([Xnum, onehot, inter])


def ridge_scores(Z: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
                 alpha: float = 10.0) -> np.ndarray:
    from sklearn.linear_model import Ridge
    mu = np.nanmean(Z[train_idx], axis=0)
    sd = np.nanstd(Z[train_idx], axis=0)
    sd[sd == 0] = 1.0
    zs = lambda a: np.nan_to_num((a - mu) / sd)  # noqa: E731  missing -> training mean
    model = Ridge(alpha=alpha).fit(zs(Z[train_idx]), y[train_idx])
    return model.predict(zs(Z[test_idx]))
