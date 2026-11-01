"""Machine-learning layer for strategies: point-in-time walk-forward models behind one function.

Agents never import sklearn, lightgbm or torch (the static scan allows only numpy and `lab.framework.*`).
A strategy builds point-in-time features with `lab.framework.api` and this module, calls `walk_forward`, and
turns the scores into weights. Everything that can leak is owned by the framework:

- **Label and purge.** `forward_return(data, h)[t]` is the total return from the open of t+1 to the open of
  t+1+h (NaN when that window is not complete). A model fitted for decisions from row r (after the close of
  r) trains only on samples s with s + h + 1 <= r, i.e. labels fully realized by the open of r. The purge uses
  the `horizon` argument, so it must equal the horizon of the label (G0 perturbation catches a longer label).
- **Schedule.** Refit on rows r = 0 mod `retrain_every` (absolute row index from the start of the data, so the
  schedule is identical on truncated data); the model of r scores rows r .. r+retrain_every-1.
- **Target.** Cross-sectional rank of the label per date (in [-0.5, 0.5]); `logit` uses its sign. No future
  statistic enters (each date is ranked on its own cross-section).
- **Determinism.** Every model gets seed + r; one thread budget; sample caps keep the run time bounded.
  Fits are memoized in-process by a hash of the training rows, so G0's repeated truncation and perturbation
  calls refit only what changed (the memo never crosses processes and touches no file).
- **Compute budget.** More than `MAX_FITS[kind]` scheduled refits raises: widen `retrain_every` in the card.

Models (`MODELS`): ridge, logit, rf (random forest), gbm (LightGBM), mlp (feed-forward) on the feature vector at
t; lstm, tcn on the last `seq_len` days of the feature vectors (need torch, extra `deep`; see `available()`).
Hyperparameters are the keys of `DEFAULTS[model]`; unknown keys raise. Put the ones worth varying in the card's
`signal.params` with a grid, as for any other parameter (G2 tests the neighbors).
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np

MODELS = ("ridge", "logit", "rf", "gbm", "mlp", "lstm", "tcn")
DEEP = ("lstm", "tcn")
MAX_FITS = {"tabular": 150, "deep": 60}
MAX_SAMPLES = {"rf": 40_000, "mlp": 100_000, "tabular": 200_000, "deep": 30_000}   # training samples per fit (deterministic subsample above)
MIN_SAMPLES = 2_000
THREADS = 2

DEFAULTS: dict[str, dict] = {
    "ridge": {"alpha": 100.0},
    "logit": {"C": 0.1},
    "rf": {"n_estimators": 60, "max_depth": 6, "min_samples_leaf": 200, "max_features": 1.0},
    "gbm": {"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 200,
            "subsample": 0.7, "colsample": 1.0},
    "mlp": {"hidden": 32, "alpha": 1e-3, "epochs": 20},
    "lstm": {"hidden": 16, "seq_len": 20, "epochs": 5, "lr": 2e-3, "weight_decay": 1e-3, "dropout": 0.1},
    "tcn": {"hidden": 16, "seq_len": 20, "levels": 3, "epochs": 5, "lr": 2e-3, "weight_decay": 1e-3,
            "dropout": 0.1},
}
LIMITS = {"hidden": (2, 64), "epochs": (1, 50), "seq_len": (5, 60), "levels": (1, 4), "n_estimators": (10, 400),
          "max_depth": (1, 12), "num_leaves": (2, 63)}


_DRY = False
DRY_CAPS = {"n_estimators": 8, "epochs": 1, "hidden": 4, "num_leaves": 4, "max_depth": 3}


def set_dry_run(on: bool) -> None:
    """`lab try` mode: the Builder checks mechanics and point-in-time behaviour on random data, where a model can
    learn nothing, so models are shrunk (few trees, one epoch, 3000 samples). Same code path, same schedule."""
    global _DRY
    _DRY = bool(on)
    _MEMO.clear()


def kind(model: str) -> str:
    return "deep" if model in DEEP else "tabular"


def available() -> dict[str, bool]:
    """Which models can run in this environment (lstm and tcn need torch)."""
    import importlib.util
    deep = importlib.util.find_spec("torch") is not None    # not imported: torch's OpenMP pool slows LightGBM down 10x
    return {m: (deep if m in DEEP else True) for m in MODELS}


# ---------------------------------------------------------------- feature and label helpers


def stack(*features: np.ndarray) -> np.ndarray:
    """(T, N) feature matrices -> one (T, N, F) float32 tensor."""
    return np.stack([np.asarray(f, dtype=np.float32) for f in features], axis=2)


def _members(x: np.ndarray, universe: np.ndarray | None) -> np.ndarray:
    ok = np.isfinite(x)
    return ok if universe is None else ok & universe


def cs_zscore(x: np.ndarray, universe: np.ndarray | None = None, clip: float = 4.0) -> np.ndarray:
    """Cross-sectional z-score per date over universe members (NaN elsewhere); clipped at +-clip."""
    x = np.asarray(x, dtype=np.float64)
    ok = _members(x, universe)
    xm = np.where(ok, x, np.nan)
    n = ok.sum(axis=1, keepdims=True)
    mu = np.where(n > 1, np.nansum(xm, axis=1, keepdims=True) / np.maximum(n, 1), np.nan)
    sd = np.sqrt(np.where(n > 1, np.nansum((xm - mu) ** 2, axis=1, keepdims=True) / np.maximum(n - 1, 1), np.nan))
    z = (xm - mu) / np.where(sd > 1e-12, sd, np.nan)
    return np.clip(z, -clip, clip)


def cs_rank(x: np.ndarray, universe: np.ndarray | None = None) -> np.ndarray:
    """Cross-sectional rank per date in [-0.5, 0.5] over universe members (NaN elsewhere)."""
    x = np.asarray(x, dtype=np.float64)
    ok = _members(x, universe)
    order = np.argsort(np.where(ok, x, np.inf), axis=1, kind="stable")
    rank = np.empty(x.shape)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(x.shape[1], dtype=float), x.shape), axis=1)
    n = ok.sum(axis=1, keepdims=True)
    return np.where(ok & (n > 1), rank / np.maximum(n - 1, 1) - 0.5, np.nan)


def forward_return(data, horizon: int) -> np.ndarray:
    """Label: total return from the open of t+1 to the open of t+1+horizon, (T, N); NaN where the window is not
    complete or the instrument cannot be bought at the open of t+1. Uses rows up to t+horizon+1: only
    `walk_forward` may consume it, and it purges accordingly."""
    h = int(horizon)
    if h < 1:
        raise ValueError("horizon must be >= 1")
    T, N = data.ret_oc.shape
    lc = np.cumsum(np.log1p(np.maximum(data.ret_co, -0.999999)), axis=0)
    lo = np.cumsum(np.log1p(np.maximum(data.ret_oc, -0.999999)), axis=0)
    y = np.full((T, N), np.nan)
    if T > h + 1:
        t = np.arange(T - h - 1)
        y[t] = np.expm1((lo[t + h] - lo[t]) + (lc[t + h + 1] - lc[t + 1]))
        y[t] = np.where(data.tradable[t + 1], y[t], np.nan)
    return y


def score_weights(score: np.ndarray, universe: np.ndarray | None, top: float = 0.2, bottom: float = 0.0,
                  min_names: int = 10) -> np.ndarray:
    """Equal-weight book from scores, per date: long the top `top` fraction, short the bottom `bottom` fraction
    (0 = long-only). Gross 1 (long-only: sum 1; long-short: 0.5 long + 0.5 short). Dates with fewer than
    `min_names` scored members get all-zero weights (a decision: flat), dates without any score stay NaN."""
    ok = _members(score, universe)
    w = np.full(score.shape, np.nan)
    have = ok.any(axis=1)
    rank = cs_rank(score, universe)                       # in [-0.5, 0.5], NaN outside
    n = ok.sum(axis=1, keepdims=True)
    long = (rank >= 0.5 - top) & ok
    short = (rank <= -0.5 + bottom) & ok if bottom > 0 else np.zeros_like(ok)
    thin = n < min_names
    gl, gs = 1.0 if not bottom else 0.5, 0.5
    wl = np.where(long, gl / np.maximum(long.sum(axis=1, keepdims=True), 1), 0.0)
    ws = np.where(short, -gs / np.maximum(short.sum(axis=1, keepdims=True), 1), 0.0)
    out = np.where(thin, 0.0, wl + ws)
    w[have] = out[have]
    return w


# ---------------------------------------------------------------- walk-forward


_MEMO: OrderedDict = OrderedDict()
_MEMO_MAX = 600


def fit_schedule(T: int, retrain_every: int, min_train_days: int, horizon: int) -> list[int]:
    """Refit rows: multiples of retrain_every with enough labelled history."""
    first = max(min_train_days + horizon + 1, retrain_every)
    return [r for r in range(0, T, retrain_every) if r >= first]


def walk_forward(X: np.ndarray, y: np.ndarray, *, horizon: int, model: str, params: dict | None = None,
                 retrain_every: int = 21, min_train_days: int = 504, train_days: int | None = None,
                 universe: np.ndarray | None = None, seed: int = 0) -> np.ndarray:
    """Point-in-time scores (T, N): row t is the prediction made after the close of t.

    X (T, N, F) features known at the close of t (use cs_zscore / cs_rank); y = forward_return(data, horizon).
    `train_days` None = expanding window, else the last train_days rows. NaN scores where there is no model yet,
    no complete feature vector (or no complete `seq_len` window), or the instrument is not a universe member."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {MODELS}")
    if not available()[model]:
        raise RuntimeError(f"model {model!r} needs torch (uv sync --extra deep)")
    hp = _hyper(model, params)
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or y.shape != X.shape[:2]:
        raise ValueError(f"X must be (T, N, F) and y (T, N); got {X.shape} and {y.shape}")
    T, N, F = X.shape
    retrain_every, min_train_days, horizon = int(retrain_every), int(min_train_days), int(horizon)
    if retrain_every < 5 or min_train_days < 60 or horizon < 1:
        raise ValueError("retrain_every >= 5, min_train_days >= 60, horizon >= 1")
    if model in DEEP and retrain_every < 21:
        raise ValueError("lstm / tcn: retrain_every >= 21")
    sched = fit_schedule(T, retrain_every, min_train_days, horizon)
    full_fits = len(fit_schedule(max(T, 6500), retrain_every, min_train_days, horizon))
    if full_fits > MAX_FITS[kind(model)]:
        raise ValueError(f"{model}: {full_fits} scheduled refits over ~25 years exceeds the limit "
                         f"{MAX_FITS[kind(model)]}; raise retrain_every (now {retrain_every})")
    L = int(hp.get("seq_len", 1)) if model in DEEP else 1
    ok_row = np.isfinite(X).all(axis=2)
    if universe is not None:
        ok_row &= universe
    if L > 1:                                            # complete window of L finite feature vectors
        c = np.cumsum(np.vstack([np.zeros((1, N)), ok_row]), axis=0)
        ok_seq = np.zeros_like(ok_row)
        ok_seq[L - 1:] = (c[L:] - c[:-L]) == L
    else:
        ok_seq = ok_row
    yr = cs_rank(y, universe)
    dig = _digests(X, yr, ok_seq)
    out = np.full((T, N), np.nan)
    for r in sched:
        lo = 0 if train_days is None else max(0, r - horizon - int(train_days))
        hi = r - horizon - 1                              # last row whose label is realized by the open of r
        if hi - lo < min_train_days // 2:
            continue
        s, j = np.nonzero(ok_seq[lo:hi + 1] & np.isfinite(yr[lo:hi + 1]))
        if len(s) < MIN_SAMPLES:
            continue
        s = s + lo
        cap = 3000 if _DRY else MAX_SAMPLES.get(model, MAX_SAMPLES[kind(model)])
        if len(s) > cap:
            keep = np.sort(np.random.default_rng(seed + r).choice(len(s), cap, replace=False))
            s, j = s[keep], j[keep]
        key = (model, tuple(sorted(hp.items())), seed, r, lo, hi, horizon, F, N,
               hashlib.blake2b(b"".join(dig[lo:hi + 1]), digest_size=16).digest())
        fitted = _MEMO.get(key)
        if fitted is None:
            Xs, ys = _gather(X, s, j, L), yr[s, j]
            fitted = _fit(model, hp, Xs, ys, seed + r)
            _MEMO[key] = fitted
            while len(_MEMO) > _MEMO_MAX:
                _MEMO.popitem(last=False)
        else:
            _MEMO.move_to_end(key)
        end = min(r + retrain_every, T)
        ts, js = np.nonzero(ok_seq[r:end])
        if len(ts):
            ts = ts + r
            out[ts, js] = _predict(model, fitted, _gather(X, ts, js, L))
    return out


def _hyper(model: str, params: dict | None) -> dict:
    hp = dict(DEFAULTS[model])
    for k, v in (params or {}).items():
        if k not in hp:
            raise ValueError(f"{model}: unknown hyperparameter {k!r}; known: {sorted(hp)}")
        hp[k] = type(DEFAULTS[model][k])(v)
    for k, (a, b) in LIMITS.items():
        b = 10 if (k == "epochs" and model in DEEP) else b
        if k in hp and not a <= hp[k] <= b:
            raise ValueError(f"{model}: {k}={hp[k]} outside [{a}, {b}]")
    if _DRY:
        hp = {k: (type(v)(min(v, DRY_CAPS[k])) if k in DRY_CAPS else v) for k, v in hp.items()}
    return hp


def _digests(X: np.ndarray, yr: np.ndarray, ok: np.ndarray) -> list[bytes]:
    yb = np.ascontiguousarray(np.nan_to_num(yr, nan=9.0).astype(np.float32))
    ob = np.ascontiguousarray(ok)
    out = []
    for t in range(len(X)):
        h = hashlib.blake2b(digest_size=8)
        h.update(X[t].tobytes())
        h.update(yb[t].tobytes())
        h.update(ob[t].tobytes())
        out.append(h.digest())
    return out


def _gather(X: np.ndarray, s: np.ndarray, j: np.ndarray, L: int) -> np.ndarray:
    if L == 1:
        return X[s, j]                                   # (n, F)
    return X[s[:, None] + np.arange(-L + 1, 1)[None, :], j[:, None]]   # (n, L, F)


# ---------------------------------------------------------------- models


def _fit(model: str, hp: dict, Xs: np.ndarray, ys: np.ndarray, seed: int):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # fixed-epoch fits by design: convergence warnings are noise
        return _fit_inner(model, hp, Xs, ys, seed)


def _fit_inner(model: str, hp: dict, Xs: np.ndarray, ys: np.ndarray, seed: int):
    seed = int(seed % (2**31 - 1))
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=hp["alpha"]).fit(Xs, ys)
    if model == "logit":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=hp["C"], max_iter=200).fit(Xs, (ys > 0).astype(int))
    if model == "rf":
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
                                   min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
                                   n_jobs=THREADS, random_state=seed).fit(Xs, ys)
        rf.n_jobs = 1                    # threaded predict sums the trees in a varying order: not bit-identical
        return rf
    if model == "gbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                                 num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"],
                                 subsample=hp["subsample"], subsample_freq=1, colsample_bytree=hp["colsample"],
                                 num_threads=THREADS, random_state=seed, deterministic=True,
                                 force_row_wise=True, verbose=-1).fit(Xs, ys)
    if model == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=(hp["hidden"], max(hp["hidden"] // 2, 2)), alpha=hp["alpha"],
                            max_iter=hp["epochs"], random_state=seed, batch_size=512).fit(Xs, ys)
    return _fit_deep(model, hp, Xs, ys, seed)


def _predict(model: str, fitted, Xs: np.ndarray) -> np.ndarray:
    if model in DEEP:
        return _predict_deep(fitted, Xs)
    if model == "logit":
        return fitted.decision_function(Xs)
    return fitted.predict(Xs)


def _net(model: str, hp: dict, F: int):
    import torch.nn as nn

    class LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn, self.drop, self.head = nn.LSTM(F, hp["hidden"], batch_first=True), nn.Dropout(hp["dropout"]), nn.Linear(hp["hidden"], 1)

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(self.drop(h[:, -1])).squeeze(-1)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, c = [], F
            for i in range(hp["levels"]):
                d = 2 ** i
                layers += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(c, hp["hidden"], 3, dilation=d), nn.ReLU(),
                           nn.Dropout(hp["dropout"])]    # causal: left padding only
                c = hp["hidden"]
            self.conv, self.head = nn.Sequential(*layers), nn.Linear(c, 1)

        def forward(self, x):
            return self.head(self.conv(x.transpose(1, 2))[:, :, -1]).squeeze(-1)

    return LSTM() if model == "lstm" else TCN()


def _fit_deep(model: str, hp: dict, Xs: np.ndarray, ys: np.ndarray, seed: int):
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(seed)
    net = _net(model, hp, Xs.shape[2])
    opt = torch.optim.AdamW(net.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    Xt, yt = torch.from_numpy(np.ascontiguousarray(Xs)), torch.from_numpy(ys.astype(np.float32))
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(hp["epochs"]):
        perm = torch.randperm(len(Xt), generator=g)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    net.eval()
    return net


def _predict_deep(net, Xs: np.ndarray) -> np.ndarray:
    import torch
    out = np.empty(len(Xs))
    with torch.no_grad():
        for i in range(0, len(Xs), 8192):
            out[i:i + 8192] = net(torch.from_numpy(np.ascontiguousarray(Xs[i:i + 8192]))).numpy()
    return out
