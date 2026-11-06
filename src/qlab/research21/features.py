"""Features of research 21 (prereg §3-§4, v1.1), computed per column chunk from `panel_r5` and `panel_r19_etf`.

Point-in-time contract: row t of every feature uses data up to the close of t, with ONE documented exception, the
earnings window E1 = [t-2, t+2] (prereg §4: the earnings date is known in advance in reality, so the post-window is a
mild look-ahead; r2 uses the strictly point-in-time window [t-2, t]). Rolling means / standard deviations of the
event measures (V, G, I) end at t-1, so the event never dilutes its own denominator (prereg v1.1 (a)).

Stored under data/derived/<snapshot>/r21/ (git-ignored), one .npy per matrix over the selected columns `cols`:
  rsi (f32), valid (bool), regime (bool, P > SMA200), exit (bool, P > SMA5), O (f64, open price index of the TR
  series), volratio / gapz / idioz (f32, NaN = cannot compute), e1 / e2 (bool, earnings in the window).
"""

import json
from pathlib import Path

import numpy as np
import polars as pl

from qlab.research19.signals import rsi_wilder, sma
from qlab.research21 import setup as S


def prior_stats(x: np.ndarray, window: int, min_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sample standard deviation of the `window` rows ending at t-1 (NaN-aware; NaN if fewer than
    `min_obs` finite values). Row t never includes x[t]."""
    ok = np.isfinite(x)
    x0 = np.where(ok, x, 0.0)
    T, N = x.shape
    pad = lambda c: np.vstack([np.zeros((1, N)), c])  # noqa: E731 - pad[k] = sum of rows 0..k-1
    s1, s2, n = pad(np.cumsum(x0, axis=0)), pad(np.cumsum(x0 * x0, axis=0)), pad(np.cumsum(ok, axis=0))
    hi = np.arange(T)
    lo = np.maximum(hi - window, 0)
    cnt, a, b = n[hi] - n[lo], s1[hi] - s1[lo], s2[hi] - s2[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = a / cnt
        var = (b - a * a / cnt) / (cnt - 1)
    enough = cnt >= min_obs
    return np.where(enough, mean, np.nan), np.where(enough, np.sqrt(np.maximum(var, 0.0)), np.nan)


def window_any(m: np.ndarray, before: int, after: int) -> np.ndarray:
    """out[t] = any(m[t-before : t+after]) along axis 0 (indices clipped at the ends)."""
    T = m.shape[0]
    c = np.vstack([np.zeros((1, m.shape[1]), dtype=np.int32), np.cumsum(m, axis=0, dtype=np.int32)])
    hi = np.minimum(np.arange(T) + after + 1, T)
    lo = np.maximum(np.arange(T) - before, 0)
    return (c[hi] - c[lo]) > 0


def compute_chunk(ret_co: np.ndarray, ret_oc: np.ndarray, listed: np.ndarray, tradable: np.ndarray,
                  dollar_volume: np.ndarray, earn: np.ndarray, etf_ret: np.ndarray) -> dict[str, np.ndarray]:
    """All features of one column chunk. `etf_ret` (T, n) is the day return of each column's sector ETF
    (all NaN where the stock has no sector ETF)."""
    valid = np.asarray(listed) & np.asarray(tradable)
    co, oc = np.asarray(ret_co, float), np.asarray(ret_oc, float)
    P = np.cumprod((1.0 + co) * (1.0 + oc), axis=0)
    O = np.vstack([np.full((1, P.shape[1]), np.nan), P[:-1]]) * (1.0 + co)      # open index (prev close x overnight)
    O[0] = np.nan
    rsi = rsi_wilder(P, valid, S.RSI_N)
    s200, s5 = sma(P, valid, S.SMA_REGIME), sma(P, valid, S.SMA_EXIT)
    with np.errstate(invalid="ignore"):
        regime, exit_ = valid & (P > s200), valid & (P > s5)

    dv = np.where(valid, np.asarray(dollar_volume, float), np.nan)
    mean_dv, _ = prior_stats(dv, S.VOL_WINDOW, S.VOL_MIN_OBS)
    with np.errstate(invalid="ignore", divide="ignore"):
        volratio = np.where(mean_dv > 0, dv / mean_dv, np.nan)

    gap = np.where(valid, co, np.nan)
    _, gsd = prior_stats(gap, S.GAP_WINDOW, S.GAP_MIN_OBS)
    with np.errstate(invalid="ignore", divide="ignore"):
        gapz = np.where(gsd > 0, gap / gsd, np.nan)

    r_day = np.where(valid, (1.0 + co) * (1.0 + oc) - 1.0, np.nan)
    diff = r_day - etf_ret
    _, dsd = prior_stats(diff, S.GAP_WINDOW, S.GAP_MIN_OBS)
    with np.errstate(invalid="ignore", divide="ignore"):
        idioz = np.where(dsd > 0, diff / dsd, np.nan)

    e = np.asarray(earn, bool)
    return {"rsi": rsi.astype(np.float32), "valid": valid, "regime": regime, "exit": exit_, "O": O,
            "volratio": volratio.astype(np.float32), "gapz": gapz.astype(np.float32),
            "idioz": idioz.astype(np.float32), "e1": window_any(e, S.E_BEFORE, S.E_AFTER),
            "e2": window_any(e, S.E_BEFORE, 0)}


DTYPES = {"rsi": np.float32, "valid": bool, "regime": bool, "exit": bool, "O": np.float64, "volratio": np.float32,
          "gapz": np.float32, "idioz": np.float32, "e1": bool, "e2": bool}


def sector_etf_columns(assets: np.ndarray, etf_cols: list[str], tickers_parquet: Path) -> np.ndarray:
    """For every stock column the index of its sector ETF in `etf_cols` (-1 = no sector or no ETF)."""
    t = (pl.read_parquet(tickers_parquet).filter(pl.col("table") == "SEP")
         .select("permaticker", "sector").unique("permaticker"))
    a = pl.DataFrame({"permaticker": assets}).with_columns(pl.col("permaticker").cast(t["permaticker"].dtype))
    sec = a.join(t, on="permaticker", how="left")["sector"].to_list()
    idx = {c: i for i, c in enumerate(etf_cols)}
    return np.array([idx[S.SECTOR_ETF[s]] if s in S.SECTOR_ETF else -1 for s in sec], dtype=np.int64)


def build(snapshot: str = S.SHARADAR, chunk: int = 400, log=print) -> Path:
    """Compute and store the features for every column that ever had ADV20 >= FEATURE_ADV (memory: one chunk of
    columns at a time, results written to memory-mapped .npy files)."""
    p, ex = S.load_stock_panel(snapshot)
    ep, _, etf_cols = S.load_etf_panel(snapshot)
    if not np.array_equal(p.dates, ep.dates):
        raise RuntimeError("stock and ETF panels have different calendars")
    adv, ok = ex["adv20"], ex["base_ok"]
    T, N = p.shape
    ever = np.zeros(N, bool)
    for c in range(0, N, 1000):
        s = slice(c, c + 1000)
        ever[s] = ((np.asarray(adv[:, s]) >= S.FEATURE_ADV) & np.asarray(ok[:, s])).any(axis=0)
    cols = np.flatnonzero(ever)
    sec = sector_etf_columns(p.assets, etf_cols, Path("data/parquet") / snapshot / "tickers.parquet")
    etf_day = np.full((T, len(etf_cols)), np.nan)
    ev = np.asarray(ep.listed) & np.asarray(ep.tradable)
    etf_day[ev] = ((1.0 + np.asarray(ep.ret_co)) * (1.0 + np.asarray(ep.ret_oc)) - 1.0)[ev]
    out = S.features_dir(snapshot)
    out.mkdir(parents=True, exist_ok=True)
    mm = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(T, len(cols)))
          for k, dt in DTYPES.items()}
    for j in range(0, len(cols), chunk):
        c = cols[j:j + chunk]
        etf_ret = np.where((sec[c] >= 0)[None, :], etf_day[:, np.maximum(sec[c], 0)], np.nan)
        f = compute_chunk(p.ret_co[:, c], p.ret_oc[:, c], p.listed[:, c], p.tradable[:, c], p.dollar_volume[:, c],
                          ex["earn8k"][:, c], etf_ret)
        for k in DTYPES:
            mm[k][:, j:j + len(c)] = f[k]
        log(f"features {min(j + chunk, len(cols))}/{len(cols)}")
    for m in mm.values():
        m.flush()
    (out / "meta.json").write_text(json.dumps({
        "snapshot": snapshot, "n_cols": int(len(cols)), "cols": cols.tolist(), "etf_cols": etf_cols,
        "sector_etf_col": sec[cols].tolist(), "feature_adv": S.FEATURE_ADV, "e_start": str(S.E_START)}))
    return out


class Features:
    """Memory-mapped features over the selected columns; `cols` maps feature column -> panel asset index."""

    def __init__(self, snapshot: str = S.SHARADAR):
        d = S.features_dir(snapshot)
        self.meta = json.loads((d / "meta.json").read_text())
        self.cols = np.array(self.meta["cols"], dtype=np.int64)
        self.m = {k: np.load(d / f"{k}.npy", mmap_mode="r") for k in DTYPES}

    def pos(self, asset: np.ndarray) -> np.ndarray:
        """Feature column of panel asset indices (must be inside `cols`)."""
        return np.searchsorted(self.cols, np.asarray(asset))
