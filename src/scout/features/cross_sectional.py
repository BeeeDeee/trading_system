"""Cross-sectional percentile ranks within the eligible set at each ts.

Features does not import universe; `snapshots` is a DataFrame passed in as data.
Lookup of eligibility rounds backward: the latest snapshot with snapshot.ts <= t.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

RANK_PAIRS: tuple[tuple[str, str], ...] = (
    ("mom_252_skip21", "mom_252_xs_pct"),
    ("vol_60", "vol_xs_pct"),
)


def eligible_mask(features: pd.DataFrame, snapshots: pd.DataFrame) -> pd.Series:
    """True iff the symbol is eligible on the latest snapshot with ts <= bar ts.

    A missing snapshot is ineligible (conservative). Ineligible names receive
    False, not a rank computed against a population they were not part of.
    """
    feat = _ts_symbol_frame(features)
    snap = _snapshots_unique(snapshots)
    feat = feat.assign(_row=range(len(feat)))
    feat_sorted = feat.sort_values(["ts", "symbol"], kind="mergesort")
    snap_sorted = snap.sort_values(["ts", "symbol"], kind="mergesort")
    before = len(feat_sorted)
    joined = pd.merge_asof(
        feat_sorted,
        snap_sorted,
        on="ts",
        by="symbol",
        direction="backward",
    )
    assert len(joined) == before, "merge_asof duplicated rows"
    joined = joined.sort_values("_row", kind="mergesort")
    mask = joined["eligible"].eq(True)
    return pd.Series(mask.to_numpy(), index=features.index, dtype=bool, name="eligible")


def cross_sectional_ranks(features: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Percentile ranks within the eligible set at each ts.

    Ineligible symbols (and names with a NaN source column) receive NaN, not
    a rank against a population they were not part of. Ties use average rank.
    """
    missing = [col for col, _ in RANK_PAIRS if col not in features.columns]
    if missing:
        raise ValueError(f"features missing rank source columns: {missing}")
    out = features.copy()
    elig = eligible_mask(out, snapshots)
    ts = _ts_series(out)
    for src, dest in RANK_PAIRS:
        ranked = out[src].where(elig)
        out[dest] = ranked.groupby(ts, sort=False).rank(pct=True, na_option="keep")
    population = elig.groupby(ts, sort=False).transform("sum")
    out["xs_population"] = population.astype("int64")
    return out


def _ts_symbol_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "ts" in frame.columns and "symbol" in frame.columns:
        return frame.loc[:, ["ts", "symbol"]].copy()
    reset = frame.reset_index()
    missing = [name for name in ("ts", "symbol") if name not in reset.columns]
    if missing:
        raise ValueError(f"features must have ts and symbol; missing {missing}")
    return reset.loc[:, ["ts", "symbol"]]


def _ts_series(frame: pd.DataFrame) -> pd.Series:
    if "ts" in frame.columns:
        return frame["ts"]
    if "ts" in frame.index.names:
        return pd.Series(frame.index.get_level_values("ts"), index=frame.index)
    raise ValueError("features must have a ts column or index level")


def _snapshots_unique(snapshots: pd.DataFrame) -> pd.DataFrame:
    required = ("ts", "symbol", "eligible")
    missing = [name for name in required if name not in snapshots.columns]
    if missing:
        raise ValueError(f"snapshots missing columns: {missing}")
    subset = snapshots.loc[:, list(required)]
    if subset.duplicated(subset=["ts", "symbol"]).any():
        raise ValueError("snapshots must be unique on (ts, symbol)")
    return subset
