"""Seed `universe_candidates.txt` from the vendor ticker table plus raw OHLCV.

Append-only. Deleting a delisted name is survivorship bias applied by hand
(project rule 6). Filters are 04-DATA_AND_UNIVERSE.md §7.1.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]

from scout.data.ingest import delisted_fraction, read_candidate_pairs
from scout.data.schemas import MIN_DELISTED_FRACTION, TICKERS_COLUMNS
from scout.utils.errors import ScoutDataError

ALLOWED_EXCHANGES: frozenset[str] = frozenset({"NYSE", "NASDAQ", "NYSEARCA", "BATS"})
# 04 §7.1 writes `{Domestic Common Stock, ETF}` and names the exclusions as
# ADRs, preferreds, warrants, units, closed-end funds, and OTC. Sharadar splits
# dual-class listings into Primary/Secondary Class (JPM, GOOGL, BRK.B). Those
# are still domestic common stock, not an excluded category.
ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {
        "Domestic Common Stock",
        "Domestic Common Stock Primary Class",
        "Domestic Common Stock Secondary Class",
        "ETF",
    }
)
MIN_LIFETIME_DOLLAR_VOLUME_USD: float = 2_000_000.0


def select_candidates(
    tickers: pd.DataFrame,
    max_dollar_volume: pd.Series,
) -> pd.DataFrame:
    """Return `asset_id,symbol` rows that pass the §7.1 seed filters.

    `max_dollar_volume` is the lifetime max of `close * volume` on unadjusted
    bars, indexed by `asset_id`. A name with no OHLCV row is excluded.
    """
    if tickers.empty:
        return pd.DataFrame(columns=["asset_id", "symbol"])
    needed = ("asset_id", "symbol", "exchange", "category")
    missing = [c for c in needed if c not in tickers.columns]
    if missing:
        raise ScoutDataError(f"tickers missing columns: {missing}")
    work = tickers.loc[:, ["asset_id", "symbol", "exchange", "category"]].copy()
    work["asset_id"] = work["asset_id"].astype(str)
    work["symbol"] = work["symbol"].astype(str)
    work["exchange"] = work["exchange"].astype(str)
    work["category"] = work["category"].astype(str)
    work = work.drop_duplicates(subset=["asset_id"], keep="first")
    work = work.loc[
        work["exchange"].isin(ALLOWED_EXCHANGES) & work["category"].isin(ALLOWED_CATEGORIES)
    ]
    dv = max_dollar_volume.astype("float64")
    dv.index = dv.index.astype(str)
    before = len(work)
    joined = work.merge(
        dv.rename("max_dollar_volume_usd"),
        left_on="asset_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if len(joined) != before:
        raise ScoutDataError("candidate/dollar-volume merge duplicated rows")
    qualified = joined["max_dollar_volume_usd"].fillna(0.0) >= MIN_LIFETIME_DOLLAR_VOLUME_USD
    out = joined.loc[qualified, ["asset_id", "symbol"]].sort_values(
        ["asset_id", "symbol"], kind="mergesort"
    )
    return out.reset_index(drop=True)


def lifetime_max_dollar_volume(ohlcv_dir: Path) -> pd.Series:
    """Max `close * volume` per asset_id across yearly raw OHLCV parquets."""
    files = sorted(
        path
        for path in ohlcv_dir.glob("*.parquet")
        if path.stem.isdigit() and path.is_file()
    )
    if not files:
        raise ScoutDataError(f"no OHLCV parquet files under {ohlcv_dir}")
    parts: list[pd.Series] = []
    for path in files:
        frame = pd.read_parquet(path, columns=["asset_id", "close", "volume"])
        if frame.empty:
            continue
        dollar = frame["close"].astype("float64") * frame["volume"].astype("float64")
        grouped = dollar.groupby(frame["asset_id"].astype(str), sort=False).max()
        parts.append(grouped)
    if not parts:
        return pd.Series(dtype="float64")
    stacked = pd.concat(parts, axis=0)
    return stacked.groupby(level=0, sort=False).max()


def parse_candidate_lines(text: str) -> list[tuple[str, str]]:
    """Parse `asset_id,symbol` pairs. Comments and blanks are ignored."""
    rows: list[tuple[str, str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) != 2:
            raise ScoutDataError(f"expected asset_id,symbol; got {stripped!r}")
        rows.append((parts[0], parts[1]))
    return rows


def append_candidates(path: Path, selected: pd.DataFrame) -> int:
    """Append newly selected pairs. Existing lines are never rewritten or deleted.

    Returns the number of rows added.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_text = path.read_text(encoding="utf-8") if path.is_file() else ""
    existing = parse_candidate_lines(existing_text)
    have = {aid for aid, _ in existing}
    additions: list[tuple[str, str]] = []
    for rec in selected.itertuples(index=False):
        aid = str(rec.asset_id)
        sym = str(rec.symbol)
        if aid in have:
            continue
        have.add(aid)
        additions.append((aid, sym))
    if not additions:
        return 0
    block = "\n".join(f"{aid},{sym}" for aid, sym in additions) + "\n"
    if existing_text and not existing_text.endswith("\n"):
        block = "\n" + block
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block)
    return len(additions)


def seed_candidates(
    tickers: pd.DataFrame,
    ohlcv_dir: Path,
    dest: Path,
) -> tuple[int, int, float]:
    """Select §7.1 candidates and append them to `dest`.

    Returns (n_selected, n_added, delisted_fraction_of_file_after).
    """
    if tickers.empty:
        raise ScoutDataError("tickers table is empty; run ingest first")
    missing = [c for c in TICKERS_COLUMNS if c not in tickers.columns]
    if missing:
        raise ScoutDataError(f"tickers missing columns: {missing}")
    max_dv = lifetime_max_dollar_volume(ohlcv_dir)
    selected = select_candidates(tickers, max_dv)
    if selected.empty:
        raise ScoutDataError("seed produced no candidates; check exchange/category filters")
    added = append_candidates(dest, selected)
    pairs = read_candidate_pairs(dest)
    fraction = delisted_fraction(pairs, tickers)
    if fraction < MIN_DELISTED_FRACTION:
        raise ScoutDataError(
            f"candidate list delisted fraction {fraction:.1%} is below "
            f"{MIN_DELISTED_FRACTION:.0%} after seed; refusing to continue "
            "(survivorship bias)"
        )
    return int(len(selected)), int(added), float(fraction)
