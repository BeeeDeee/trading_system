"""Convert a raw Sharadar snapshot (bulk *.csv.zip) to Parquet, one file per table.

Streams each CSV out of its zip to a temporary file and lets DuckDB convert it with a
bounded memory budget, so it works on a small machine. Raw zips are never modified.

Usage: python scripts/raw_to_parquet.py data/raw/sharadar_YYYY-MM-DD [table ...]
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import duckdb

DATE_COLUMNS = {"date", "lastupdated", "datekey", "reportperiod", "calendardate", "filingdate",
                "transactiondate", "firstadded", "firstpricedate", "lastpricedate",
                "firstquarter", "lastquarter"}


def convert(zip_path: Path, out_dir: Path, tmp_dir: Path) -> None:
    table = zip_path.name.removesuffix(".csv.zip")
    out = out_dir / f"{table}.parquet"
    if out.exists():
        print(f"{table}: exists, skipping")
        return
    with zipfile.ZipFile(zip_path) as zf:
        (member,) = zf.namelist()
        csv_path = tmp_dir / member
        with zf.open(member) as src, open(csv_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 << 20)

    con = duckdb.connect()
    con.execute("SET memory_limit='1500MB'; SET threads=2; SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{tmp_dir}'")
    rel = con.sql(f"SELECT * FROM read_csv('{csv_path}', header=true, sample_size=-1)")
    casts = [f"TRY_CAST(\"{c}\" AS DATE) AS \"{c}\"" if c in DATE_COLUMNS and t == "VARCHAR"
             else f'"{c}"' for c, t in zip(rel.columns, rel.types)]
    tmp_out = out.with_suffix(".parquet.tmp")
    con.execute(f"COPY (SELECT {', '.join(casts)} FROM read_csv('{csv_path}', header=true, "
                f"sample_size=-1)) TO '{tmp_out}' (FORMAT parquet, COMPRESSION zstd, "
                f"ROW_GROUP_SIZE 500000)")
    rows = con.execute(f"SELECT count(*) FROM read_parquet('{tmp_out}')").fetchone()[0]
    con.close()
    tmp_out.rename(out)
    csv_path.unlink()
    print(f"{table}: {rows:,} rows -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main() -> None:
    raw = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    out_dir = Path("data/parquet") / raw.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir="data") as tmp:
        for zip_path in sorted(raw.glob("*.csv.zip"), key=lambda p: p.stat().st_size):
            if not wanted or zip_path.name.removesuffix(".csv.zip") in wanted:
                convert(zip_path, out_dir, Path(tmp))


if __name__ == "__main__":
    main()
