"""Data audit of a Sharadar snapshot (spec §4.6) -> docs/audit/<snapshot>.md

Usage: python scripts/audit_sharadar.py sharadar_YYYY-MM-DD
Needs data/parquet/<snapshot>/ and data/derived/<snapshot>/ (scripts/build_bars.py).
"""

import sys
from datetime import date
from pathlib import Path

import duckdb
import polars as pl

snapshot = sys.argv[1]
P = Path("data/parquet") / snapshot
DER = Path("data/derived") / snapshot
BARS = f"read_parquet('{DER / 'bars' / '*.parquet'}')"
t = lambda name: f"read_parquet('{P / f'{name}.parquet'}')"  # noqa: E731

con = duckdb.connect()
con.execute("SET memory_limit='1500MB'; SET threads=2; SET preserve_insertion_order=false")
sql = lambda q: con.sql(q).pl()  # noqa: E731
out: list[str] = []


def md(df: pl.DataFrame, floats: int = 4) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:,.{floats}f}"
        if isinstance(v, int):
            return f"{v:,}" if abs(v) >= 10_000 else str(v)  # keep years readable
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in df.rows())


def section(title: str, text: str = "", table: pl.DataFrame | None = None, floats: int = 4):
    out.append(f"\n## {title}\n")
    if text:
        out.append(text + "\n")
    if table is not None:
        out.append(md(table, floats) + "\n")


# 1. Inventory -----------------------------------------------------------------------------------
inv = []
for f in sorted(P.glob("*.parquet")):
    n = con.sql(f"SELECT count(*) FROM read_parquet('{f}')").fetchone()[0]
    inv.append((f.stem, n, round(f.stat().st_size / 1e6, 1)))
section("Inventář", table=pl.DataFrame(inv, schema=["tabulka", "řádků", "MB"], orient="row"), floats=1)

# 2. Prices --------------------------------------------------------------------------------------
section("Ceny (stocks)", "Kontroly neplatných hodnot a duplicit na celé tabulce SEP.", sql(f"""
    SELECT count(*) AS rows, count(DISTINCT ticker) AS tickers, min(date)::VARCHAR AS first,
      max(date)::VARCHAR AS last,
      count(*) FILTER (WHERE open IS NULL OR open <= 0) AS bad_open,
      count(*) FILTER (WHERE close IS NULL OR close <= 0 OR closeadj <= 0 OR closeunadj <= 0) AS bad_close,
      count(*) FILTER (WHERE volume = 0) AS zero_volume,
      (SELECT count(*) FROM (SELECT ticker, date FROM {t('stocks')} GROUP BY ALL HAVING count(*) > 1)) AS duplicates
    FROM {t('stocks')}"""))
section("Řádky s nulovým objemem", "Podíl řádků, které jsou kopií předchozího dne (O=H=L=C=předchozí close).",
        sql(f"""WITH s AS (SELECT *, lag(close) OVER (PARTITION BY ticker ORDER BY date) pc
                           FROM {t('stocks')})
                SELECT count(*) AS zero_volume_rows,
                  avg((open = high AND high = low AND low = close AND close = pc)::INT) AS stale_copy_share
                FROM s WHERE volume = 0"""))

# 3. Calendar ------------------------------------------------------------------------------------
cal = sql(f"""SELECT (SELECT count(DISTINCT date) FROM {t('stocks')}) AS sep_days,
                     (SELECT count(*) FROM {t('funds')} WHERE ticker = 'SPY') AS spy_days,
                     (SELECT count(*) FROM (SELECT DISTINCT date FROM {t('stocks')} EXCEPT
                       SELECT date FROM {t('funds')} WHERE ticker = 'SPY')) AS only_sep,
                     (SELECT count(*) FROM (SELECT date FROM {t('funds')} WHERE ticker = 'SPY' EXCEPT
                       SELECT DISTINCT date FROM {t('stocks')})) AS only_spy""")
closures = ["2001-09-11", "2001-09-12", "2001-09-13", "2001-09-14", "2004-06-11", "2007-01-02",
            "2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09"]
present = con.sql(f"""SELECT count(*) FROM (SELECT DISTINCT date FROM {t('stocks')})
                      WHERE date IN ({', '.join(f"DATE '{d}'" for d in closures)})""").fetchone()[0]
section("Obchodní kalendář", f"Mimořádné uzávěry NYSE přítomné v datech: {present} z {len(closures)} "
        "(má být 0).", cal)

# 4. Bars ----------------------------------------------------------------------------------------
section("Normalizované bars", "Stav řádků po normalizaci (univerzum kategorií, spec §5.1).", sql(f"""
    SELECT status, count(*) AS rows, count(DISTINCT permaticker) AS securities
    FROM {BARS} GROUP BY 1 ORDER BY 2 DESC"""))
section("Pokrytí v čase", "Počet titulů s alespoň jedním řádkem v daném období a počet delistingů.",
        sql(f"""SELECT (year(date) // 5) * 5 AS od_roku, count(DISTINCT permaticker) AS tituly,
                  count(*) FILTER (WHERE status = 'delisted') AS delistingy,
                  avg((status = 'no_open')::INT) AS podil_no_open
                FROM {BARS} GROUP BY 1 ORDER BY 1"""))
section("Extrémní denní výnosy", "Close-to-close total return; `liquid` = předchozí close ≥ 5 USD a "
        "dollar volume ≥ 1 mil. USD.", sql(f"""
    WITH r AS (SELECT (1 + ret_co) * (1 + ret_oc) - 1 AS c2c,
                 lag(close_u) OVER w AS pc, lag(dollar_volume) OVER w AS pdv
               FROM {BARS} WHERE status IN ('active', 'no_open')
               WINDOW w AS (PARTITION BY permaticker ORDER BY date))
    SELECT count(*) FILTER (WHERE abs(c2c) > 0.5) AS abs_gt_50pct,
           count(*) FILTER (WHERE c2c > 5) AS gt_500pct,
           count(*) FILTER (WHERE abs(c2c) > 0.5 AND pc >= 5 AND pdv >= 1e6) AS liquid_abs_gt_50pct,
           count(*) FILTER (WHERE c2c > 5 AND pc >= 5 AND pdv >= 1e6) AS liquid_gt_500pct
    FROM r"""))

# 5. Delistings ----------------------------------------------------------------------------------
dl = pl.read_parquet(DER / "delistings.parquet")
section("Delistingy", "Klasifikace podle akcí v poslední obchodní den (spec §4.5). "
        "`unknown` bez jakékoli akce řeší normalizace zvlášť.", dl.group_by("kind").agg(
            n=pl.len(), median_last_close=pl.col("last_close").median(),
            last_close_ge_1=(pl.col("last_close") >= 1).sum(),
            with_consideration=pl.col("consideration_per_share").is_not_null().sum(),
            implausible=(pl.col("raw_consideration").is_not_null()
                         & pl.col("consideration_per_share").is_null()).sum(),
        ).sort("n", descending=True), floats=2)
acq = dl.filter(pl.col("consideration_per_share").is_not_null()).with_columns(
    ret=pl.col("consideration_per_share") / pl.col("last_close") - 1)
section("Protihodnota akvizic vůči poslední ceně", "", pl.DataFrame({
    "kvantil": ["1 %", "5 %", "25 %", "50 %", "75 %", "95 %", "99 %"],
    "výnos": [acq["ret"].quantile(q) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]}))
term = sql(f"""SELECT count(*) AS delisting_rows, avg(terminal_ret) AS mean_terminal,
                 count(*) FILTER (WHERE terminal_ret = -1) AS minus_100pct,
                 count(*) FILTER (WHERE terminal_ret = 0) AS zero
               FROM {BARS} WHERE status = 'delisted'""")
section("Terminální výnosy v bars", "", term)

# 6. Sanity check vs SPY / RSP -------------------------------------------------------------------
sp = pl.read_parquet(P / "sp500.parquet")
current = set(sp.filter(pl.col("action") == "current")["ticker"])
events = sp.filter(pl.col("action").is_in(["added", "removed"])).sort("date", descending=True)
open_end: dict[str, date | None] = {tk: None for tk in current}
intervals = []
for d, action, tk in events.select("date", "action", "ticker").iter_rows():
    if action == "added":
        intervals.append((tk, d, open_end.pop(tk, None)))
    else:
        open_end[tk] = d
intervals += [(tk, None, end) for tk, end in open_end.items()]
iv = pl.DataFrame(intervals, schema={"ticker": pl.Utf8, "start": pl.Date, "end": pl.Date},
                  orient="row")
con.register("iv", iv)
snap_check = sql(f"""
    WITH snaps AS (SELECT date, ticker FROM {t('sp500')} WHERE action = 'historical'),
    recon AS (SELECT s.date, iv.ticker FROM (SELECT DISTINCT date FROM snaps) s JOIN iv
              ON (iv.start IS NULL OR s.date >= iv.start) AND (iv.end IS NULL OR s.date < iv.end))
    SELECT count(DISTINCT date) AS snapshots,
           (SELECT count(*) FROM snaps) AS snapshot_members,
           (SELECT count(*) FROM snaps JOIN recon USING (date, ticker)) AS matched
    FROM snaps""")
con.execute(f"""CREATE TEMP TABLE members AS
    SELECT CAST(tk.permaticker AS BIGINT) AS permaticker, iv.ticker, iv.start, iv.end
    FROM iv JOIN {t('tickers')} tk ON tk.ticker = iv.ticker AND tk."table" = 'SEP'""")
idx = sql(f"""
    WITH r AS (
        SELECT b.date, (1 + b.ret_co) * (1 + b.ret_oc) - 1 AS r,
               lag(d.marketcap) OVER (PARTITION BY b.permaticker ORDER BY b.date) AS mc
        FROM {BARS} b JOIN members m ON m.permaticker = b.permaticker
             AND (m.start IS NULL OR b.date >= m.start) AND (m.end IS NULL OR b.date < m.end)
        LEFT JOIN {t('daily')} d ON d.ticker = m.ticker AND d.date = b.date)
    SELECT date, count(*) AS n, sum(mc * r) / sum(mc) AS cw, avg(r) AS ew FROM r
    GROUP BY 1 ORDER BY 1""")
etf = sql(f"""SELECT date, ticker, closeadj / lag(closeadj) OVER (PARTITION BY ticker ORDER BY date) - 1 AS r
              FROM {t('funds')} WHERE ticker IN ('SPY', 'RSP')""").pivot(on="ticker", index="date",
                                                                         values="r")
j = idx.join(etf, on="date").drop_nulls(["cw", "SPY"]).with_columns(y=pl.col("date").dt.year())
yearly = j.group_by("y").agg(
    cw=(1 + pl.col("cw")).product() - 1, spy=(1 + pl.col("SPY")).product() - 1,
    ew=(1 + pl.col("ew")).product() - 1,
    rsp=pl.when(pl.col("RSP").is_not_null().all()).then((1 + pl.col("RSP")).product() - 1),
).sort("y").with_columns(td_cw_spy=pl.col("cw") - pl.col("spy"), td_ew_rsp=pl.col("ew") - pl.col("rsp"))
worst = yearly["td_cw_spy"].abs().max()
section("Sanity check: rekonstruovaný S&P 500 vs. SPY a RSP",
        f"Členství S&P 500 k datu z událostí added/removed, ověřeno proti kvartálním snímkům "
        f"(tabulka níže). Cap-weighted podle `daily.marketcap` předchozího dne, EW denně "
        f"rebalancovaný. Členů s daty denně: min {idx['n'].min()}, medián {idx['n'].median():.0f}.\n\n"
        f"{md(snap_check)}\n\n**Největší roční |cw − SPY|: {worst:.2%}** (tolerance spec §4.6: ~1 p.b.; "
        f"SPY nese poplatek ~0,09 % ročně). Průměr cw − SPY: {yearly['td_cw_spy'].mean():.2%}, "
        f"průměr ew − RSP (od 2004): {yearly.filter(pl.col('y') >= 2004)['td_ew_rsp'].mean():.2%}.",
        yearly)

header = (f"# Datový audit – {snapshot}\n\nVygenerováno `scripts/audit_sharadar.py` "
          f"({date.today()}). Interpretace a rozhodnutí: `docs/DATA_FINDINGS.md`.\n")
dest = Path("docs/audit") / f"{snapshot}.md"
dest.parent.mkdir(parents=True, exist_ok=True)
dest.write_text(header + "\n".join(out))
print(f"wrote {dest}; worst |cw - SPY| = {worst:.2%}")
