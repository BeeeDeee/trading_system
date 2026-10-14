"""Forward paper test of research 17 (docs/research17/FORWARD.md): data after 2026-09-30, panel, evaluation.

Data: daily 15m ZIPs of data.binance.vision (sha256 verified, only days that are already published), funding
from the Binance REST endpoint, membership from the monthly 1d ZIPs. The panel is built exactly like
`scripts/r15_build_panel.py`; evaluation reuses `qlab.research17.engine` unchanged on a daily grid that starts
2026-10-01 (the grid constants of `setup` are patched for the duration of an evaluation).
"""

import hashlib
import io
import json
import urllib.error
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from qlab.research8 import download as dl
from qlab.research15 import universe as uni
from qlab.research15.evaluate import by_year, stats
from qlab.research16 import engine as G
from qlab.research17 import engine as R, setup as T
from qlab.research15.setup import epoch_15m

FORWARD_START = date(2026, 10, 1)
FORWARD_END = date(2027, 3, 31)
BASE_SNAPSHOT = "binance_2026-10-03"
LAST_BASE_MONTH = "2026-09"
MIN_EVENTS = 30
N_RND, N_BOOT = 1000, 2000
RAW = Path("data/raw/binance_forward17")
BASE_RAW = Path("data/raw") / BASE_SNAPSHOT
FAPI = "https://fapi.binance.com/fapi/v1/fundingRate"
FUNDING_PAGE = 1000                     # records per REST call (the endpoint maximum)
RULE_FILES = ["src/qlab/research15/strategy.py", "src/qlab/research15/universe.py",
              "src/qlab/research16/engine.py", "src/qlab/research16/families.py",
              "src/qlab/research17/engine.py", "src/qlab/research17/evaluate.py", "src/qlab/research17/setup.py"]
LOCK = Path("docs/research17/forward_lock.json")
SCENARIOS = [R.Scenario("base"), R.Scenario("cost75", cost=7.5e-4), R.Scenario("x2", cost=2 * T.COST),
             R.Scenario("delay1", delay=1), R.Scenario("nofund", use_funding=False),
             R.Scenario("nolag", lag=False), R.Scenario("btc_perp_leader", leader="perp")]


def rule_hash(files: list[str] = RULE_FILES) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + Path(f).read_bytes() + b"\0")
    return h.hexdigest()


# ---------------------------------------------------------------- download

def kline_key(market: str, sym: str, day: date, interval: str = "15m") -> str:
    root = "futures/um" if market == "perp" else "spot"
    return f"data/{root}/daily/klines/{sym}/{interval}/{sym}-{interval}-{day:%Y-%m-%d}.zip"


def fetch_file(key: str, dest: Path) -> bool:
    """Download `key` from data.binance.vision and verify its CHECKSUM; True if the file is new. Idempotent."""
    marker = dest.with_suffix(".zip.ok")
    if dest.exists() and marker.exists():
        return False
    blob = dl._get(f"{dl.HTTP}/{key}")
    expected = dl._get(f"{dl.HTTP}/{key}.CHECKSUM").decode().split()[0]
    got = hashlib.sha256(blob).hexdigest()
    if got != expected:
        raise ValueError(f"checksum mismatch {key}: {got} != {expected}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    marker.write_text(expected + "\n")
    return True


def fetch_days(market: str, sym: str, first: date, last: date, out: Path = RAW) -> dict:
    """Daily 15m ZIPs from `first` to `last`; the first day that is not published yet (404) ends the loop."""
    d = out / f"{market}_15m" / sym
    new, last_ok, day = 0, None, first
    while day <= last:
        try:
            new += fetch_file(kline_key(market, sym, day), d / f"{sym}-15m-{day:%Y-%m-%d}.zip")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            break
        last_ok, day = day, day + timedelta(days=1)
    return {"market": market, "symbol": sym, "new": new, "last_day": str(last_ok) if last_ok else None}


def fetch_months(kind: str, sym: str, out: Path, first_month: str | None, last_month: str) -> dict:
    """Monthly ZIPs of `kind` (see `research8.download.PREFIX`) for months in [first_month, last_month]."""
    keys, _ = dl.s3_list(dl.PREFIX[kind].format(sym=sym), delimiter=False)
    zips = [k for k in keys if k.endswith(".zip") and (m := dl.MONTH_RE.search(k))
            and (first_month is None or m.group(1) >= first_month) and m.group(1) <= last_month]
    new = sum(fetch_file(k, out / kind / sym / k.rsplit("/", 1)[1]) for k in zips)
    return {"kind": kind, "symbol": sym, "files": len(zips), "new": new}


def fetch_funding(sym: str, out: Path = RAW, get=dl._get) -> dict:
    """Funding records since the forward start from the REST endpoint; the raw JSON is kept and only extended."""
    f = out / "funding_rest" / f"{sym}.json"
    have = json.loads(f.read_text()) if f.exists() else []
    start = max((r["t"] for r in have), default=int(datetime(*FORWARD_START.timetuple()[:3], tzinfo=timezone.utc)
                                                    .timestamp() * 1000) - 1) + 1
    new = 0
    while True:
        rows = json.loads(get(f"{FAPI}?symbol={sym}&startTime={start}&limit={FUNDING_PAGE}"))
        if not rows:
            break
        have += [{"t": int(r["fundingTime"]), "rate": float(r["fundingRate"])} for r in rows]
        new += len(rows)
        start = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < FUNDING_PAGE:
            break
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(sorted({r["t"]: r for r in have}.values(), key=lambda r: r["t"])))
    return {"symbol": sym, "new_funding": new}


def last_complete_month(today: date) -> str:
    first = today.replace(day=1) - timedelta(days=1)
    return f"{first.year}-{first.month:02d}"


# ---------------------------------------------------------------- membership

def volumes_of(sym: str) -> tuple[np.ndarray, np.ndarray]:
    """Daily quote volume of a perp from the base snapshot and the forward monthly 1d ZIPs."""
    parts = [uni.daily_quote_volume(p / "perp_1d" / sym) for p in (BASE_RAW, RAW)]
    d, q = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    d, idx = np.unique(d, return_index=True)
    return d, q[idx]


def forward_months() -> list[date]:
    out, y, m = [], FORWARD_START.year, FORWARD_START.month
    while (y, m) <= (FORWARD_END.year, FORWARD_END.month):
        out.append(date(y, m, 1))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def check_stable(old: dict[str, list[str]], new: dict[str, list[str]]) -> None:
    """A month that already has members must keep them (data revisions would change the test)."""
    for m, v in old.items():
        if v and new.get(m) != v:
            raise ValueError(f"membership of {m} changed: {v} -> {new.get(m)}")


def refresh_members(today: date, workers: int = 8, out: Path = RAW) -> dict:
    """Download the missing monthly 1d ZIPs of every eligible perp and compute the membership (prereg §3)."""
    spot_symbols = set(dl.list_symbols("spot_1d"))
    elig = uni.eligible(dl.list_symbols("perp_1d"), spot_symbols)
    last = last_complete_month(today)

    def one(sym: str) -> dict:
        known = (BASE_RAW / "perp_1d" / sym).exists()
        try:
            return fetch_months("perp_1d", sym, out, "2026-10" if known else None, last)
        except Exception as e:
            return {"symbol": sym, "error": f"{type(e).__name__}: {e}"}

    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(one, elig))
    errors = [r for r in res if "error" in r]
    members = uni.monthly_members({s: volumes_of(s) for s in elig}, forward_months())
    f = out / "members.json"
    if f.exists():
        check_stable(json.loads(f.read_text())["members"], members)
    out.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(), "last_complete_month": last,
                             "members": members, "errors": errors}, indent=1))
    return {"members": members, "errors": errors, "eligible": len(elig)}


# ---------------------------------------------------------------- panel

def _read_klines(files: list[Path]) -> np.ndarray:
    rows = []
    for f in files:
        with zipfile.ZipFile(f) as z:
            txt = z.read(z.namelist()[0]).decode()
        lines = [ln for ln in txt.splitlines() if ln and ln[0].isdigit()]
        if lines:
            rows.append(np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",", ndmin=2, usecols=range(6)))
    return np.vstack(rows) if rows else np.zeros((0, 6))


def funding_records(sym: str) -> list[tuple[int, float]]:
    """(time ms, rate): monthly ZIPs of the base snapshot, then the REST records of the forward period."""
    recs = []
    for f in sorted((BASE_RAW / "funding" / sym).glob(f"{sym}-fundingRate-*.zip")):
        with zipfile.ZipFile(f) as z:
            for ln in z.read(z.namelist()[0]).decode().splitlines()[1:]:
                if ln:
                    p = ln.split(",")
                    recs.append((int(p[0]), float(p[2])))
    rest = RAW / "funding_rest" / f"{sym}.json"
    if rest.exists():
        recs += [(r["t"], r["rate"]) for r in json.loads(rest.read_text())]
    return recs


def build_panel(files: list[Path], funding: list[tuple[int, float]] | None = None, first_grid: int = 0) -> dict:
    """Panel dict (t, o, h, l, c, real, funding) from kline ZIPs; same rules as `scripts/r15_build_panel.py`:
    first record of a duplicate timestamp wins, the close is carried over missing candles (`real` False),
    funding goes to the 15m candle that contains its time. The panel ends at the last candle that exists."""
    k = _read_klines(files)
    ts = k[:, 0].astype(np.int64)
    ts = np.where(ts > 10**14, ts // 1000, ts)                   # microseconds since 2025
    q, idx = np.unique(ts // 900_000, return_index=True)
    o, h, lo, c = (k[idx, j] for j in (1, 2, 3, 4))
    t0 = max(first_grid, int(q[0]) // 16 * 16)
    n = int(q[-1]) + 1 - t0
    pos = q - t0
    ok = pos >= 0
    real = np.zeros(n, bool)
    real[pos[ok]] = True
    arr = {key: np.full(n, np.nan) for key in "ohlc"}
    for key, v in zip("ohlc", (o, h, lo, c)):
        arr[key][pos[ok]] = v[ok]
    first = int(np.argmax(real))
    for t in range(first + 1, n):
        if not real[t]:
            arr["c"][t] = arr["c"][t - 1]
            arr["o"][t] = arr["h"][t] = arr["l"][t] = arr["c"][t]
    fund = np.full(n, np.nan)
    for ms, rate in funding or []:
        i = ms // 900_000 - t0
        if 0 <= i < n:
            fund[i] = rate
    return {"t": t0 + np.arange(n, dtype=np.int64), "o": arr["o"], "h": arr["h"], "l": arr["l"], "c": arr["c"],
            "real": real, "funding": fund}


def kline_files(market: str, sym: str) -> list[Path]:
    """Every 15m ZIP of the symbol: base snapshot, then forward (monthly for new symbols, daily)."""
    files = []
    for root in (BASE_RAW, RAW):
        files += sorted((root / f"{market}_15m" / sym).glob(f"{sym}-15m-*.zip"))
    return files


def truncate(d: dict, asof: date) -> dict:
    n = int((d["t"] < epoch_15m(asof) + 96).sum())
    return {k: v[:n] for k, v in d.items()}


def last_day(d: dict) -> date:
    return (np.datetime64(int(d["t"][-1]) * 900, "s").astype("datetime64[D]")).astype(object)


def load_panels(syms: list[str]) -> dict:
    """{'BTCUSDT@spot': ..., 'BTCUSDT@perp': ..., 'SYM': perp panel} for the symbols that were downloaded."""
    out = {"BTCUSDT@spot": build_panel(kline_files("spot", "BTCUSDT"))}
    for s in sorted({"BTCUSDT", "ETHUSDT", *syms}):
        files = kline_files("perp", s)
        if files:
            out[f"{s}@perp" if s in ("BTCUSDT", "ETHUSDT") else s] = build_panel(files, funding_records(s))
    return out


# ---------------------------------------------------------------- evaluation

@contextmanager
def forward_window(asof: date):
    """The grid constants of `setup` (read at call time by `engine`) cover FORWARD_START .. asof."""
    old = T.START, T.END
    T.START, T.END = FORWARD_START, asof
    try:
        yield
    finally:
        T.START, T.END = old


def data_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(RAW.rglob("*.ok")):
        h.update(str(p.relative_to(RAW)).encode() + p.read_bytes())
    for p in sorted((RAW / "funding_rest").glob("*.json")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def evaluate(panels: dict, members: dict, asof: date, n_rnd: int = N_RND, n_boot: int = N_BOOT) -> dict:
    """Primary and descriptive results of the forward test up to `asof` (FORWARD.md §3-4). `panels` as from
    `load_panels`; `members` {'YYYY-MM': [symbols]} (a month with an empty list is pending and not traded)."""
    pend = sorted(m for m in {f"{d.year}-{d.month:02d}" for d in forward_months()
                              if d <= asof.replace(day=1)} if not members.get(m))
    ok_members = {m: v for m, v in members.items() if v}
    followers = sorted({s for v in ok_members.values() for s in v if s in panels})
    pan = {k: truncate(v, asof) for k, v in panels.items()}
    with forward_window(asof):
        leaders = {"spot": R.leader(pan["BTCUSDT@spot"]), "perp": R.leader(pan["BTCUSDT@perp"])}
        s0f = int(epoch_15m(FORWARD_START))
        res, lights, stale = {}, {}, {}
        for sym in followers:
            a = G.Asset.build(pan[sym])
            res[sym] = R.run_follower(a, leaders, SCENARIOS, max(s0f - int(a.t[0]), 0),
                                      R.member_mask(a, sym, ok_members))
            lights[sym] = R.light(a)
            if last_day(pan[sym]) < asof:
                stale[sym] = str(last_day(pan[sym]))
        eth = G.Asset.build(pan["ETHUSDT@perp"])
        eth_ref = R.run_follower(eth, leaders, [SCENARIOS[0]], max(s0f - int(eth.t[0]), 0), None, weight=1.0)["base"]
        nd = T.n_days()
        zero = np.zeros(nd)

        def portfolio(name):
            return sum((r[name].daily for r in res.values()), zero)

        def trades(name):
            xs = [r[name].rets for r in res.values() if len(r[name].rets)]
            return np.concatenate(xs) if xs else np.zeros(0)

        base = {s: r["base"] for s, r in res.items() if len(r["base"].entries)}
        daily = portfolio("base")
        sp = stats(daily, trades("base") if len(trades("base")) else None)
        mean, lo, hi, n_events = R.event_ci([r.entries for r in base.values()], [r.rets for r in base.values()],
                                            n_boot=n_boot)
        p_null, rnd = None, None
        if n_events:
            ev = R.events_of(base)
            bt = pan["BTCUSDT@spot"]
            times = R.candidate_times(bt["t"], bt["real"], s0f, int(bt["t"][-1]) - R.HOLD - 1)
            rnd = np.array([stats(R.null_run(lights, ev, times, np.random.default_rng(sd))[0])["sharpe"]
                            for sd in range(n_rnd)])
            p_null = float((1 + (rnd >= sp["sharpe"]).sum()) / (1 + n_rnd))
        dates = (T.day0() + np.arange(nd)).astype("datetime64[D]")
        per_sym = {s: {"n": int(len(r.rets)), "mean": float(r.rets.mean()),
                       "contribution": float(r.daily.sum())} for s, r in base.items()}
        out = {
            "asof": str(asof), "days": int(nd), "period": [str(FORWARD_START), str(asof)],
            "members_used": ok_members, "pending_months": pend, "followers": followers, "stale_symbols": stale,
            "n_events": n_events, "n_trades": int(len(trades("base"))), "P_stats": sp,
            "mean_trade_ci90_events": [mean, lo, hi],
            "primary_criterion": {"sharpe_gt_0": bool(sp["sharpe"] > 0), "mean_trade_gt_0": bool(mean > 0),
                                  "min_events": MIN_EVENTS, "enough_events": bool(n_events >= MIN_EVENTS),
                                  "meets": bool(sp["sharpe"] > 0 and mean > 0)} if n_events else None,
            "null_p": p_null, "null_median": float(np.median(rnd)) if rnd is not None else None,
            "scenarios": {sc.name: stats(portfolio(sc.name), trades(sc.name) if len(trades(sc.name)) else None)
                          for sc in SCENARIOS},
            "eth_reference_weight1": stats(eth_ref.daily, eth_ref.rets if len(eth_ref.rets) else None),
            "per_symbol": per_sym,
        }
        out["daily"] = {"dates": [str(d) for d in dates], "ret": [float(x) for x in daily]}
        out["trades"] = sorted(({"sym": s, "entry": str(np.datetime64(int(t) * 900, "s")), "ret": float(x)}
                                for s, r in base.items() for t, x in zip(r.entries, r.rets)),
                               key=lambda x: x["entry"])
        m = dates.astype("datetime64[M]")
        out["by_month"] = {str(k): float(np.prod(1 + daily[m == k]) - 1) for k in np.unique(m)}
        if len(dates) > 1:
            out["by_year"] = by_year(dates, daily)
    return out
