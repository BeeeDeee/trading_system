"""Stáhne 5m klines (REST) a denní bookDepth + metrics (data.binance.vision), uloží kompaktně do npz."""
import io, sys, zipfile, datetime as dt, concurrent.futures as cf, time
import httpx, numpy as np

OUT = sys.argv[1]
DAYS = int(sys.argv[2])
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
end_day = dt.date(2026, 10, 19)
days = [end_day - dt.timedelta(d) for d in range(DAYS - 1, -1, -1)]
t0 = int(dt.datetime.combine(days[0], dt.time(), dt.timezone.utc).timestamp() * 1000)
t1 = int(dt.datetime.combine(end_day + dt.timedelta(1), dt.time(), dt.timezone.utc).timestamp() * 1000)
cli = httpx.Client(timeout=30)


def klines(sym):
    rows, s = [], t0
    while s < t1:
        r = cli.get("https://fapi.binance.com/fapi/v1/klines",
                    params=dict(symbol=sym, interval="5m", startTime=s, endTime=t1 - 1, limit=1500)).json()
        if not r:
            break
        rows += r
        s = r[-1][0] + 300_000
        time.sleep(0.2)
    a = np.array([[float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[7]), float(x[10])] for x in rows])
    return a  # open_time, o, h, l, c, qv, tbqv


def zipcsv(url):
    for _ in range(3):
        try:
            r = cli.get(url)
            if r.status_code == 404:
                return None
            z = zipfile.ZipFile(io.BytesIO(r.content))
            return z.read(z.namelist()[0]).decode().splitlines()[1:]
        except Exception:
            time.sleep(1)
    return None


def ts(s):
    return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() * 1000


def bookdepth(sym, day):
    lines = zipcsv(f"https://data.binance.vision/data/futures/um/daily/bookDepth/{sym}/{sym}-bookDepth-{day}.zip")
    if not lines:
        return np.zeros((0, 3))
    out = []
    for ln in lines:
        p = ln.split(",")
        out.append((ts(p[0]), float(p[1]), float(p[3])))
    return np.array(out)  # ts, pct, notional


def metrics(sym, day):
    lines = zipcsv(f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day}.zip")
    if not lines:
        return np.zeros((0, 2))
    return np.array([(ts(l.split(",")[0]), float(l.split(",")[3])) for l in lines])  # ts, oi_value


res = {}
with cf.ThreadPoolExecutor(16) as ex:
    for sym in SYMS:
        res[f"{sym}_k"] = klines(sym)
        bd = list(ex.map(lambda d: bookdepth(sym, d), days))
        res[f"{sym}_bd"] = np.concatenate([b for b in bd if len(b)])
        mt = list(ex.map(lambda d: metrics(sym, d), days))
        res[f"{sym}_m"] = np.concatenate([m for m in mt if len(m)])
        print(sym, res[f"{sym}_k"].shape, res[f"{sym}_bd"].shape, res[f"{sym}_m"].shape, flush=True)
np.savez_compressed(OUT, **res)
