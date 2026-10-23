import io, sys, zipfile, datetime as dt, concurrent.futures as cf, time
import httpx, numpy as np
SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
end_day = dt.date(2026, 10, 19); days = [end_day - dt.timedelta(d) for d in range(89, -1, -1)]
cli = httpx.Client(timeout=30)
def ts(s): return dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc).timestamp() * 1000
def get(sym, day):
    for _ in range(3):
        try:
            r = cli.get(f"https://data.binance.vision/data/futures/um/daily/metrics/{sym}/{sym}-metrics-{day}.zip")
            z = zipfile.ZipFile(io.BytesIO(r.content)); L = z.read(z.namelist()[0]).decode().splitlines()[1:]
            return np.array([(ts(l.split(",")[0]), float(l.split(",")[2]), float(l.split(",")[3])) for l in L])
        except Exception: time.sleep(1)
    return np.zeros((0, 3))
res = {}
with cf.ThreadPoolExecutor(16) as ex:
    for s in SYMS:
        res[s] = np.concatenate([a for a in ex.map(lambda d: get(s, d), days) if len(a)])
np.savez_compressed(sys.argv[1], **res)
