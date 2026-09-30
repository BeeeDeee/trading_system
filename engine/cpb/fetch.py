"""Live market data from public exchange APIs (no keys needed except the optional CoinGecko demo key).

Every call goes through http.Recorder, so each body is stored verbatim with URL, time and sha256.
Normalized formats (shared with the simulator):
  daily candle   [date, open, high, low, close, base_volume, quote_volume_usd]   (UTC day, closed)
  intraday bar   [open_ms, open, high, low, close]
Prices are only ever parsed from API responses here. Nothing is estimated or interpolated.
"""
import json
import urllib.parse

from . import canon
from .http import FetchError

BINANCE_HOSTS = ("https://api.binance.com", "https://data-api.binance.vision")
DAY_MS = 86_400_000
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def _f(x):
    v = float(x)
    if v != v:
        raise ValueError("NaN")
    return v


def book_summary(bids, asks):
    """bids/asks = [(price, qty)] best first."""
    if not bids or not asks or bids[0][0] <= 0 or asks[0][0] < bids[0][0]:
        return None
    mid = (bids[0][0] + asks[0][0]) / 2
    bd = sum(p * q for p, q in bids)
    ad = sum(p * q for p, q in asks)
    return {"mid": mid, "spread_bps": (asks[0][0] - bids[0][0]) / mid * 1e4, "bid_depth": bd, "ask_depth": ad,
            "imbalance": (bd - ad) / (bd + ad) if bd + ad > 0 else None,
            "range_pct": (asks[-1][0] - bids[-1][0]) / mid * 100, "levels": min(len(bids), len(asks))}


class LiveMarket:
    simulated = False

    def __init__(self, rec, cg_key=None):
        self.rec = rec
        self.cg_key = cg_key

    # ------------------------------------------------------------------ Binance
    def _binance(self, path, name, kind, meta=None, compress=False):
        err = None
        for host in BINANCE_HOSTS:
            try:
                return self.rec.get(host + path, name, kind, meta, compress=compress), host
            except FetchError as e:
                err = e
        raise err

    def exchange_status(self, pairs=None):
        """{pair: status}. With pairs: small query; if a pair no longer exists (HTTP 400) the full list is
        fetched (gzip-stored) and missing pairs are reported as "MISSING"."""
        if pairs:
            q = urllib.parse.quote(json.dumps(sorted(pairs), separators=(",", ":")))
            try:
                data, _ = self._binance(f"/api/v3/exchangeInfo?symbols={q}", "binance_exchangeinfo", "exchange_info", compress=True)
                return {s["symbol"]: s["status"] for s in data["symbols"]}
            except FetchError:
                pass
        data, _ = self._binance("/api/v3/exchangeInfo?permissions=SPOT", "binance_exchangeinfo_full", "exchange_info", compress=True)
        st = {s["symbol"]: s["status"] for s in data["symbols"]}
        if pairs:
            return {p: st.get(p, "MISSING") for p in pairs}
        return st

    def binance_quote_volume_24h(self, pairs):
        q = urllib.parse.quote(json.dumps(sorted(pairs), separators=(",", ":")))
        data, _ = self._binance(f"/api/v3/ticker/24hr?symbols={q}", "binance_24hr", "volume_24h")
        return {t["symbol"]: _f(t["quoteVolume"]) for t in data}

    def daily_candles(self, coin, pair, start_date, end_date):
        """Closed daily candles with open date in [start_date, end_date]. Falls back per coin."""
        errors = []
        for src in ("binance", "okx", "coinbase", "kraken"):
            try:
                rows = getattr(self, f"_daily_{src}")(coin, pair, start_date, end_date)
                rows = [r for r in rows if start_date <= r[0] <= end_date]
                if rows:
                    return rows, src
                errors.append(f"{src}: no rows")
            except (FetchError, KeyError, ValueError, TypeError, IndexError) as e:
                errors.append(f"{src}: {e}")
        raise FetchError(f"{coin}: no daily data ({'; '.join(errors)})")

    def _daily_binance(self, coin, pair, start_date, end_date):
        start, end = canon.date_ms(start_date), canon.date_ms(end_date) + DAY_MS - 1
        out = []
        while start <= end:
            data, host = self._binance(
                f"/api/v3/klines?symbol={pair}&interval=1d&startTime={start}&endTime={end}&limit=1000",
                f"binance_1d_{coin}", "daily", {"coin": coin, "pair": pair})
            if not data:
                break
            for k in data:
                if k[6] >= canon.date_ms(end_date) + DAY_MS:   # not closed yet
                    continue
                out.append([canon.ms_date(k[0]), _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[5]), _f(k[7])])
            start = data[-1][0] + DAY_MS
            if len(data) < 1000:
                break
        return out

    def _daily_okx(self, coin, pair, start_date, end_date):
        n = canon.days_between(start_date, end_date) + 1
        if n > 300:
            raise ValueError("okx fallback limited to 300 days")
        after = canon.date_ms(end_date) + DAY_MS
        data = self.rec.get(f"https://www.okx.com/api/v5/market/candles?instId={coin}-USDT&bar=1Dutc&after={after}&limit={min(n, 300)}",
                            f"okx_1d_{coin}", "daily", {"coin": coin})
        if data.get("code") != "0":
            raise ValueError(f"okx code {data.get('code')}")
        return sorted([canon.ms_date(int(k[0])), _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[5]), _f(k[7])]
                      for k in data["data"] if k[8] == "1")

    def _daily_coinbase(self, coin, pair, start_date, end_date):
        if canon.days_between(start_date, end_date) + 1 > 300:
            raise ValueError("coinbase fallback limited to 300 days")
        s, e = start_date + "T00:00:00Z", canon.add_days(end_date, 1) + "T00:00:00Z"
        data = self.rec.get(f"https://api.exchange.coinbase.com/products/{coin}-USD/candles?granularity=86400&start={s}&end={e}",
                            f"coinbase_1d_{coin}", "daily", {"coin": coin})
        rows = []
        for t, lo, hi, op, cl, vol in data:
            d = canon.ms_date(t * 1000)
            if d <= end_date:
                rows.append([d, _f(op), _f(hi), _f(lo), _f(cl), _f(vol), _f(vol) * _f(cl)])
        return sorted(rows)

    def _daily_kraken(self, coin, pair, start_date, end_date):
        k = {"BTC": "XBT", "DOGE": "XDG"}.get(coin, coin)
        since = canon.date_ms(start_date) // 1000 - 1
        data = self.rec.get(f"https://api.kraken.com/0/public/OHLC?pair={k}USD&interval=1440&since={since}",
                            f"kraken_1d_{coin}", "daily", {"coin": coin})
        if data.get("error"):
            raise ValueError(f"kraken {data['error']}")
        res = [v for key, v in data["result"].items() if key != "last"][0]
        rows = []
        for t, op, hi, lo, cl, vwap, vol, cnt in res:
            d = canon.ms_date(int(t) * 1000)
            if d <= end_date:          # the last Kraken row is the running day; end_date is always closed
                rows.append([d, _f(op), _f(hi), _f(lo), _f(cl), _f(vol), _f(vol) * _f(vwap)])
        return sorted(rows)

    def intraday(self, coin, pair, start_ms, end_ms, interval="5m"):
        """Bars with open time in [start_ms, end_ms) that are fully closed before end_ms."""
        step = INTERVAL_MS[interval]
        out, s = [], start_ms
        errors = []
        try:
            while s < end_ms:
                data, _ = self._binance(
                    f"/api/v3/klines?symbol={pair}&interval={interval}&startTime={s}&endTime={end_ms - 1}&limit=1000",
                    f"binance_{interval}_{coin}", "intraday", {"coin": coin, "pair": pair}, compress=True)
                if not data:
                    break
                out += [[k[0], _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4])] for k in data if k[0] + step <= end_ms]
                s = data[-1][0] + step
                if len(data) < 1000:
                    break
            return out, "binance"
        except FetchError as e:
            errors.append(str(e))
        # fallback OKX (history-candles, newest first, 100 per page)
        try:
            out, after = [], end_ms
            while after > start_ms:
                data = self.rec.get(f"https://www.okx.com/api/v5/market/history-candles?instId={coin}-USDT&bar={interval}&after={after}&limit=100",
                                    f"okx_{interval}_{coin}", "intraday", {"coin": coin}, compress=True)
                rows = data.get("data") or []
                if not rows:
                    break
                for k in rows:
                    t = int(k[0])
                    if start_ms <= t and t + step <= end_ms:
                        out.append([t, _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4])])
                after = int(rows[-1][0])
            return sorted(out), "okx"
        except (FetchError, ValueError, KeyError) as e:
            errors.append(str(e))
        raise FetchError(f"{coin}: no intraday data ({'; '.join(errors)})")

    def ticker_prices(self, pairs):
        """Fill snapshot: best bid/ask of all pairs in one call (bookTicker) + Binance server time.
        Returns ({pair: (bid, ask)}, server_ms, source)."""
        q = urllib.parse.quote(json.dumps(sorted(pairs), separators=(",", ":")))
        data, host = self._binance(f"/api/v3/ticker/bookTicker?symbols={q}", "binance_snapshot", "snapshot")
        tm, _ = self._binance("/api/v3/time", "binance_time", "server_time")
        return {t["symbol"]: (_f(t["bidPrice"]), _f(t["askPrice"])) for t in data}, int(tm["serverTime"]), "binance"

    def ticker_price_fallback(self, coin):
        data = self.rec.get(f"https://www.okx.com/api/v5/market/ticker?instId={coin}-USDT", f"okx_ticker_{coin}", "snapshot", {"coin": coin})
        d = data["data"][0]
        return (_f(d["bidPx"]), _f(d["askPx"])), "okx"

    # ------------------------------------------------------------------ cross-checks
    def second_close(self, coin, date):
        """Close of `date` from the first non-Binance exchange that lists the coin (OKX, Coinbase, Kraken)."""
        errors = []
        for src in ("okx", "coinbase", "kraken"):
            try:
                rows = [r for r in getattr(self, f"_daily_{src}")(coin, None, date, date) if r[0] == date]
                if rows:
                    return rows[0][4], src
            except (FetchError, KeyError, ValueError, TypeError, IndexError) as e:
                errors.append(f"{src}: {e}")
        raise FetchError(f"{coin} {date}: žádný druhý zdroj ({'; '.join(errors)})")

    def okx_close(self, coin, date):
        after = canon.date_ms(date) + DAY_MS
        data = self.rec.get(f"https://www.okx.com/api/v5/market/candles?instId={coin}-USDT&bar=1Dutc&after={after}&limit=1",
                            f"okx_check_{coin}", "crosscheck_close", {"coin": coin, "date": date})
        rows = [k for k in data.get("data", []) if canon.ms_date(int(k[0])) == date and k[8] == "1"]
        if not rows:
            raise FetchError(f"okx: no closed candle for {coin} {date}")
        return _f(rows[0][4]), "okx"

    def reference_price(self, coin, cg_id):
        """Independent aggregate spot price (CoinGecko with demo key, else CoinPaprika)."""
        if self.cg_key and cg_id:
            try:
                data = self.rec.get(f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd",
                                    f"coingecko_price_{coin}", "crosscheck_price", {"coin": coin},
                                    headers={"x-cg-demo-api-key": self.cg_key})
                return _f(data[cg_id]["usd"]), "coingecko"
            except (FetchError, KeyError, ValueError):
                pass
        data = self.rec.get(f"https://api.coinpaprika.com/v1/tickers?quotes=USD", "coinpaprika_prices", "crosscheck_price", {"coin": coin},
                            compress=True) if not hasattr(self, "_pap") else self._pap
        self._pap = data
        cands = sorted((r["rank"] or 10 ** 9, r["quotes"]["USD"]["price"]) for r in data if r["symbol"] == coin)
        if not cands:
            raise FetchError(f"coinpaprika: {coin} not found")
        return _f(cands[0][1]), "coinpaprika"

    # ------------------------------------------------------------------ hourly data
    def hourly_candles(self, coin, pair, start_ms, end_ms):
        """Closed 1h candles with open time in [start_ms, end_ms): [open_ms, o, h, l, c, quote_vol, taker_buy_quote_vol]."""
        out, s = [], start_ms
        try:
            while s < end_ms:
                data, _ = self._binance(f"/api/v3/klines?symbol={pair}&interval=1h&startTime={s}&endTime={end_ms - 1}&limit=1000",
                                        f"binance_1h_{coin}", "hourly", {"coin": coin, "pair": pair}, compress=True)
                if not data:
                    break
                out += [[k[0], _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[7]), _f(k[10])] for k in data if k[0] + 3_600_000 <= end_ms]
                s = data[-1][0] + 3_600_000
                if len(data) < 1000:
                    break
            return out, "binance"
        except FetchError as e:
            err = str(e)
        # fallback OKX: no taker-buy split (None)
        try:
            out, after = [], end_ms
            while after > start_ms:
                data = self.rec.get(f"https://www.okx.com/api/v5/market/history-candles?instId={coin}-USDT&bar=1H&after={after}&limit=100",
                                    f"okx_1h_{coin}", "hourly", {"coin": coin}, compress=True)
                rows = data.get("data") or []
                if not rows:
                    break
                for k in rows:
                    t = int(k[0])
                    if start_ms <= t and t + 3_600_000 <= end_ms and k[8] == "1":
                        out.append([t, _f(k[1]), _f(k[2]), _f(k[3]), _f(k[4]), _f(k[7]), None])
                after = int(rows[-1][0])
            return sorted(out), "okx"
        except (FetchError, ValueError, KeyError) as e:
            raise FetchError(f"{coin}: no hourly data (binance: {err}; okx: {e})")

    def second_hourly_close(self, coin, open_ms):
        """Close of the 1h candle opening at open_ms from OKX, Coinbase or Kraken (first that lists the coin)."""
        errs = []
        try:
            return self.okx_hourly_close(coin, open_ms)
        except (FetchError, KeyError, ValueError) as e:
            errs.append(f"okx: {e}")
        try:
            s = canon.ms_iso(open_ms)
            e = canon.ms_iso(open_ms + 3_600_000)
            data = self.rec.get(f"https://api.exchange.coinbase.com/products/{coin}-USD/candles?granularity=3600&start={s}&end={e}",
                                f"coinbase_1h_check_{coin}", "crosscheck_close", {"coin": coin})
            rows = [r for r in data if r[0] * 1000 == open_ms]
            if rows:
                return _f(rows[0][4]), "coinbase"
            errs.append("coinbase: no candle")
        except (FetchError, KeyError, ValueError, TypeError) as ex:
            errs.append(f"coinbase: {ex}")
        try:
            k = {"BTC": "XBT", "DOGE": "XDG"}.get(coin, coin)
            data = self.rec.get(f"https://api.kraken.com/0/public/OHLC?pair={k}USD&interval=60&since={open_ms // 1000 - 1}",
                                f"kraken_1h_check_{coin}", "crosscheck_close", {"coin": coin}, compress=True)
            res = [v for key, v in data["result"].items() if key != "last"][0]
            rows = [r for r in res if int(r[0]) * 1000 == open_ms]
            if rows:
                return _f(rows[0][4]), "kraken"
            errs.append("kraken: no candle")
        except (FetchError, KeyError, ValueError, TypeError, IndexError) as ex:
            errs.append(f"kraken: {ex}")
        raise FetchError(f"{coin} {open_ms}: žádný druhý zdroj ({'; '.join(errs)})")

    def okx_hourly_close(self, coin, open_ms):
        data = self.rec.get(f"https://www.okx.com/api/v5/market/history-candles?instId={coin}-USDT&bar=1H&after={open_ms + 3_600_000}&limit=1",
                            f"okx_1h_check_{coin}", "crosscheck_close", {"coin": coin}, compress=False)
        rows = [k for k in data.get("data", []) if int(k[0]) == open_ms and k[8] == "1"]
        if not rows:
            raise FetchError(f"okx: no 1h candle {coin} {open_ms}")
        return _f(rows[0][4]), "okx"

    def book(self, coin, pair, levels=100):
        """Order book summary of the top `levels` levels: spread, quote depth per side, imbalance."""
        data, _ = self._binance(f"/api/v3/depth?symbol={pair}&limit={levels}", f"binance_depth_{coin}", "depth", {"coin": coin}, compress=True)
        bids = [(_f(p), _f(q)) for p, q in data["bids"]]
        asks = [(_f(p), _f(q)) for p, q in data["asks"]]
        return book_summary(bids, asks)

    def derivatives(self, coin):
        """Perpetual futures sentiment (Binance USDT-M): funding rate, OI change 1h, long/short account ratio. None if no perp."""
        sym = coin + "USDT"
        try:
            pi, _ = self._fapi(f"/fapi/v1/premiumIndex?symbol={sym}", f"fapi_premium_{coin}", coin)
            oi, _ = self._fapi(f"/futures/data/openInterestHist?symbol={sym}&period=1h&limit=2", f"fapi_oi_{coin}", coin)
            ls, _ = self._fapi(f"/futures/data/globalLongShortAccountRatio?symbol={sym}&period=1h&limit=1", f"fapi_ls_{coin}", coin)
        except FetchError:
            return None
        oi_chg = (_f(oi[-1]["sumOpenInterestValue"]) / _f(oi[-2]["sumOpenInterestValue"]) - 1) if len(oi) >= 2 and _f(oi[-2]["sumOpenInterestValue"]) > 0 else None
        return {"funding": _f(pi["lastFundingRate"]), "mark": _f(pi["markPrice"]), "oi_usd": _f(oi[-1]["sumOpenInterestValue"]) if oi else None,
                "oi_chg_1h": oi_chg, "ls_ratio": _f(ls[0]["longShortRatio"]) if ls else None}

    def _fapi(self, path, name, coin):
        return self.rec.get("https://fapi.binance.com" + path, name, "derivatives", {"coin": coin}, retries=1), "binance-futures"

    def okx_ticker_mid(self, coin):
        (b, a), src = self.ticker_price_fallback(coin)
        return (b + a) / 2, src

    # ------------------------------------------------------------------ market caps (universe)
    def market_caps(self, n):
        """[{id, symbol, name, market_cap, rank}] top n by market cap. CoinGecko (demo key), fallback CoinPaprika."""
        if self.cg_key:
            try:
                data = self.rec.get(f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={n}&page=1",
                                    "coingecko_markets", "market_caps", headers={"x-cg-demo-api-key": self.cg_key})
                return [{"id": c["id"], "symbol": c["symbol"].upper(), "name": c["name"], "market_cap": _f(c["market_cap"] or 0),
                         "rank": c["market_cap_rank"]} for c in data], "coingecko"
            except (FetchError, KeyError, ValueError, TypeError):
                pass
        data = self.rec.get(f"https://api.coinpaprika.com/v1/tickers?quotes=USD", "coinpaprika_markets", "market_caps", compress=True)
        rows = sorted((r for r in data if r.get("rank")), key=lambda r: r["rank"])[:n]
        return [{"id": r["id"], "symbol": r["symbol"].upper(), "name": r["name"], "market_cap": _f(r["quotes"]["USD"]["market_cap"]),
                 "rank": r["rank"]} for r in rows], "coinpaprika"
