import json

API = "https://api.upbit.com/v1"
START = "2019-09-01"      # rows before this date are not kept (request budget, see known_biases)
BUDGET = 480              # the framework allows 500 requests; keep a margin

# Binance USD-M perp bases that are liquid enough to matter, fetched first (budget order).
MAJORS = ["BTC", "ETH", "XRP", "ADA", "DOGE", "LTC", "BCH", "TRX", "XLM", "ETC", "LINK", "DOT",
          "ATOM", "SOL", "AVAX", "MATIC", "POL", "SAND", "MANA", "AXS", "NEAR", "ALGO", "XTZ", "AAVE", "UNI",
          "FIL", "ICP", "EOS", "TON", "FTM"]
# cheap in requests (listed on Upbit KRW after 2021, few pages)
NEWER = ["APT", "ARB", "OP", "SUI", "SEI", "INJ", "TIA", "WLD", "EGLD", "GRT", "IMX", "MINA", "STX", "COMP",
         "CRV", "ENS", "1INCH", "SNX", "MKR", "GALA", "FLOW", "PEPE", "SHIB", "BONK"]
# older markets (many pages), as far as the budget goes
OLDER = ["VET", "THETA", "NEO", "QTUM", "ONT", "IOTA", "BAT", "ZRX", "DASH", "ZEC", "ZIL", "WAVES", "HBAR",
         "KAVA", "CHZ", "ENJ", "ANKR", "SXP", "YFI", "SUSHI", "LRC"]

# Tickers whose identity changed or was reused on Upbit/Binance (Terra): never emitted.
SKIP = {"LUNA", "LUNC", "LUNA2", "UST", "USTC"}

# Delisted KRW markets that the market list no longer shows: probed, taken only if the API still answers.
DELISTED_PROBES = ["BCHA", "EMC2", "GAME", "MEDX", "QKC", "VIB"]


class _Budget:
    def __init__(self, get):
        self.get = get
        self.used = 0

    def left(self):
        return BUDGET - self.used

    def call(self, url, params):
        self.used += 1
        return self.get(url, params=params)


def _market_history(b, market):
    """Newest-first pagination with `to` (exclusive). Returns list of (date, close, value) or None
    if the market is unknown or the budget ran out before the history was complete."""
    out = []
    to = None
    pages = 0
    while pages < 25:
        if b.left() <= 0:
            return None
        params = {"market": market, "count": "200"}
        if to:
            params["to"] = to
        try:
            body = b.call(API + "/candles/days", params)
            doc = json.loads(body)
        except Exception:
            return None
        pages += 1
        if not isinstance(doc, list) or not doc:
            break
        oldest = None
        for c in doc:
            dt = c.get("candle_date_time_utc")
            tp = c.get("trade_price")
            vv = c.get("candle_acc_trade_price")
            if dt is None or tp is None or vv is None:
                continue
            d = dt[:10]
            if oldest is None or dt < oldest:
                oldest = dt
            out.append((d, float(tp), float(vv)))
        if len(doc) < 200 or oldest is None or oldest[:10] <= START:
            break
        to = oldest[:19] + "Z"
    return out


def fetch(get):
    """Upbit KRW-market daily candles (UTC day, 00:00-24:00). Key BASE.close = trade_price (KRW),
    BASE.value_krw = candle_acc_trade_price (KRW). The newest day seen anywhere is dropped (the
    current day's candle is still forming)."""
    b = _Budget(get)
    body = b.call(API + "/market/all", {"isDetails": "false"})
    listed = []
    for m in json.loads(body):
        code = m.get("market", "")
        if code.startswith("KRW-"):
            listed.append(code[4:])
    listed_set = set(listed)
    probes = [x for x in DELISTED_PROBES if x not in listed_set]
    order = [x for x in MAJORS if x in listed_set] + probes
    order += [x for x in NEWER if x in listed_set] + [x for x in OLDER if x in listed_set]
    order += [x for x in listed if x not in set(order)]
    data = {}
    for base in order:
        if base in SKIP:
            continue
        if b.left() < 3:
            break
        h = _market_history(b, "KRW-" + base)
        if h:
            data[base] = h
    gmax = ""
    for base in data:
        for d, c, v in data[base]:
            if d > gmax:
                gmax = d
    rows = []
    for base in sorted(data):
        seen = set()
        for d, c, v in data[base]:
            if d < START or d >= gmax or d in seen:
                continue
            seen.add(d)
            rows.append((d, base + ".close", c))
            rows.append((d, base + ".value_krw", v))
    return rows
