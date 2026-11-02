import json
import re

BASE = "https://community-api.coinmetrics.io/v4"

# Coin Metrics ids whose ticker was reused by a different asset on Binance (or whose
# supply identity changed through a redenomination/new chain): dropped, never guessed.
DROP = {"luna", "luna2", "lunc", "ust", "ustc", "rep", "repv2", "bcc", "ant", "mir",
        "bnb",  # CM 'bnb' is the ERC-20 token up to the 2019 mainnet swap, not today's BNB
        "avaxc", "avaxp", "avaxx"}  # AVAX chain-specific assets, not the Binance AVAX perp

# Binance perp bases that may be served although the community catalog does not list them;
# requested explicitly (unserved ones are skipped by the API flags, not guessed).
EXTRA = ["sol", "trx", "avax", "atom", "near", "apt", "arb", "op", "matic", "pol", "ftm", "vet", "hbar",
         "fil", "ape", "sand", "axs", "gala", "grt", "enj", "chz", "lrc", "ens", "imx", "inj", "sui",
         "sei", "ton", "shib", "pepe", "floki", "bonk", "xec", "eos", "iota", "ont", "qtum", "zil",
         "rune", "kava", "ksm", "egld", "theta", "ar", "stx", "mina", "cake", "rose", "ankr", "bnt",
         "storj", "iost", "ctsi", "ach", "dydx", "gmx", "snx", "ocean", "skl", "celr", "audio", "ray",
         "waves", "xrp", "ada", "dot", "doge", "ltc", "link", "uni", "aave"]

# Binance perps quoted per 1000 units: supply growth is unit-free, so the same Coin Metrics
# asset is also emitted under the 1000-prefixed Binance id.
MULT1000 = {"shib": "1000SHIB", "pepe": "1000PEPE", "floki": "1000FLOKI", "bonk": "1000BONK", "xec": "1000XEC"}


def _community_assets(get):
    """Asset ids that the community tier serves for SplyCur at 1d."""
    body = get(BASE + "/catalog/asset-metrics", params={"metrics": "SplyCur"})
    doc = json.loads(body)
    assets = []
    for item in doc.get("data", []):
        if item.get("metric") != "SplyCur":
            continue
        for fr in item.get("frequencies", []):
            if fr.get("frequency") == "1d":
                assets.extend(fr.get("assets", []))
    return sorted(set(assets))


def fetch(get):
    """Coin Metrics Community API, SplyCur (current supply, native units), frequency 1d, full history.
    Key = Binance perp id <TICKER>USDT.P of the Coin Metrics asset id (upper-cased). End-of-UTC-day values."""
    listed = _community_assets(get)
    assets = sorted(set(listed) | set(EXTRA))
    assets = [a for a in assets if re.fullmatch(r"[a-z0-9]+", a) and a not in DROP]
    rows = []
    seen = set()
    chunk = 8
    for i in range(0, len(assets), chunk):
        batch = assets[i:i + chunk]
        url = BASE + "/timeseries/asset-metrics"
        params = {
            "assets": ",".join(batch),
            "metrics": "SplyCur",
            "frequency": "1d",
            "start_time": "2009-01-01",
            "page_size": "10000",
            "paging_from": "start",
            "ignore_forbidden_errors": "true",
            "ignore_unsupported_errors": "true",
        }
        guard = 0
        while url and guard < 40:
            guard += 1
            body = get(url, params=params)
            params = None
            doc = json.loads(body)
            for item in doc.get("data", []):
                v = item.get("SplyCur")
                if v is None or v == "":
                    continue
                asset = item["asset"]
                d = item["time"][:10]
                val = float(v)
                if not (val == val) or val in (float("inf"), float("-inf")):
                    continue
                ids = [asset.upper() + "USDT.P"]
                if asset in MULT1000:
                    ids.append(MULT1000[asset] + "USDT.P")
                for key in ids:
                    if (d, key) in seen:
                        continue
                    seen.add((d, key))
                    rows.append((d, key, val))
            url = doc.get("next_page_url")
    return rows
