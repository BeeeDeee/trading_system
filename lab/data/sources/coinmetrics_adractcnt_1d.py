import json

BASE = "https://community-api.coinmetrics.io/v4"

# Ambiguous or reused tickers dropped, same list as the coinmetrics_supply_1d ingest.
DROP = {
    "luna", "luna2", "lunc", "ust", "ustc", "rep", "repv2", "bcc", "ant", "mir", "bnb",
    "avaxc", "avaxp", "avaxx",
}


def _assets_with_metric(get):
    out = []
    try:
        out = _parse_v2(get(BASE + "/catalog-v2/asset-metrics", params={"metrics": "AdrActCnt"}))
    except Exception:
        out = []
    if not out:
        out = _parse_v1(get(BASE + "/catalog/asset-metrics", params={"metrics": "AdrActCnt"}))
    return out


def _parse_v1(body):
    # legacy shape: data = [{"metric": "AdrActCnt", "frequencies": [{"frequency": "1d", "assets": [...]}]}]
    out = []
    for item in json.loads(body).get("data", []):
        if item.get("metric") != "AdrActCnt":
            continue
        for f in item.get("frequencies", []):
            if f.get("frequency") == "1d":
                for a in f.get("assets", []):
                    if a.lower() not in DROP:
                        out.append(a.lower())
    return sorted(set(out))


def _parse_v2(body):
    out = []
    for item in json.loads(body).get("data", []):
        asset = item.get("asset") or item.get("id")
        if not asset:
            continue
        ok = False
        for m in item.get("metrics", []):
            name = m.get("metric") if isinstance(m, dict) else m
            if name != "AdrActCnt":
                continue
            freqs = m.get("frequencies") if isinstance(m, dict) else None
            if not freqs:
                ok = True
            else:
                for f in freqs:
                    if f.get("frequency") == "1d":
                        ok = True
        if ok and asset.lower() not in DROP:
            out.append(asset.lower())
    return sorted(set(out))


def fetch(get):
    assets = _assets_with_metric(get)
    rows = {}
    for asset in assets:
        key = asset.upper() + "USDT.P"
        body = get(
            BASE + "/timeseries/asset-metrics",
            params={
                "assets": asset,
                "metrics": "AdrActCnt",
                "frequency": "1d",
                "page_size": "10000",
                "paging_from": "start",
            },
        )
        pages = 0
        while True:
            doc = json.loads(body)
            for item in doc.get("data", []):
                v = item.get("AdrActCnt")
                t = item.get("time")
                if v is None or t is None:
                    continue
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if val != val or val < 0:
                    continue
                rows[(t[:10], key)] = val
            nxt = doc.get("next_page_url")
            pages += 1
            if not nxt or pages > 5:
                break
            body = get(nxt)
    return [(d, k, v) for (d, k), v in sorted(rows.items())]
