import json
import datetime


def fetch(get):
    """DefiLlama stablecoins API: total circulating supply of USD-pegged stablecoins, all chains, daily."""
    body = get("https://stablecoins.llama.fi/stablecoincharts/all")
    data = json.loads(body)
    by_date = {}
    for item in data:
        ts = int(item["date"])
        day = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
        tot = item.get("totalCirculatingUSD") or {}
        v = tot.get("peggedUSD")
        if v is None:
            continue
        v = float(v)
        if v <= 0:
            continue
        by_date[day] = v
    rows = []
    for day in sorted(by_date):
        rows.append((day, "total", by_date[day]))
    return rows
