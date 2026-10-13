"""Scout briefs: steer each `propose` run toward the least explored horizon and asset group.

Left alone, the Scout converges on the cheapest instruments and one horizon (four of its first five cards were
SPY/IEF or monthly switches). The brief is chosen deterministically from the registry: the (horizon, asset
group) cell with the fewest hypotheses, ties broken in a fixed order, and a multi-timeframe requirement while
fewer than half of the cards combine timeframes. It is a direction for diversity, not a quality waiver.
"""

from lab.framework.blackboard import Lab

HORIZONS = (("scalp", "minutes to a day (5m / 15m candles)", 0, 1), ("short", "1-5 trading days", 1, 5), ("swing", "1-4 weeks", 5, 25),
            ("medium", "1-6 months", 25, 130), ("long", "6-24 months", 130, 10**6))
GROUPS = {"crypto": "crypto (Binance spot or perpetuals, optionally the hourly-derived daily features)",
          "us_equity": "US single stocks, cross-sectional (sharadar_sep, liq_n or sp500)",
          "us_etf": "US ETFs across asset classes (sharadar_sfp)",
          "cross_asset": "cross-asset (signals from one market, trades in another; FRED macro series allowed)"}


def horizon(card: dict) -> str:
    days = float((card.get("holding_period") or {}).get("typical_days") or 0)
    return next(h for h, _, lo, hi in HORIZONS if lo <= days < hi)


def group(card: dict) -> str:
    classes = set(card.get("asset_classes") or [])
    if len(classes) > 1 or "macro" in classes:
        return "cross_asset"
    if classes & {"crypto_spot", "crypto_perp"}:
        return "crypto"
    return "us_equity" if "us_equity" in classes else "us_etf"


def multi_timeframe(card: dict) -> bool:
    return len(set((card.get("signal") or {}).get("timeframes") or [])) >= 2


def choose(lab: Lab) -> dict:
    import json
    cards = [json.loads(h["card_json"]) for h in lab.hypotheses()]
    counts = {(h, g): 0 for h, *_ in HORIZONS for g in GROUPS if h != "scalp" or g == "crypto"}   # scalping: crypto only
    for c in cards:
        counts[(horizon(c), group(c))] += 1
    # the scalping cell counts double (owner, 2026-10-13): it is proposed about twice as often as any other cell
    (h, g), _ = min(counts.items(), key=lambda kv: kv[1] / (SCALP_WEIGHT if kv[0][0] == "scalp" else 1.0))   # dict order breaks ties
    mtf = sum(map(multi_timeframe, cards)) < len(cards) / 2
    return {"horizon": h, "group": g, "multi_timeframe": mtf}


SCALP_WEIGHT = 2.0
SCALP_TEXT = (
    "propose. Brief for this run (diversity of the lab's portfolio of ideas): SCALPING. A signal on 5m or 15m candles "
    "of Binance USD-M perpetuals (dataset binance_perp_15m; use '5m' only if the card truly needs it, 5m data may not "
    "exist yet), holding a few candles to a few hours. Fields: o h l c, quote volume, number of trades, taker-buy "
    "volume, funding. Universe: crypto_top_n (n 5-30) or a short instrument list; long and short allowed. List the "
    "timeframe ('15m') in signal.timeframes, put typical_days below 1 in holding_period, and do NOT declare a "
    "mechanism_test (the intraday judge has none). Think about who pays you: forced liquidations, funding "
    "settlement, session opens, order-flow imbalance (taker-buy share), range compression and expansion. A scalp "
    "pays two taker fees plus slippage per trade (at least 13 bps round trip even for BTC), so rare, high-"
    "conviction entries survive and frequent small ones do not.")


def text(brief: dict) -> str:
    if brief["horizon"] == "scalp":
        return SCALP_TEXT + (" The brief is a direction, not a waiver: if no idea with a real mechanism fits it, take "
                             "the closest fit and say why in notes.")
    span = next(s for h, s, *_ in HORIZONS if h == brief["horizon"])
    out = (f"propose. Brief for this run (diversity of the lab's portfolio of ideas): holding horizon "
           f"{brief['horizon']} ({span}); asset group: {GROUPS[brief['group']]}.")
    if brief["multi_timeframe"]:
        out += (" Combine timeframes: a signal on one timeframe confirmed or filtered by another (e.g. a slow "
                "weekly/monthly regime confirming a faster daily entry, or hourly-derived features gating a "
                "daily signal); list them in signal.timeframes.")
    out += " Declare a mechanism_test (event + primary horizon) so the judge tests the mechanism before the strategy."
    return out + (" The brief is a direction, not a waiver: if no idea with a real mechanism fits it, take the "
                  "closest fit and say why in notes.")
