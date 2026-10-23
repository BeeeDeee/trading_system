from decimal import Decimal as D

import pytest

from arena.binance.filters import SymbolFilters, quantize_to_step, round_callback_rate


def _sym(tick="0.10", step="0.001", min_notional="50", symbol="BTCUSDT", mmax="120"):
    return SymbolFilters.from_exchange_info({
        "symbol": symbol, "status": "TRADING", "underlyingType": "COIN", "triggerProtect": "0.0300",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": tick},
            {"filterType": "LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": "1000"},
            {"filterType": "MARKET_LOT_SIZE", "stepSize": step, "minQty": step, "maxQty": mmax},
            {"filterType": "MIN_NOTIONAL", "notional": min_notional},
        ],
    })


def test_quantize_non_power_of_ten_step():
    assert quantize_to_step(D("1.27"), D("0.05"), "ROUND_FLOOR") == D("1.25")
    assert quantize_to_step(D("1.27"), D("0.05"), "ROUND_CEILING") == D("1.30")


def test_round_price_keeps_tick_exponent():
    f = _sym()
    assert str(f.round_price(D("65123.456"), "down")) == "65123.40"
    assert str(f.round_price("65123.401", "up")) == "65123.50"


def test_stop_price_is_never_further_from_market():
    f = _sym()
    # LONG: stop pod cenou → nahoru (blíž)
    px, changed = f.stop_price(D("64000.03"), "LONG")
    assert px == D("64000.10") and changed
    # SHORT: stop nad cenou → dolů (blíž)
    px, changed = f.stop_price(D("66000.07"), "SHORT")
    assert px == D("66000.00") and changed
    px, changed = f.stop_price(D("66000.10"), "SHORT")
    assert not changed


def test_take_profit_rounds_away_from_market():
    f = _sym()
    assert f.take_profit_price(D("66000.01"), "LONG") == D("66000.10")
    assert f.take_profit_price(D("64000.09"), "SHORT") == D("64000.00")


def test_qty_for_notional_and_min_notional():
    f = _sym()
    assert f.qty_for_notional(200, D("65000")) == D("0.003")
    with pytest.raises(ValueError, match="minQty"):
        f.qty_for_notional(60, D("65000"))   # 0.000923 → 0.000
    with pytest.raises(ValueError, match="MIN_NOTIONAL"):
        f.qty_for_notional(60, D("40000"))   # 0.001 × 40000 = 40 < 50
    doge = _sym(tick="0.000010", step="1", min_notional="5", symbol="DOGEUSDT", mmax="30000000")
    assert doge.qty_for_notional(150, D("0.08")) == D("1875")


def test_callback_rate_percent_rounding_and_clip():
    assert round_callback_rate(0.46) == D("0.5")
    assert round_callback_rate(0.04) == D("0.1")
    assert round_callback_rate(14.0) == D("10")
    assert round_callback_rate(1.25) == D("1.3")
