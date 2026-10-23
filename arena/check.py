"""`arena check` (P0): konektivita, čas, oprávnění klíče, futures, one-way, margin/páka, poplatky, filtry."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import websockets

from arena.binance import errors
from arena.binance.filters import parse_exchange_info
from arena.binance.rest import RestClient
from arena.binance.signing import Signer
from arena.config import Config

SPOT_REST = "https://api.binance.com"   # jen pro GET /sapi/v1/account/apiRestrictions (mainnet)


@dataclass
class CheckReport:
    rows: list[tuple[str, str, str]] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))

    @property
    def failed(self) -> bool:
        return any(s == "FAIL" for s, _, _ in self.rows)

    @property
    def incomplete(self) -> bool:
        return any(s == "SKIP" for s, _, _ in self.rows)

    def render(self) -> str:
        return "\n".join(f"[{s:4}] {n}{(': ' + d) if d else ''}" for s, n, d in self.rows)


async def _ws_probe(url: str, want: set[str], timeout_s: float = 8.0) -> dict[str, int]:
    """Poslouchá, dokud nepřijde aspoň 1 zpráva z každého streamu `want`, nebo do timeoutu."""
    got: dict[str, int] = {}
    async with websockets.connect(url, open_timeout=10) as ws:
        loop = asyncio.get_running_loop()
        end = loop.time() + timeout_s
        while loop.time() < end and not want <= set(got):
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=max(0.01, end - loop.time())))
            except asyncio.TimeoutError:
                break
            stream = msg.get("stream", "?")
            got[stream] = got.get(stream, 0) + 1
    return got


async def run_check(cfg: Config, signer: Signer | None) -> CheckReport:
    rep = CheckReport()
    rep.add("OK", "konfigurace", f"env={cfg.env} config_hash={cfg.config_hash} variant={len(cfg.variants)}")
    ep = cfg.endpoints
    async with RestClient(ep.rest, signer=signer) as rest:
        # čas
        try:
            off = await rest.sync_time()
            st = "OK" if abs(off) < 1000 else "WARN"
            rep.add(st, "čas", f"offset {off} ms, rtt {rest.clock.last_rtt_ms} ms ({ep.rest})")
        except Exception as e:  # noqa: BLE001
            rep.add("FAIL", "čas / konektivita REST", repr(e))
            return rep

        # filtry
        info = await rest.exchange_info()
        filters = parse_exchange_info(info, cfg.universe)
        prices = {p["symbol"]: Decimal(p["markPrice"]) for p in await rest.premium_index()
                  if p["symbol"] in cfg.universe}
        for sym in cfg.universe:
            f = filters.get(sym)
            if f is None:
                rep.add("FAIL", f"filtry {sym}", "symbol v exchangeInfo chybí")
                continue
            st = "OK" if f.status == "TRADING" and f.underlying_type == "COIN" else "FAIL"
            detail = (f"tick {f.tick_size} step {f.step_size} mstep {f.market_step_size} "
                      f"minNotional {f.min_notional} status {f.status} underlying {f.underlying_type}")
            slot = cfg.live.slots.get(sym)
            if slot is not None and sym in prices:
                try:
                    qty = f.qty_for_notional(Decimal(str(slot.notional_usdt)), prices[sym])
                    detail += f" | live {slot.variant} {slot.notional_usdt} USDT → qty {qty}"
                except ValueError as e:
                    st = "FAIL"
                    detail += f" | live notional neprojde: {e}"
            rep.add(st, f"filtry {sym}", detail)

        fi = {x["symbol"]: x for x in await rest.funding_info()}
        rep.add("OK", "funding interval",
                ", ".join(f"{s} {fi.get(s, {}).get('fundingIntervalHours', 8)} h" for s in cfg.universe))

        # WS routy (nástraha 2)
        sym = cfg.universe[0].lower()
        for name, url, want in (
            ("WS /market", f"{ep.ws_market}/stream?streams={sym}@kline_5m/{sym}@aggTrade/{sym}@markPrice@1s",
             {f"{sym}@aggTrade", f"{sym}@markPrice@1s"}),
            ("WS /public", f"{ep.ws_public}/stream?streams={sym}@bookTicker/{sym}@depth@100ms",
             {f"{sym}@bookTicker", f"{sym}@depth@100ms"}),
        ):
            try:
                got = await _ws_probe(url, want)
                missing = want - set(got)
                rep.add("FAIL" if missing else "OK", name, f"{got}" + (f" chybí {sorted(missing)}" if missing else ""))
            except Exception as e:  # noqa: BLE001
                rep.add("FAIL", name, repr(e))

        # podepsaná část
        if signer is None:
            prefix = "BINANCE_TESTNET_" if cfg.env == "testnet" else "BINANCE_"
            rep.add("SKIP", "klíč", f"není nastavený ({prefix}API_KEY v env nebo /etc/arena/env)")
            return rep
        await _check_signed(cfg, rest, signer, rep)
    return rep


async def _check_signed(cfg: Config, rest: RestClient, signer: Signer, rep: CheckReport) -> None:
    try:
        acct = await rest.account()
    except errors.BinanceError as e:
        rep.add("FAIL", "futures účet", f"{e} (futures nepovolené? klíč bez oprávnění? IP whitelist?)")
        return
    usdt = next((a for a in acct.get("assets", []) if a.get("asset") == "USDT"), {})
    rep.add("OK", "futures účet", f"USDT wallet {usdt.get('walletBalance', '?')}, "
                                  f"available {acct.get('availableBalance', '?')}")

    if cfg.env == "mainnet":
        try:
            async with RestClient(SPOT_REST, signer=signer, clock=rest.clock) as spot:
                r: dict[str, Any] = await spot.signed("GET", "/sapi/v1/account/apiRestrictions")
            ok = (not r.get("enableWithdrawals")) and r.get("enableFutures") and r.get("ipRestrict")
            rep.add("OK" if ok else "FAIL", "oprávnění klíče",
                    f"withdrawals={r.get('enableWithdrawals')} futures={r.get('enableFutures')} "
                    f"ipRestrict={r.get('ipRestrict')} (D9: bez výběrů, s IP whitelistem)")
        except errors.BinanceError as e:
            rep.add("WARN", "oprávnění klíče", f"apiRestrictions nejde přečíst: {e}")
    else:
        rep.add("SKIP", "oprávnění klíče", "apiRestrictions jen na mainnetu")

    dual = await rest.position_side_dual()
    rep.add("FAIL" if dual else "OK", "one-way mód", f"dualSidePosition={dual} (nástraha 4)")

    try:
        acfg = await rest.account_config()
        multi = acfg.get("multiAssetsMargin")
        rep.add("OK" if not multi else "WARN", "accountConfig", f"multiAssetsMargin={multi} "
                f"canTrade={acfg.get('canTrade')} feeTier={acfg.get('feeTier')}")
    except errors.BinanceError as e:
        rep.add("WARN", "accountConfig", str(e))

    sc = {x["symbol"]: x for x in await rest.symbol_config()}
    positions = {p["symbol"]: p for p in await rest.position_risk() if Decimal(p.get("positionAmt", "0")) != 0}
    for sym in cfg.universe:
        c = sc.get(sym, {})
        mt, lev = c.get("marginType"), c.get("leverage")
        good = mt == "ISOLATED" and int(lev or 0) == cfg.live.leverage
        detail = f"marginType={mt} leverage={lev}"
        if sym in positions:
            detail += f" | OTEVŘENÁ POZICE {positions[sym].get('positionAmt')}"
        # P0 nic nenastavuje; nastaví to start live (SPEC §9), když není otevřená pozice
        rep.add("OK" if good else "WARN", f"margin/páka {sym}", detail + ("" if good else
                f" (cíl ISOLATED/{cfg.live.leverage}x, nastaví se při startu live)"))
        cr = await rest.commission_rate(sym)
        rep.add("OK", f"poplatky {sym}", f"maker {cr.get('makerCommissionRate')} taker {cr.get('takerCommissionRate')}")
