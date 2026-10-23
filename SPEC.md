# arena5m: experimentální framework pro 5m crypto strategie na Binance

Verze specifikace: 1.1 (2026-10-21). Autor návrhu: Claude Opus 5.5. Implementaci dělá jiný model podle tohoto dokumentu.

Změny 1.0 → 1.1 (na pokyn vlastníka): (1) zrušena omezení disku a paměti (migrace na větší server), surová data se
nově nahrávají (D6, §4.4); (2) rodina B `mr` nahrazena rodinou `obi` (nerovnováha hloubky knihy + objem, §5.5),
podklad je explorativní analýza v `research/obi_prior/`; (3) místo `depth20` se drží lokální plná kniha z diff streamu
(§4.3), protože pásmo ±1 % se do 20 ani do 1000 úrovní u BTC/ETH nevejde.
Pro implementujícího: čti celý dokument. Sekce 0 a 1 jsou závazná rozhodnutí; neměň je bez souhlasu vlastníka. Kde je něco
označeno **OVĚŘIT**, ověř to proti aktuální dokumentaci Binance nebo živému API dřív, než na tom postavíš kód.

---

## 0. Co tenhle framework je a co není

**Účel:** levně a poctivě zjistit, jestli některá z 30 variant 3 strategií má na 5m timeframe kladnou hranu **po reálných
nákladech**, a změřit ty náklady na skutečných fillech na Binance.

**Očekávání (otevřeně):** většina variant bude po nákladech záporná. Čísla změřená 2026-10-21 z Binance USDⓈ-M perpů:

| Symbol | sd 5m výnosu | spread | průměrný 5m obrat |
|---|---|---|---|
| BTCUSDT | 9,4 bps | 0,01 bps | 30,6 M USDT |
| ETHUSDT | 14,4 bps | 0,04 bps | 23,4 M USDT |
| SOLUSDT | 16,7 bps | 0,94 bps | 4,7 M USDT |
| XRPUSDT | 15,7 bps | 0,74 bps | 1,8 M USDT |
| DOGEUSDT | 17,2 bps | 1,22 bps | 0,9 M USDT |

Round trip taker/taker na futures VIP0 stojí 2 × 5 bps = 10 bps plus spread a slippage, takže zhruba 11–14 bps. To je
**víc než jedna směrodatná odchylka 5m svíčky na BTC**. Vlastníkův vlastní výzkum (research 15/16) ukázal, že držení kratší
než zhruba 2 dny prohrává na 15 bps round tripu. Z toho plyne návrhové pravidlo:

> **5m je timeframe pro signál a načasování vstupu, ne pro délku držení.** Cílové pohyby musí být ≥ 50–100 bps, takže
> trailing callback je typicky 0,3–1,5 % a maximální držení 1–8 h. Varianty, které chtějí brát 10–20 bps, se netestují:
> jsou mrtvé už podle aritmetiky.

**Není to:** HFT, market making ani optimalizátor parametrů. Framework neladí parametry za běhu. Parametry jsou zmrazené
v konfiguraci a každá změna znamená novou variantu s novým ID.

---

## 1. Závazná rozhodnutí

### D1. Venue: Binance USDⓈ-M perpetual futures
- Taker 0,05 % a maker 0,02 % (VIP0, skutečné sazby účtu čti z `GET /fapi/v1/commissionRate`). Na spotu je to 0,10 %, takže
  spot je pro 5m 2× dražší. Futures navíc umožňují shorty.
- Režim **one-way** (`dualSidePosition=false`), margin **ISOLATED**, páka 2× na každém symbolu. Velikost pozice určuje
  notional, ne páka. Páka 2× jen drží margin nízko; při stopu ≤ 3 % je likvidace daleko.
- **OVĚŘIT (P0):** účet musí mít futures povolené (EU/MiCA omezení). Když nemá, záložní varianta je spot long-only se
  `STOP_LOSS` + `trailingDelta` v BIPS (filtr `TRAILING_DELTA`, typicky 10–2000). Executor proto stav za rozhraním
  `Venue`, ale implementuj jen futures, dokud vlastník neřekne jinak.
- Ideálně dedikovaný **sub-account** jen pro bota, aby ruční obchody vlastníka nekolidovaly s pozicemi bota.
  **OVĚŘIT**, jestli to úroveň účtu dovoluje. Když ne, platí pravidlo: na symbolech z universa se ručně neobchoduje.

### D2. Všechno běží v paperu, živě jen několik slotů (problém nettingu)
Slot = (varianta, symbol). Máme 3 rodiny × 10 variant × 5 symbolů (rodina C jen 4 symboly) = **140 slotů** + kontroly.
Na jednom účtu v one-way módu se pozice na stejném symbolu **sčítají**: 10 variant na BTCUSDT by na burze byla jedna
pozice a trailing stopy jednotlivých variant by se navzájem rušily. Proto:

- **Každý slot vždy běží v PAPER módu** s realistickým simulátorem fillů (sekce 7), který je kalibrovaný živými obchody.
- **LIVE slotů je max. 1 na symbol na účet** (tedy max. 5). Validace konfigurace jinou konfiguraci odmítne.
- Živý slot má zároveň paper dvojče, které dostává identické signály. Rozdíl live − paper na páru obchodů je
  **chyba nákladového modelu**. To je hlavní výstup live části.
- Primární statistické rozhodování stojí na paper datech všech slotů (velké N). Live data dávají skutečné náklady a
  skutečné P&L, ne statistickou sílu.

### D3. Trailing stop nativně na burze přes Algo Order API
Od 2025-12-09 se podmíněné příkazy na USDⓈ-M futures (STOP_MARKET, TAKE_PROFIT_MARKET, TRAILING_STOP_MARKET…) zadávají
**jen** přes `POST /fapi/v1/algoOrder`. Starý `POST /fapi/v1/order` je odmítne s chybou `-4120 STOP_ORDER_SWITCH_ALGO`.
Hodně knihoven a návodů (python-binance, starší ccxt) je kvůli tomu zastaralých. **Proto žádná trading knihovna,
vlastní tenký klient (sekce 6).**

Parametry trailing příkazu:
```
POST /fapi/v1/algoOrder
  algoType=CONDITIONAL  type=TRAILING_STOP_MARKET  symbol=..  side=SELL|BUY  quantity=..
  callbackRate=0.5        # v procentech, min 0.1, max 10 (OVĚŘIT krok; zaokrouhluj na 0.1)
  activatePrice=..        # volitelné; bez něj se trailuje od aktuální ceny hned
  workingType=CONTRACT_PRICE   # last price; paper simulátor simuluje přesně tohle
  reduceOnly=true  clientAlgoId=<deterministické, ^[\.A-Z\:/a-z0-9_-]{1,36}$>  newOrderRespType=RESULT
```
Další algo endpointy: `DELETE /fapi/v1/algoOrder` (podle algoId nebo clientAlgoId), `GET /fapi/v1/openAlgoOrders`,
`GET /fapi/v1/algoOrder`, `GET /fapi/v1/allAlgoOrders`, `DELETE /fapi/v1/algoOpenOrders`. Události chodí v user data
streamu jako `ALGO_UPDATE`; samotný fill po spuštění je normální `ORDER_TRADE_UPDATE` market příkazu.
**OVĚŘIT** názvy polí v `ALGO_UPDATE`, hlavně vazbu algoId → orderId spuštěného market příkazu.

**Ke každé pozici patří vždy i tvrdý stop** (`STOP_MARKET`, algo, reduceOnly), a to ze dvou důvodů:
1. Trailing s `activatePrice` nechrání pozici, dokud se neaktivuje.
2. Pojistka pro případ, že trailing příkaz selže nebo je zrušen.

### D4. Vstupy, výstupy, časování
- Signál se počítá **jen na uzavřené 5m svíčce** (kline event s `k.x == true`). Nikdy na rozpracované.
- Vstup: `MARKET` (taker) hned po uzavření svíčky. Maker vstupy (post-only GTX) jsou mimo v1, protože férově simulovat
  frontu a adverse selection v paperu nejde.
- Výstupy: trailing (burza), tvrdý stop (burza), volitelný take-profit jako `LIMIT` reduceOnly GTC (maker, burza), časový
  výstup (bot: nejdřív zruší algo příkazy, pak `MARKET` reduceOnly), kill switch (bot).
- Jedna pozice na slot, žádné přikupování. Signály během otevřené pozice nebo cooldownu se zalogují jako `suppressed`.

### D5. Universe: zmrazené
`BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT`. Je to vybrané podle obratu a spreadu 2026-10-21 jen z perpů s
`underlyingType == "COIN"`. Seznam podle 24h obratu je plný TradFi perpů (SOXL, XAU, akcie) a krátkodobých pumpů (ZEC,
NEAR), proto se universe během experimentu **neotáčí**. Skript `arena universe-check` jen vypíše 30denní obraty a
spready pro kontrolu; seznam nemění.

### D6. Úložiště: SQLite (WAL), jeden zapisující proces, plus surová data v souborech
SQLite drží 5m bary, featury knihy na bar, pásma knihy po 30 s, signály, příkazy, filly a obchody. Surové aggTrades se
nahrávají do denních souborů mimo DB (§4.4), aby šlo každou hypotézu později přehrát offline a forenzně rozebrat
slippage libovolného obchodu. Retence se nastavuje v konfiguraci, default bez mazání.

### D7. Dashboard: statická stránka a JSON snapshot
Engine každých 15 s atomicky zapíše (`tmp` + `os.replace`) `snapshot.json` do `/srv/arena/site/data/`. Statická
`index.html` + `app.js` ho polluje. Servíruje to stávající Caddy portal na `/arena/`, stejně jako `/lead/` (vzor:
`portal/install_lead.sh`, `portal/patch_caddyfile_lead.py`). Nepotřebuje to druhý proces ani webový framework.
**Na `/arena/*` dej Caddy `basic_auth`**, protože stránka ukazuje zůstatky a pozice.

### D8. Proces a provoz
Jeden asyncio proces `arena run` pod systémovým uživatelem `arena`, systemd `Restart=always`. Po
restartu se stav obnoví z DB a vždy se nejdřív zrekonciluje s burzou (sekce 9). Alerty jdou přes ntfy (téma jako u
crypto bota, `NTFY_TOPIC` v env souboru).

### D9. Bezpečnost klíčů
API klíč typu **Ed25519** s povoleným jen čtením a futures obchodováním, **bez výběrů**, s IP whitelistem na IP
serveru. Ukládá se v `/etc/arena/env` (mode 600, vlastník arena): `BINANCE_API_KEY`, `BINANCE_PRIVATE_KEY_PATH`.
Testnet klíče jsou oddělené.

---

## 2. Architektura

```
        Binance WS /market                 Binance WS /public            Binance WS /private (listenKey)
  kline_5m, aggTrade, markPrice@1s          bookTicker, depth@100ms       ORDER_TRADE_UPDATE, ALGO_UPDATE, ACCOUNT_UPDATE
              │                                     │                                  │
              ▼                                     ▼                                  ▼
       ┌─────────────┐   uzavřený bar   ┌──────────────────┐                  ┌──────────────────┐
       │  BarStore   │ ───────────────▶ │ FeatureEngine    │                  │ LiveExecutor     │◀── REST /fapi (podepsané)
       │ (+backfill) │                  └────────┬─────────┘                  │ (stavový automat)│
       └─────────────┘                           ▼                            └────────┬─────────┘
              ▲                        ┌──────────────────┐   Intent                   │
              │ REST klines            │ Strategy sloty   │ ─────────┬────────────────▶│ (jen LIVE sloty)
                                       │ 140 + kontroly   │          │                 │
                                       └──────────────────┘          ▼                 ▼
       tick stream (aggTrade, bookTicker) ───────────────▶ ┌──────────────────┐   ┌──────────┐
                                                           │ PaperExecutor    │   │Reconciler│ každých 60 s
                                                           │ + TrailingSim    │   └──────────┘
                                                           └────────┬─────────┘
                                                                    ▼
                         ┌───────────┐     ┌──────────────┐    ┌─────────┐      ┌────────────────────┐
                         │ RiskGuard │     │ IncomePoller │───▶│ SQLite  │─────▶│ Stats + Snapshot   │──▶ /srv/arena/site/data/*.json
                         └───────────┘     │ /fapi/v1/    │    │  (WAL)  │      └────────────────────┘
                                           │ income       │    └─────────┘
                                           └──────────────┘
```

---

## 3. Struktura repozitáře

```
arena5m/
  SPEC.md  CLAUDE.md  pyproject.toml  README.md
  config/
    arena.yaml                  # globální konfigurace (sekce 11)
    strategies/brk.yaml obi.yaml ll.yaml ctl.yaml
  arena/
    __init__.py  cli.py  config.py  clock.py  util.py
    binance/
      signing.py                # Ed25519 (cryptography) + HMAC fallback
      rest.py                   # tenký async klient (httpx), retry, rate-limit hlavičky, mapování chyb
      ws.py                     # reconnect, ping/pong, 24h rotace, stream router
      filters.py                # exchangeInfo → tick/step/minNotional, zaokrouhlování (Decimal)
      errors.py
    data/
      bars.py                   # BarStore, backfill, detekce mezer
      book.py                   # lokální plná kniha (snapshot + diff), pásma ±x %, simulace market fillu přes knihu
      ticks.py                  # dispatcher aggTrade/bookTicker → TrailingSim
      recorder.py               # surové aggTrades do denních souborů, pásma knihy po 30 s (§4.4)
      book_validate.py          # porovnání lokální knihy s bookDepth z data.binance.vision (§4.3)
    features.py
    strategies/
      base.py  brk.py  obi.py  ll.py  ctl.py  registry.py
    execution/
      base.py                   # Executor protocol, Position, TradeRecord
      paper.py  trailing_sim.py  live.py  sizing.py  ids.py
    reconcile.py  risk.py  income.py
    db.py  schema.sql
    stats.py  snapshot.py  alerts.py
    engine.py                   # orchestrace
    replay.py                   # běh engine nad nahranými daty (determinismus, offline test)
  dashboard/
    index.html  app.js  style.css
  deploy/
    arena.service  install.sh  patch_caddyfile_arena.py
  research/
    obi_prior/                  # explorativní analýza, ze které vychází rodina obi (§5.5); skripty + README
  tests/
    ...
```

Závislosti: `httpx`, `websockets`, `cryptography`, `pyyaml`, `numpy`; testy `pytest`, `pytest-asyncio`. Žádný pandas
v hot path. Python 3.12 (na serveru je), správa přes `uv`.

---

## 4. Datová vrstva

### 4.1 Streamy
Od 2026-04-23 platí jen routované URL (staré `/ws` a `/stream` bez cesty dostávají jen public data):
- `wss://fstream.binance.com/market/stream?streams=btcusdt@kline_5m/btcusdt@aggTrade/btcusdt@markPrice@1s/...`
- `wss://fstream.binance.com/public/stream?streams=btcusdt@bookTicker/btcusdt@depth@100ms/...` (diff depth, ne
  partial `depth20`; **OVĚŘIT**, že diff depth je na routě `/public`)
- `wss://fstream.binance.com/private/ws?listenKey=<key>&events=ORDER_TRADE_UPDATE/ACCOUNT_UPDATE/ALGO_UPDATE`
  (**OVĚŘIT** formát parametru `events` a to, jestli `ALGO_UPDATE` vyžaduje explicitní uvedení)

Base URL dej do konfigurace (testnet má jiné). WS klient: ping/pong podle Binance, preventivní reconnect každých 23 h,
exponenciální backoff, a po každém reconnectu **backfill mezery přes REST**. listenKey: `POST /fapi/v1/listenKey`,
keepalive `PUT` každých 30 min. Při `listenKeyExpired` vytvoř nový klíč a spusť plnou rekonciliaci.

### 4.2 BarStore
- Uloží bar jen při `k.x == true`. Klíč je `(symbol, open_time)`.
- Při startu a po reconnectu stáhne `GET /fapi/v1/klines` od posledního uloženého baru. Warmup historie = max
  `warmup_bars` všech strategií + 288 (min. 3 dny, tj. 864 barů).
- Detekce mezer: když `open_time` nového baru ≠ předchozí + 300 000 ms, backfill. Když backfill selže, slot pro ten
  symbol je `stale` a nesmí generovat vstupy.
- Událost „bar uzavřen“ se pro strategie vydá, až jsou uzavřené bary **všech** symbolů pro daný čas, nebo po timeoutu
  3 s od hranice (rodina C potřebuje BTC i alt ze stejného baru). Symbol, který nedorazil, se pro ten bar přeskočí a
  zaloguje se.

### 4.3 Book a ticky
- `book.py` drží na symbol **lokální plnou knihu**: REST snapshot `GET /fapi/v1/depth?limit=1000` + diff stream
  `depth@100ms` podle postupu Binance „How to manage a local order book correctly“ (zahodit eventy s `u` < `lastUpdateId`,
  první použitý event musí mít `U ≤ lastUpdateId ≤ u`, každý další `pu` == předchozí `u`; jinak resync). **OVĚŘIT**
  aktuální znění postupu pro USDⓈ-M. Úrovně s množstvím 0 se mažou. Resnapshot preventivně každých 30 min a po každém
  resyncu (váha 20, zanedbatelné).
- Proč plná kniha: 1000 úrovní snapshotu sahá 2026-10-21 jen ±0,16 % u BTC a ±0,45 % u ETH (SOL/XRP/DOGE přes ±7 %).
  Pásmo ±1 % se tedy u BTC/ETH složí ze snapshotu a úrovní, které diff stream průběžně doplní. Úroveň daleko od ceny,
  která se od snapshotu nezměnila, v lokální knize chybí; proto validace níže.
- `bands(t)` vrátí pro pásma `p ∈ {0.2, 1, 2, 5}` % kumulativní notional bid (cena ≥ mid × (1 − p/100)) a ask
  (cena ≤ mid × (1 + p/100)), `mid` = střed best bid/ask. Vzorkuje se **každých 30 s na sekundě :00 a :30**, aby
  živý výpočet odpovídal historickému datasetu `bookDepth` (snapshot ~30 s). Vzorek se nepočítá (a bar dostane
  `book_ok = 0`), když kniha není synchronizovaná nebo je diff stream starší než 2 s.
- `simulate_market(side, qty, t)` projde lokální knihu a vrátí VWAP. Když kniha není synchronizovaná nebo je starší než
  1 s, použije bookTicker + penalizaci `stale_book_penalty_bps`.
- `ticks.py` posílá každý aggTrade do `TrailingSim` jen pro otevřené paper pozice daného symbolu. Pořadí je podle
  `T`/`a` (trade time / aggId).
- **Validace knihy** (`arena book-validate --date D`, denně cronem pro včerejšek): stáhne
  `data.binance.vision/data/futures/um/daily/bookDepth/{SYM}/{SYM}-bookDepth-{D}.zip` (sloupce `timestamp, percentage,
  depth, notional`; kumulativní notional pro ±0,2/1/2/3/4/5 %) a porovná se zaznamenanými pásmy (párování na nejbližší
  timestamp ± 5 s). Výstup na symbol a pásmo: korelace `obi`, medián |Δ notional| v %. Výsledek do `events` a na
  dashboard. **OVĚŘIT**, jestli `percentage` je vůči mid, nebo vůči best bid/ask (validace to ukáže).

### 4.4 Recorder (surová data)
- Každý aggTrade všech symbolů universa se zapíše do `/srv/arena/raw/aggTrades/{SYM}/{YYYY-MM-DD}.csv.gz` (sloupce
  jako v `data.binance.vision`: `agg_trade_id, price, quantity, first_trade_id, last_trade_id, transact_time,
  is_buyer_maker`). Soubor se rotuje o UTC půlnoci, zápis po blocích (flush ≤ 5 s), po pádu se pokračuje novým
  segmentem `{date}.partN.csv.gz`. Mezery (výpadek) se doplní z REST `/fapi/v1/aggTrades` jen pro otevřené paper
  pozice (§7); úplná data za den se dají dotáhnout z `data.binance.vision` (T+1).
- Pásma knihy (30 s vzorky) jdou do tabulky `book_bands`, agregace na bar do `bar_book` (§12).
- Surové diff depth eventy se defaultně **nenahrávají** (`recorder.raw_depth: false`); zapnout jde pro forenzní účely.
- Retence: `recorder.retention_days` (default `null` = nemazat).

---

## 5. Featury a strategie

### 5.1 Featury (`features.py`, počítané inkrementálně na uzavřeném baru, jen z minulých a aktuálního baru)
| Název | Definice |
|---|---|
| `ret1` | ln(c_t / c_{t−1}) |
| `sigma288` | sd(`ret1`) přes posledních 288 barů (1 den) |
| `atr14_pct` | Wilder ATR(14) / c_t × 100 |
| `lvz288` | z-skóre ln(volume) vůči posledním 288 barům (bez aktuálního) |
| `don_hi_N`, `don_lo_N` | max high / min low barů t−N … t−1 (**bez** aktuálního) |
| `ema_n` | EMA close, α = 2/(n+1) |
| `beta288` | OLS beta `ret1_alt` na `ret1_btc` přes 288 barů |
| `cvd12`, `cvd12_z` | Σ(2·tbqv − qv) / Σ qv přes bary t−11 … t (taker quote delta); z-skóre vůči 288 barům (bez aktuálního) |
| `obi{p}` | průměr přes 30s vzorky knihy v [open_time, open_time + 300 s) z (bid_p − ask_p)/(bid_p + ask_p), p ∈ {0.2, 1, 2} (§4.3) |
| `obi{p}_z` | z-skóre `obi{p}` vůči posledním 288 barům s `book_ok = 1` (bez aktuálního); méně než 200 platných → NaN |
| `book_ok` | 1, když bar má ≥ 8 z 10 platných 30s vzorků knihy |

Test bez look-ahead (povinný): změna libovolného baru s indexem > t nezmění žádnou featuru ani signál v čase ≤ t.

### 5.2 Rozhraní strategie
```python
@dataclass(frozen=True)
class Intent:
    side: Literal["LONG", "SHORT"]
    callback_pct: float | None       # 0.1..10, zaokrouhleno na 0.1; None = bez trailingu (jen ll.v09)
    activate_offset_pct: float | None  # None = trailing hned od vstupu; jinak aktivace při zisku X %
    hard_stop_pct: float             # vzdálenost tvrdého stopu od vstupu, v %
    take_profit_px: float | None     # absolutní cena LIMIT reduceOnly, nebo None
    max_hold_bars: int
    reason: str                      # krátký popis pro log

class Strategy(Protocol):
    family: str            # "brk" | "obi" | "ll" | "ctl"
    variant_id: str        # "brk.v00"
    params: Mapping[str, Any]
    warmup_bars: int
    symbols: tuple[str, ...]
    def on_bar(self, ctx: BarContext) -> Intent | None: ...
```
`BarContext`: `symbol`, `bar_time`, read-only numpy pohledy na posledních `warmup_bars` barů tohoto symbolu a BTC,
featury, stav slotu (`flat | in_position | cooldown`). Strategie je **čistá funkce**: žádné I/O, žádný čas zdi, žádná
náhoda (výjimka `ctl` má RNG seedovaný hashem `(slot_id, bar_time)`). Cooldown po výstupu je `cooldown_bars`
(default 3) a hlídá ho engine, ne strategie.

Společná pravidla pro výstupy (pokud varianta neříká jinak):
`callback_pct = clip(round1(k_cb × atr14_pct), 0.2, 3.0)`, `hard_stop_pct = max(2 × callback_pct, 1.0)`.

### 5.3 Varianty: zásada
10 variant **není grid**. Je to základ V00 a 9 jednorozměrných odchylek, takže každé srovnání Vxx vs V00 odpovídá na
jednu otázku (citlivost na lookback, práh, šířku trailingu, filtr, směr, délku držení). Variant ID se nikdy nerecykluje.

### 5.4 Rodina A: `brk`, breakout s potvrzením objemem (BTC, ETH, SOL, XRP, DOGE)
LONG: `c_t > don_hi_N` a `lvz288 ≥ vz`. SHORT: `c_t < don_lo_N` a `lvz288 ≥ vz`.

| ID | N | vz | k_cb | směr | max_hold | jiné |
|---|---|---|---|---|---|---|
| v00 | 24 | 1.0 | 4 | oba | 36 | základ |
| v01 | 12 | 1.0 | 4 | oba | 36 | kratší lookback |
| v02 | 48 | 1.0 | 4 | oba | 36 | delší lookback |
| v03 | 24 | — | 4 | oba | 36 | bez objemového filtru |
| v04 | 24 | 2.0 | 4 | oba | 36 | přísný objem |
| v05 | 24 | 1.0 | 2.5 | oba | 36 | užší trailing |
| v06 | 24 | 1.0 | 7 | oba | 36 | širší trailing |
| v07 | 24 | 1.0 | 4 | jen LONG | 36 | test asymetrie (research 15: shorty prohrávaly) |
| v08 | 24 | 1.0 | 4 | oba | 96 | dlouhé držení 8 h |
| v09 | 24 | 1.0 | 2.5 | oba | 36 | `activate_offset_pct = 1.5 × callback`, do aktivace drží tvrdý stop `2 × callback` |

### 5.5 Rodina B: `obi`, nerovnováha hloubky knihy a objem (BTC, ETH, SOL, XRP, DOGE)
Hypotéza: když je v pásmu ±p % kolem ceny výrazně víc klidové likvidity na jedné straně, než je v posledním dni
obvyklé, cena se během hodin posune nahoru při převaze bidů a dolů při převaze asků. Nejde o top of book: ten na 5m
a nákladech taker/taker nic nenese (viz analýza níže a varianta v04, která to kontroluje).

`obi_z = obi{p}_z` (§5.1). LONG když `obi_z ≥ zt` a `book_ok`, SHORT když `obi_z ≤ −zt` a `book_ok`. Bar s
`book_ok = 0` signál nikdy nevydá a zaloguje se `stale`. `warmup_bars = 288` platných barů knihy; dokud kniha nemá
historii (nový start bez `bar_book`), rodina mlčí. Historii lze předplnit z `bookDepth` (`arena backfill-book --days N`),
featura se pak počítá stejně jako v §4.3 (30s vzorky).

| ID | p (%) | zt | směr | max_hold | k_cb | jiné |
|---|---|---|---|---|---|---|
| v00 | 1 | 2.5 | oba | 48 | 5 | základ |
| v01 | 1 | 2.0 | oba | 48 | 5 | volnější práh |
| v02 | 1 | 3.0 | oba | 48 | 5 | přísný práh |
| v03 | 2 | 2.5 | oba | 48 | 5 | širší pásmo |
| v04 | 0.2 | 2.5 | oba | 48 | 5 | blízká kniha (≈ top of book): kontrola, že edge nesedí v nejbližších úrovních |
| v05 | 1 | 2.5 | jen LONG | 48 | 5 | test asymetrie |
| v06 | 1 | 2.5 | oba | 96 | 5 | dlouhé držení 8 h |
| v07 | 1 | 2.5 | oba | 48 | 8 | širší trailing |
| v08 | 1 | 2.5 | oba | 48 | 5 | + objem: taker flow nesmí jít proti (LONG `cvd12_z ≥ 0`, SHORT `cvd12_z ≤ 0`) |
| v09 | 1 | 2.5 | oba | 48 | 5 | + TP LIMIT reduceOnly (maker) na `c_t × (1 ± 2 × callback_pct/100)` |

Pozor (v09): TP LIMIT se při zadání nesmí okamžitě vyplnit (cena musí být na „správné straně“ trhu). Když by se
vyplnila, TP se nezadá a zaloguje se `tp_skipped_marketable`. Paper to dělá stejně.

**Podklad pro priory (explorativní, in-sample, `research/obi_prior/`).** 90 dní (2026-07-22 až 2026-10-19), 5 symbolů,
5m klines + `bookDepth` + `metrics` z data.binance.vision. Výnos ve směru signálu při pevném držení (bez trailingu),
nepřekrývající se signály, SE clusterovaná po dnech UTC:

| Signál | 4 h (48 barů) | 8 h (96 barů) | signálů/den (5 sym.) |
|---|---|---|---|
| kniha ±1 %, `obi_z` ≥ 2 | +7,8 ± 4,0 bps | +11,4 ± 6,8 | 12 |
| kniha ±1 %, `obi_z` ≥ 2,5 | +17,7 ± 5,8 | +26,2 ± 7,5 | 7 |
| kniha ±1 %, `obi_z` ≥ 3 | +25,9 ± 10,0 | +44,5 ± 11,0 | 4 |
| kniha ±2 %, ≥ 2,5 | +4,8 ± 6,0 | +6,0 ± 8,1 | 7 |
| kniha ±0,2 %, ≥ 2,5 | +7,5 ± 4,7 | +4,1 ± 6,5 | 10 |
| kniha ±5 %, ≥ 2,5 | −2,1 ± 6,9 | +3,8 ± 11,3 | 7 |
| taker flow: „absorpce“ (silný prodej, cena drží → LONG) | −18,6 ± 9,3 | −24,7 ± 14,1 | 2 |
| taker flow + cena ve stejném směru (pokračování) | +10,4 ± 9,7 | +6,7 ± 13,7 | 4 |
| fade 1h pohybu z ≥ 2,5 (obdoba bývalého `mr`) | −10,0 ± 6,6 | −9,4 ± 11,3 | 8 |
| pokles OI (flush) → fade | +2,6 ± 8,0 | −0,5 ± 12,8 | 4 |
| náhodné vstupy (kontrola) | −15,8 ± 9,6 | −30,8 ± 12,8 | 10 |

Další zjištění: taker delta baru vysvětluje výnos **téhož** baru (Spearman 0,59), ale budoucí výnos ≥ 1 h nepredikuje
(IC ≈ 0). Parciální IC `obi1`/`obi2` po odečtení minulého výnosu (1 h, 4 h) a taker flow zůstává 0,035–0,055, takže
nejde jen o převlečený mean reversion. Efekt `obi1` je kladný ve všech třech třetinách období (6, 13, 34 bps při 4 h).
Bez kalibrovaných nákladů (~12 bps round trip) je hrubá hrana v00 v tomto vzorku zhruba +6 bps/obchod při 4 h.

**Proč to není důkaz:** ~60 testovaných kombinací na stejných 90 dnech, práh 2,5 vybraný až po pohledu na 2,0; náhodná
kontrola sama vyšla t ≈ −2,4, takže SE jsou spíš podhodnocené; pevné držení není trailing; období je jeden režim trhu.
Čekání je, že forward hrana bude výrazně menší, klidně nulová. Parametry tabulky jsou zmrazené priory a forward paper je
jediný test, který platí. Analýza se po pre-registraci (§14.1) už neopakuje s jinými prahy pod stejnými ID.

### 5.6 Rodina C: `ll`, BTC vede, alt zaostává (ETH, SOL, XRP, DOGE; BTC je jen signál)
5m verze vlastníkova LEAD (research 17/18). Zároveň měří to, co LEAD forward test potřebuje: **skutečnou slippage
vstupu po skoku BTC**.
`rb = ret1_btc / sigma288_btc`. LONG alt když `rb ≥ k` a `ret1_alt ≤ λ × beta288 × ret1_btc`.

| ID | k | λ | směr | max_hold | k_cb | jiné |
|---|---|---|---|---|---|---|
| v00 | 3.0 | 0.3 | LONG | 48 | 5 | základ |
| v01 | 2.5 | 0.3 | LONG | 48 | 5 | volnější spouštěč |
| v02 | 4.0 | 0.3 | LONG | 48 | 5 | přísný spouštěč |
| v03 | 3.0 | 0.0 | LONG | 48 | 5 | alt plochý nebo dolů (obdoba V05_lag0) |
| v04 | 3.0 | 0.3 | LONG | 12 | 5 | držení 1 h |
| v05 | 3.0 | 0.3 | LONG | 96 | 5 | držení 8 h |
| v06 | 3.0 | 0.3 | oba | 48 | 5 | SHORT zrcadlově (`rb ≤ −k`) |
| v07 | 3.0 | 0.3 | LONG | 48 | 5 | spouštěč na 2 barech: `(ret1+ret1_{t−1})_btc / (sigma288·√2)` |
| v08 | 3.0 | 0.3 | LONG | 48 | 5 | navíc BTC close > ema 576 (2 dny), režimový filtr |
| v09 | 3.0 | 0.3 | LONG | 48 | — | bez trailingu: jen tvrdý stop 2 % + časový výstup (kontrola výstupního mechanismu) |

### 5.7 Kontrolní rodina: `ctl` (jen paper, nepočítá se mezi „3 strategie“)
Pro každou rodinu a symbol jeden slot `ctl.<family>.<symbol>`: náhodný vstup s pravděpodobností p na bar, kde p je
empirická frekvence vstupů V00 té rodiny na tom symbolu za posledních 7 dní (prior z konfigurace, dokud nejsou data).
Směr se losuje podle podílu LONG/SHORT u V00. Výstupy jsou **identické s V00**. Bez kontroly nejde říct, jestli
výsledek dělá signál, nebo samotný výstupní mechanismus (trailing má vlastní statistický otisk).

---

## 6. Binance klient (`arena/binance/`)

- Podpis: Ed25519 přes `cryptography`, payload = query string, `signature` = base64. Hlavička `X-MBX-APIKEY`.
  `timestamp` = lokální čas + offset z `GET /fapi/v1/time` (přepočítat každých 10 min), `recvWindow=5000`.
- Endpointy, které potřebuješ (nic víc):
  - veřejné: `/fapi/v1/time`, `/fapi/v1/exchangeInfo`, `/fapi/v1/klines`, `/fapi/v1/aggTrades`, `/fapi/v1/fundingInfo`,
    `/fapi/v1/premiumIndex`
  - účet: `/fapi/v1/commissionRate`, `/fapi/v3/account` nebo `/fapi/v2/account` (**OVĚŘIT** aktuální verzi),
    `/fapi/v3/positionRisk` (**OVĚŘIT**), `/fapi/v1/income`, `/fapi/v1/userTrades`, `/fapi/v1/positionSide/dual` (GET),
    `/fapi/v1/leverage`, `/fapi/v1/marginType`
  - příkazy: `POST/GET/DELETE /fapi/v1/order`, `/fapi/v1/openOrders`, `POST/GET/DELETE /fapi/v1/algoOrder`,
    `/fapi/v1/openAlgoOrders`, `/fapi/v1/listenKey`
- Ceny a množství jen jako `Decimal`, zaokrouhlené podle `PRICE_FILTER.tickSize` a `LOT_SIZE.stepSize`
  (`MARKET_LOT_SIZE` pro market). Zaokrouhlení stopů vždy **konzervativně** (stop dál od ceny se nesmí stát; zaokrouhli
  k bližší straně v rámci 1 ticku a zaloguj). Kontrola `MIN_NOTIONAL.notional` (2026-10-21: BTC 50, ETH 20, ostatní 5
  USDT, čti živě).
- Chyby, které musí být explicitně ošetřené: `-1021` (timestamp → resync času, 1 retry), `-2021` (would immediately
  trigger), `-2022` (ReduceOnly rejected → pozice už je pryč, rekonciliuj), `-4120` (špatný endpoint = bug, crash
  testem), `-1003`/HTTP 429 (zpomal podle `Retry-After`), HTTP 418 (ban → zastav live, alert), 5xx/timeout (stav
  příkazu je **neznámý** → dotaz podle `origClientOrderId`/`clientAlgoId`, nikdy slepý retry).
- Idempotence: `newClientOrderId` / `clientAlgoId` = `a5-{slot_short}-{trade_seq}-{kind}` (≤ 36 znaků, regex výše),
  `kind ∈ {E, TS, HS, TP, TX, EM}`. `slot_short` je stabilní krátký hash slot_id (6 znaků base32).

---

## 7. Paper executor (realismus je tady nejdůležitější)

Každý slot má paper pozici. Fixní notional **1000 USDT na obchod**, bez skládání, takže výsledky jsou přímo v bps a
srovnatelné napříč symboly.

**Vstup:** decision time `t_d` = čas přijetí kline eventu. Fill time = `t_d + entry_latency_ms` (default 250, kalibruje
se). Cena = `book.simulate_market(side, qty)` k fill time (BUY bere ask, SELL bid, projde lokální knihu). Poplatek =
skutečná taker sazba účtu.

**TrailingSim, přesně podle pravidla burzy na last price (CONTRACT_PRICE):**
- uzavírání LONG (SELL trailing): pokud je `activatePrice`, aktivuje se při prvním obchodu `≥ activatePrice`, jinak hned
  při vstupu s HWM = vstupní cena. Po aktivaci `HWM = max(HWM, last)`, spustí se, když `last ≤ HWM × (1 − cb/100)`.
- uzavírání SHORT zrcadlově (LWM, `last ≥ LWM × (1 + cb/100)`).
- po spuštění se market fill simuluje na `t_trigger + trigger_latency_ms` (default 100) přes knihu + taker fee.
- tvrdý stop: `last ≤ stop_px` (LONG) → market jako výše.
- TP LIMIT: fill **jen když proběhne obchod striktně za limitní cenou** (dotek nestačí, kvůli frontě). Maker fee, cena =
  limit.
- časový výstup: na uzavření baru `entry_bar + max_hold_bars`, market přes knihu.
- když v jedné tickové dávce nastanou dva výstupy, rozhoduje pořadí ticků; při shodě časů vyhrává horší výsledek
  (konzervativně).

**Funding:** pro každou otevřenou pozici v čase fundingu (`fundingIntervalHours` z `/fapi/v1/fundingInfo`, default 8):
`pnl -= sign × notional × fundingRate` (LONG platí při kladném rate). Rate se bere z `markPrice` streamu (pole `r`)
těsně před časem fundingu.

**Výpadek:** po restartu se pro otevřené paper pozice stáhnou chybějící aggTrades přes REST (`/fapi/v1/aggTrades`,
okna ≤ 1 h) a přehrají se do TrailingSim. Kniha za výpadek není, takže fill = last price ± `outage_penalty_bps`
(default 2) a obchod dostane příznak `tainted=1`. Statistiky ukazují výsledky s tainted obchody i bez nich.

**Kalibrace** (`arena calibrate`): z párů live/paper dvojčat spočítá medián a p90 rozdílu vstupní a výstupní ceny v bps
a latence. Navrhne nové `entry_latency_ms`, `trigger_latency_ms` a přirážku `extra_slippage_bps` na symbol, ale
**nepřepisuje konfiguraci sama**. Vlastník je potvrdí, tím vznikne nová `cost_model_version` a statistiky se dají
filtrovat podle verze.

---

## 8. Live executor: stavový automat jedné pozice

```
FLAT ──signal──▶ ENTRY_SENT ──fill──▶ PROTECTING ──všechny ack──▶ PROTECTED ──trigger/TP/time/kill──▶ EXITING ──▶ FLAT
                     │                    │ (do 2 s, 3 pokusy)                                         │
                     │ reject/timeout     └──selhání ochrany──▶ EMERGENCY_CLOSE (market reduceOnly) ───┘
                     └──▶ FLAT (log)
```
1. **Vstup:** `POST /fapi/v1/order` `type=MARKET newOrderRespType=RESULT`. Při timeoutu nejdřív dotaz podle
   `origClientOrderId`, nikdy nový příkaz.
2. **Ochrana** hned po fillu, quantity = `executedQty`: (a) tvrdý `STOP_MARKET` algo reduceOnly, (b)
   `TRAILING_STOP_MARKET` algo reduceOnly (s `activatePrice`, pokud ho intent má; při `-2021` zadej bez aktivace a
   zaloguj), (c) volitelně `LIMIT` reduceOnly GTC TP. Když (a) nebo (b) neprojde do 2 s / 3 pokusů → **EMERGENCY_CLOSE**
   + alert. **Invariant: živá pozice nikdy nezůstane bez tvrdého stopu déle než 2 s.**
3. **Výstup přes burzu:** `ALGO_UPDATE`/`ORDER_TRADE_UPDATE` → zruš sourozence (ostatní algo příkazy a TP) a zapiš obchod.
4. **Časový výstup / kill:** zruš algo příkazy a TP, potom `MARKET reduceOnly`. Když přijde `-2022`, pozice mezitím
   skončila jinak: rekonciliuj a dopočti.
5. **Náklady skutečně:** fee z `ORDER_TRADE_UPDATE` (pole `n` a `N`), realized PnL (`rp`). Každých 10 min
   `GET /fapi/v1/income` (COMMISSION, FUNDING_FEE, REALIZED_PNL), páruje se se symbolem a časem. **Income je ground
   truth.** Rozdíl proti lokálnímu ledgeru > 0,01 USDT → varování na dashboardu.
6. **Měřené latence:** bar close → odeslání, odeslání → ack, ack → fill, trigger → fill (z časů událostí burzy).

---

## 9. Rekonciliace (`reconcile.py`), při startu a každých 60 s

Porovná DB (očekávané pozice a příkazy live slotů) s `positionRisk`, `openOrders` a `openAlgoOrders`:
- pozice na burze, v DB žádná → `FOREIGN_POSITION`: alert, žádná automatická akce, live na tom symbolu se zmrazí.
- pozice v DB i na burze bez tvrdého stopu nebo trailingu → okamžitě doplnit ochranu (krok 2 výše).
- pozice v DB, na burze žádná → dohledat výstup přes `userTrades`/`income`, zapsat obchod `exit_reason=reconciled`.
- osiřelý algo/limit příkaz s prefixem `a5-` bez pozice → zrušit.
- příkazy bez prefixu `a5-` → nikdy nesahat (patří vlastníkovi).
- nesoulad množství → alert + zmrazit slot.

Při startu: zkontrolovat one-way mód, ISOLATED a páku na každém symbolu (nastavit, pokud to jde bez otevřené pozice,
jinak alert a nestartovat live).

---

## 10. Risk guard (`risk.py`)

| Pravidlo | Default | Akce |
|---|---|---|
| live notional na slot | 150 USDT (BTC 200 kvůli stepSize 0.001) | pevné |
| max. otevřených live pozic | 5 | odmítnout vstup |
| denní ztráta live slotu | 3 % jeho alokace | slot `disabled` do další UTC půlnoci |
| denní ztráta účtu (wallet + unrealized vs. 00:00 UTC) | 3 % | zavřít vše, `live_halt`, alert, ruční reset |
| kumulativní DD live slotu | 15 % alokace | slot `disabled` natrvalo (ručně zpět) |
| stale data (bar > 20 s po hranici, book > 5 s) | | žádné nové vstupy na symbolu |
| WS private odpojený > 30 s | | žádné nové live vstupy, rekonciliace po reconnectu |
| běh live bez `live.enabled: true` **a** flagu `--live` | | live executor se nevytvoří |

Kill switch: soubor `/srv/arena/KILL` (nebo `arena kill`). Když existuje, live se zavře a nic nového se neotevře.
Paper běží dál.

---

## 11. Konfigurace

`config/arena.yaml`:
```yaml
env: testnet            # testnet | mainnet
universe: [BTCUSDT, ETHUSDT, SOLUSDT, XRPUSDT, DOGEUSDT]
endpoints:
  mainnet: {rest: https://fapi.binance.com, ws_market: wss://fstream.binance.com/market,
            ws_public: wss://fstream.binance.com/public, ws_private: wss://fstream.binance.com/private}
  testnet: {rest: https://testnet.binancefuture.com, ws_market: "OVĚŘIT", ws_public: "OVĚŘIT", ws_private: "OVĚŘIT"}
paper: {notional_usdt: 1000, entry_latency_ms: 250, trigger_latency_ms: 100, stale_book_penalty_bps: 1,
        outage_penalty_bps: 2, extra_slippage_bps: {}, cost_model_version: 1}
live:
  enabled: false
  leverage: 2
  slots:                      # max. 1 na symbol (validace)
    BTCUSDT: {variant: brk.v00, notional_usdt: 200}
    ETHUSDT: {variant: ll.v00,  notional_usdt: 150}
    SOLUSDT: {variant: obi.v09, notional_usdt: 150}   # v09 kvůli měření TP LIMIT (maker výstup)
    XRPUSDT: {variant: brk.v00, notional_usdt: 150}
    DOGEUSDT: {variant: ll.v00, notional_usdt: 150}
risk: {account_daily_loss_pct: 3, slot_daily_loss_pct: 3, slot_max_dd_pct: 15, max_live_positions: 5}
engine: {cooldown_bars: 3, bar_close_wait_s: 3, snapshot_every_s: 15, reconcile_every_s: 60}
recorder: {raw_dir: /srv/arena/raw, raw_aggtrades: true, raw_depth: false, retention_days: null}
book: {bands_pct: [0.2, 1, 2, 5], sample_every_s: 30, resnapshot_every_min: 30, max_diff_age_s: 2}
paths: {db: /srv/arena/arena.db, site: /srv/arena/site, kill_file: /srv/arena/KILL}
alerts: {ntfy_topic_env: NTFY_TOPIC}
```
Počáteční live sloty pokrývají všechny tři typy výstupu (trailing, TP limit přes `obi.v09`, vstup po skoku BTC) a
všech 5 symbolů.
Jsou to **měřicí sondy**, ne sázka na nejlepší variantu. Očekávaná cena měření při ~3 obchodech/den/slot a 13 bps
nákladů: zhruba 0,1 USDT/den/slot za poplatky, plus šum P&L.

`config/strategies/*.yaml` obsahuje jen parametry variant (tabulky v sekci 5). Hash celé konfigurace
(`config_hash`) se zapisuje ke každému signálu a obchodu. Změna parametrů existující varianty je **chyba validace**:
musí vzniknout nové ID.

---

## 12. Databáze (`schema.sql`)

```sql
CREATE TABLE bars(symbol TEXT, open_time INTEGER, o REAL, h REAL, l REAL, c REAL, v REAL, qv REAL, n INTEGER,
                  tbv REAL, tbqv REAL, PRIMARY KEY(symbol, open_time));      -- tbqv = kline pole `Q`
CREATE TABLE book_bands(symbol TEXT, ts INTEGER, pct REAL, bid_notional REAL, ask_notional REAL, mid REAL,
                  source TEXT CHECK(source IN ('live','bookDepth')), PRIMARY KEY(symbol, ts, pct, source));
CREATE TABLE bar_book(symbol TEXT, open_time INTEGER, pct REAL, obi REAL, n_samples INTEGER, book_ok INTEGER,
                  source TEXT, PRIMARY KEY(symbol, open_time, pct));
CREATE TABLE slots(slot_id TEXT PRIMARY KEY, family TEXT, variant_id TEXT, symbol TEXT, params_json TEXT,
                   params_hash TEXT, is_control INTEGER, created_at INTEGER);
CREATE TABLE slot_state(slot_id TEXT, mode TEXT CHECK(mode IN ('paper','live')), status TEXT, state_json TEXT,
                   updated_at INTEGER, PRIMARY KEY(slot_id, mode));
CREATE TABLE signals(id INTEGER PRIMARY KEY, slot_id TEXT, bar_time INTEGER, side TEXT, intent_json TEXT,
                   action TEXT,   -- entered | suppressed_position | suppressed_cooldown | blocked_risk | stale
                   decided_at INTEGER, config_hash TEXT);
CREATE TABLE trades(trade_id TEXT PRIMARY KEY, slot_id TEXT, mode TEXT, twin_trade_id TEXT, signal_id INTEGER,
                   side TEXT, qty REAL, notional REAL,
                   decision_px REAL, arrival_mid REAL, entry_px REAL, entry_time INTEGER, entry_fee REAL,
                   exit_trigger_px REAL, exit_px REAL, exit_time INTEGER, exit_fee REAL, exit_reason TEXT,
                   funding REAL, gross_pnl REAL, net_pnl REAL, gross_bps REAL, net_bps REAL,
                   entry_slip_bps REAL, exit_slip_bps REAL, mfe_bps REAL, mae_bps REAL, bars_held INTEGER,
                   tainted INTEGER DEFAULT 0, cost_model_version INTEGER, config_hash TEXT);
CREATE TABLE orders(client_id TEXT PRIMARY KEY, trade_id TEXT, kind TEXT, is_algo INTEGER, exchange_id TEXT,
                   status TEXT, request_json TEXT, last_response_json TEXT, sent_at INTEGER, ack_at INTEGER);
CREATE TABLE fills(exchange_trade_id TEXT PRIMARY KEY, client_id TEXT, trade_id TEXT, px REAL, qty REAL,
                   fee REAL, fee_asset TEXT, is_maker INTEGER, ts INTEGER, realized_pnl REAL);
CREATE TABLE income(tran_id TEXT PRIMARY KEY, symbol TEXT, income_type TEXT, amount REAL, asset TEXT, ts INTEGER,
                   trade_id TEXT);
CREATE TABLE equity(ts INTEGER PRIMARY KEY, wallet REAL, unrealized REAL, margin_used REAL);
CREATE TABLE latencies(trade_id TEXT, kind TEXT, ms INTEGER);
CREATE TABLE events(id INTEGER PRIMARY KEY, ts INTEGER, level TEXT, component TEXT, msg TEXT, data_json TEXT);
CREATE TABLE config_versions(config_hash TEXT PRIMARY KEY, yaml TEXT, activated_at INTEGER);
```
Všechny časy jsou v ms UTC. Definice slippage: `entry_slip_bps = side × (entry_px − decision_px) / decision_px × 1e4`
(kladná = horší). `decision_px` = close signálního baru. `exit_slip_bps` se počítá proti `exit_trigger_px` (pro
trailing teoretická úroveň `HWM × (1 − cb)`, pro čas close baru).

---

## 13. Statistiky (`stats.py`)

Počítá se pro slot, pro variantu (sloučeně přes symboly) a pro rodinu, zvlášť paper a live, s filtrem
`cost_model_version` a `exclude_tainted`:

- `n`, obchody/den, expozice %, průměrné držení, mix exit_reason
- průměr a medián `gross_bps` a `net_bps`, sd, win rate, MFE/MAE
- **95% CI průměru net_bps**: blokový bootstrap po dnech UTC (2000 vzorků, seed fixní)
- náklady na obchod v bps: fee, entry slip, exit slip, funding. Break-even hrubá hrana. Net při 1,5× a 2× nákladech.
- denní P&L řada → Sharpe (×√365), max DD, Calmar
- **Δ proti kontrole** (`ctl` stejné rodiny a symbolu): rozdíl průměrů net_bps s bootstrap CI a p-hodnotou
- **Holm** přes 30 variant (p vs kontrola), **DSR** s N = 30 a rozptylem Sharpe napříč variantami
- **potřebné n**: `n_req = ((z × sd) / mean)²` pro z = 2 a z = 3 a odhad, kolik dní ještě zbývá při aktuální frekvenci
- live: model error = `live.net_bps − paper_twin.net_bps` (medián, p90) rozložený na vstup, výstup, fee, timing

Orientační síla testu (rozhodnutí se dělá na skutečném sd po 2 týdnech): při sd ≈ 60 bps/obchod a skutečné hraně
5 bps je potřeba ~580 obchodů na t = 2 a ~1300 na t = 3 (Holm přes 30 variant). Varianta sloučená přes 5 symbolů se
~15 obchody/den se tam dostane za ~40, resp. ~90 dní. **Dřív se nerozhoduje.**

Status badge pro variantu: `PŘÍLIŠ BRZY` (n < 200) · `ZÁPORNÁ` (horní mez CI < 0) · `NEJASNÁ` · `KANDIDÁT`
(splňuje pravidla 14.2) · `LIVE`.

---

## 14. Experimentální protokol

### 14.1 Před startem (pre-registrace)
`docs/PREREG.md` (zapíše vlastník, commitne se **před** prvním paper barem): verze konfigurace, seznam variant,
rozhodovací pravidla 14.2, termíny kontrol. Kumulativní N trialů se zapisuje do vlastníkova registru (sem 30 variant;
kontroly se nepočítají).

### 14.2 Rozhodovací pravidla (jen v pevných termínech: den 45 a den 90 od startu)
Varianta je **KANDIDÁT**, když platí všechno najednou:
1. sloučené paper `n ≥ 500`, bez tainted obchodů
2. průměr net_bps > 0 při 1,5× nákladech (model kalibrovaný ≥ 30 live obchody na symbol)
3. Holm-adjustované p vs kontrola < 0,05
4. kladný průměr net_bps na ≥ 4 z 5 symbolů (C: 3 ze 4)
5. žádný kalendářní měsíc s net < −3 sd měsíčního P&L

Kandidát dostane **live slot** místo sondy na svém nejlepším symbolu a dalších 60 dní forward na čistých datech.
Dashboard ukazuje průběžné statistiky, ale rozhodnutí mimo termíny je zakázané (optional stopping).

### 14.3 Co se nikdy nedělá
Ladit parametry podle dashboardu a pokračovat pod stejným ID. Vybrat nejlepší ze 30 a hlásit jeho Sharpe bez DSR.
Pustit live víc slotů na jeden symbol. Mazat obchody (taintované se jen označí).

---

## 15. Dashboard (`dashboard/`, snapshot z `snapshot.py`)

Soubory v `/srv/arena/site/data/`: `snapshot.json` (každých 15 s), `variants/<variant_id>.json` (každých 5 min:
denní equity paper/live, gross vs net křivka, posledních 200 obchodů), `costs.json` (každých 5 min).

Záložky:
1. **Přehled:** equity účtu (wallet, unrealized), dnešní live P&L, dnešní náklady (fee, funding), otevřené live pozice
   (vstup, aktuální cena, aktuální úroveň trailingu = HWM × (1 − cb), vzdálenost v bps, algoId), počet otevřených paper
   pozic, health (zpoždění posledního baru na symbol, stáří booku, sync lokální knihy a podíl barů s `book_ok`, poslední
   `book-validate`, stav WS, poslední rekonciliace, kill switch, chyby
   za 24 h).
2. **Žebříček:** tabulka 30 variant + 3 kontroly: n, obchody/den, net bps ± CI, gross bps, náklady bps, Sharpe, Δ vs
   kontrola, p_holm, DSR, n_req, badge. Nad tabulkou trvalé upozornění: „pořadí nejlepších variant je zkreslené
   výběrem, rozhoduj podle p_holm a DSR“. Pod tím heatmapa varianta × symbol (net bps, s n v buňce, šedá když n < 30).
3. **Detail varianty** (klik): kumulativní net bps a gross bps v čase (rozdíl = náklady), stejná křivka kontroly,
   rozpad podle symbolů, exit_reason, histogram net_bps, posledních 200 obchodů.
4. **Náklady:** pro každý symbol live: fee bps, rozdělení entry a exit slippage (medián, p90), latence, funding.
   Paper vs live model error a aktuální `cost_model_version`.
5. **Log:** posledních 200 událostí `events`.

Technika: vanilla JS, žádný build. Grafy Chart.js `4.4.1` z `https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js`
(pinnutá verze). Tmavý i světlý režim. Na mobilu čitelné (tabulka se scrolluje uvnitř kontejneru, ne celá stránka).
Když je `snapshot.json` starší než 60 s, nahoře se ukáže červený pruh „data neaktuální“.

Nasazení: `deploy/install.sh` (sudo, idempotentní): uživatel `arena`, `/srv/arena{,/site,/site/data}`,
`/etc/arena/env`, systemd unit, Caddy blok přes `patch_caddyfile_arena.py` (vzor `portal/patch_caddyfile_lead.py`:
vložit `redir /arena /arena/` a `handle_path /arena/* { basic_auth {...}; root * /srv/arena/site; file_server }`,
záloha, `caddy validate`, reload) a záložka „Arena“ v `portal/index.html`.

---

## 16. CLI

```
arena check            # P0: konektivita, čas, oprávnění klíče, futures povoleny, one-way, margin/páka, commissionRate, filtry
arena universe-check   # 30denní obrat a spread universa (jen výpis)
arena run [--live]     # hlavní proces
arena replay --from --to [--ticks DIR] [--book DIR]   # offline běh nad bary (+ aggTrades a bookDepth z data.binance.vision nebo z recorderu)
arena backfill-book --days N   # předplní bar_book z bookDepth (source=bookDepth), aby obi nemusela čekat 288 barů
arena book-validate --date D   # lokální kniha vs bookDepth za den D (§4.3)
arena stats [--variant X] [--mode paper|live]
arena calibrate        # návrh parametrů nákladového modelu z live/paper dvojčat
arena reconcile        # jednorázová rekonciliace, výpis rozdílů, bez akcí (--fix provede akce ze sekce 9)
arena kill | arena unkill
arena flatten          # zruší všechny a5- příkazy, zavře všechny a5 pozice (s potvrzením)
```

---

## 17. Fáze implementace a akceptační kritéria

Každá fáze končí zelenými testy a krátkým zápisem do `CHANGELOG.md`. Do další fáze se nepostupuje, dokud předchozí
nesplňuje kritéria.

| Fáze | Obsah | Hotovo, když |
|---|---|---|
| **P0** | skeleton, config + validace, `binance/` klient, `arena check` | `arena check` na mainnet (read-only klíč) projde; testy podpisu proti příkladu z Binance docs; validace odmítne 2 live sloty na 1 symbolu a změnu parametrů existující varianty |
| **P1** | WS, BarStore, backfill, lokální kniha, recorder, book-validate | 24 h běh: 0 chybějících barů, bary bitově shodné s REST `klines`; test reconnectu (umělé odpojení) doplní mezeru; test synchronizace knihy (mezera v `pu` → resync); `book-validate` za den běhu: korelace `obi1` a `obi2` vůči `bookDepth` ≥ 0,9 a medián \|Δ notional\| ≤ 10 % na každém symbolu. **Když BTC/ETH neprojdou, zastav se a zeptej se vlastníka** (rodina `obi` by na nich měřila něco jiného, než z čeho vznikly priory) |
| **P2** | features, 4 rodiny, registry | golden testy featur na fixním CSV; look-ahead test (sekce 5.1); každá varianta má test „signál ano/ne“ na syntetických datech; `obi` featury z `bookDepth` souboru se shodují s výstupem `research/obi_prior/` na stejných datech |
| **P3** | paper executor, TrailingSim, funding, replay | testy TrailingSim na syntetických tickových cestách (aktivace, LONG/SHORT, souběh výstupů, TP „striktně za cenou“); `arena replay` dvakrát po sobě dá identické obchody; 7 dní paper bez pádu |
| **P4** | stats, snapshot, dashboard, deploy | statistiky ověřené ručním výpočtem na malém vzorku; stránka běží za Caddy s basic_auth; banner neaktuálnosti funguje |
| **P5** | live executor, rekonciliace, risk, income na **testnetu** | 200 round tripů bez jediné nechráněné pozice > 2 s; `kill -9` mezi vstupem a ochranou → po restartu ochrana doplněna nebo pozice zavřena; osiřelé algo příkazy zrušeny; `-2022` race ošetřen. **OVĚŘIT**, že testnet podporuje `/fapi/v1/algoOrder`; když ne, P5 přes mainnet s 1 slotem na min. notional |
| **P6** | mainnet: 1 live slot (XRPUSDT, min. notional), 7 dní | ledger = income endpoint ± 0,01 USDT; latence a slippage se zapisují |
| **P7** | mainnet: 5 live sond | po 30 live obchodech na symbol `arena calibrate` → vlastník potvrdí `cost_model_version: 2` |

---

## 18. Nástrahy (čti před psaním kódu)

1. Podmíněné příkazy jen přes `/fapi/v1/algoOrder`. Test, který mockuje REST, musí selhat, když kód pošle
   `TRAILING_STOP_MARKET` na `/fapi/v1/order`.
2. Routované WS URL (`/market`, `/public`, `/private`). Kline a aggTrade na `/public` nepřijdou.
3. Nikdy nerozhoduj na baru s `k.x == false`.
4. `reduceOnly` funguje jen v one-way módu. Při startu ověř `GET /fapi/v1/positionSide/dual` = false.
5. `callbackRate` je v **procentech** (0.5 = 0,5 %), spotový `trailingDelta` je v **BIPS**. Nepleť je.
6. `Decimal` všude, kde jde o ceny a množství na burzu. `float` jen ve statistikách.
7. Po timeoutu je stav příkazu neznámý: dotaz podle client ID, ne retry.
8. listenKey vyprší po 60 min bez keepalive. WS spojení Binance ukončí po 24 h.
9. Funding interval se liší podle symbolu (4 h / 8 h), čti `fundingInfo`.
10. Logy rotuj (journald limit nebo `RotatingFileHandler` 50 MB × 3).
11. Čas: všechno ms UTC, offset proti serveru Binance, ne lokální hodiny.
12. Paper i live musí používat **stejnou** funkci pro výpočet `Intent` → příkazy (callback rounding, ceny stopů),
    jinak dvojčata nejsou srovnatelná.
13. Dashboard nesmí číst DB přímo z webu. Jen snapshot JSON.
14. Žádné tajné klíče v repu, v logu ani ve snapshotu.
15. Lokální kniha: bez kontroly `pu` == předchozí `u` se kniha tiše rozjede a `obi` bude šum. Každá mezera = resync a
    bar s chybějícími vzorky dostane `book_ok = 0`, nikdy „poslední známá hodnota“.
16. `metrics` z data.binance.vision má `create_time` = **začátek** 5m periody, hodnota platí na jejím konci (ověřeno
    2026-10-21 přes implikovanou cenu `sum_open_interest_value / sum_open_interest`). Kdo ji spáruje s close baru se
    stejným časem, má look-ahead (v analýze to dalo falešnou korelaci 0,85). `bookDepth` timestamp sedí na reálný čas.
17. `sum_open_interest_value` obsahuje cenu; změna OI se počítá z `sum_open_interest` (kontrakty).

---

## 19. Mimo rozsah v1 (možná později)
Maker vstupy (GTX) s modelem fronty. Sub-accounty pro víc live slotů na symbol. Spot venue. Dynamický výběr variant
(bandit). Další signály z order flow (OI, likvidace přes `forceOrder`); OI flush v analýze `research/obi_prior/` nic
neukázal. Data se dají přidat do `features.py` bez změny architektury. Napojení na Research Lab: export obchodů jako lab-karty.
