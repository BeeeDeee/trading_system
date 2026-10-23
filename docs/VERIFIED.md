# Ověřené body (SPEC „OVĚŘIT“)

Zdroj „docs“ = https://developers.binance.com (stažený `llms-full.txt`), „živě“ = read-only volání veřejného API.

| Datum | Bod | Endpoint / zdroj | Co platí |
|---|---|---|---|
| 2026-10-23 | §4.1 routy WS | docs (Important WebSocket Change Notice, Connect) + živě | `/public`: `bookTicker`, `depth`, `depth@100ms`, partial depth. `/market`: `aggTrade`, `kline_*`, `markPrice@1s`. Živě: `kline_5m`, `aggTrade` a `markPrice@1s` chodí na `/market/stream`. `bookTicker` a `depth@100ms` chodí na `/public/stream`. `aggTrade`+`kline_5m` na `/public` nepřišlo nic (0 zpráv za 8 s). |
| 2026-10-23 | §4.1 WS limity | docs (Connect) | Spojení platí 24 h. Server posílá ping každé 3 min a odpojí, když do 10 min nepřijde pong. Max. 10 příchozích zpráv/s a 1024 streamů na spojení. Combined stream obal: `{"stream":..,"data":..}`. |
| 2026-10-23 | §4.1 private | docs (User Data Streams, Change Notice) | Base `wss://fstream.binance.com/private`. Dvě formy: `/private/ws/<listenKey>` (User Data Streams) a `/private/ws?listenKey=<k>&events=A/B` (Change Notice, oddělovač `/`). Docs neříkají, co se pošle bez `events`. **Zvoleno:** `/private/ws/<listenKey>` (všechny eventy včetně `listenKeyExpired`). Že chodí `ALGO_UPDATE`, ověřím empiricky na testnetu (P5). |
| 2026-10-23 | §4.1 listenKey | docs | Platí 60 min, `PUT` prodlouží o 60 min; `-1125` = klíč neexistuje → `POST` nový. `POST` při aktivním klíči vrátí stávající. |
| 2026-10-23 | §4.3 lokální kniha | docs (How to manage a local order book correctly) | Postup sedí se SPEC: buffer, snapshot `limit=1000`, zahodit `u < lastUpdateId`, první event `U ≤ lastUpdateId ≤ u`, dál `pu == předchozí u`, jinak znovu od snapshotu. Množství je absolutní, 0 = smazat, mazání neexistující úrovně je normální. Živě: `GET /fapi/v1/depth?limit=1000` vrací `lastUpdateId, E, T, bids, asks`, váha 20. |
| 2026-10-23 | §6 Ed25519 podpis | docs (Spot SIGNED Ed25519, FIX Logon) | Payload = query string (percent-encoded), podpis Ed25519, base64, v URL percent-encoded. Futures docs mají jen HMAC vektor (`3c6612…6af9`). Ed25519 test vektor (klíč + payload + podpis) je jen ve FIX docs a použil jsem ho v testu. Ed25519 příklad ve spot REST docs ukazuje RSA výstup (chyba docs), proto se nepoužívá. |
| 2026-10-23 | §6 účet | docs (changelog) | `GET /fapi/v3/account` a `GET /fapi/v3/positionRisk` existují a vracejí jen symboly s pozicí nebo příkazem. Konfigurace (marginType, leverage) je v `GET /fapi/v1/symbolConfig`, režim účtu v `GET /fapi/v1/accountConfig`. **Zvoleno:** v3 + symbolConfig. |
| 2026-10-23 | D3 algo | docs (New Algo Order, changelog 2025-11-06) | `POST/GET/DELETE /fapi/v1/algoOrder`, `DELETE /fapi/v1/algoOpenOrders`, `GET /fapi/v1/openAlgoOrders`, `GET /fapi/v1/allAlgoOrders`. `callbackRate` 0,1–20 (SPEC omezuje na ≤ 10, to je přísnější a v pořádku), krok docs neuvádí → ověřit v P5. `workingType` default `CONTRACT_PRICE`. `clientAlgoId` regex `^[\.A-Z\:/a-z0-9_-]{1,36}$` sedí. Trailing `-2021`: SELL `activatePrice` musí být nad last, BUY pod. Váha 0 na IP limit, 1 na order count. |
| 2026-10-23 | D3 `ALGO_UPDATE` | docs | Statusy `NEW, CANCELED, TRIGGERING, TRIGGERED, FINISHED, REJECTED, EXPIRED`. Pole `o.ia` (od 2026-08-21) říká, jestli je trailing aktivovaný, a trailing může poslat dva `NEW` (ia false → true). **Názvy polí v `o` (vazba algoId → orderId) docs inline neuvádějí → empiricky v P5.** |
| 2026-10-23 | §11 testnet | docs (General Info) + živě | REST `https://demo-fapi.binance.com`, WS `wss://demo-fstream.binance.com`. Živě: `/market/stream` a `/public/stream` na demo-fstream fungují (kline, aggTrade, markPrice, depth, bookTicker). Staré `testnet.binancefuture.com` i `stream.binancefuture.com` zatím odpovídají, ale docs je už neuvádějí → používá se `demo-*`. Universe je na testnetu celé. |
| 2026-10-23 | §6 filtry | živě `exchangeInfo` | BTC tick 0.10 step 0.001 minNotional 50; ETH 0.01 / 0.001 / 20; SOL 0.0100 / 0.01 / 5; XRP 0.0001 / 0.1 / 5; DOGE 0.000010 / 1 / 5. `MARKET_LOT_SIZE.stepSize` = `LOT_SIZE.stepSize` u všech 5. Všechny 5 `underlyingType=COIN`, `TRADING`. `triggerProtect` 0.03–0.10. |
| 2026-10-23 | §7 funding | živě `fundingInfo` | Všech 5 symbolů má `fundingIntervalHours=8`. |
| 2026-10-23 | §6 chyby / limity | docs (General Info) | Hlavička `X-MBX-USED-WEIGHT-1M` (per IP), `X-MBX-ORDER-COUNT-*`. 429 = limit, 418 = ban (2 min až 3 dny). 503 „Unknown error“ = stav neznámý, ověřit dotazem. 503 „Service Unavailable“ = selhalo, retry s backoffem. `-1008` throttling (reduceOnly a cancel výjimka). `timestamp < serverTime + 1000` a `serverTime − timestamp ≤ recvWindow`. |

## Zbývá ověřit (s klíčem nebo na testnetu)
- D1: futures povolené na účtu, sub-account (`arena check` s klíčem).
- `ALGO_UPDATE` pole a vazba na `ORDER_TRADE_UPDATE` spuštěného příkazu (P5).
- Krok `callbackRate` (P5: zadat 0.15 a sledovat odmítnutí/zaokrouhlení).
- Testnet podpora `/fapi/v1/algoOrder` a private route na `demo-fstream` (P5).
- §4.3 `bookDepth.percentage` vůči mid nebo best (P1, `book-validate`).
