# CHANGELOG

## P0: skeleton, konfigurace, Binance klient, `arena check` (2026-10-23)

Hotovo:
- `pyproject.toml` (uv, Python 3.12), balík `arena/`, CLI `arena`.
- `arena/config.py`: načtení `config/arena.yaml` + `config/strategies/*.yaml`, validace (zmrazené universe,
  routované WS URL, max. 1 live slot na symbol: YAML loader odmítá duplicitní klíče, kontrolní rodina nesmí live,
  varianta musí běžet na symbolu slotu), `config_hash`, zámek parametrů variant `config/variants.lock.json`.
  Klíče z env nebo `/etc/arena/env` (Ed25519, HMAC fallback, testnet s prefixem `BINANCE_TESTNET_`).
- `arena/binance/`: Ed25519/HMAC podpis, tenký async REST klient (čas burzy s offsetem, `-1021` resync +
  1 retry, 429/`-1003` pauza podle `Retry-After`, 418 ban zastaví klienta, timeout/5xx na příkazových
  endpointech = `UnknownOrderStatus` bez retry, GET retry s backoffem, sledování `X-MBX-USED-WEIGHT-1M`
  a order count). Guard: podmíněný typ na `/fapi/v1/order` nebo `batchOrders` se neodešle. Filtry z
  `exchangeInfo` v `Decimal`, konzervativní zaokrouhlení stopů, `callbackRate` v procentech po 0,1.
- `arena check`: čas, filtry a live notional proti filtrům, funding interval, WS routy `/market` a `/public`.
  S klíčem navíc futures účet, oprávnění klíče (`apiRestrictions`, jen mainnet), one-way, accountConfig,
  margin type a páka na symbol, otevřené pozice, commissionRate.
- `docs/VERIFIED.md`: ověřené body OVĚŘIT (WS routy, private stream, lokální kniha, Ed25519, v3 účet,
  algo endpointy, testnet URL, filtry, funding).

Odchylky od SPEC (bez dopadu na D1–D9):
- Testnet URL: `https://demo-fapi.binance.com` a `wss://demo-fstream.binance.com/{market,public,private}`
  (aktuální docs). SPEC měl `testnet.binancefuture.com`.
- Účet přes `/fapi/v3/account`, `/fapi/v3/positionRisk`, margin a páka přes `/fapi/v1/symbolConfig`.
- Navíc `GET /sapi/v1/account/apiRestrictions` (api.binance.com, read-only) kvůli kontrole D9.
- `brk.v09`: do aktivace platí společné pravidlo `hard_stop = max(2 × callback, 1 %)`. SPEC říká jen
  „2 × callback“, minimum 1 % je ze společných pravidel §5.2.
- Priory `ctl` (pravděpodobnost vstupu na bar): obi 0,0049 z `research/obi_prior`, brk 0,007 a ll 0,0035
  jsou odhady. Po 7 dnech je nahradí empirická frekvence V00.

Testy: 36 unit testů zelených (podpis proti vektorům z Binance docs: HMAC z futures docs, Ed25519 z FIX docs;
mock REST; validace konfigurace). `pytest -m net`: veřejná část `arena check` na mainnetu i testnetu projde.

**Akceptace P0 neuzavřena:** `arena check` s read-only klíčem na mainnetu (klíč zatím není). Spustí vlastník:
`uv run arena check --env mainnet`. Zjištění: testnet má u BTC `stepSize` 0.0001, mainnet 0.001.
