# arena5m

Experimentální framework pro 5m strategie na Binance USDⓈ-M perpech. Závazná specifikace: `SPEC.md`,
pokyny pro implementaci: `CLAUDE.md`, ověřené body API: `docs/VERIFIED.md`, průběh fází: `CHANGELOG.md`.

## Rychlý start
```
uv sync
uv run pytest                 # unit testy (síťové přeskočené)
uv run pytest -m net          # živé read-only testy
uv run arena check            # P0 kontrola (env z config/arena.yaml, default testnet)
uv run arena check --env mainnet
```

## Klíče (D9)
Klíče se čtou z proměnných prostředí, jinak z `/etc/arena/env` (mode 600, vlastník `arena`). Do repa nepatří.
```
# mainnet: Ed25519, jen čtení + futures, bez výběrů, IP whitelist
BINANCE_API_KEY=...
BINANCE_PRIVATE_KEY_PATH=/etc/arena/binance_ed25519.pem
# testnet (demo-fapi.binance.com), Ed25519 nebo HMAC
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_PRIVATE_KEY_PATH=...    # nebo BINANCE_TESTNET_API_SECRET=...
NTFY_TOPIC=...
```

## Varianty
Parametry variant jsou v `config/strategies/*.yaml` a jejich hashe v `config/variants.lock.json`.
Změna parametrů existující varianty je chyba validace. Nová varianta dostane nové ID a zaregistruje se
příkazem `uv run arena lock-variants`.
