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

## Nasazení na nový server
```
git clone -b arena5m git@github.com:BeeeDeee/trading_system.git ~/ccode/arena5m
cd ~/ccode/arena5m && uv sync && uv run pytest
deploy/install.sh            # sudo, idempotentní: uživatel arena, /srv/arena, systemd, Caddy /arena/, portál
sudo -e /etc/arena/env       # klíče (viz výše); pak: sudo systemctl restart arena
uv run arena backfill-book --days 3 --data-root /srv/arena   # volitelně: obi nemusí čekat 288 barů (jako arena)
```
Data (`/srv/arena/arena.db`, `/srv/arena/raw`) nejsou v gitu. Při migraci běžícího experimentu je zkopíruj
po zastavení služby (`systemctl stop arena`), jinak se `tainted` označí obchody přes výpadek.

## Provoz
- `systemctl status arena`, log `/srv/arena/log/arena.log`, dashboard `/arena/`.
- `arena kill` / `arena unkill`: kill switch (live se zavře, paper běží dál).
- `arena reconcile [--fix]`, `arena flatten`, `arena risk-reset`, `arena stats`, `arena calibrate`.
- Live jen s `live.enabled: true` v `config/arena.yaml` **a** `arena run --live` (v unitu `ExecStart`).
- `env` = obchodní účet (testnet/mainnet), `data_env` = tržní data (vždy mainnet, jen veřejná volání).
