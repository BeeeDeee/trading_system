# Go-live runbook (P5 → P6 → P7)

**Aktuální rozhodnutí vlastníka (2026-10-25): obchoduje se jen na demo/testnetu (demo.binance.com), žádné reálné peníze.**
Platí kroky 0A, 1 a 3. Kroky 2, 4 a 5 (mainnet, P6/P7) jsou ODLOŽENÉ a vyžadují nový výslovný souhlas vlastníka.

Pořadí je záměrné. Každý krok má kritérium, bez kterého se nejde dál.

## 0A. Co musí udělat vlastník (demo, nyní)
1. Ed25519 pár na serveru: `sudo openssl genpkey -algorithm ed25519 -out /etc/arena/testnet_ed25519.pem`
   (chown arena, mode 600), veřejný klíč: `openssl pkey -in ... -pubout`.
2. demo.binance.com → API Management → Self-generated (Ed25519), čtení + futures (když Ed25519 nejde, HMAC:
   `BINANCE_TESTNET_API_SECRET`). One-way mode, demo zůstatek USDT.
3. `/etc/arena/env`: `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_PRIVATE_KEY_PATH`, `NTFY_TOPIC`.
4. `docs/PREREG.md`: doplnit datum startu a commitnout.
5. `uv run arena check` (default env testnet).

## 0B. Odloženo: mainnet (jen s novým výslovným souhlasem)
1. **Binance klíče** (Ed25519, viz D9):
   - testnet/demo (demo.binance.com → API): Ed25519 klíč, futures povoleno.
   - mainnet: nový Ed25519 klíč, oprávnění jen *Reading* + *Enable Futures*, **bez withdrawals**, IP whitelist na 62.83.71.235.
     Účet musí mít aktivní USDⓈ-M futures, **one-way mode**, kolaterál ≥ 100 USDT (P6: 1 slot), pro P7 ≥ 300 USDT.
2. `ntfy` topic (alerty) → `NTFY_TOPIC`.
3. Na serveru: `/etc/arena/env` + pem soubory (mode 600, vlastník arena), viz README.
4. `docs/PREREG.md` doplnit datum startu a commitnout.
5. Výslovný souhlas s mainnet live v konverzaci (CLAUDE.md) – před krokem 4.

## 1. Nasazení paper (bez live)
    git pull && uv run pytest            # 158 testů
    deploy/install.sh                    # server
    systemctl status arena ; journalctl -u arena -n 50
Kritérium: 24 h běh, `arena bars-check --hours 24` bez mezer, `arena book-validate` ≥ 0,9 (BTC/ETH musí projít, jinak stop).

## 2. (ODLOŽENO, mainnet) P0 akceptace: `arena check --env mainnet` (read-only) – musí projít celé, vč. one-way a oprávnění klíče.

## 3. P5 testnet (paralelně s krokem 1)
V `/opt/arena5m/config/arena.yaml`: `env: testnet` (už je), `live.enabled: true` (zapne vlastník/asistent až po klíčích); nasadit drop-in `deploy/arena-live.conf`.
Ověřit: `/fapi/v1/algoOrder` na demo, vazba `ALGO_UPDATE.ai` ↔ fill, `arena reconcile`, `kill -9` test, 200 round tripů,
žádná pozice bez tvrdého stopu > 2 s. Zapsat do `docs/VERIFIED.md` a `CHANGELOG.md`.
Když demo `algoOrder` nepodporuje: SPEC §17 P5 → přes mainnet s 1 slotem na min. notional (= P6).

## 4. (ODLOŽENO, mainnet) P6 mainnet, 1 slot (XRPUSDT, min. notional), 7 dní
V `config/arena.yaml`:
    env: mainnet
    live:
      enabled: true
      leverage: 2
      slots:
        XRPUSDT: {variant: brk.v00, notional_usdt: 10}   # ≥ minNotional (ověř `arena check`), 2x páka
    risk: {account_daily_loss_pct: 3, slot_daily_loss_pct: 3, slot_max_dd_pct: 15, max_live_positions: 1}
Pak: `arena check --env mainnet` → `sudo systemctl restart arena` (s drop-inem), sledovat dashboard + `journalctl -u arena -f`.
Nouzově: `arena kill` (zavře live, paper běží), `arena flatten --yes`.
Kritérium: ledger = income endpoint ± 0,01 USDT, latence a slippage v `costs.json`, žádný FOREIGN_POSITION / nechráněná pozice.

## 5. (ODLOŽENO, mainnet) P7: 5 live sond (původní sloty z repa), po 30 obchodech na symbol `arena calibrate` → vlastník potvrdí `cost_model_version: 2`.
