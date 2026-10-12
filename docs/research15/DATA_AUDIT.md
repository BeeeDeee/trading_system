# Výzkum 15 – audit 15m dat

Snapshot `binance_2026-10-03` · vytvořeno `scripts/r15_download.py` + `scripts/r15_build_panel.py --secondary` ·
2026-10-12. Všechny zipy ověřeny proti `.CHECKSUM` při stažení; surová data se nemění.

## Primární symboly

| Zdroj | zipů | první svíčka UTC | svíček | chybí | mezer | nejdelší mezera (svíček) | mimo mřížku | špatné OHLC | funding záznamů | SHA256 sady zipů |
|---|---|---|---|---|---|---|---|---|---|---|
| spot/BTCUSDT | 110 | 2017-08-17T04:00:00 | 319 294 | 562 | 32 | 133 | 81 | 0 | 0 | `c3b7f48ff7a057e4…` |
| spot/ETHUSDT | 110 | 2017-08-17T04:00:00 | 319 294 | 562 | 32 | 133 | 81 | 0 | 0 | `c3f61b06a44092ac…` |
| perp/BTCUSDT | 81 | 2020-01-01T00:00:00 | 236 640 | 0 | 0 | 0 | 0 | 0 | 7 395 | `32ec1313558cb766…` |
| perp/ETHUSDT | 81 | 2020-01-01T00:00:00 | 236 640 | 0 | 0 | 0 | 0 | 0 | 7 395 | `19239cf40749734d…` |

Spot BTC/ETH končí 2026-09-30 23:45 UTC. Mimo 15m mřížku je 81 svíček po výpadku Binance 2018-02-09 (čas
posunutý o 28 min, zaokrouhlí se dolů; duplicity nevznikají). Mezery spotu jsou krátké výpadky burzy, svíčka se
přenese (OHLC = předchozí close), příznak `real = False`. Perp BTC/ETH nemá žádnou mezeru. Funding BTC/ETH od
2020-01-01 po 8 h (7 395 záznamů), před tím paušálně 0,01 % / 8 h.

## Sekundární koš (105 perpů)

- Duplicity: žádné. Špatné OHLC: žádné. Svíčky mimo mřížku: žádné.
- Společné mezery ve zdrojových datech Binance u altcoinů: **2022-02-26 → 02-28 (288 svíček)** a
  **2022-04-01 → 04-02 (192 svíček)**; 26 perpů má přesně tyto dvě mezery (480 svíček). BTC/ETH je nemají.
- Ukončené nebo přejmenované kontrakty (po poslední svíčce `real = False`, neobchoduje se): EOS (do 2025-05-21),
  MATIC (2024-09-11), LUNA (2022-05-13), RNDR (2024-07-30), TOMO (2024-05-28), YFII (2022-04-12), ostatní viz
  `runs/r15_audit_all.json`.
- `HYPEUSDT` nemá v datech žádný funding → pro celé období paušál 0,01 % / 8 h.
- Koš je jen popisný (prereg §9).

Pozn.: celý audit (po symbolech) je v `runs/r15_audit_all.json` (mimo git, `runs/` je ignorováno).
