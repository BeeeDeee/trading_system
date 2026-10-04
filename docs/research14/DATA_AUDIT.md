# Výzkum 14 – audit hodinových dat

Výstup `scripts/r14_build_panel.py binance_2026-10-03` (zdroj: `market_relations_2026_oct/data/raw/binance_1h/spot_1h`, jen čtení).

| Symbol | Zipů | Svíček | První hodina | Duplicity | Posunuté časy | Chybné OHLC | Chybějící hodiny | Mezer | Nejdelší (h) | Funding od | SHA256 zipů |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | 109 | 79117 | 2017-08-17T04:00:00 | 0 | 43 | 0 | 127 | 28 | 32 | 2020-01-01T00:00:00 | `c2d30b863d5bc148…` |
| ETHUSDT | 109 | 79117 | 2017-08-17T04:00:00 | 0 | 43 | 0 | 127 | 28 | 32 | 2020-01-01T00:00:00 | `b7756021a80fb486…` |
| BNBUSDT | 106 | 77181 | 2017-11-06T03:00:00 | 0 | 43 | 0 | 120 | 27 | 32 | 2020-02-10T08:00:00 | `d90b9b7b224de018…` |
| LTCUSDT | 105 | 76293 | 2017-12-13T03:00:00 | 0 | 43 | 0 | 120 | 27 | 32 | 2020-01-09T08:00:00 | `ff36aa37e57c5b88…` |
| ADAUSDT | 101 | 73325 | 2018-04-17T04:00:00 | 0 | 0 | 0 | 87 | 25 | 10 | 2020-01-19T16:00:00 | `4335fcfcbfec93ce…` |
| XRPUSDT | 100 | 72913 | 2018-05-04T08:00:00 | 0 | 0 | 0 | 87 | 25 | 10 | 2020-01-06T08:00:00 | `a7b517b43871dce1…` |
| ETCUSDT | 99 | 71993 | 2018-06-12T02:00:00 | 0 | 0 | 0 | 77 | 24 | 10 | 2020-01-16T08:00:00 | `75d5c39c9f339c8d…` |
| TRXUSDT | 99 | 71998 | 2018-06-11T11:00:00 | 0 | 0 | 0 | 87 | 25 | 10 | 2020-01-15T08:00:00 | `decbeb00bc7e066f…` |
| XLMUSDT | 100 | 72264 | 2018-05-31T09:00:00 | 0 | 0 | 0 | 87 | 25 | 10 | 2020-01-19T16:00:00 | `9874c85f055111a0…` |
| NEOUSDT | 106 | 76845 | 2017-11-20T03:00:00 | 0 | 43 | 0 | 120 | 27 | 32 | 2020-02-17T08:00:00 | `704c3fea5b06b9d1…` |
| EOSUSDT | 85 | 61231 | 2018-05-28T05:00:00 | 0 | 0 | 0 | 11196 | 26 | 11109 | 2020-01-08T08:00:00 | `17a91483d1fb7cc5…` |

- Posunuté časy (43 svíček) = restart Binance 2018-02-09, svíčky v :28; zaokrouhleno dolů na hodinu.
- EOSUSDT: dlouhá mezera = ukončení obchodování 2025-05 (jen sekundární koš, popisně).
