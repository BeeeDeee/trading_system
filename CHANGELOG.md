# CHANGELOG

Stav akceptace (SPEC §17) k 2026-10-23. ✅ = splněno a ověřeno, ⏳ = čeká na nasazení, klíče nebo dlouhý běh.

| Fáze | Kód + testy | Akceptace |
|---|---|---|
| P0 | ✅ | ⏳ `arena check --env mainnet` s read-only klíčem (spustí vlastník) |
| P1 | ✅ | ⏳ 24h běh, bary vs REST, `book-validate` ≥ 0,9 (po nasazení na novém serveru) |
| P2 | ✅ | ✅ |
| P3 | ✅ | ⏳ 7 dní paper bez pádu |
| P4 | ✅ | ⏳ stránka za Caddy s basic_auth (install.sh na novém serveru) |
| P5 | ✅ (falešná burza) | ⏳ testnet: klíče, 200 round tripů, `kill -9`, ověřit algoOrder na testnetu |
| P6, P7 | – | – (mainnet live jen s výslovným souhlasem vlastníka) |

## Exec probe (rodina xp), 2026-10-25

- Důvod: strategie dávají ~7 live vstupů denně na 5 slotů, takže 200 round tripů z P5 by trvalo týdny. `xp.v00` vstupuje
  náhodně (RNG z hashe slotu a baru, deterministické) s p = 0,15 na bar ve stavu flat, ≈ 100 round tripů denně na 5 symbolů.
- Rotuje 5 výstupních mechanismů, aby každý symbol procvičil všechny cesty: trailing od vstupu, trailing s aktivací
  (`activatePrice`/`-2021`), jen tvrdý stop + časový výstup (bez trailingu), TP LIMIT, úzký trailing + úzký stop.
- Mimo experiment: stats, žebříček, heatmapa, DSR, `calibrate`, `costs.json`, „dnes“ a signály 24 h ignorují `xp.*`
  (nepočítá se do N). Validace odmítne `xp` s `env: mainnet` (před mainnetem smazat `config/strategies/xp.yaml`).
- `config/arena.yaml`: všech 5 live slotů = `xp.v00` (100 USDT), slotové limity risk guardu uvolněné na 100 %
  (probe záměrně ztrácí náklady), účetní limit 3 % platí. Po akceptaci P5 vrátit strategie (viz komentář v souboru).
- Testy: `tests/test_probe.py` (determinismus, jen flat, všechny mechanismy dávají platné úrovně přes `plan_exits`,
  vyloučení ze statistik, zákaz na mainnetu). 163 testů.

## P5: live executor, rekonciliace, risk, income (kód; testnet čeká na klíče)

- `execution/live.py`: stavový automat ENTRY_SENT → PROTECTING → PROTECTED → EXITING (+ EMERGENCY_CLOSE).
  Vstup MARKET `RESULT`; po timeoutu dotaz podle `origClientOrderId`, nikdy druhý příkaz. Ochrana: tvrdý
  `STOP_MARKET` a `TRAILING_STOP_MARKET` přes `/fapi/v1/algoOrder` (reduceOnly, `CONTRACT_PRICE`,
  quantity = executedQty), ≤ 3 pokusy a ≤ 2 s, jinak EMERGENCY_CLOSE + alert. Trailing `-2021` s activatePrice
  → znovu bez aktivace. TP LIMIT reduceOnly GTC (přeskočí se, když by se hned vyplnil). Výstup přes
  `ALGO_UPDATE` (`ai` → orderId fillu) a `ORDER_TRADE_UPDATE` (fee `n`, `rp`), zrušení sourozenců. Časový výstup
  a kill: nejdřív zrušit algo a TP, pak MARKET reduceOnly; `-2022` → rekonciliace. Stínový TrailingSim na
  aggTrades dává HWM, takže `exit_trigger_px` trailingu = teoretická úroveň (§12). Latence: rozhodnutí → odeslání,
  odeslání → ack, ack → fill, trigger → fill. Intent → úrovně přes stejnou `plan_exits` jako paper.
- `reconcile.py` (§9): FOREIGN_POSITION → zmrazit, chybějící HS/TS → doplnit, pozice zmizela → výstup z
  `userTrades` (`reconciled`), osiřelé `a5-` příkazy zrušit, cizí nikdy, nesoulad množství → zmrazit;
  nedokončený vstup po pádu (ENTRY_SENT) se dohledá a ochrání nebo zahodí.
- `risk.py` (§10): limity slotu a účtu (alokace slotu = jeho live notional), live_halt s ručním resetem
  (`arena risk-reset`), zmrazení symbolu, stale data, private WS > 30 s, live jen s `live.enabled` a `--live`.
- `live_runtime.py`: listenKey (keepalive 30 min, `listenKeyExpired` → nový klíč + rekonciliace),
  `/private/ws/<listenKey>`, startovní kontrola one-way + ISOLATED + páka (nastaví jen bez pozice), rekonciliace
  60 s, equity 60 s, income 10 min a ledger vs income (> 0,01 USDT → varování), kill switch 1 s.
- CLI: `reconcile [--fix]`, `kill`, `unkill`, `flatten` (s potvrzením), `risk-reset`, `stats`, `calibrate`.
- Testy (`tests/test_live.py`, falešná burza): happy path, selhání tvrdého stopu → emergency close, ochrana
  > 2 s → emergency close, `-2021`, timeout vstupu (dotaz, ne nový příkaz), `kill -9` po vstupu i po odeslání
  vstupu → restart doplní ochranu, osiřelé `a5-` zrušeny a cizí ne, FOREIGN_POSITION, nesoulad množství, časový
  výstup (zrušení algo před market), race `-2022` → `reconciled`, 200 round tripů s náhodnými chybami bez jediné
  pozice bez tvrdého stopu. Žádný podmíněný typ nikdy nešel na `/fapi/v1/order`.
- Pole `ALGO_UPDATE` podle oficiálního connectoru (docs/VERIFIED.md); empiricky ověřit na testnetu.

## P4: statistiky, snapshot, dashboard, nasazení (kód; nasazení po migraci)

- `stats.py` (§13): n, obchody/den, expozice, držení, mix výstupů, gross/net (průměr, medián, sd, win rate),
  MFE/MAE, 95% CI blokovým bootstrapem po dnech (2000, fixní seed), náklady v bps (fee, slip, funding),
  break-even, net při 1,5× a 2× nákladech, denní P&L → Sharpe, max DD, Calmar, Δ vs kontrola s bootstrap CI a
  jednostranným p, Holm, DSR (N = počet variant), n_req pro t = 2 a 3 a zbývající dny, badge podle §13/§14.2.
  Ověřeno ručním výpočtem na malém vzorku (`tests/test_stats.py`).
- `snapshot.py`: `snapshot.json` každých 15 s (atomicky), `variants/<id>.json` a `costs.json` každých 5 min.
  Statistiky se počítají ve vlákně s vlastním DB spojením, aby bootstrap neblokoval čtení WS.
- `dashboard/` (vanilla JS, Chart.js 4.4.1 z cdnjs): Přehled, Žebříček (upozornění na výběrové zkreslení,
  heatmapa varianta × symbol, šedá při n < 30), Detail varianty, Náklady, Log. Tmavý i světlý režim, tabulky
  scrollují ve vlastním kontejneru, červený pruh při snapshotu starším než 60 s, přepínač „bez tainted“.
- `deploy/`: `arena.service` (Restart=always, log do `/srv/arena/log` s rotací 50 MB × 3), denní
  `arena-book-validate.timer` (03:30 UTC), idempotentní `install.sh`, `patch_caddyfile_arena.py`
  (basic_auth na `/arena/*` převezme přihlašovací řádek webu), `patch_portal_arena.py` (záložka Arena).
  Oba patche vyzkoušeny na kopiích aktuálních souborů, upravený Caddyfile prošel `caddy validate`.

## P3: paper executor, TrailingSim, funding, replay

- `TrailingSim` přesně podle §7 (aktivace, HWM/LWM, tvrdý stop, TP striktně za cenou, při stejném čase obchodu
  vyhrává horší výsledek). Paper: fill vstupu v t_d + 250 ms přes lokální knihu (stale → bookTicker / poslední
  obchod + penalizace), výstup trigger + 100 ms, TP maker za limit, časový výstup, funding z markPrice (`r` těsně
  před časem fundingu), výpadek: přehrání aggTrades z REST (okno ≤ 1 h + stránkování fromId), fill ± 2 bps,
  `tainted = 1`. Stav pozic v `slot_state`, cooldown a pořadová čísla obchodů se obnoví z DB.
- `slots.py`: SlotRunner (featury → strategie → signály → executor), sdílený live během i replayem.
  Signály se zapisují s akcí `entered | suppressed_position | suppressed_cooldown`. Bar bez knihy u `obi` jde do
  `events` (ne jako řádek signálu pro každý z 50 slotů).
- `replay.py` + `arena replay --from --to [--ticks DIR] [--book DIR] [--src-db]`: stejná cesta nad nahranými
  daty, fill z prvního obchodu po latenci (bez knihy), rozhodnutí 500 ms po uzavření baru (živě změřeno
  0,2–1,4 s). Test: dva běhy → identické obchody (otisk SHA-256).
- Testy: TrailingSim na syntetických cestách, paper (latence, poplatky, TP, časový výstup, funding, výpadek,
  obnova stavu), replay determinismus.

## P2: featury, 4 rodiny, registry

- `features.py`: všechny featury §5.1 jako čisté funkce nad numpy. Konvence, kde SPEC nechává volnost:
  sd a z-skóre s ddof = 0 a ignorováním NaN (jako research), z-skóre jen s plným oknem 288; `sigma288` a `beta288`
  přes 288 barů **bez** aktuálního; `lvz288` z ln(quote volume); ATR (okno 288) a EMA (okno 2n) mají pevné okno
  se seedem na začátku, takže live i replay dají totéž.
- Strategie brk/obi/ll/ctl, registry: 140 strategických slotů + 14 kontrol, globální warmup 1440 barů.
- `arena backfill-book --days N` (živá data mají přednost, INSERT OR IGNORE).
- Testy: featury proti naivní referenční implementaci na fixním CSV (BTC/ETH, 1500 barů), look-ahead test
  (změna barů > t nemění featury ani signály ≤ t), signál ano/ne pro každou z 33 variant, `obi` z `bookDepth`
  bitově shodné s algoritmem `research/obi_prior/analyze.py` (2 dny BTC) a `obi1_z` shodné na barech s plným oknem.

## P1: WS, BarStore, backfill, lokální kniha, recorder, book-validate (kód; 24h běh po nasazení)

- WS klient (`binance/ws.py`): routované URL, `stream` mód, reconnect s backoffem, rotace po 23 h, idle
  timeout, vlastní ping 20 s; čtení přes `async for` + hlídací task.
- **Tržní data vždy z mainnetu** (`data_env: mainnet`), `env` určuje jen obchodní účet. Výchozí `env: testnet`
  by jinak bral bary a knihu z testnetu, kde neplatí validace proti `bookDepth` ani priory `obi`.
- **depth a bookTicker na oddělených spojeních.** Změřeno 2026-10-23: spojení s bookTicker Binance shazuje
  (3–15× za minutu i u nečinného klienta, ~2 % CPU), samotný `depth@100ms` za 10 min ani jednou. Na společném
  spojení by každé shození znamenalo resync knihy a ztracené vzorky.
- BarStore: jen `x == true`, warmup okno při startu, detekce mezer, nedoplněná díra drží symbol `stale`
  a zkouší se znovu. Koordinátor uzavření baru (všechny symboly nebo 3 s), bary z backfillu spustí rozhodování
  jen v okně čekání. Finální kline chodí typicky 0,2–0,5 s po hranici, ojediněle až ~3 s.
- Lokální kniha podle postupu Binance, resync při mezeře v `pu` nebo překřížení; preventivní resnapshot
  přes stínovou knihu (vzdálené úrovně mimo rozsah snapshotu se převezmou z běžící knihy).
- Recorder: aggTrades do `{raw}/aggTrades/{SYM}/{date}.csv.gz` (formát data.binance.vision, flush ≤ 5 s,
  Z_SYNC_FLUSH, po restartu `partN`), pásma knihy každých 30 s na :00/:30, `bar_book` (obi = průměr vzorků,
  `book_ok` ≥ 8/10).
- `arena book-validate --date D`, `arena universe-check`.
- Živé ověření (krátké běhy, mainnet veřejná data): 5 knih synchronizovaných, 0 resynců za 10 min, 10/10
  vzorků na bar, bary bez mezer. **24h akceptace teprve po nasazení.**

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
