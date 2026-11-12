# Výzkum 23 – VWAP Wave (Drysdale): čtyři setupy kolem session VWAP a jejích pásem (pre-registrace)

Verze 1.0 · 2026-11-12 · Stav: **návrh**. Po commitu verze 1.0 platí: každá změna = nový pokus v registru a
řádek v §12.

## 1. Zdroj a otázka

Zdroj: leták „VWAP Wave – Core Setup Guide“ (Chris Drysdale, Drysdale Trading Group, 7 stran, soubor
`vwap_strategy.pdf` v pracovním adresáři vlastníka, necommitnutý). Popisuje 8 principů a 4 diskreční setupy na ES
(S&P 500 E-mini) 5m s VWAP a jejími směrodatnými pásmy („value area“). Leták **neobsahuje žádná čísla výkonnosti**
(win rate, P&L, počet obchodů); ukázkové grafy jsou vybrané dny z 18. 3. 2025. Tvrzení „velmi výdělečné“ pochází
z marketingu, ne z dokumentu.

**Otázka:** Má některý ze 4 setupů, převedený na pevná pravidla, čistý zisk po nákladech, který **nevysvětlí
náhodné vstupy téhož dne, se stejnou stranou a stejnou vzdáleností stopu a cíle**, a přežije dvojnásobné náklady
a plnění o svíčku později?

## 2. Poctivá omezení předem

- **Trh.** Leták cílí na ES. Intradenní data US indexu nemám (Dukascopy vrací 429, placené zdroje čekají na
  rozhodnutí vlastníka). Tato verze testuje **BTCUSDT a ETHUSDT perpy 5m** (Binance, skutečný objem). Krypto není
  ES: nemá RTH session ani otevírací aukci, „session“ je UTC den. Negativní výsledek na kryptu **nevyvrací**
  tvrzení pro ES. Pozitivní výsledek je jen kandidát. Test na ES/SPX bude samostatný dodatek (v1.1) se stejnými
  pravidly a vlastním holdoutem a jeho pokusy se přičtou do N.
- **Diskreční → pevná pravidla.** „Acceptance“, „first sign of strength“, „a few points“ a „band area“ jsou
  v letáku nedefinované. Moje převedení (§4) je jedno z mnoha možných. Volby dělám předem, podle textu a grafů,
  bez pohledu na data. Výsledek platí pro tuto operacionalizaci, ne pro „diskrétního tradera“.
- **Data jsou spotřebovaná.** Holdout 2023-01 → 2026-09 na BTC/ETH viděly výzkumy 15–17 a Lab (jiné rodiny,
  15m). Pásma VWAP nebyla testována nikdy. Historický test může rodinu jen **zamítnout, nebo poslat na forward**.
- **Apriorní očekávání:** obchody trvají minuty až hodiny. Cíle na BTC 5m jsou typicky 0,2–0,6 %, náklady
  15 bps na kolo sežerou velkou část. Realisticky čekám 0 setupů přes holdout. Výzkum 16 ukázal, že krátké držení
  na BTC/ETH při těchto nákladech prodělává ve všech rodinách.

## 3. Data a etapy

- Binance USDⓈ-M perpy BTCUSDT, ETHUSDT, 5m klines ze snapshotu `binance_2026-10-03`
  (`data/raw/binance_2026-10-03/perp_5m/`), 2020-01-01 → 2026-09-30, sloupce o, h, l, c, objem (base),
  quote objem. Chybějící svíčka: close se přenese, objem 0, `real = False`. Funding z panelu Labu
  `lab_intraday_5m` (sazba vypořádaná na začátku svíčky).
- **Vývoj:** 2020-01-01 → 2022-12-31. **Holdout:** 2023-01-01 → 2026-09-30 (trezor `runs/vault_r23/`,
  otevřen jednou). Podúseky holdoutu: 2023-01..2024-12 a 2025-01..2026-09.

## 4. Pravidla (vše z dat uzavřených do konce 5m svíčky *t*; rozhodnutí po close *t*)

**Session a VWAP.** Session = UTC den (00:00–23:55, 288 svíček), kotva 00:00 UTC (výchozí kotva „Session“
VWAP v TradingView na kryptu). Cena svíčky `p_i = quote_volume_i / volume_i` (přesný VWAP svíčky; bez objemu
`(h+l+c)/3`). Od začátku session do *t*: `V = Σv·p / Σv`, `σ = sqrt(Σv·p² / Σv − V²)`. Pásma `U = V + kσ`,
`L = V − kσ`; **value area** = [L, U]. Primárně **k = 1**. Svíčky bez objemu do součtů nevstupují.

**Společné konstanty (pevné, mimo mřížku).** ATR = průměr true range 14 posledních svíček (přes hranice
session). Buffer stopu `buf = 0,25·ATR` („a few points“: na ES ≈ 2 body při 5m ATR ≈ 8). Tolerance blízkosti
`tol = 0,1·kσ` („band area“). Vstupní okno: pozice svíčky v session 6 až 275 (bez prvních 30 min, kdy jsou
pásma degenerovaná, a bez poslední hodiny). Signál jen na reálné svíčce s `σ > 0`. „Síla“ (long) = close nad
high referenční svíčky, „slabost“ (short) = close pod low referenční svíčky.

Popis je pro long. Short je zrcadlový (U↔L, high↔low, nad↔pod). Obě strany se obchodují vždy (leták je
obousměrný). Stavy se nulují na začátku session.

| Setup | Kontext | Vstup (mřížka *E*) | Stop | Cíl (mřížka *X*) |
|---|---|---|---|---|
| **S1 PDC** – Price Discovery Continuation | průraz z value area: close > U | `BRK`: první close > U (předchozí close ≤ U). `PB`: po „acceptance“ (≥ 3 closes > U v řadě, **nebo** close ≥ V + (k+0,5)σ) svíčka *b* s low ≤ U + tol = backtest. Signál = první close > high[*b*] v (*b*, *b*+6]. Nová backtest svíčka (další dotyk) nahradí *b*. Close < V ruší acceptance i backtest. | `BRK`: low[t] − buf. `PB`: min low od *b* − buf | `EOD` (jen stop a konec session) / `2R` |
| **S2 FADE** – Fade Value Area Extremes | ≥ 6 posledních closes (do *t*−1) uvnitř [L, U] | test *b*: high ≥ U − tol a close ≤ U (short). `TST`: signál hned na *b*. `REJ`: první close < low[*b*] v (*b*, *b*+3]; close > U test ruší. | max high od *b* + buf | `VWAP` (V při signálu) / `OPP` (L při signálu) |
| **S3 RTV** – Return to Value | předtím ≥ 3 closes < L v řadě a v té řadě low ≤ L − 0,5σ (čas **a** vzdálenost) | návrat: close > L po close < L. `BRK`: signál na návratu. `PB`: po ≥ 2 closes ≥ L svíčka *b* s low ≤ L + tol a close ≥ L; signál = první close > high[*b*] v (*b*, *b*+6]. Close < L ruší (zpět venku). Platnost 24 svíček od návratu. | `BRK`: low[t] − buf. `PB`: min low od *b* − buf | `VWAP` / `OPP` (U) |
| **S4 BOUNCE** – VWAP Bounce | v posledních 24 svíčkách proběhl návrat do value area (close uvnitř po close venku, kterékoli pásmo); close[t−1] > V a uvnitř [L, U] | test *b*: low ≤ V + tol a close > V. `TST`: signál na *b*. `STR`: první close > high[*b*] v (*b*, *b*+3]; close < V ruší. | min low od *b* − buf | `BAND` (U při signálu) / `2R` |

**Mřížka:** každý setup 2 vstupy × 2 cíle = **16 variant** při k = 1. **Robustnost** (počítá se do N):
primární variantu každého setupu přepočítám s k = 2 (4 varianty). **N = 20.** Kumulativně se do DSR hlásí i
N = 20 + 46 (výzkumy 15–17 na týchž BTC/ETH datech).

**Primární varianty (podle textu letáku, předem):** S1 `PB_EOD` („let the trade breathe“), S2 `REJ_VWAP` („VWAP
serves as a key magnet“), S3 `PB_VWAP` (principy 3–4), S4 `STR_BAND`.

## 5. Obchod

- Plnění za close svíčky signálu *t* (v nepřetržitém trhu prakticky open *t*+1). Stop a cíl se hlídají od *t*+1:
  stop má přednost ve stejné svíčce, mezera přes stop se plní za open. Cíl je limit: plní se za cenu cíle,
  nebo za open, pokud je open za cílem. Vynucený výstup za close poslední svíčky session (23:55 UTC).
  `2R` = vstup ± 2·|vstup − stop|. Cíl `VWAP/OPP/BAND` je pevná cena určená při signálu.
- Signál se vynechá, pokud stop nebo cíl leží na špatné straně plnění (risk ≤ 0 nebo zisk do cíle ≤ 0).
  **Žádný filtr minimálního rizika** (leták žádný nemá; náklady to potrestají samy).
- Na aktivum a variantu nejvýš jedna pozice. Signály během pozice se ignorují, další vstup nejdřív svíčku po
  výstupu. Počet obchodů za den není omezen.
- **Náklady 7,5 bps na stranu** (taker 5 + skluz 2,5, jako výzkumy 15–16), zátěž ×2. Funding se platí na
  svíčkách vypořádání, přes které se pozice drží. Pozice = 100 % rukávu, portfolio 50/50 BTC/ETH, denní výnos
  = součet čistých výnosů obchodů dne (všechny obchody končí v den vstupu).

## 6. Nula a benchmarky

- **B_RND (hlavní nula, 500 běhů, seed 0..499):** pro každý obchod varianty náhodný vstup **téhož UTC dne**
  ve vstupním okně, na reálné svíčce bez otevřené pozice, **se stejnou stranou, stejnou relativní vzdáleností
  stopu a stejnou relativní vzdáleností cíle** (u `EOD` bez cíle), výstup podle stejných pravidel a nejpozději na
  konci session. Testuje se načasování vstupu v rámci dne, ne strana, volatilita dne ani tvar výstupu.
  p = (1 + #{běhy s průměrným čistým obchodem ≥ P}) / (1 + 500).
- Reportuje se i hrubý výsledek (náklady 0) a long/short zvlášť.

## 7. Vývojová brána (2020-01 → 2022-12)

Pro každý setup vyberu **P_f** = variantu s nejvyšším vývojovým Sharpe (denní čisté výnosy × √365) mezi 4
variantami při k = 1. Setup postupuje na holdout, jen když P_f splní **všechno**: Sharpe > 0, průměrný čistý
obchod > 0, ≥ 200 obchodů, p B_RND ≤ 0,10. Holdout se neotevře, pokud bránou neprojde nic.

## 8. Kritéria holdoutu (pro každý postupující setup, P_f)

1. Sharpe > 0 a 90% bootstrap interval průměrného čistého obchodu (iid, 2000, seed 0) celý nad 0.
2. B_RND p ≤ Holmova hranice (α = 0,05 přes postupující setupy).
3. Sharpe > 0 při nákladech ×2.
4. Sharpe > 0 při plnění o svíčku později (svíčka *t*+1; signál se zahodí, pokud mezitím padl stop).
5. Sharpe > 0 v obou podúsecích holdoutu.
6. Průměrný čistý obchod > 0 na BTC i na ETH.

**Verdikt:** setup splní všech 6 → **kandidát na forward test** (papírový, 6 měsíců, samostatná pre-registrace).
Jinak zamítnuto. DSR (N = 20 a 66) se reportuje, verdikt nemění.

## 9. Robustnost (po §8, verdikt nemění)

Všech 20 variant na holdoutu (popisně), k = 2 pro primární varianty, hrubé výsledky, long vs short, rozpad
podle roku, podle výstupu (stop / cíl / konec session), podle hodiny vstupu (UTC) a podle volatility dne
(tercily ATR, protože leták tvrdí „higher volatility days are better“ u S4).

## 10. Implementace

`src/qlab/research23/` (`data`, `vwap`, `setups`, `engine`, `evaluate`), `scripts/r23_dev.py`,
`scripts/r23_final.py freeze|run`, testy `tests/test_research23.py` (point-in-time pásem a signálů:
oříznutí budoucnosti nemění minulé signály; simulátor na ručních scénářích; nula zachovává počty, strany a dny).
Registr `runs/registry/research23.sqlite`.

## 11. Lab

Stejné čtyři setupy jdou jako položky fronty vlastníka (`lab/briefs/owner_queue.yaml`, VWAP-C1..C4, jedna
rodina `session-vwap-bands-crypto`). Lab je otestuje nezávisle svým enginem a branami (G0–G4) na
`binance_perp_5m`. Lab má jiné náklady, plnění za open *t*+1 a jiný vesmír, takže výsledky nebudou identické.

## 12. Změny po verzi 1.0

| Datum | Změna | Důvod |
|---|---|---|
