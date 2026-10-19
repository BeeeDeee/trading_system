# H2_btc200 – předem dané varianty výstupu

Verze 0.1 · 2026-10-19 · Popisná studie; pravidlo a rozhodnutí hlavního forward testu H2 se nemění.

## Otázka

Změní alternativní výstupy historické a budoucí výsledky H2 `LEAD_K3_BTC200`? Srovnávají se různé TP, SL,
kombinace bracket stop/cíle a trailing stop. Sada má 30 konfigurací včetně kontrolní základní varianty. Parametry
se po zobrazení výsledků neupravují a žádná varianta nerozhoduje o úspěchu H2 v `PREREGISTRATION.md`.

## Základní vstupy (beze změny)

Signál, BTC MA200 filtr, univerzum, 15m data, náklady a funding jsou totožné s H2 v `PREREGISTRATION.md` §§4–5.
Výchozí časový výstup nastává po 16 svíčkách (4 h). Všechny stop/cíl hodnoty se počítají od vstupní ceny.

## Zmrazená mřížka

| Skupina | Hodnoty |
|---|---|
| Kontrola | Bez TP/SL, pouze časový výstup po 4 h |
| Samotný cíl | +0,5 %, +1 %, +1,5 %, +2 %, +3 %, +4 % |
| Samotný stop | −0,5 %, −1 %, −1,5 %, −2 %, −3 %, −4 % |
| Stop + cíl | (−0,5 %, +1 %), (−1 %, +2 %), (−1,5 %, +3 %), (−2 %, +4 %), (−3 %, +6 %) |
| Trailing stop | aktivace 0 / +0,5 / +1 / +2 %; vzdálenost pod dosavadním maximem podle dvojic v `research18/exits.py` |

Trailing stop se aktivuje po dosažení aktivačního zisku. Poté se stop počítá jako dosavadní maximum high minus
stanovená vzdálenost. Nová úroveň se smí použít až od následující svíčky, aby se na jedné 15m svíčce nepoužila
informace o jejím high před vyhodnocením low. Pokud v jedné svíčce zasáhne stop i cíl, počítá se konzervativně stop.
Při gapu přes úroveň se plní za open, pokud je pro strategii horší než cena stopu/cíle. Všechny varianty mají stále
tvrdý časový výstup po 4 h.

## Vyhodnocení a omezení

- Výsledky se hlásí vedle původního H2: Sharpe, CAGR, max drawdown, počet obchodů, průměr obchodu a výsledky po
  letech; náklady/funding se počítají stejně jako v H2.
- 30 variant jsou průzkumná mřížka. Nejlepší historická varianta není potvrzená strategie a její výběr nepřepisuje
  H2. Zohlední se počet zkoušených variant.
- Když se mřížka spustí na datech forwardu, je to popisný výsledek; H2 forward zůstává posuzován výhradně podle
  `PREREGISTRATION.md` §§6–7.
- Intrabar pořadí stop/cíle nelze z 15m OHLC přesně zjistit; stop-first je záměrně konzervativní aproximace.

## Log rozhodnutí

| Datum | Změna |
|---|---|
| 2026-10-19 | Uživatelsky požadována samostatná mřížka TP/SL/trailing stop. Předem stanovena sada 30 konfigurací; žádné výsledky této mřížky nebyly při volbě parametrů použity. |
