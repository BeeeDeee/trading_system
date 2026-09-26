# Předběžná zjištění o datech Sharadar

Datum: 2026-09-26 · Snapshot: `data/raw/sharadar_2026-09-25/` · Stav: **předběžné**. Formální
datový audit (spec §4.6) je součástí Fáze 1; tento dokument shrnuje, co víme z dokumentace Sharadaru
a z malých tabulek (tickers, actions, sp500, events, metrics, descriptions). Cenové tabulky
(stocks, funds, daily) zatím analyzované nejsou.

## 1. Snapshot

| Tabulka | Zip | Poznámka |
|---|---|---|
| stocks (SEP) | 945 MB | ceny akcií, od 1997-12-31 do 2026-09-25 |
| funds (SFP) | 291 MB | ETF, fondy **a indexy** (`^GSPC`, `^VIX`, `^RUT`, `^IXIC`, `^DJI`) |
| daily | 731 MB | denní market cap, EV, valuační poměry |
| fundamentals (SF1) | 631 MB | ve v1 nepoužito |
| holdings (SF3), insiders (SF2), holdings_ticker, holdings_investor | 849 MB | ve v1 nepoužito, staženo do zásoby |
| tickers | 5 MB | 74 271 řádků, z toho 20 989 pro SEP |
| actions | 11 MB | 708 997 corporate actions, 31 typů |
| sp500 | 0,3 MB | kvartální snímky od 1998-03 + události added/removed |
| events | 12 MB | 8-K event kódy od 1993 |
| metrics | 1,5 MB | **jen aktuální stav**, pro backtest nepoužitelné |
| descriptions | 20 kB | datový slovník |
| FRED `DTB3` | – | 3M T-bill, 1954 → 2026-09-24 (`data/raw/fred_2026-09-25/`) |

- Stažení: `scripts/download_sharadar.sh` (bulk API `api.sharadar.com/v1.0`, `years=full`).
  Kontrolní součty v `SHA256SUMS`.
- Předplatné končí 2026-09-27; data se dál neaktualizují. Pro paper/live trading by se předplatné
  obnovilo a stejný skript stáhne nový snapshot.
- Licence Personal Use pokrývá výzkum, backtesting i obchodování vlastního účtu.

## 2. Potvrzené vlastnosti (z dokumentace Sharadaru)

| Téma | Zjištění | Dopad na spec |
|---|---|---|
| OHLC | `open/high/low/close` upravené o splity, **ne** o dividendy a spinoffy | Potvrzuje P1 |
| `closeadj` | Upravené o splity, dividendy **i spinoffy** (spinoff: prodej na open, zpětný nákup parenta na open) | P6 odpadá; spinoffy řeší TR faktor |
| `closeunadj` | Neupravené | Filtry ceny z něj (P1) |
| Neupravené OHL | `open_u = open × closeunadj / close` | Potvrzuje §4.4 |
| Objem | Upravený o splity (historie × poměr splitu); `volume_u = volume × close / closeunadj`; `close × volume` = skutečný dollar volume | Dollar volume lze počítat přímo |
| TR open | `open_adj = open × closeadj / close` | Potvrzuje P2 |
| Úpravy | Zpětné (backward); dnešní upravená cena = tržní cena | Upravené ceny se mezi snapshoty mění → neměnný snapshot nutný |
| Tickery | Recyklované; delistovaný dostane číselný suffix; `permaticker` je stabilní ID třídy akcií | Potvrzuje P4 |
| `exchange` v tickers | Poslední primární burza, ne historie | Potvrzuje P3 |
| Fundamenty | `ARQ/ARY/ART` point-in-time podle data podání u SEC, `MR*` restatované | Potvrzuje P9 |
| Survivorship | Sharadar odhaduje 99 % pokrytí delistovaných titulů | OK |

## 3. Zjištění z tabulek

### 3.1 Securities master (tickers, table = SEP)

- 20 989 titulů, `permaticker` i `ticker` jsou v rámci SEP unikátní.
- Kategorie (kolik titulů / z toho delistovaných):

| Kategorie | Titulů | Delistováno |
|---|---|---|
| Domestic Common Stock | 13 528 | 10 316 |
| Domestic Common Stock Primary Class | 2 197 | 1 039 |
| Domestic Common Stock Secondary Class | 1 338 | 1 007 |
| ADR Common Stock (+ Primary/Secondary) | 2 285 | 1 264 |
| Domestic / ADR / Canadian Preferred | 1 237 | 780 |
| Canadian Common Stock (+ třídy) | 404 | 245 |

- **Dvě třídy akcií jedné firmy** (např. GOOGL/GOOG) mají kategorie Primary Class a Secondary Class.
  Do univerza patří jen Primary Class, jinak se stejná firma objeví dvakrát.
- `exchange` u delistovaných ukazuje poslední burzu (NASDAQ, NYSE, NYSEMKT), ne „DELISTED".
  Pole `isdelisted` je snapshot a nesmí se použít (P3).

### 3.2 Indexy a VIX

`funds` obsahuje kategorii `IDX`: `^GSPC` (S&P 500), `^VIX`, `^RUT`, `^IXIC`, `^DJI`, vše od
1997-12-31. VIX je tedy k dispozici pro ML features (§11.3). Dále SPY (od 1997-12-31), RSP
(od 2003-05-01), IWM (od 2000-05-26).

### 3.3 Corporate actions

31 typů akcí. Důležité pro simulaci:

| Akce | Počet | Význam pole `value` |
|---|---|---|
| dividend | 555 746 | hotovostní dividenda upravená o splity; `date` = ex-date |
| split | 12 985 | nový počet akcií na 1 starou; `date` = účinnost |
| delisted | 18 239 | **konečná tržní kapitalizace**, ne výplata; `date` = poslední obchodní den |
| acquisitionby | 8 111 | konečná tržní kapitalizace kupovaného |
| acquisitioncash | 5 319 | **hotovost na akcii (USD)** |
| acquisitionstock | 2 437 | **počet akcií kupujícího na 1 akcii** |
| bankruptcyliquidation | 3 329 | konečná tržní kapitalizace |
| regulatorydelisting / voluntarydelisting | 980 / 435 | konečná tržní kapitalizace |
| spinoff / spinoffdividend | 572 / 525 | poměr akcií / USD hodnota na akcii parenta |
| spacmerger / spacunitseparation | 818 / 1 409 | SPAC události |
| exchangefrom/to | 3 895 | **historické změny burzy** |
| sicchangefrom/to | 2 929 | **historické změny SIC kódu** |
| tickerchangefrom/to | 13 131 | změny tickeru |

Poznámky:
- `split` obsahuje i budoucí data (ohlášené splity s datem po 2026-09-25). Při stavbě časové řady
  se akce po posledním obchodním dni ignorují.
- Historie burzy a SIC kódu umožňuje rekonstruovat stav k datu `t`. FAQ Sharadaru tvrdí, že historie
  burzy není k dispozici; akce `exchangefrom/to` ji ale částečně obsahují. Ověří audit.

### 3.4 Důvody delistingu (Domestic Common Stock, 12 362 delistovaných)

| Důvod | Počet | Terminální hodnota |
|---|---|---|
| Akvizice (`acquisitionby`) | 6 917 | Z toho 4 988 s hotovostí na akcii a 2 298 s akciovou protihodnotou → **přesná výplata** |
| Fúze (`mergerto`) | 118 | Stejně jako akvizice |
| Bankrot / likvidace | 3 129 | Bez údaje o výplatě |
| Regulatorní delisting | 826 | Bez údaje |
| Dobrovolný delisting | 216 | Bez údaje |
| Bez uvedeného důvodu | 1 156 (9 %) | Bez údaje |

### 3.5 SPAC (blank-check společnosti)

- 1 641 titulů s aktuálním SIC 6770 (blank checks), 1 433 domácích akcií s akcí `spacmerger` nebo
  `spacunitseparation`. Většina od roku 2020 (vlny 2021, 2025–26).
- Před fúzí se SPAC obchoduje kolem 10 USD (hodnota trustu) s téměř nulovou volatilitou a často
  vysokým objemem.
- **Riziko 1:** strategie `low_vol` by vybírala SPAC schránky jako „nejméně volatilní akcie".
- **Riziko 2:** likvidace SPAC (vrácení ~10 USD z trustu) je v actions označená jako
  `bankruptcyliquidation` (např. AACB, ANSC v 2026). Pravidlo „bankrot = −100 %" by tu bylo hrubě
  špatně.
- Po fúzi SPAC změní SIC kód (`sicchangefrom 6770`), takže status SPAC k datu `t` lze určit z historie.

### 3.6 Ostatní tabulky

- `sp500`: kvartální snímky složení od 1998-03-31 + denní události `added`/`removed` (i starší,
  od 1957). Členství k datu `t` je rekonstruovatelné.
- `metrics`: jen poslední stav pro každý ticker (30 917 řádků). **Nepoužívat.**
- `events`: 2,5 mil. záznamů 8-K event kódů. Ve v1 nepoužito.

## 4. Dopady na specifikaci (zapracováno v §4 a §5)

1. P6 (spinoffy) vyřešeno – `closeadj` je obsahuje.
2. Politika delistingu (§4.5): akvizice podle `acquisitioncash`/`acquisitionstock`; likvidace SPAC
   jako výplata hodnoty trustu (poslední cena), ne −100 %.
3. Univerzum (§5.1): vyloučit SPAC (SIC 6770 k datu `t`), jen Primary Class u dvou tříd akcií.
4. VIX a indexy potvrzeny (§4.2).
5. `metrics` vyřazeno.

## 5. Otevřené pro formální audit (Fáze 1)

- Cenová data: pokrytí po letech, duplicity, `open = 0`, extrémní výnosy, chování `closeadj/close`
  v ex-div den, soulad posledního obchodního dne s datem delistingu.
- Kolik bankrotů má poslední cenu výrazně nad nulou (dopad politiky −100 %).
- Stabilita kategorie `Primary/Secondary Class` v čase (je to snapshot).
- Sanity check: rekonstruovaný cap-weighted index (z `daily`) vs. SPY a `^GSPC`.
- Rozhodnutí, jak naložit s 1 156 delistingy bez důvodu (podle poslední ceny a tržní kapitalizace).
