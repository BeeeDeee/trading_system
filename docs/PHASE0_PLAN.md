# Fáze 0 – plán

Cíl: základ frameworku ověřený na syntetických datech, kde známe správný výsledek (spec §15).
Každý krok = vlastní testy + commit. Reálná data se ve Fázi 0 nepoužívají.

**Stav 2026-09-26: Fáze 0 dokončena, brána splněna** (48 testů: golden testy, test úniku a parita
enginů zelené na syntetických datech).

| Krok | Obsah | Hotovo, když |
|---|---|---|
| 0.1 | Kostra balíčku `qlab`, schéma normalizovaných dat (`bars`), generátor syntetického trhu (splity, dividendy, halty, open = 0, IPO, delisting: akvizice / bankrot) | Testy schématu a generátoru zelené |
| 0.2 | Vektorový engine (spec §7.1–7.2), nákladový model (§7.3), úrok z hotovosti (§7.4) | Golden testy: buy-and-hold = TR řada titulu, cash = T-bill, náklady přesně podle vzorce, delisting podle politiky |
| 0.3 | Ledger engine (kusy, hotovost, příkazy, fixní poplatky, no-trade band) + paritní test | Parita s vektorovým enginem na náhodných strategiích |
| 0.4 | Test úniku budoucnosti (useknutí dat v `t` + perturbace dat po `t`) jako obecný nástroj | Nástroj chytí záměrně „děravou" feature; poctivé features projdou |
| 0.5 | Statistika (PSR/DSR, PBO přes CSCV, stationary block bootstrap), registr pokusů, OOS vault, kostra reportu | Testy proti známým hodnotám a simulacím |

## Výsledky

| Krok | Moduly | Poznámka |
|---|---|---|
| 0.1 | `qlab.data.schema`, `normalize`, `synthetic`, `panel` | Normalizace rekonstruuje skutečný TR index na 1e-9 přes splity, dividendy, halty a delistingy |
| 0.2 | `qlab.engine.vector`, `costs` | ~0,34 s na kandidáta při 7 000 dnech × 1 000 titulů |
| 0.3 | `qlab.engine.ledger` | Parita s vektorovým enginem < 1e-10 na 9 scénářích; test našel a opravil chybu fixních poplatků při nedostatku hotovosti |
| 0.4 | `qlab.validation.leakage`, `qlab.features.basic` | Chytí 4 typy úniku (budoucí řádek, centrované okno, normalizace přes celý vzorek, normalizace koncovou hodnotou) |
| 0.5 | `qlab.validation.stats`, `registry`, `vault`, `metrics` | DSR ověřen Monte Carlem, PBO: šum ≈ 0,5, skutečná schopnost ≈ 0 |

Přesunuto do pozdějších fází (vyžaduje reálné výsledky):
- Hansen SPA test a grafy reportu → Fáze 3.
- Odhad spreadu z OHLC (Abdi–Ranaldo) → Fáze 1 (feature nad reálnými daty; engine už bere
  libovolnou matici nákladů).

## Rozhodnutí přijatá při návrhu

- **Dividendy a spinoffy v obou enginech** se modelují přes total-return ceny (`closeadj` a z ní
  odvozený `open_adj`), tj. automatická reinvestice do stejného titulu. Explicitní hotovostní
  dividendy nejsou ve v1 potřeba (čistý výzkum, bez daní) a zbytečně by komplikovaly spinoffy.
- **Ledger engine** proto počítá v „TR jednotkách" titulu. Dolarové hodnoty pozic, obchodů a poplatků
  jsou skutečné; cena za kus v logu je upravená cena. Pro reporty se dá přepočítat faktorem.
- **Delisting:** v datech je řádek pro první obchodní den po posledním obchodu se stavem `delisted`,
  `ret_co = terminal_ret` a `ret_oc = 0`. Engine pozici na open tohoto dne převede na hotovost.
- **Kalendář** syntetických dat: pracovní dny (Po–Pá). NYSE kalendář přijde ve Fázi 1.
