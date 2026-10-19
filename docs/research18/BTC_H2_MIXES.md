# BTC buy-and-hold + H2 mixy

Verze 0.1 · 2026-10-19 · Popisný post hoc experiment. Výsledek nemění primární H2 ani jeho forward rozhodnutí.

## Otázka

Jak se v období 2020-05-01 až 2026-09-30 změnily výnos a drawdown při kombinaci BTC spot buy-and-hold a H2_btc200 oproti samotnému BTC a samotnému H2?

## Předem dané nastavení

- BTC složka drží BTC spot a nemá poplatek za obchody.
- H2 složka je základní H2_btc200 s původním časovým výstupem po 4 h, 10% vahou na každý altcoinový obchod, 10 bps nákladem na stranu a původním funding modelem.
- Portfolio se denně vrací na cílové váhy: denní výnos je vážený součet denního BTC spot výnosu a denního H2 portfolia.
- Kandidátní váhy BTC/H2: 90/10, 80/20, 70/30, 60/40, 50/50, 40/60, 25/75 a 20/80. Větší H2 váhy zahrnují scénáře s potenciálně vyšším drawdownem.
- Metriky: CAGR, celkový výnos, max drawdown, anualizovaný Sharpe, nejhorší kalendářní rok a roční výnosy. Srovnání obsahuje samostatně BTC B&H a H2.
- Experiment používá stejný historický snapshot a období jako předchozí H2 výstupní běhy. Jde o post hoc výběrově zkreslený výsledek, nikoli o potvrzení strategie.

## Rozhodovací log

| Datum | Změna |
|---|---|
| 2026-10-19 | Na žádost vlastníka předem určena sada osmi mixů BTC/H2 pro popisné srovnání. Výběr mixu z této historie nebude vydáván za OOS důkaz. |
